"""LoRA fine-tune of Matcha for Taiwan accent — base FROZEN (preserves correctness), tiny
low-rank adapters on the encoder + duration-predictor 1x1 convs learn the accent/prosody shift.

Decoder stays frozen -> mel distribution stays BASE -> reuse the clean original vocoder (no echo).
Train LIGHT (few steps) — accent appears early; over-training is what hurt correctness.

  python -m matcha8k.lora_finetune --root matcha_eval/tw_combined --out matcha8k/ft_runs/lora --steps 1500
"""
import argparse, os, sys, json, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "third_party/Matcha-TTS"); sys.path.insert(0, ".")
from matcha8k.finetune import build_model, TWSet, collate, wav_to_mel  # reuse
from matcha8k.frontend import MatchaFrontend


class LoRAConv1d(nn.Module):
    """Wrap a frozen Conv1d (kernel=1 typical) with a parallel low-rank path: y = base(x) + B(A(x))*s."""
    def __init__(self, base: nn.Conv1d, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        cin, cout, k = base.in_channels, base.out_channels, base.kernel_size[0]
        self.A = nn.Conv1d(cin, r, k, padding=base.padding, bias=False)
        self.B = nn.Conv1d(r, cout, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)                       # start as no-op
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.s


def inject_lora(model, r=16, alpha=32):
    # collect targets first (do NOT mutate while iterating the module tree)
    targets = []
    for parent_name, parent in model.named_modules():
        if isinstance(parent, LoRAConv1d):
            continue
        for child_name, child in parent.named_children():
            full = f"{parent_name}.{child_name}"
            if isinstance(child, nn.Conv1d) and child.kernel_size[0] == 1 and \
               (".attn_layers." in full or child_name in ("proj",) or ".dp." in full or ".sdp." in full):
                targets.append((parent, child_name, child))
    for parent, child_name, child in targets:
        setattr(parent, child_name, LoRAConv1d(child, r, alpha))
    return len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="matcha_eval/tw_combined")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)        # LoRA tolerates higher LR
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    model = build_model().to(dev)
    mm, ms = float(model.mel_mean), float(model.mel_std)
    for p in model.parameters():
        p.requires_grad = False                              # freeze ALL base
    n_lora = inject_lora(model, args.r, args.alpha)
    model = model.to(dev).train()
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"LoRA injected into {n_lora} convs; trainable {trn/1e6:.3f}M / {tot/1e6:.1f}M ({100*trn/tot:.2f}%)")

    ids_path = args.root + "_ids.json"
    ids_cache = json.load(open(ids_path)) if os.path.exists(ids_path) else {}
    ds = TWSet(args.root, None, mm, ms, ids_cache)
    print(f"dataset {len(ds)} clips")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step = 0; it = iter(dl); t0 = time.time()
    while step < args.steps:
        try: X, xl, Y, yl = next(it)
        except StopIteration: it = iter(dl); X, xl, Y, yl = next(it)
        X, xl, Y, yl = X.to(dev), xl.to(dev), Y.to(dev), yl.to(dev)
        dur, prior, diff, _ = model(x=X, x_lengths=xl, y=Y, y_lengths=yl, spks=None, out_size=128)
        loss = dur + prior + diff
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); step += 1
        if step % args.log_every == 0:
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"step {step} loss {loss.item():.3f} (dur {dur.item():.3f} prior {prior.item():.3f} diff {diff.item():.3f}) {sps:.1f} it/s", flush=True)
        if step % args.save_every == 0:
            save_merged(model, args, os.path.join(args.out, f"lora_merged_step{step}.bin"))
            save_merged(model, args, os.path.join(args.out, "last.bin"))
            print(f"saved merged ckpt @ step {step}", flush=True)
    print("LoRA fine-tune done")


@torch.no_grad()
def save_merged(model, args, path):
    """Fold LoRA deltas into a fresh base model and save its plain Matcha state_dict (model.* prefix)."""
    merged = build_model()                                    # fresh base structure
    msd = merged.state_dict()
    # collect LoRA-merged conv weights keyed by their matcha module path
    for name, mod in model.named_modules():
        if isinstance(mod, LoRAConv1d):
            W = mod.base.weight.data
            A = mod.A.weight.data; B = mod.B.weight.data      # A:[r,cin,k], B:[cout,r,1]
            if W.shape[2] == 1:
                delta = (B[:, :, 0] @ A[:, :, 0]).unsqueeze(-1) * mod.s
            else:
                delta = torch.einsum('or,rik->oik', B[:, :, 0], A) * mod.s
            msd[name + ".weight"].copy_(W + delta)
            if mod.base.bias is not None:
                msd[name + ".bias"].copy_(mod.base.bias.data)
    # non-LoRA params: copy from the (frozen) live model's base
    live = {n: m for n, m in model.named_modules()}
    for k in msd:
        if k in dict(model.state_dict()):
            msd[k].copy_(model.state_dict()[k])
    torch.save({"model." + k: v for k, v in msd.items()}, path)


if __name__ == "__main__":
    main()
