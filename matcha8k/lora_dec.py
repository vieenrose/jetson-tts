"""Research-backed accent PEFT for Matcha (zh-CN -> zh-TW), per:
- 2305.11320 (zh-CN->zh-TW): freeze encoder+variance+embeddings, adapt ONLY the decoder, ~1% params.
- 2509.22727 (flow-matching/F5 DiaMoE): LoRA on attention Q/V only, tiny alpha, frozen backbone.
- 2305.04816 / 2401.03538: freezing the text encoder prevents pronunciation/content forgetting.

So: FREEZE encoder, duration predictor, embeddings, all decoder convs/resnets. LoRA ONLY the CFM
decoder's transformer attention to_q/to_v (nn.Linear). Gentle (alpha≈rank), low LR. Duration
predictor stays FROZEN (we saw its LoRA diverge -> broken timing).
"""
import argparse, os, sys, json, time, math, torch, torch.nn as nn
sys.path.insert(0, "third_party/Matcha-TTS"); sys.path.insert(0, ".")
from matcha8k.finetune import build_model, TWSet, collate


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.s


def inject(model, r, alpha, targets=("to_q", "to_v")):
    found = []
    for pname, parent in model.decoder.named_modules():
        for cname, child in parent.named_children():
            if isinstance(child, nn.Linear) and cname in targets:
                found.append((parent, cname, child))
    for parent, cname, child in found:
        setattr(parent, cname, LoRALinear(child, r, alpha))
    return len(found)


@torch.no_grad()
def save_merged(model, path):
    merged = build_model(); msd = merged.state_dict()
    live = dict(model.state_dict())
    # copy all non-LoRA params straight through
    for k in msd:
        if k in live:
            msd[k].copy_(live[k])
    # fold LoRA deltas into the corresponding decoder Linear weights
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            delta = (mod.B.weight @ mod.A.weight) * mod.s     # [out,in]
            msd[name + ".weight"].copy_(mod.base.weight + delta)
            if mod.base.bias is not None:
                msd[name + ".bias"].copy_(mod.base.bias)
    torch.save({"model." + k: v for k, v in msd.items()}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="matcha_eval/tw_combined")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    model = build_model().to(dev)
    mm, ms = float(model.mel_mean), float(model.mel_std)
    for p in model.parameters():
        p.requires_grad = False
    n = inject(model, args.r, args.alpha)
    model = model.to(dev).train()
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"LoRA on {n} decoder-attn Linears (to_q/to_v); trainable {trn/1e6:.3f}M/{tot/1e6:.1f}M ({100*trn/tot:.2f}%)")

    ids = args.root + "_ids.json"
    ids_cache = json.load(open(ids)) if os.path.exists(ids) else {}
    ds = TWSet(args.root, None, mm, ms, ids_cache)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step = 0; it = iter(dl); t0 = time.time()
    while step < args.steps:
        try: X, xl, Y, yl = next(it)
        except StopIteration: it = iter(dl); X, xl, Y, yl = next(it)
        X, xl, Y, yl = X.to(dev), xl.to(dev), Y.to(dev), yl.to(dev)
        dur, prior, diff, _ = model(x=X, x_lengths=xl, y=Y, y_lengths=yl, spks=None, out_size=128)
        loss = diff + prior        # duration predictor frozen -> dur_loss not optimized
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); step += 1
        if step % args.log_every == 0:
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"step {step} diff {diff.item():.3f} prior {prior.item():.3f} {sps:.1f} it/s", flush=True)
        if step % args.save_every == 0:
            save_merged(model, os.path.join(args.out, f"merged_step{step}.bin"))
            save_merged(model, os.path.join(args.out, "last.bin"))
            print(f"saved @ {step}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
