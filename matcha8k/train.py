"""Train the 8 kHz Matcha vocoder (mel -> 8 kHz) by distilling the 16k vocoder's output.

Reuses the melo student's losses + discriminators. PESQ-NB tracked on a held-out split.
  python -m matcha8k.train --device cuda:0 --out matcha8k/runs/vocos8k
"""
import argparse, os, time, json, numpy as np, torch
from pesq import pesq as pesq_fn

from matcha8k.models import VocosMel8k
from matcha8k.dataset import make_loaders, load_full, TARGET_SR
from student.models import MultiDiscriminator
from student.losses import MultiResSTFTLoss, TelephonyMelLoss, generator_adv_loss, discriminator_loss


def evaluate(G, val, device, max_items=48):
    G.eval(); pesqs = 0; vals = []
    with torch.no_grad():
        for base, _ in val[:max_items]:
            mel, wav = load_full(base)
            y = G(mel.unsqueeze(0).to(device))[0, 0].cpu().numpy()
            n = min(len(y), len(wav))
            if n < TARGET_SR // 2:
                continue
            try:
                vals.append(pesq_fn(TARGET_SR, wav[:n], y[:n], "nb"))
            except Exception:
                pass
    G.train()
    return (float(np.mean(vals)) if vals else 0.0), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="matcha_eval/pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seg-frames", type=int, default=48)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--lambda-mel", type=float, default=30.0)
    ap.add_argument("--lambda-stft", type=float, default=2.0)
    ap.add_argument("--lambda-fm", type=float, default=2.0)
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    ap.add_argument("--init-from", default=None, help="warm-start generator G from this .pt checkpoint")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=2)

    ds, dl, val = make_loaders(args.root, seg_frames=args.seg_frames, batch=args.batch, workers=args.workers)
    print(f"[matcha8k] train {len(ds)} val {len(val)} steps {args.steps}")
    G = VocosMel8k().to(dev)
    D = MultiDiscriminator().to(dev)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=dev, weights_only=False)
        G.load_state_dict(ck["G"]); print(f"[matcha8k] warm-started G from {args.init_from} (PESQ {ck.get('pesq')})")
    print(f"[matcha8k] generator params {sum(p.numel() for p in G.parameters())/1e6:.2f}M")
    optG = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.8, 0.99))
    stft_loss = MultiResSTFTLoss().to(dev)
    mel_loss = TelephonyMelLoss().to(dev)

    step, best, t0 = 0, -1.0, time.time()
    it = iter(dl)
    while step < args.steps:
        try:
            mel, y = next(it)
        except StopIteration:
            it = iter(dl); mel, y = next(it)
        mel, y = mel.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        y_hat = G(mel)
        n = min(y_hat.shape[-1], y.shape[-1]); y_hat, y = y_hat[..., :n], y[..., :n]
        gan_on = step >= args.warmup

        if gan_on:
            outs = D(y, y_hat.detach())
            d_loss = discriminator_loss(outs)
            optD.zero_grad(set_to_none=True); d_loss.backward(); optD.step(); d_val = d_loss.item()
        else:
            d_val = 0.0
        sc, lm = stft_loss(y_hat.squeeze(1), y.squeeze(1))
        mel_l = mel_loss(y_hat.squeeze(1), y.squeeze(1))
        recon = args.lambda_mel * mel_l + args.lambda_stft * (sc + lm)
        if gan_on:
            outs = D(y, y_hat); adv, fm = generator_adv_loss(outs)
            g_loss = recon + args.lambda_adv * adv + args.lambda_fm * fm; adv_v, fm_v = adv.item(), fm.item()
        else:
            g_loss = recon; adv_v = fm_v = 0.0
        optG.zero_grad(set_to_none=True); g_loss.backward(); optG.step()
        step += 1

        if step % args.log_every == 0:
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"[matcha8k] step {step} g {g_loss.item():.3f} mel {mel_l.item():.3f} "
                  f"sc {sc.item():.3f} lm {lm.item():.3f} adv {adv_v:.3f} fm {fm_v:.3f} d {d_val:.3f} {sps:.1f} it/s", flush=True)
        if step % args.eval_every == 0:
            pq, npe = evaluate(G, val, dev)
            tag = ""
            if pq > best:
                best = pq; torch.save({"G": G.state_dict(), "step": step, "pesq": pq}, os.path.join(args.out, "best.pt")); tag = " *BEST*"
            torch.save({"G": G.state_dict(), "step": step, "pesq": pq}, os.path.join(args.out, "last.pt"))
            with open(os.path.join(args.out, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "pesq_nb": pq, "n": npe, "mel": float(mel_l.detach()), "best": best}) + "\n")
            print(f"[matcha8k] EVAL step {step} PESQ-NB {pq:.3f} (n={npe}) best {best:.3f}{tag}", flush=True)
    print(f"[matcha8k] done. best PESQ-NB {best:.3f}")


if __name__ == "__main__":
    main()
