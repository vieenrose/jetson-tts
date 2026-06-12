"""Train an 8 kHz student vocoder by distilling MeloTTS zh_en's decoder.

GAN training (MPD+MSD) + multi-res STFT + telephony-band mel. PESQ-NB tracked on a held-out
val split; best-by-PESQ checkpoint saved. One student per GPU (run two processes for A/B).

  python -m student.train --arch vocos   --device cuda:1 --out student/runs/vocos
  python -m student.train --arch hifigan --device cuda:0 --out student/runs/hifigan
"""
import argparse, os, time, json, numpy as np, torch
import torch.nn.functional as F
from pesq import pesq as pesq_fn

from .audio_config import TARGET_SR
from .models import build_generator, MultiDiscriminator
from .losses import MultiResSTFTLoss, TelephonyMelLoss, generator_adv_loss, discriminator_loss
from .dataset import make_loaders, load_full


def evaluate(G, g_vec, val, device, max_items=48):
    G.eval()
    pesqs, errs = [], 0
    with torch.no_grad():
        for base, _ in val[:max_items]:
            z, wav = load_full(base)
            z = z.unsqueeze(0).to(device)
            gg = g_vec.unsqueeze(0).to(device)
            y = G(z, gg)[0, 0].cpu().numpy()
            n = min(len(y), len(wav))
            if n < TARGET_SR // 2:        # need >=0.5s for PESQ-NB
                continue
            try:
                pesqs.append(pesq_fn(TARGET_SR, wav[:n], y[:n], "nb"))
            except Exception:
                errs += 1
    G.train()
    return (float(np.mean(pesqs)) if pesqs else 0.0), len(pesqs), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["hifigan", "vocos"])
    ap.add_argument("--root", default="data/pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seg-z", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1000, help="recon-only steps before GAN")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--lambda-mel", type=float, default=30.0)
    ap.add_argument("--lambda-stft", type=float, default=2.0)
    ap.add_argument("--lambda-fm", type=float, default=2.0)
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=2)

    ds, dl, val = make_loaders(args.root, seg_z=args.seg_z, batch=args.batch, workers=args.workers)
    g_vec = ds.g.to(dev)                         # [256,1]
    print(f"[{args.arch}] train items {len(ds)}  val {len(val)}  steps {args.steps}")

    G = build_generator(args.arch).to(dev)
    D = MultiDiscriminator().to(dev)
    nG = sum(p.numel() for p in G.parameters())
    print(f"[{args.arch}] generator params {nG/1e6:.2f}M")
    optG = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.8, 0.99))
    stft_loss = MultiResSTFTLoss().to(dev)
    mel_loss = TelephonyMelLoss().to(dev)

    step, best, t0 = 0, -1.0, time.time()
    data_iter = iter(dl)
    while step < args.steps:
        try:
            z, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            z, y = next(data_iter)
        z, y = z.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        gb = g_vec.unsqueeze(0).expand(z.shape[0], -1, -1)
        y_hat = G(z, gb)
        n = min(y_hat.shape[-1], y.shape[-1])
        y_hat, y = y_hat[..., :n], y[..., :n]
        gan_on = step >= args.warmup

        # ---- D step
        if gan_on:
            outs = D(y, y_hat.detach())
            d_loss = discriminator_loss(outs)
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()
            d_val = d_loss.item()
        else:
            d_val = 0.0

        # ---- G step
        sc, lm = stft_loss(y_hat.squeeze(1), y.squeeze(1))
        mel = mel_loss(y_hat.squeeze(1), y.squeeze(1))
        recon = args.lambda_mel * mel + args.lambda_stft * (sc + lm)
        if gan_on:
            outs = D(y, y_hat)
            adv, fm = generator_adv_loss(outs)
            g_loss = recon + args.lambda_adv * adv + args.lambda_fm * fm
            adv_v, fm_v = adv.item(), fm.item()
        else:
            g_loss = recon
            adv_v = fm_v = 0.0
        optG.zero_grad(set_to_none=True)
        g_loss.backward()
        optG.step()
        step += 1

        if step % args.log_every == 0:
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"[{args.arch}] step {step} g {g_loss.item():.3f} mel {mel.item():.3f} "
                  f"sc {sc.item():.3f} lm {lm.item():.3f} adv {adv_v:.3f} fm {fm_v:.3f} "
                  f"d {d_val:.3f} {sps:.1f} it/s", flush=True)
        if step % args.eval_every == 0:
            pesq_nb, npe, errs = evaluate(G, g_vec, val, dev)
            tag = ""
            if pesq_nb > best:
                best = pesq_nb
                torch.save({"G": G.state_dict(), "arch": args.arch, "step": step,
                            "pesq": pesq_nb}, os.path.join(args.out, "best.pt"))
                tag = " *BEST*"
            torch.save({"G": G.state_dict(), "arch": args.arch, "step": step,
                        "pesq": pesq_nb}, os.path.join(args.out, "last.pt"))
            with open(os.path.join(args.out, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "pesq_nb": pesq_nb, "n": npe,
                                    "mel": float(mel), "best": best}) + "\n")
            print(f"[{args.arch}] EVAL step {step} PESQ-NB {pesq_nb:.3f} (n={npe},err={errs}) best {best:.3f}{tag}", flush=True)

    print(f"[{args.arch}] training done. best PESQ-NB {best:.3f}")


if __name__ == "__main__":
    main()
