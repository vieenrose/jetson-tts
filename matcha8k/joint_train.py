"""Joint acoustic+vocoder co-training for zh-TW accent (breaks the acoustic<->vocoder coupling).

Deployed arch is UNCHANGED (Matcha n_steps=3 + VocosMel8k, fp32, sherpa drop-in = Nano-OK); this only
changes how they're trained: end-to-end on the TW (qwen) audio so the vocoder tracks the accented
decoder's actual CFM mels. Encoder + duration predictor are FROZEN (correct content + timing); the CFM
decoder + vocoder are trained together.

Per step (teacher-forced for alignment):
  encoder(text)[frozen] -> mu_x ; MAS-align vs qwen mel -> mu_y
  mel = decoder(mu_y, n_steps, temp)[grad] ; audio_hat = vocoder(mel*std+mean)[grad]
  loss = vocoder(mel L1 + multi-STFT + GAN vs qwen 8k) + lambda_cfm * CFM_loss
Eval: free-running synthesise -> vocoder -> X-ASR CER (run separately).
"""
import argparse, os, sys, json, glob, math, time, numpy as np, torch, torch.nn as nn, soundfile as sf, librosa
sys.path.insert(0, "third_party/Matcha-TTS"); sys.path.insert(0, ".")
import matcha.utils.monotonic_align as monotonic_align
from matcha.utils.model import sequence_mask
from matcha.utils.audio import mel_spectrogram
from matcha8k.finetune import build_model
from matcha8k.models import VocosMel8k
from student.models import MultiDiscriminator
from student.losses import MultiResSTFTLoss, TelephonyMelLoss, generator_adv_loss, discriminator_loss

MEL = dict(n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000)
HOP8 = 128


class JointSet(torch.utils.data.Dataset):
    def __init__(self, corpus, ids_cache, mm, ms):
        self.items = []
        for mf in sorted(glob.glob(os.path.join(corpus, "shard*", "manifest.jsonl"))):
            for l in open(mf, encoding="utf-8"):
                try: r = json.loads(l)
                except Exception: continue
                if os.path.exists(r["wav"]) and ids_cache.get(r["id"]):
                    self.items.append((r["id"], r["wav"]))
        self.ids_cache, self.mm, self.ms = ids_cache, mm, ms

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        uid, wav = self.items[i]
        ids = self.ids_cache[uid]
        w, sr = sf.read(wav, dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        w16 = librosa.resample(w, orig_sr=sr, target_sr=16000, res_type="soxr_hq") if sr != 16000 else w
        w16 = np.clip(w16, -1, 1)
        mel = mel_spectrogram(torch.from_numpy(w16).unsqueeze(0), MEL["n_fft"], MEL["num_mels"], MEL["sampling_rate"],
                              MEL["hop_size"], MEL["win_size"], MEL["fmin"], MEL["fmax"], center=False)[0]
        meln = (mel - self.mm) / self.ms                            # [80,T]
        w8 = np.clip(librosa.resample(w16, orig_sr=16000, target_sr=8000, res_type="soxr_hq"), -1, 1)
        return torch.tensor(ids, dtype=torch.long), meln, torch.from_numpy(w8)


def collate(batch):
    batch = [b for b in batch if b[1].shape[1] >= 16]
    xs, mels, w8s = zip(*batch)
    xl = torch.tensor([len(x) for x in xs])
    Tm = max(m.shape[1] for m in mels); Tm = ((Tm + 3) // 4) * 4    # U-Net needs %4
    yl = torch.tensor([m.shape[1] for m in mels])
    X = torch.zeros(len(xs), max(xl), dtype=torch.long)
    Y = torch.zeros(len(xs), 80, Tm)
    W = torch.zeros(len(xs), Tm * HOP8)
    for i, (x, m, w) in enumerate(zip(xs, mels, w8s)):
        X[i, :len(x)] = x; Y[i, :, :m.shape[1]] = m
        n = min(len(w), Tm * HOP8); W[i, :n] = w[:n]
    return X, xl, Y, yl, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="matcha_eval/tw_qwen_corpus")
    ap.add_argument("--ids", default="matcha_eval/tw_combined_ids.json")
    ap.add_argument("--voc-init", default="matcha8k/runs/vocos8k/best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-dec", type=float, default=1e-5)
    ap.add_argument("--lr-voc", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.667)
    ap.add_argument("--lambda-cfm", type=float, default=1.0)
    ap.add_argument("--lambda-mel", type=float, default=30.0)
    ap.add_argument("--lambda-stft", type=float, default=2.0)
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    ap.add_argument("--lambda-fm", type=float, default=2.0)
    ap.add_argument("--seg-frames", type=int, default=200, help="mel-frame crop for vocoder+GAN (memory bound)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    M = build_model().to(dev)
    mm, ms = float(M.mel_mean), float(M.mel_std)
    for n, p in M.named_parameters():                              # freeze encoder + duration predictor
        if not n.startswith("decoder."):
            p.requires_grad = False
    M.encoder.eval()                                               # frozen encoder in eval mode
    G = VocosMel8k().to(dev)
    ck = torch.load(args.voc_init, map_location=dev, weights_only=False)
    G.load_state_dict(ck["G"]); print(f"warm-started vocoder from {args.voc_init} (PESQ {ck.get('pesq')})")
    D = MultiDiscriminator().to(dev)
    dec_params = [p for n, p in M.named_parameters() if p.requires_grad]
    print(f"trainable: decoder {sum(p.numel() for p in dec_params)/1e6:.1f}M + vocoder {sum(p.numel() for p in G.parameters())/1e6:.1f}M")

    ids_cache = json.load(open(args.ids))
    ds = JointSet(args.corpus, ids_cache, mm, ms)
    print(f"dataset {len(ds)} clips")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    optG = torch.optim.AdamW(dec_params + list(G.parameters()), lr=args.lr_voc, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(D.parameters(), lr=args.lr_voc, betas=(0.8, 0.99))
    # separate (smaller) LR for the decoder via param groups
    optG = torch.optim.AdamW([{"params": dec_params, "lr": args.lr_dec},
                              {"params": list(G.parameters()), "lr": args.lr_voc}], betas=(0.8, 0.99))
    stft_loss = MultiResSTFTLoss().to(dev); mel_loss = TelephonyMelLoss().to(dev)

    def tf_mel_and_cfm(X, xl, Y, yl):
        with torch.no_grad():
            mu_x, logw, x_mask = M.encoder(X, xl, None)
            y_mask = sequence_mask(yl, Y.shape[-1]).unsqueeze(1).to(x_mask)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            const = -0.5 * math.log(2 * math.pi) * M.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, device=dev)
            ys = torch.matmul(factor.transpose(1, 2), Y ** 2)
            ymu = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), Y)
            msq = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            attn = monotonic_align.maximum_path(ys - ymu + msq + const, attn_mask.squeeze(1))
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
        mel = M.decoder(mu_y, y_mask, args.n_steps, temperature=args.temp)     # [B,80,T] grad
        cfm, _ = M.decoder.compute_loss(x1=Y, mask=y_mask, mu=mu_y)            # anchor decoder to real mels
        return mel, y_mask, cfm

    step = 0; it = iter(dl); t0 = time.time()
    while step < args.steps:
        try: X, xl, Y, yl, W = next(it)
        except StopIteration: it = iter(dl); X, xl, Y, yl, W = next(it)
        X, xl, Y, yl, W = X.to(dev), xl.to(dev), Y.to(dev), yl.to(dev), W.to(dev)
        mel, y_mask, cfm = tf_mel_and_cfm(X, xl, Y, yl)
        # crop a fixed mel window for the vocoder+GAN (bounds memory; standard vocoder-training trick)
        Tfull = mel.shape[-1]; seg = min(args.seg_frames, Tfull)
        s0 = int(torch.randint(0, Tfull - seg + 1, (1,)).item())
        melc = mel[:, :, s0:s0 + seg]
        wc = W[:, s0 * HOP8:(s0 + seg) * HOP8]
        audio = G(melc * ms + mm).squeeze(1)                          # [B,seg*HOP8]
        n = min(audio.shape[-1], wc.shape[-1]); audio, w = audio[..., :n], wc[..., :n]
        gan_on = step >= args.warmup
        if gan_on:
            outs = D(w.unsqueeze(1), audio.detach().unsqueeze(1))
            d_loss = discriminator_loss(outs)
            optD.zero_grad(set_to_none=True); d_loss.backward(); optD.step(); d_val = d_loss.item()
        else:
            d_val = 0.0
        sc, lm = stft_loss(audio, w); mel_l = mel_loss(audio, w)
        g = args.lambda_mel * mel_l + args.lambda_stft * (sc + lm) + args.lambda_cfm * cfm
        if gan_on:
            outs = D(w.unsqueeze(1), audio.unsqueeze(1)); adv, fm = generator_adv_loss(outs)
            g = g + args.lambda_adv * adv + args.lambda_fm * fm; adv_v, fm_v = adv.item(), fm.item()
        else:
            adv_v = fm_v = 0.0
        optG.zero_grad(set_to_none=True); g.backward()
        torch.nn.utils.clip_grad_norm_(dec_params + list(G.parameters()), 5.0); optG.step()
        step += 1
        if step % args.log_every == 0:
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"step {step} g {g.item():.3f} mel {mel_l.item():.3f} cfm {cfm.item():.3f} "
                  f"adv {adv_v:.3f} d {d_val:.3f} {sps:.1f} it/s", flush=True)
        if step % args.save_every == 0:
            torch.save({"model." + k: v for k, v in M.state_dict().items()}, os.path.join(args.out, f"acoustic_step{step}.bin"))
            torch.save({"G": G.state_dict(), "step": step}, os.path.join(args.out, f"vocoder_step{step}.pt"))
            torch.save({"model." + k: v for k, v in M.state_dict().items()}, os.path.join(args.out, "acoustic_last.bin"))
            torch.save({"G": G.state_dict(), "step": step}, os.path.join(args.out, "vocoder_last.pt"))
            print(f"saved @ {step}", flush=True)
    print("joint training done")


if __name__ == "__main__":
    main()
