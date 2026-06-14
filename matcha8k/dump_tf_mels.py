"""Teacher-forced mel dump for vocoder co-training (fixes acoustic<->vocoder coupling).

For each corpus utterance: encoder(text) -> MAS align against the REAL Qwen mel -> aligned mu_y
at the ground-truth length -> CFM decoder samples a mel of that length. This TTS mel is what the
acoustic ACTUALLY produces, and it aligns frame-by-frame with the real 8 kHz audio -> train the
vocoder on (TTS_mel, real_8k_audio). Standard HiFi-GAN-style vocoder fine-tuning on TTS features.
"""
import argparse, os, sys, json, glob, math, numpy as np, torch, soundfile as sf, librosa
sys.path.insert(0, "third_party/Matcha-TTS"); sys.path.insert(0, ".")
import matcha.utils.monotonic_align as monotonic_align
from matcha.utils.model import sequence_mask, generate_path
from matcha.utils.audio import mel_spectrogram
from matcha8k.finetune import build_model
from matcha8k.frontend import MatchaFrontend

MEL = dict(n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000)


@torch.no_grad()
def tf_mel(model, x, x_len, y, y_len, n_steps=10, temperature=0.0):
    """Replicate Matcha.forward up to aligned mu_y, then sample a GT-length mel from the CFM decoder."""
    mu_x, logw, x_mask = model.encoder(x, x_len, None)
    y_mask = sequence_mask(y_len, y.shape[-1]).unsqueeze(1).to(x_mask)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    const = -0.5 * math.log(2 * math.pi) * model.n_feats
    factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
    y_sq = torch.matmul(factor.transpose(1, 2), y ** 2)
    y_mu = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
    mu_sq = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
    log_prior = y_sq - y_mu + mu_sq + const
    attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
    mel = model.decoder(mu_y, y_mask, n_steps, temperature=temperature)   # [B,80,T] normalized (temp=0 -> deterministic)
    return mel * model.mel_std + model.mel_mean                     # denormalize -> raw log-mel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", default="matcha_eval/tw_qwen_corpus")
    ap.add_argument("--ids", default="matcha_eval/tw_combined_ids.json")
    ap.add_argument("--out", default="matcha_eval/tf_vocoder_pairs")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.667, help="CFM temperature (match inference noise_scale)")
    ap.add_argument("--n-steps", type=int, default=3, help="CFM ODE steps (match inference / steps-N export)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = build_model()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True); model = model.to(dev).eval()
    mm, ms = float(model.mel_mean), float(model.mel_std)
    ids_cache = json.load(open(args.ids)) if os.path.exists(args.ids) else {}
    fe = MatchaFrontend() if not ids_cache else None
    sub = os.path.join(args.out, f"shard{args.shard:02d}"); os.makedirs(sub, exist_ok=True)
    rows = []
    for mf in sorted(glob.glob(os.path.join(args.corpus, "shard*", "manifest.jsonl"))):
        for l in open(mf, encoding="utf-8"):
            try: r = json.loads(l)
            except Exception: continue
            rows.append(r)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    mfp = open(os.path.join(sub, "manifest.jsonl"), "a"); n = 0
    for r in rows:
        uid = r["id"]; wav = r["wav"]
        if not os.path.exists(wav): continue
        ids = ids_cache.get(uid) or (fe.text_to_ids(r["text"]) if fe else None)
        if not ids or len(ids) < 3: continue
        w, sr = sf.read(wav, dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        w16 = librosa.resample(w, orig_sr=sr, target_sr=16000, res_type="soxr_hq") if sr != 16000 else w
        w16 = np.clip(w16, -1, 1)
        y = mel_spectrogram(torch.from_numpy(w16).unsqueeze(0), MEL["n_fft"], MEL["num_mels"],
                            MEL["sampling_rate"], MEL["hop_size"], MEL["win_size"], MEL["fmin"], MEL["fmax"],
                            center=False)
        yn = ((y - mm) / ms).to(dev)
        T = yn.shape[-1]; Tp = ((T + 3) // 4) * 4              # CFM U-Net needs len % 4 == 0
        if Tp != T:
            yn = torch.nn.functional.pad(yn, (0, Tp - T))
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        try:
            mel = tf_mel(model, x, torch.tensor([x.shape[1]], device=dev), yn, torch.tensor([Tp], device=dev),
                         n_steps=args.n_steps, temperature=args.temp)[0].cpu().numpy()
            mel = mel[:, :T]                                    # trim pad back to real audio length
        except Exception as e:
            print(f"[skip {uid}] {e}", file=sys.stderr); continue
        w8 = np.clip(librosa.resample(w16, orig_sr=16000, target_sr=8000, res_type="soxr_hq"), -1, 1)
        if mel.shape[1] < 16: continue
        np.save(os.path.join(sub, f"{uid}.mel.npy"), mel.astype(np.float16))
        sf.write(os.path.join(sub, f"{uid}.wav"), w8, 8000, subtype="PCM_16")
        mfp.write(json.dumps({"id": uid, "mel_frames": int(mel.shape[1]), "n8k": int(len(w8))}) + "\n")
        n += 1
        if n % 500 == 0: mfp.flush(); print(f"[shard{args.shard}] {n}", flush=True)
    mfp.close(); print(f"[shard{args.shard}] DONE {n} TF pairs -> {sub}")


if __name__ == "__main__":
    main()
