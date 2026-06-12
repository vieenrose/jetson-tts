#!/usr/bin/env python3
"""Quality gate: student vs teacher (downsampled-to-8k) on a held-out eval set.

Metrics (clean 8 kHz and through the simulated G.711 mu-law channel):
  - PESQ-NB  (the telephony metric)
  - MCD      (mel-cepstral distortion, dB; lower=better)
Saves student/teacher WAVs (clean + G.711) for the ear-test, including code-mixed-name lines.
The teacher target is melo's 44.1k output soxr-resampled to 8k (== the training target).
"""
import argparse, os, sys, json, glob, numpy as np, torch, soundfile as sf
import librosa
from pesq import pesq as pesq_fn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.export_dec_onnx import load_student
from scripts.g711 import roundtrip as g711_roundtrip
from student.dataset import scan_items, load_g, load_full
from student.audio_config import TARGET_SR


def mcd(ref, deg, sr=TARGET_SR, n_mfcc=25):
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    cr = librosa.feature.mfcc(y=ref.astype(np.float32), sr=sr, n_mfcc=n_mfcc, n_fft=256, hop_length=64)
    cd = librosa.feature.mfcc(y=deg.astype(np.float32), sr=sr, n_mfcc=n_mfcc, n_fft=256, hop_length=64)
    m = min(cr.shape[1], cd.shape[1])
    diff = cr[1:, :m] - cd[1:, :m]               # drop c0 (energy)
    dist = np.sqrt((diff ** 2).sum(0))
    return float((10.0 / np.log(10)) * np.sqrt(2) * dist.mean())


def safe_pesq(ref, deg):
    n = min(len(ref), len(deg))
    if n < TARGET_SR // 2:
        return None
    try:
        return float(pesq_fn(TARGET_SR, ref[:n], deg[:n], "nb"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/pairs")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--wav-out", default="eval/wavs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    G, ck = load_student(args.ckpt, device=args.device)
    g_vec = load_g(args.root).to(args.device)
    items = sorted(scan_items(args.root), key=lambda x: x[0])
    items = items[:: max(1, len(items) // args.n)][:args.n]
    os.makedirs(args.wav_out, exist_ok=True)

    rows = []
    for k, (base, _) in enumerate(items):
        z, ref = load_full(base)
        with torch.no_grad():
            y = G(z.unsqueeze(0).to(args.device), g_vec.unsqueeze(0).to(args.device))[0, 0].cpu().numpy()
        n = min(len(y), len(ref))
        y, ref = y[:n], ref[:n]
        ref_t = g711_roundtrip(ref); y_t = g711_roundtrip(y)
        rows.append({
            "pesq": safe_pesq(ref, y), "mcd": mcd(ref, y),
            "pesq_g711": safe_pesq(ref_t, y_t), "mcd_g711": mcd(ref_t, y_t),
        })
        if k < 12:                                # save first 12 for ear-test
            uid = os.path.basename(base)
            sf.write(f"{args.wav_out}/{uid}.teacher.wav", ref, TARGET_SR, subtype="PCM_16")
            sf.write(f"{args.wav_out}/{uid}.student.wav", y, TARGET_SR, subtype="PCM_16")
            sf.write(f"{args.wav_out}/{uid}.student.g711.wav", y_t, TARGET_SR, subtype="PCM_16")

    def agg(key):
        v = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(v)) if v else float("nan")
    summary = {"ckpt": args.ckpt, "arch": ck["arch"], "step": ck.get("step"), "n": len(rows),
               "PESQ_NB": agg("pesq"), "PESQ_NB_g711": agg("pesq_g711"),
               "MCD_dB": agg("mcd"), "MCD_dB_g711": agg("mcd_g711")}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.wav_out, "..", f"quality_{ck['arch']}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
