"""Build (mel[80,T] @62.5Hz, 8 kHz wav) pairs from the Qwen TW corpus for vocoder training.

mel = matcha mel (center=False, n_fft1024/hop256/win1024/fmin0/fmax8000) of the qwen wav @16k.
target = qwen wav resampled to 8k. Lengths align with VocosMel8k ISTFTHead ((T-1)*128).
Reuses the matcha8k/ dataset+train (point --root here). Sharded, resumable.
"""
import argparse, os, sys, json, glob, numpy as np, torch, soundfile as sf, librosa
sys.path.insert(0, "third_party/Matcha-TTS")
from matcha.utils.audio import mel_spectrogram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="matcha_eval/tw_qwen_corpus")
    ap.add_argument("--out", default="matcha_eval/tw_vocoder_pairs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    sub = os.path.join(args.out, f"shard{args.shard:02d}")
    os.makedirs(sub, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(args.corpus, "shard*", "*.wav")))
    wavs = [w for i, w in enumerate(wavs) if i % args.num_shards == args.shard]
    manifest = os.path.join(sub, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for l in open(manifest):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    mf = open(manifest, "a", encoding="utf-8")
    n = 0
    for wpath in wavs:
        uid = os.path.splitext(os.path.basename(wpath))[0]
        if uid in done:
            continue
        w, sr = sf.read(wpath, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        w16 = librosa.resample(w, orig_sr=sr, target_sr=16000, res_type="soxr_hq") if sr != 16000 else w
        w16 = np.clip(w16, -1, 1)
        mel = mel_spectrogram(torch.from_numpy(w16).unsqueeze(0), 1024, 80, 16000, 256, 1024, 0, 8000,
                              center=False)[0].numpy()                      # [80,T]
        w8 = np.clip(librosa.resample(w16, orig_sr=16000, target_sr=8000, res_type="soxr_hq"), -1, 1)
        if mel.shape[1] < 16:
            continue
        np.save(os.path.join(sub, f"{uid}.mel.npy"), mel.astype(np.float16))
        sf.write(os.path.join(sub, f"{uid}.wav"), w8, 8000, subtype="PCM_16")
        mf.write(json.dumps({"id": uid, "mel_frames": int(mel.shape[1]), "n8k": int(len(w8))}) + "\n")
        n += 1
        if n % 500 == 0:
            mf.flush(); print(f"[shard{args.shard}] {n} pairs", flush=True)
    mf.close()
    print(f"[shard{args.shard}] DONE {n} pairs -> {sub}", flush=True)


if __name__ == "__main__":
    main()
