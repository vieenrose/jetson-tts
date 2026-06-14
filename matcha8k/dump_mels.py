"""Stage B: (mel, 8kHz target) pairs from Matcha acoustic+vocoder, given token sentences.

For each token-id sentence (from dump_tokens.py): acoustic ONNX -> mel[80,T]; vocos-16k ONNX ->
mag/x/y; iSTFT (n_fft1024/hop256/win1024, hann, center) -> 16k wav; soxr -> 8k target.
mel and 8k-wav are paired (same forward). Saves mel f16 + 8k wav + manifest.
Resumable; shardable by --shard/--num-shards.
"""
import argparse, os, sys, json, numpy as np, onnxruntime as ort, librosa, soundfile as sf

TEACHER_SR = 16000


def istft16(out):
    mag, x, y = out
    S = (mag * x + 1j * mag * y)[0]               # [513, T]
    return librosa.istft(S, n_fft=1024, hop_length=256, win_length=1024,
                         window="hann", center=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="matcha_eval/matcha-icefall-zh-en")
    ap.add_argument("--tokens", default="matcha_eval/tokens.jsonl")
    ap.add_argument("--out", default="matcha_eval/pairs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sub = os.path.join(args.out, f"shard{args.shard:02d}")
    os.makedirs(sub, exist_ok=True)

    so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
    am = ort.InferenceSession(f"{args.model_dir}/model-steps-3.onnx", so, providers=["CPUExecutionProvider"])
    voc = ort.InferenceSession(f"{args.model_dir}/vocos-16khz-univ.onnx", so, providers=["CPUExecutionProvider"])

    rows = [json.loads(l) for l in open(args.tokens, encoding="utf-8")]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    manifest = os.path.join(sub, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for l in open(manifest):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    mf = open(manifest, "a", encoding="utf-8")
    n_ok, frames_tot = 0, 0
    for k, r in enumerate(rows):
        uid = f"s{args.shard:02d}_{k:06d}"
        if uid in done:
            continue
        ids = r["ids"]
        if len(ids) < 3:
            continue
        x = np.array([ids], np.int64)
        try:
            mel = am.run(None, {"x": x, "x_length": np.array([len(ids)], np.int64),
                                "noise_scale": np.array([args.noise_scale], np.float32),
                                "length_scale": np.array([1.0], np.float32)})[0]   # [1,80,T]
            out = voc.run(None, {"mels": mel.astype(np.float32)})
            wav16 = istft16(out)
        except Exception as e:
            print(f"[skip {uid}] {type(e).__name__}: {e}", file=sys.stderr); continue
        wav8 = np.clip(librosa.resample(wav16.astype(np.float32), orig_sr=TEACHER_SR,
                                        target_sr=8000, res_type="soxr_hq"), -1, 1)
        T = mel.shape[2]
        np.save(os.path.join(sub, f"{uid}.mel.npy"), mel[0].astype(np.float16))   # [80,T]
        sf.write(os.path.join(sub, f"{uid}.wav"), wav8, 8000, subtype="PCM_16")
        mf.write(json.dumps({"id": uid, "mel_frames": int(T), "n8k": int(len(wav8))}) + "\n")
        n_ok += 1; frames_tot += T
        if n_ok % 500 == 0:
            mf.flush(); print(f"[{sub}] {n_ok} pairs, ~{frames_tot/62.5/3600:.2f}h", flush=True)
    mf.close()
    print(f"[{sub}] DONE {n_ok} pairs, ~{frames_tot/62.5/3600:.3f}h")


if __name__ == "__main__":
    main()
