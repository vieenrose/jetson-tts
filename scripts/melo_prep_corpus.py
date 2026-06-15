"""Prepare the TW corpus for MeloTTS fine-tuning: resample to 44.1 kHz + write metadata.list
(path|ZH|ZH|text). Then run melo's preprocess_text.py for g2p -> train.list/val.list.

  python scripts/melo_prep_corpus.py --corpus matcha_eval/tw_qwen_corpus --out melo_ft --limit 0
  cd third_party/MeloTTS/melo && python preprocess_text.py \
      --metadata <abs>/melo_ft/metadata.list --train-path <abs>/melo_ft/train.list \
      --val-path <abs>/melo_ft/val.list --config_path <abs>/melo_ft/config.json
"""
import argparse, os, json, glob, numpy as np, soundfile as sf, librosa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="matcha_eval/tw_qwen_corpus")
    ap.add_argument("--out", default="melo_ft")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "wav44"), exist_ok=True)
    rows = []
    for mf in sorted(glob.glob(os.path.join(args.corpus, "shard*", "manifest.jsonl"))):
        for l in open(mf, encoding="utf-8"):
            try: rows.append(json.loads(l))
            except Exception: pass
    if args.limit:
        rows = rows[: args.limit]
    out = open(os.path.join(args.out, "metadata.list"), "w", encoding="utf-8"); n = 0
    for r in rows:
        if not os.path.exists(r["wav"]):
            continue
        w, sr = sf.read(r["wav"], dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        w44 = librosa.resample(w, orig_sr=sr, target_sr=44100, res_type="soxr_hq") if sr != 44100 else w
        p = os.path.abspath(os.path.join(args.out, "wav44", f"{r['id']}.wav"))
        sf.write(p, np.clip(w44, -1, 1), 44100)
        out.write(f"{p}|ZH|ZH|{r['text'].strip()}\n"); n += 1
        if n % 1000 == 0: out.flush(); print(f"  {n}", flush=True)
    out.close()
    print(f"wrote {n} clips @44.1k -> {args.out}/metadata.list")


if __name__ == "__main__":
    main()
