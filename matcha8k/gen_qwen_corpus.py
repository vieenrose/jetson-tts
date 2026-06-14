"""Phase 2: generate the TW-accented (code-mixed) corpus with Qwen3-TTS Base voice-cloning.

Clones the edge-tts young-TW-female reference once, then reads our corpus text -> single-voice
TW-accent audio (24k). Output: 24k wav + transcript manifest. Sharded by --shard/--num-shards
(one process per GPU), resumable. These (text, audio) pairs feed the Matcha accent fine-tune.
"""
import argparse, os, json, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

REF_TEXT = "你好,很高興為您服務。今天天氣很好,希望您有美好的一天。我們會盡力協助您解決問題。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-TTS-1.7B-Base")
    ap.add_argument("--ref", default="models/tw_ref/hsiaochen_16k.wav")
    ap.add_argument("--corpus", default="data/text/train.tsv")
    ap.add_argument("--out", default="matcha_eval/tw_qwen_corpus")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()
    sub = os.path.join(args.out, f"shard{args.shard:02d}")
    os.makedirs(sub, exist_ok=True)

    rows = []
    for l in open(args.corpus, encoding="utf-8"):
        l = l.rstrip("\n")
        if not l:
            continue
        uid, text = (l.split("\t", 1) if "\t" in l else (f"u{len(rows):06d}", l))
        rows.append((uid, text))
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]

    manifest = os.path.join(sub, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for l in open(manifest):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    todo = [(u, t) for u, t in rows if u not in done]
    print(f"[shard{args.shard}] {len(todo)} to generate ({len(done)} done)", flush=True)

    model = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device,
                                          dtype=torch.bfloat16, attn_implementation=args.attn)
    prompt = model.create_voice_clone_prompt(ref_audio=args.ref, ref_text=REF_TEXT)
    mf = open(manifest, "a", encoding="utf-8")
    n = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        texts = [t for _, t in chunk]
        try:
            wavs, sr = model.generate_voice_clone(text=texts, language="Chinese",
                                                  voice_clone_prompt=prompt)
        except Exception as e:
            print(f"[batch {i} err] {type(e).__name__}: {e}", flush=True)
            continue
        for (uid, text), w in zip(chunk, wavs):
            p = os.path.join(sub, f"{uid}.wav")
            sf.write(p, w, sr)
            mf.write(json.dumps({"id": uid, "text": text, "wav": p, "sr": int(sr)},
                                ensure_ascii=False) + "\n")
            n += 1
        mf.flush()
        if n % 200 < args.batch:
            print(f"[shard{args.shard}] {n}/{len(todo)} done", flush=True)
    mf.close()
    print(f"[shard{args.shard}] DONE {n} clips -> {sub}", flush=True)


if __name__ == "__main__":
    main()
