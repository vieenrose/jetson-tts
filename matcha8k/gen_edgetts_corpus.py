"""Phase 2 data: generate a single-voice Taiwan-Mandarin corpus with Edge-TTS.

Edge-TTS zh-TW-HsiaoChenNeural (authentic TW accent) reads our corpus text -> clean single-voice
24k mp3 -> 22.05k mono wav + transcript manifest. Resumable, rate-limit tolerant.

NOTE: Edge read-aloud TTS is a ToS gray area for bulk dataset generation — this is for the Phase-2
PROTOTYPE (validate the accent fine-tune pipeline). Swap to a licensed source for the final ship.
"""
import argparse, asyncio, os, json, subprocess, sys

VOICE = "zh-TW-HsiaoChenNeural"


async def synth_one(edge_tts, text, voice, mp3_path):
    await edge_tts.Communicate(text, voice).save(mp3_path)


def to_wav(mp3, wav, sr):
    r = subprocess.run(["ffmpeg", "-y", "-i", mp3, "-ar", str(sr), "-ac", "1", wav],
                       capture_output=True)
    return r.returncode == 0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/text/train.tsv")
    ap.add_argument("--out", default="matcha_eval/tw_edgetts")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    import edge_tts
    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    rows = []
    for l in open(args.corpus, encoding="utf-8"):
        l = l.rstrip("\n")
        if not l:
            continue
        uid, text = (l.split("\t", 1) if "\t" in l else (f"u{len(rows):06d}", l))
        rows.append((uid, text))
    if args.limit:
        rows = rows[: args.limit]
    manifest = os.path.join(args.out, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for l in open(manifest):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    mf = open(manifest, "a", encoding="utf-8")
    sem = asyncio.Semaphore(args.concurrency)
    n_ok = [0]; n_err = [0]

    async def work(uid, text):
        if uid in done:
            return
        wav = os.path.join(args.out, "wav", f"{uid}.wav")
        mp3 = os.path.join(args.out, "wav", f"{uid}.mp3")
        async with sem:
            for attempt in range(3):
                try:
                    await synth_one(edge_tts, text, args.voice, mp3)
                    break
                except Exception as e:
                    if attempt == 2:
                        n_err[0] += 1; print(f"[err {uid}] {e}", file=sys.stderr); return
                    await asyncio.sleep(2 * (attempt + 1))
        if not to_wav(mp3, wav, args.sr):
            n_err[0] += 1; return
        os.remove(mp3)
        mf.write(json.dumps({"id": uid, "text": text, "wav": wav}, ensure_ascii=False) + "\n")
        mf.flush()
        n_ok[0] += 1
        if n_ok[0] % 200 == 0:
            print(f"{n_ok[0]} done, {n_err[0]} err", flush=True)

    # batch to avoid creating 15k tasks at once
    B = 64
    todo = [(u, t) for u, t in rows if u not in done]
    for i in range(0, len(todo), B):
        await asyncio.gather(*(work(u, t) for u, t in todo[i:i + B]))
    mf.close()
    print(f"DONE {n_ok[0]} ok, {n_err[0]} err -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
