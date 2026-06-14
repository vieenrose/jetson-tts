"""ASR intelligibility gate for zh/en TTS output (faster-whisper large-v3).

Transcribes TTS wavs and scores against the reference text:
  - Chinese CER (character error rate over Han chars)
  - English word recall (fraction of reference English words present in the hypothesis)
These objectively measure intelligibility + English retention across systems/checkpoints.
"""
import argparse, os, re, glob, json, sys
import numpy as np


def han(s):
    return "".join(c for c in s if "一" <= c <= "鿿")


def en_words(s):
    return re.findall(r"[A-Za-z][A-Za-z'']*", s)


def cer(ref, hyp):
    ref, hyp = han(ref), han(hyp)
    if not ref:
        return None
    # Levenshtein
    import numpy as np
    d = np.zeros((len(ref) + 1, len(hyp) + 1), int)
    d[:, 0] = np.arange(len(ref) + 1); d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    return d[len(ref), len(hyp)] / len(ref)


def en_recall(ref, hyp):
    rw = [w.lower() for w in en_words(ref)]
    if not rw:
        return None
    hw = set(w.lower() for w in en_words(hyp))
    return sum(w in hw for w in rw) / len(rw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="jsonl with {wav,text} or {id,text}+--wav-dir")
    ap.add_argument("--pairs", nargs="*", help="wav=text pairs or a dir glob via --glob")
    ap.add_argument("--glob", help="glob of wavs; text from sibling .txt or --texts")
    ap.add_argument("--texts", help="tsv id<TAB>text for --glob mode (basename match)")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device, compute_type="float16")

    items = []  # (wav, ref_text, tag)
    if args.manifest:
        for l in open(args.manifest, encoding="utf-8"):
            r = json.loads(l); items.append((r["wav"], r["text"], r.get("id", "")))
    elif args.glob:
        texts = {}
        if args.texts:
            for l in open(args.texts, encoding="utf-8"):
                if "\t" in l: k, v = l.rstrip("\n").split("\t", 1); texts[k] = v
        for w in sorted(glob.glob(args.glob)):
            base = os.path.basename(w)
            ref = texts.get(base, "")
            items.append((w, ref, base))

    def transcribe(w):
        segs, _ = model.transcribe(w, language=None, beam_size=5)
        return "".join(s.text for s in segs).strip()

    for wav, ref, tag in items:
        hyp = transcribe(wav)
        c = cer(ref, hyp); er = en_recall(ref, hyp)
        print(json.dumps({"tag": tag, "ref": ref, "hyp": hyp,
                          "cer": round(c, 3) if c is not None else None,
                          "en_recall": round(er, 3) if er is not None else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
