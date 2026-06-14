#!/usr/bin/env python3
"""Apply Taiwan-Mandarin reading overrides to a sherpa-onnx lexicon.txt (in place copy).

- char_overrides.tsv: systematic single-char CN->TW reading swaps. Applied to single-char entries
  AND propagated into multi-char word entries that use the standard CN reading for that char
  (1:1 char<->syllable alignment only, to avoid clobbering polyphone-disambiguated words).
- word_overrides.tsv: irregular whole-word TW readings (override/insert the word entry verbatim).

Front-end only; no model retrain. Output is a drop-in lexicon.txt for the same model dir.
"""
import argparse, os


def load_tsv(path):
    rows = []
    for l in open(path, encoding="utf-8"):
        l = l.rstrip("\n")
        if not l or l.startswith("#"):
            continue
        rows.append(l.split("\t"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--char-overrides", default="data/tw_readings/char_overrides.tsv")
    ap.add_argument("--word-overrides", default="data/tw_readings/word_overrides.tsv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    char_ov = {r[0]: (r[1], r[2]) for r in load_tsv(args.char_overrides)}   # char -> (cn, tw)
    word_ov = {r[0]: r[1] for r in load_tsv(args.word_overrides)}           # word -> "tw1 tw2"

    n_char_single, n_char_word, n_word, n_lines = 0, 0, 0, 0
    seen_words = set()
    out = open(args.out, "w", encoding="utf-8")
    for line in open(args.lexicon, encoding="utf-8"):
        parts = line.split()
        if not parts:
            out.write(line); continue
        w, syl = parts[0], parts[1:]
        seen_words.add(w)
        if w in word_ov:                                  # whole-word override
            out.write(f"{w} {word_ov[w]}\n"); n_word += 1; n_lines += 1; continue
        if len(w) == 1 and w in char_ov:                  # single-char entry
            cn, tw = char_ov[w]
            new = [tw if s == cn else s for s in syl]
            if new != syl:
                n_char_single += 1
            out.write(f"{w} {' '.join(new)}\n"); n_lines += 1; continue
        if len(w) == len(syl) and any(c in char_ov for c in w):   # 1:1 word, propagate char rule
            new = list(syl); changed = False
            for i, c in enumerate(w):
                if c in char_ov and syl[i] == char_ov[c][0]:
                    new[i] = char_ov[c][1]; changed = True
            if changed:
                n_char_word += 1
            out.write(f"{w} {' '.join(new)}\n"); n_lines += 1; continue
        out.write(line); n_lines += 1
    # insert any word-overrides that weren't already present
    for w, tw in word_ov.items():
        if w not in seen_words:
            out.write(f"{w} {tw}\n"); n_word += 1
    out.close()
    print(f"wrote {args.out}: {n_lines} lines | char-single {n_char_single}, char-in-words {n_char_word}, "
          f"word-overrides {n_word}")


if __name__ == "__main__":
    main()
