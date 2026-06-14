"""Stage A: capture exact Matcha token-id sentences from sherpa's debug log.

Run with the venv that has sherpa_onnx (e.g. /home/luigi/realtime-bot/venv/bin/python).
Feeds corpus text through sherpa (debug=True), parses every `new sentence: [..]` token-id list
(Chinese via lexicon pinyin tokens, English via espeak IPA), writes them to tokens.jsonl.
Each sentence becomes one vocoder training unit.
"""
import sys, os, re, json, tempfile

M = sys.argv[1] if len(sys.argv) > 1 else "matcha_eval/matcha-icefall-zh-en"
CORPUS = sys.argv[2] if len(sys.argv) > 2 else "data/text/train.tsv"
OUT = sys.argv[3] if len(sys.argv) > 3 else "matcha_eval/tokens.jsonl"
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0

import sherpa_onnx

cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model=f"{M}/model-steps-3.onnx", vocoder=f"{M}/vocos-16khz-univ.onnx",
            lexicon=f"{M}/lexicon.txt", tokens=f"{M}/tokens.txt", data_dir=f"{M}/espeak-ng-data"),
        num_threads=4, provider="cpu", debug=True),
    rule_fsts=f"{M}/date-zh.fst,{M}/number-zh.fst,{M}/phone-zh.fst", max_num_sentences=1)
tts = sherpa_onnx.OfflineTts(cfg)

lines = []
for l in open(CORPUS, encoding="utf-8"):
    l = l.rstrip("\n")
    if not l:
        continue
    lines.append(l.split("\t", 1)[1] if "\t" in l else l)
if LIMIT:
    lines = lines[:LIMIT]

pat = re.compile(r"new sentence: \[([0-9, ]+)\]")
n_sent = 0
with open(OUT, "w", encoding="utf-8") as out:
    for idx, text in enumerate(lines):
        # redirect C-level stderr (sherpa debug logs) to a temp file per call, then parse it
        tf = tempfile.TemporaryFile()
        old = os.dup(2)
        os.dup2(tf.fileno(), 2)
        try:
            tts.generate(text, sid=0, speed=1.0)
        finally:
            os.dup2(old, 2)
            os.close(old)
        tf.seek(0)
        buf = tf.read().decode("utf-8", "ignore")
        tf.close()
        for m in pat.finditer(buf):
            ids = [int(x) for x in m.group(1).split(",")]
            if len(ids) >= 2:
                out.write(json.dumps({"src": idx, "ids": ids}) + "\n")
                n_sent += 1
        if (idx + 1) % 500 == 0:
            out.flush()
            print(f"{idx+1}/{len(lines)} lines, {n_sent} sentences", flush=True)
print(f"DONE {len(lines)} lines -> {n_sent} sentences -> {OUT}")
