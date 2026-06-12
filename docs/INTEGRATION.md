# INTEGRATION.md — installing the 8 kHz distilled MeloTTS on the Jetson Nano

This is a **drop-in replacement** for the existing `vits-melo-tts-zh_en/` model directory. The
stock `sherpa-onnx-offline-tts` binary runs it **unmodified** — the only thing that changes is
the model directory you point at; the audio comes out at 8 kHz instead of 44.1 kHz.

## What's in the package
```
vits-melo-tts-zh_en-8k/
  model.onnx        # melo enc/flow + distilled 8kHz vocoder, metadata sample_rate=8000
  tokens.txt        # identical to the stock model
  lexicon.txt       # identical to the stock model
```
`tokens.txt` and `lexicon.txt` are byte-identical to the stock export — only `model.onnx`
differs (new decoder, `sample_rate=8000`).

## Install (next to the existing model)
```bash
# on the device, beside the current vits-melo-tts-zh_en/
cp -r vits-melo-tts-zh_en-8k /path/to/models/
```

## Invocation — nothing changes but the model dir
```bash
sherpa-onnx-offline-tts \
  --vits-model=vits-melo-tts-zh_en-8k/model.onnx \
  --vits-tokens=vits-melo-tts-zh_en-8k/tokens.txt \
  --vits-lexicon=vits-melo-tts-zh_en-8k/lexicon.txt \
  --vits-dict-dir=<same jieba dict dir as before> \
  --num-threads=4 \
  --output-filename=out.wav \
  "幫您轉接給 Kevin 陳經理,他的分機是 533。"
```
`out.wav` is now **8 kHz mono**. sherpa-onnx reads `sample_rate=8000` from the model metadata, so
the WAV header and any downstream resampler pick it up automatically.

## Why this is safe
- Identical graph inputs/outputs (`x, x_lengths, tones, sid, noise_scale, length_scale,
  noise_scale_w` -> `y`) and dtypes as the official melo export.
- Same front end: tokens/lexicon/jieba unchanged; bert is zeroed inside the graph exactly as the
  stock export does (the binary never computes BERT). The student decoder was distilled on the
  **bert=0** z distribution, matching what the device actually produces.
- fp32, opset 17, no custom ops — runs on the sherpa-onnx-pinned ORT (~1.17) CPU build.
- Voice/prosody/front end are untouched; identical text yields the same speech, just 8 kHz.

## zh-TW (Traditional Chinese) input — supported, with front-end caveats
Traditional Chinese input **works**. Melo's Chinese frontend does **no** Traditional→Simplified
conversion — it feeds characters straight to `pypinyin`, which reads Traditional correctly.
Verified: 15/15 Traditional/Simplified pairs (幫轉經陳鐵颱濕會廣聯…) give identical pinyin; full
zh-TW lines phonemize cleanly (e.g. `分機533` → 五百三十三 via cn2an). The whole training corpus
is Traditional and synthesizes fine. The on-device `lexicon.txt` covers the CJK range incl.
Traditional chars, so device-side lookup behaves the same.

Two caveats, **both in the front end** (token/pinyin generation) — the decoder distillation does
**not** touch them, so they are identical to the stock 44.1 kHz model:

1. **Accent is Mainland Putonghua, not Taiwan Guoyu.** `MeloTTS-Chinese` is trained on Mainland
   Mandarin, so Taiwan-specific readings come out Mainland: 垃圾 `lā jī` (TW lè sè), 和(conj.)
   `hé` (TW hàn), 研究 `yán jiū` (TW yán jiù), 企業 `qǐ yè` (TW qì yè). Characters are right;
   accent/prosody are not Taiwanese.
2. **Polyphone errors exist** (inherent to the frontend, not Traditional-specific): e.g. 銀行 →
   `yín xíng` (should be *háng*), 長度 → `zhǎng dù` (should be *cháng*). Common words mostly fine.

**Implication:** whatever zh-TW quality the stock 44.1 kHz model gives, this 8 kHz drop-in
reproduces **identically** (phonemes/accent/prosody come from the unchanged teacher). Fixing the
accent/polyphones is a separate front-end effort (Taiwan-tuned lexicon, or OpenCC + Taiwan-reading
overrides) — out of scope for the decoder replacement.

See `DEVICE_ACCEPTANCE.md` for the speed/quality numbers to reproduce on the Nano.
