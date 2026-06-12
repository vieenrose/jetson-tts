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

See `DEVICE_ACCEPTANCE.md` for the speed/quality numbers to reproduce on the Nano.
