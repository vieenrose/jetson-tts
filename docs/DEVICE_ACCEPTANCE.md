# DEVICE_ACCEPTANCE.md — vits-melo-tts-zh_en-8k (distilled 8 kHz vocoder)

Blind-handoff checklist for the Jetson Nano gen1 (4× Cortex-A57, ONNX Runtime CPU via
sherpa-onnx). All numbers below are **host x86 gates** measured on the training box; the device
side reproduces them on the A57 and confirms against the predicted bounds.

Host: AMD Ryzen 9 9950X3D, onnxruntime **1.17.0** CPU, fp32, opset 17.
Predicted A57 = host × **6–8** (the device project's measured host→A57 factor).

## 1. Exact CLI (drop-in — nothing changes but the model dir)
```bash
sherpa-onnx-offline-tts \
  --vits-model=vits-melo-tts-zh_en-8k/model.onnx \
  --vits-tokens=vits-melo-tts-zh_en-8k/tokens.txt \
  --vits-lexicon=vits-melo-tts-zh_en-8k/lexicon.txt \
  --vits-dict-dir=<same jieba dict dir as the stock model> \
  --num-threads=4 \
  --output-filename=out.wav \
  "幫您轉接給 Kevin 陳經理,他的分機是 533。"
```
`out.wav` is **8 kHz mono** (sherpa reads `sample_rate=8000` from model metadata).
Verified end-to-end through the actual `sherpa_onnx` OfflineTts engine (same C++ core as the
device binary): model loads, reports sample_rate 8000, generates the code-mixed lines.

Speaker note: this is a single-speaker model. For MeloTTS, sherpa hardcodes the graph speaker to
`metadata.speaker_id` (=1) regardless of the `--sid` you pass, so the voice is exactly melo's
own — the distillation target. A "use sid=0" info message is harmless.

## 2. Speed gate — full model RTF (host x86 ORT CPU), lower is better
| threads | RTF | predicted A57 (×6–8) |
|--------:|-----|----------------------|
| 1 | 0.025 | 0.15–0.20 |
| 2 | 0.019 | 0.11–0.15 |
| 4 | **0.013** | **0.08–0.10** |

- Gate: host 4-thread full-model RTF ≤ 0.05 → **PASS (0.013)**. Predicted A57 RTF ≈ 0.08–0.10,
  well under the ≤0.3–0.4 target (was 2–3 with the stock 44.1 kHz model → unusable).
- Per-node profile: **decoder = 12.6 %** of full-model compute → PASS (gate ≤ 25 %). The original
  bottleneck (88 % in /dec at 44.1 kHz) is gone; enc/flow now dominate.
- Reproduce on device: `python scripts/bench_full_onnx.py --onnx model.onnx` (RTF + dec share).

## 3. Quality gate — student vs teacher (melo 44.1 kHz → 8 kHz), 120 held-out code-mixed utts
| metric | clean 8 kHz | through G.711 μ-law |
|--------|------------:|--------------------:|
| PESQ-NB (↑) | **2.90** | 2.89 |
| MCD-DTW dB (↓, 13 MCEP) | **10.8** | 10.8 |

- PESQ-NB ~2.9 on the telephony metric; the simulated G.711 channel barely changes it (−0.01),
  confirming the 8 kHz output sits cleanly inside the phone band.
- Reproduce: `python scripts/eval_quality.py --ckpt <best.pt> --root data/eval_pairs`.

## 4. Ear-test set (code-mixed names, pre-rendered through the G.711 channel)
`eval/wavs_vocos/*.{teacher,student,student.g711}.wav` — teacher (8 kHz reference), student
(clean 8 kHz), and student through G.711 μ-law roundtrip, for the product's defining lines:
English names in Chinese (Kevin 陳經理 / Amy 林 / Jason 王襄理), tech terms (Wi-Fi, Server, Zoom),
extensions and phone numbers. `eval/sherpa_proof/sherpa_0*.wav` are the same lines rendered by the
**stock sherpa-onnx engine** (the true drop-in path).
Check: code-mixed names stay crisp and natural; no metallic buzz/clicks at clause boundaries;
extensions/digits intelligible.

## 5. Reference comparison (the two students)
| student | PESQ-NB | MCD-DTW dB | 4-thr RTF | dec share | verdict |
|---------|--------:|-----------:|----------:|----------:|---------|
| **Vocos8k (shipped)** | **2.90** | **10.8** | **0.013** | **12.6 %** | wins on all gates |
| HiFiGAN8k | 2.70 | 11.5 | 0.018 | 38.7 % | RTF ok, fails dec-share gate |

## Package contents
```
model.onnx   127 MB  melo enc/flow + distilled Vocos8k decoder, sample_rate=8000, opset17, fp32
tokens.txt           identical to the stock vits-melo-tts-zh_en model
lexicon.txt          identical to the stock vits-melo-tts-zh_en model
DEVICE_ACCEPTANCE.md this file
```
See repo `docs/INTEGRATION.md` for install steps and the zh-TW (Traditional) input notes.
