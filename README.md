# jetson-tts — 8 kHz vocoder distillation for MeloTTS zh_en

Replace the 44.1 kHz HiFi-GAN decoder of `vits-melo-tts-zh_en` (MeloTTS) with a retrained
**lightweight 8 kHz vocoder**, so the full TTS runs in real time on a **Jetson Nano gen1 CPU**
(4× Cortex-A57, ONNX Runtime CPU via sherpa-onnx). Target use: a phone attendant whose audio
leaves through an 8 kHz G.711 channel — rendering 44.1 kHz wastes ≥80 % of vocoder compute.

The voice, prosody, and front end are **unchanged** — only the decoder is retrained, distilled
from MeloTTS's own output. Same text → same speech, just 8 kHz and cheap.

## Approach
- **Teacher / data:** stock MeloTTS zh_en synthesizes a zh/en code-mixed receptionist corpus;
  we dump the decoder-input latent `z[192,T]` + speaker embedding `g[256]` paired with the
  audio downsampled to 8 kHz. No human speech dataset needed. (`scripts/dump_teacher.py`)
  - **Must dump with `--zero-bert`**: the deployed sherpa-onnx binary never computes BERT, so a
    faithful drop-in distills on the bert=0 latent distribution.
- **Students (two, compared):**
  - `HiFiGAN8k` — re-derived HiFi-GAN upsample stack for 8 kHz (1.0M params).
  - `Vocos8k` — ConvNeXt @ 125 Hz + iSTFT head (3.7M params; the expected speed winner).
  - Shared front end resamples `z` 86.13 Hz → 125 Hz so an integer ×64 lands on **exactly 8000 Hz**.
- **Export:** drop-in `model.onnx` (melo enc/flow + student dec, `sample_rate=8000`, opset 17,
  fp32, no custom ops) that the stock `sherpa-onnx-offline-tts` runs unmodified. (`scripts/export_full_onnx.py`)
- **Gates:** host x86 ORT-CPU RTF @1/2/4 threads (predicts A57 RTF ×6–8), PESQ-NB + MCD vs the
  teacher, ear-test through a simulated G.711 μ-law channel.

## Layout
```
student/        student vocoders, losses, dataset, training, audio framing config
scripts/        teacher dump, corpus gen, ONNX export (dec-only + full), bench, eval, G.711
docs/           ENV_SETUP, TEACHER_DECODER_CONFIG, ORT_COMPAT, INTEGRATION, (DEVICE_ACCEPTANCE)
data/text/      code-mixed corpus (rendered pairs are gitignored — regenerate via scripts)
third_party/    melo lazy-import patch + sherpa-onnx melo export reference
```

## Reproduce
See `docs/ENV_SETUP.md` (isolated venv, cu128 torch, MeloTTS zh/en-only install + patch), then:
```bash
python scripts/build_corpus_text.py --n 15000 --out data/text/corpus.tsv
python scripts/dump_teacher.py --corpus data/text/train.tsv --out data/pairs --zero-bert  # per GPU shard
python -m student.train --arch vocos   --device cuda:1 --out student/runs/vocos
python -m student.train --arch hifigan --device cuda:0 --out student/runs/hifigan
python scripts/export_full_onnx.py --ckpt student/runs/vocos/best.pt --out-dir export/vits-melo-tts-zh_en-8k
```

## License
Code: MIT. Derives from MeloTTS (MIT, MyShell.ai) and references the sherpa-onnx (k2-fsa, Apache-2.0)
melo export script. Attribution retained in `third_party/`.
