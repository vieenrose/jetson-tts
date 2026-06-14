# jetson-tts — lightweight 8 kHz zh/en (and zh-TW) TTS for the Jetson Nano

Distill best-in-class open-source **Chinese/English code-mixed** TTS models into **8 kHz** drop-in
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) model dirs that run in real time on a **Jetson
Nano gen1 CPU** (4× Cortex-A57). Target: a phone attendant whose audio leaves through an 8 kHz G.711
channel — rendering 44.1/16 kHz wastes most of the vocoder compute.

## Released models (Hugging Face)
| model | what it is | status |
|---|---|---|
| [Luigi/vits-melo-tts-zh_en-8k](https://huggingface.co/Luigi/vits-melo-tts-zh_en-8k) | MeloTTS zh_en, HiFi-GAN decoder distilled → 8 kHz Vocos | **device-accepted on real Nano**, PESQ-NB 2.90 |
| [Luigi/matcha-zh-en-8k](https://huggingface.co/Luigi/matcha-zh-en-8k) | Matcha-TTS zh-en, 16 kHz vocoder distilled → 8 kHz | best code-mixing; PESQ-NB 3.80 |
| [Luigi/matcha-zh-tw-en-8k](https://huggingface.co/Luigi/matcha-zh-tw-en-8k) | matcha-zh-en-8k + **Taiwan-readings lexicon** | **zh-TW deliverable** (readings ✓, accent: see below) |
| [Luigi/zh-en-tts-8k-comparison](https://huggingface.co/datasets/Luigi/zh-en-tts-8k-comparison) | A/B listening sets (dataset) | — |

Measured Jetson Nano RTF (melo-8k, stock `sherpa-onnx-offline-tts`, ORT CPU): **0.79 / 0.44 / 0.34** at
1/2/4 threads (~10× faster than the stock 44.1 kHz melo), under real-time at every thread count. The
host→A57 factor for this conv-heavy workload measured **~×13**.

> **Speaker id:** all models are single-speaker. sherpa-onnx hardcodes the graph speaker to the
> model's metadata `speaker_id` (melo: 1; matcha: the single voice), so `--sid` is effectively
> ignored — any value, including the default `0`, produces correct audio (sherpa may log a benign
> "Use sid=0" notice). The speaker embedding only matters if you bypass sherpa and drive the raw
> ONNX yourself (then MeloTTS needs `emb_g(1)`, not 0). Verified in sherpa-onnx for both models.

## Two distillation tracks

### A. MeloTTS zh_en → 8 kHz (`student/`)
VITS-family: dump the decoder-input latent `z[192,T]` + speaker emb `g[256]` from stock melo
(**with `--zero-bert`** to match the deployed sherpa graph), pair with 8 kHz audio, train a
lightweight vocoder. Two students compared: `HiFiGAN8k` (1.0M) and the winner `Vocos8k` (3.7M,
ConvNeXt @125 Hz + iSTFT head). `z` is resampled 86.13→125 Hz so an integer ×64 lands on exactly 8000 Hz.

### B. Matcha-TTS zh-en → 8 kHz (`matcha8k/`)
2-stage (acoustic → mel, separate Vocos vocoder). Keep the acoustic model; distill a new **8 kHz Vocos
vocoder** (mel[80,T] → mag/x/y, sherpa iSTFT n_fft=512/hop=128). Tokens captured from sherpa's own
debug frontend; vocoder trained on real (mel, 8 kHz) pairs. PESQ-NB 3.80.

## zh-TW (Taiwan Mandarin) — what works and what doesn't
- **Readings (字音) — ✅ shipped.** A curated CN→TW reading-override lexicon (`data/tw_readings/`,
  `scripts/apply_tw_lexicon.py`): 垃圾→lèsè, 期→qí, 究→jiù, 質→zhí, 危→wéi, 企→qì… On-device lexicon
  swap only, no retrain, keeps English/code-mixing. Validated. → `Luigi/matcha-zh-tw-en-8k`.
- **Accent (腔調) — ✗ out of scope for this edge stack.** Authentic Taiwan accent (reduced retroflex,
  TW prosody) requires changing the acoustic model. Extensive attempts (full fine-tune; LoRA on
  encoder/duration/decoder-attention; teacher-forced-mel vocoder co-training) could not reach clean
  quality: changing the accent breaks the base acoustic↔vocoder co-training, and re-pairing on
  flow-matching mels caps quality (PESQ ~1.2). Heavy LLM-TTS teachers (BreezyVoice/CosyVoice,
  Qwen3-TTS) do accent+quality natively but are far too large for the Nano. It's a device-capability
  limit, not a tuning bug. See `docs/TW_ACCENT_RESEARCH.md` and `docs/ZH_TW_PLAN.md`.

A clean way to *generate* TW-accented code-mixed audio offline (e.g. as a teacher): **Qwen3-TTS-Base
voice-cloning** an edge-tts zh-TW reference produced excellent results — but only as an offline data
generator, not an edge-deployable model.

## Evaluation tooling
- Host x86 ORT-CPU RTF @1/2/4 threads + per-node profile (`scripts/bench_*.py`).
- PESQ-NB + MCD-DTW vs teacher, simulated G.711 μ-law channel (`scripts/eval_quality.py`, `scripts/g711.py`).
- **ASR intelligibility gate** for zh/en (English-word recall + CER): faster-whisper and the
  **X-ASR-zh-en** sherpa Zipformer (`matcha8k/asr_verify.py`, `matcha8k/xasr_verify.py`). Use ≥70-word
  eval sets — small sets are too noisy (a lesson learned the hard way).

## Layout
```
student/        MeloTTS 8k vocoders, losses, dataset, training, audio framing
matcha8k/       Matcha 8k vocoder, frontend, ASR gate, accent experiments (lora_dec, dump_tf_mels)
scripts/        teacher dump, corpus gen, ONNX export, bench, eval, G.711, TW lexicon
docs/           ENV_SETUP, *_CONFIG, ORT_COMPAT, INTEGRATION, DEVICE_ACCEPTANCE, ZH_TW_PLAN, TW_ACCENT_RESEARCH
data/           code-mixed corpus + TW reading overrides (rendered audio is gitignored — regenerate via scripts)
```

## GPU offload (moved out)
The ggml-CUDA Maxwell-GPU offload work lives in its own repo:
**https://github.com/vieenrose/edge-speech-gpu-bench**.

## License
Code: MIT. Derives from MeloTTS (MIT, MyShell.ai), Matcha-TTS (MIT) / csukuangfj/dengcunqin matcha
zh-en, references sherpa-onnx (k2-fsa, Apache-2.0). Vocoder arch inspired by Vocos (MIT). Attribution
retained in `third_party/`.
