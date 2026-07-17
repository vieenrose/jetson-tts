# jetson-tts — lightweight 8 kHz zh/en (and zh-TW) TTS for the Jetson Nano

Distill best-in-class open-source **Chinese/English code-mixed** TTS models into **8 kHz** drop-in
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) model dirs that run in real time on a **Jetson
Nano gen1 CPU** (4× Cortex-A57). Target: a phone attendant whose audio leaves through an 8 kHz G.711
channel — rendering 44.1/16 kHz wastes most of the vocoder compute.

## 🚀 Live demo
Try all three 8 kHz models in the browser — pick a backend, type zh/en code-mixed text, hear it
stream: **[Sherpa-TTS Space](https://huggingface.co/spaces/Luigi/sherpa-zh-en-tts)**. The `📞`
backends (MeloTTS-8k, Matcha-8k, Matcha-zh-TW-8k) are the drop-in models from this repo; the demo
also keeps the original dual-engine (aishell3/Breeze2 + Silero) and 44.1 kHz MeloTTS backends for A/B.

## ⭐ PrimeTTS — tiny bilingual zh-TW + English (4.63M, 8 kHz, CPU)

**[PrimeTTS demo →](https://huggingface.co/spaces/Luigi/PrimeTTS-app)** ·
**[model + full training recipe →](https://huggingface.co/Luigi/PrimeTTS)** ·
base arch: [`owensong/Inflect-Nano-v1`](https://huggingface.co/owensong/Inflect-Nano-v1)

> **Streaming variant (PrimeTTS v2.1, MB-iSTFT-VITS):** the causal cached-conv vocoder +
> token-streaming encoder for intra-phrase incremental output on the Nano. Trained weights/exports
> are on the Hub (`Luigi/PrimeTTS`: `v21_streaming/`, `v2stream_streaming/`); the **full self-contained
> reproduction recipe** (training-code patch, launch scripts, configs, frontend, arch spec) is in
> [`reproduce/primetts-v21-streaming/`](reproduce/primetts-v21-streaming/README.md). Streaming
> inference reference: [`streaming/`](streaming/). Design: [`docs/streaming-arch-design.md`](docs/streaming-arch-design.md).

A third, from-scratch-distilled track: **PrimeTTS** packs Mandarin (Taiwan) **and** English into the
**frozen 4.63M-parameter Inflect-Nano** architecture (depthwise Conv-FFN, no attention; ~3.47M acoustic
+ ~1.17M Snake-HiFiGAN vocoder), emitting **8 kHz** and running torch-free on CPU via `onnxruntime`.
One model, one voice — zh-TW, English, and code-mix through a single unified bopomofo+arpabet frontend.

The result that matters: **held-out Mandarin CER fell from ~0.88 to ~0.06** at this size — proving the
architecture was never capacity-limited. Two levers did it, both architecture-frozen:
1. **Phone-level forced alignment** (espeak phoneme-CTC + `torchaudio.forced_align`) replacing crude
   char/letter-CTC splits — the prior "capacity wall" was mis-aligned training targets.
2. **Diverse, well-covered training text** (per-language) — the narrow original corpus left most
   characters/words unseen at eval time.

Applied symmetrically to English, the same recipe yields a genuinely **bilingual** model
(zh-only ≈ 0.13, English ≈ 0.16 in one 4.63M net). Teacher: BreezyVoice (single "mark" voice);
eval: offline X-ASR (sherpa zipformer zh-en). Lives in `mossnano/zhtw8k/`.

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
- **Accent (腔調) — ✗ out of scope for this edge stack (thoroughly tested).** Authentic Taiwan accent
  (reduced retroflex, TW prosody) requires fine-tuning the acoustic model on TW-accented audio. Every
  approach tried — across **both** architectures — destroyed zh/en correctness (ASR-measured) while
  creating the accent:
  - **Matcha:** full fine-tune (zh CER 0.38→0.55, dropped words); LoRA on encoder/duration/decoder-attn
    (garbled); teacher-forced-mel vocoder co-training and joint acoustic+vocoder co-training (CER 0.9+).
  - **MeloTTS** (end-to-end VITS, no vocoder coupling): full FT, frozen-text-encoder, and low-LR +
    discriminator-warmup all collapsed content to babble (CER 0.7–0.99, English recall ~0) vs base 0.357/0.59.
  - **Root cause:** the only available TW-accent teacher is a *different-speaker* clone (Qwen3-TTS +
    edge-tts ref). Forcing a lightweight single-speaker model to that voice drags its flow/decoder off the
    content manifold regardless of LR, freezing, or warmup. It's a **data problem** (need TW-accent audio in
    the base model's own voice, or real recorded zh-TW), not a tuning knob — and heavy LLM-TTS teachers that
    do accent+quality natively are far too large for the Nano. Full ASR-gated logs in the memory plan;
    see `docs/TW_ACCENT_RESEARCH.md`, `docs/ZH_TW_PLAN.md`, `docs/MELO_TW_ACCENT_PLAN.md`.

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
