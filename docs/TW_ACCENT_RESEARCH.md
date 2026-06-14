# Taiwan-Mandarin accent for our edge TTS — methods & data evaluation (Phase 2 research)

Goal: add authentic Taiwan-Mandarin **accent (腔調)** — reduced retroflex (zh/ch/sh→z/c/s), no
erhua, fewer neutral tones, Taiwan prosody — to our lightweight Matcha-8k while **keeping the good
English + zh/en code-mixing** and staying ONNX/sherpa-onnx edge-deployable. The **readings layer
(字音)** is already shipped (`Luigi/matcha-zh-tw-en-8k`); this is only about the acoustic accent.

Evidence base: deep-research workflow (5 angles, 20 sources, 25 claims verified, 12 confirmed via
3-vote adversarial check) + direct corpus verification. Confidence noted per item. Sources listed at end.

## Feasibility gate (resolved)
- **Trainable checkpoint EXISTS.** The Matcha zh-en source (modelscope `dengcunqin/matcha_tts_zh_en_20251010`)
  ships `pytorch_model.bin` + `vocab_tts.txt` + test scripts — not just ONNX. So acoustic fine-tuning is
  possible without retraining from scratch. (csukuangfj HF mirror is ONNX-only.)

## Method comparison (scored for OUR constraints)
| method | needs PyTorch ckpt? | preserves EN/code-mix? | edge-deployable after? | quality ceiling | effort | fit |
|---|---|---|---|---|---|---|
| **(a) Fine-tune acoustic on TW speech + EN retention** | yes | yes (with retention mix + low LR) | yes (same arch→ONNX) | = data quality | med | **BEST** |
| (a′) PEFT/adapter fine-tune (freeze base) | yes | **yes — designed to avoid forgetting** | yes (fold adapter at export) | high | low-med | **strong** |
| (b) Train from scratch multi-corpus | yes | rebuild | yes | high but **fails low-resource** (misalignment) | high | poor |
| (c) Accent embedding / multi-accent conditioning (global+phoneme, GRL speaker-disentangle) | yes (arch change) | yes | yes if kept small | high, controllable | high | overkill (we want 1 accent) |
| (c′) Accent-Vector (task vector θ_pre+α·τ); DART/Accent-VITS disentangle | yes (arch/recipe) | yes | varies | high, α-controllable | high | future option |
| (d) Accent/voice conversion (post-proc or data-aug) | no (separate model) | n/a (offline aug) | adds runtime cost if online | medium | med | **as data-aug only** |
| (e) Synthetic distillation from BreezyVoice teacher | no (teacher offline) | yes (mix EN from matcha) | yes | **capped at teacher + TTS-on-TTS loss** | med | fallback/single-voice |
| (f) Rule-based phoneme/prosody hacks | no | yes | yes | low (readings only, not 腔調) | low | already done (Phase 1) |
| (g) Self-supervised / cross-lingual transfer | yes | yes | yes | high | high | research, not now |

### Key verified findings
- **PEFT beats full fine-tune for adaptation while avoiding forgetting** (Interspeech 2025, 3-0): adapter
  approach adapted a TTS to a new language with **12 h single-speaker data on one GPU**, updating only
  **1.72% of params (5.81M/335.8M)**, and *beat* full fine-tune on pronunciation (CER 4.91 vs 5.67) and
  naturalness (NMOS 3.88 vs 3.58) — while **keeping the base model's existing languages** by freezing it.
  → Directly supports method (a′) for "add TW accent without breaking English."
- **Train-from-scratch fails in low-resource** flow-matching settings (text-speech misalignment); fine-tune
  from a pretrained checkpoint succeeds (same paper, 3-0). → rules out (b).
- **Accent can be a controllable vector**: "Accent Vector" τ=θ_ft−θ_pre, inference θ_pre+α·τ with α scaling
  strength (3-0); and it can be derived **without accent-specific data** from native speech. → gives a future
  dial for "how Taiwanese" and a path if clean TW data is scarce.
- **Speaker/accent disentanglement is real but hard**: GRL adversarial accent encoder (3-0), Accent-VITS
  two-stage text→accent→wave (3-0, needs full VITS), DART ML-VAE+VQ (3-0). Powerful but heavier than we
  need for a single target accent + flexible voice.
- **BreezyVoice = CosyVoice-family** (S3 tokenizer + LLM + OT-CFM + g2p) (2-0) → teacher-only, not edge.

## Data sources for Taiwan **Mandarin** (Guoyu/Huayu) speech
| corpus | hours / speakers | grade | sr | license | verdict for us |
|---|---|---|---|---|---|
| **Common Voice zh-TW** (cv 25) | ~80 h validated / ~131 h, 2317 spk | crowd/ASR-grade, noisy, multi-spk | 48k (mp3) | CC0 | **primary** — only large, free, real TW-Mandarin set. Needs cleaning + speaker selection/denoise |
| **TAT / TAT-TTS** | TAT 300 h; TAT-TTS 40 h (4 spk, 48k, clean) | clean (TTS) | 48k | ACLCLP application | ❌ **Taiwanese HOKKIEN (台語, 台羅/白話字), NOT Mandarin** — wrong language (verified) |
| FSW (Formosa Speech in the Wild) | large | ASR-grade | — | application | mixed; mostly Hokkien/ASR — verify per-subset |
| MATBN Mandarin news | ~198 h | broadcast, multi-spk | 16k | ACLCLP/LDC | TW-Mandarin but news-read, multi-spk, licensing |
| TCC300 | ~300 spk read | read, multi-spk | 16k | ACLCLP | TW-Mandarin read speech; licensing; older |
| **BreezyVoice synthetic** | unlimited / 1 chosen voice | clean, consistent | model SR | Apache-2.0 | **single-voice TW corpus on demand** — but quality capped at teacher (TTS-on-TTS) |
| Taiwan podcast/audiobook + ASR pipeline | unlimited | self-labeled (Whisper) | varies | per-source | DIY single-voice clean data if a good public-domain TW speaker is found |
| TW-accented **English / code-mixed** speech | ~none public | — | — | — | **the real scarcity** — essentially must be synthetic or self-collected |

Net: **there is no large, clean, single-speaker, TTS-grade Taiwan-MANDARIN corpus that is freely
licensed.** Common Voice zh-TW is the only sizable free real option but is multi-speaker/noisy. The
clean Taiwan corpora (TAT-TTS) are Hokkien. This scarcity is the central constraint and drives the recommendation.

## Accent evaluation (how to gate Phase 2)
- **Native-listener MOS / preference (A/B)** on accent authenticity — primary gate (you = native judge).
- **Accent-classifier accuracy** — train/borrow a CN-vs-TW Mandarin classifier; measure % of outputs
  judged TW (objective, automatable).
- **Phonetic feature checks** — retroflex energy (zh/ch/sh spectral centroid / sibilant contrast), erhua
  absence, neutral-tone rate, final-merge (ㄣ/ㄥ). Scriptable acoustic probes for regression tracking.
- **Retention guards (must not regress):** PESQ/MCD vs current matcha-8k on Chinese; English/code-mix
  intelligibility (ASR-CER on the zh/en eval set); the existing comparison dataset as the A/B harness.

## Recommended approach (for our constraints)
**Primary: PEFT/adapter fine-tune of the Matcha acoustic model (method a′), data = cleaned Common Voice
zh-TW + an English/code-mix retention set, voice optionally normalized via one consistent target.**

Why: PEFT is the verified sweet spot — small data, single GPU, *preserves English* by construction,
stays the same architecture (so it re-exports to ONNX/sherpa and we just re-distill the 8k vocoder with
our existing matcha8k pipeline). It sidesteps train-from-scratch failure and the heavy disentanglement
architectures we don't need for one accent.

Pipeline:
1. **Data prep**: pull Common Voice zh-TW (~80 h), filter by quality/SNR, optionally restrict to a few
   clean speakers (or one) to control voice; force-align; build text+audio pairs. Add a **retention set**
   generated from our current matcha (English + zh/en code-mixed) so the adapter can't forget English.
2. **Voice consistency** (phone-attendant wants one voice): either pick the single cleanest CV speaker, or
   use accent/voice conversion (method d) **offline** to normalize many CV speakers → one target timbre,
   yielding a larger single-voice TW-Mandarin set without an online VC cost.
3. **PEFT fine-tune** the acoustic ckpt (freeze base, adapters; low LR; mix TW-Chinese + EN-retention each
   batch). Track accent classifier + retention CER.
4. **Re-distill the 8k vocoder** on mels from the fine-tuned acoustic model (reuse `matcha8k/`).
5. **Export drop-in**, verify in sherpa, gate on native MOS + retention.

**Fallback / accelerator: BreezyVoice as offline single-voice data generator (method e)** if cleaned CV
data proves too noisy or voice-inconsistent. Accept the teacher ceiling; use ASR-filtering to drop bad
teacher utterances. Possibly **hybrid**: CV real data to anchor authentic accent + BreezyVoice/VC to fill
clean single-voice + code-mixed coverage.

**Optional dial: Accent-Vector (α)** — compute τ from the fine-tune and expose α to tune "how Taiwanese"
at inference if full strength sounds off.

## Main risks & mitigations
- **No clean single-voice TW-Mandarin data** (central risk) → speaker selection in CV, or VC-normalize to one
  voice, or BreezyVoice synthetic; accept a voice-identity change (product allows it).
- **Catastrophic forgetting of English/code-mix** → PEFT with frozen base + retention data + low LR; gate on
  English CER every checkpoint.
- **CV noise/quality** → SNR/҂duration filtering, denoise, forced-alignment dropout of bad segments.
- **TTS-on-TTS ceiling (if BreezyVoice path)** → use as supplement not sole source; ASR-confidence filtering.
- **Voice inconsistency from multi-speaker CV** → restrict speakers or VC-normalize offline.
- **Accent under/over-shoot** → Accent-Vector α control + native A/B iteration.
- **License**: CV CC0 ✓, BreezyVoice Apache-2.0 ✓, sherpa Apache-2.0 ✓, Matcha MIT ✓. TAT needs application
  (and is Hokkien anyway). Keep attribution.

## Sources (verified)
- Multi-scale accent embedding + GRL speaker disentangle — arxiv.org/html/2406.10844
- Accent-VITS two-stage text→accent→wave — arxiv.org/pdf/2312.16850
- Accent Vector (task-vector, α-control, no accent data) — arxiv.org/html/2603.07534
- PEFT/adapter cross-lingual adaptation (12 h, 1.72% params, anti-forgetting; scratch fails) — isca-archive.org/interspeech_2025/kwon25_interspeech.pdf
- DART ML-VAE+VQ accent/speaker disentangle — arxiv.org/abs/2410.13342
- BreezyVoice (CosyVoice family, heavy) — arxiv.org/abs/2501.17790 ; huggingface.co/MediaTek-Research/BreezyVoice
- TAT-TTS = Taiwanese Hokkien (台語), not Mandarin — sites.google.com/nycu.edu.tw/fsw/home/tat-tts-corpus (verified)
- Common Voice zh-TW ~80 h validated / 2317 spk — commonvoice.mozilla.org/zh-TW (cv-corpus-25)
- Trainable Matcha ckpt (pytorch_model.bin) — modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010
