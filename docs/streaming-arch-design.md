# Streaming zh-TW + English TTS on Jetson Nano gen-1 — Architecture Design

Status: proposal (design only; no code/training in this doc)
Baseline: PrimeTTS v2 / v2.1 = MB-iSTFT-VITS, 34.7M, 16 kHz, whole-utterance
Target device: Tegra X1 (Maxwell sm_53, 2 s kernel watchdog, no CUDA-graph replay; Cortex-A57 fp32-only CPU)

---

## Executive summary (5 lines)

1. **Recommended: a hybrid (d)** — *phrase-incremental input* (split text at punctuation/prosodic breaks, reuse the existing encoder/duration/flow per phrase, no arch change) driving a *streaming causal vocoder* (cached-conv MB-iSTFT + PQMF/iSTFT overlap-add) for true incremental output.
2. **Ship it in phases**: Phase-1 MVP is pure **chunked-reuse (option a)** — zero retrain, reuse v2.1 weights, phrase-split + crossfade — which alone gets first-audio to ~200-320 ms for a short opening phrase.
3. **The vocoder is the one high-value new component** (Phase 2): convert the MB-iSTFT GAN to causal cached-conv with a 24-frame chunk / 4-frame lookahead; **warm-start finetune** from v2.1 GAN (no from-scratch). Flow is nearly frame-local and chunkable with a tiny left cache; the windowed encoder and deterministic duration predictor are already streaming-friendly.
4. **Nano fit**: fixed 24-frame chunk graphs keep every kernel in the millisecond range → far under the 2 s watchdog, bound the launch-overhead blowup, and dodge the long-sequence kernel that tripped the watchdog in whole-utterance v2. Per-chunk RTF ≈ 0.45 GPU / 0.60 CPU (< 1 → playback never starves). **Reject AR frame-level (option c)** — per-frame launch overhead on Maxwell is fatal.
5. **Runtime: RapidSpeech.cpp ggml-CUDA** for the streaming decoder (explicit ring-buffer conv/KV state + fixed-shape per-chunk graph), ORT-CPU fp32 as the portable fallback and for the g2pw/g2p_en frontend, which stays CPU and chunks at punctuation to preserve polyphone context.

---

## 1. Component diagram (recommended hybrid)

```
                       ┌─────────────────────────── streams in (LLM tokens / typed text) ──────────────────────────┐
                       │                                                                                            │
   text tokens ──▶  PHRASE CHUNKER (CPU)                                                                            │
                  · buffer until a phrase boundary: 。！？，、；: / EN . ! ? , ; : / soft cap ~10-12 chars          │
                  · never split inside a word / zh compound (keeps g2pw polyphone context intact)                  │
                  · emit phrase P0, P1, P2 … as boundaries arrive                                                   │
                       │                                                                                            │
                       ▼   (per phrase Pi)                                                                          │
   ┌───────────────────────────────────────────── PER-PHRASE ACOUSTIC (reuse v2.1, unchanged) ──────────────────┐ │
   │  g2pw (bopomofo+polyphone) + g2p_en (arpabet)  →  88-sym 3-embed (phone+tone+lang)                          │ │
   │        ▼                                                                                                    │ │
   │  Text encoder  (windowed rel-pos attn, window=4, 6L, hidden 192)   ── already local context                │ │
   │        ▼                                                                                                    │ │
   │  Duration predictor (deterministic) → length-regulate            ── naturally per-phoneme incremental      │ │
   │        ▼                                                                                                    │ │
   │  Normalizing FLOW (4× ResidualCouplingLayer + WN)                 ── frame-local; chunk w/ small L-cache    │ │
   │        ▼  latent z frames for phrase Pi (held in a frame queue)                                             │ │
   └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
                       │  z-frames pushed into a rolling queue                                                     │
                       ▼                                                                                            │
   ┌──────────────── STREAMING CAUSAL VOCODER (NEW; warm-start finetune of v2.1 GAN) ────────────────┐            │
   │  pull fixed 24-frame chunks (+4-frame right lookahead) from the queue                            │            │
   │  causal HiFiGAN-style upsample+ResBlocks  → cached conv-state ring buffers (per layer)           │            │
   │  PQMF synthesis (4 subbands)  → carry filter-tap tail                                            │            │
   │  per-band iSTFT (gen_istft_n_fft=16)      → overlap-add tail (n_fft-hop ≈ 1 frame)               │◀───────────┘
   │  → 24 frames of 16 kHz PCM per chunk                                                             │
   └─────────────────────────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
   Audio ring buffer  → 20-40 ms crossfade at phrase seams (Phase-1 MVP) → speaker output (streamed)
```

Frame-rate anchor used throughout: MB-iSTFT-VITS at 16 kHz, `hop_length` 256 ⇒ **62.5 frames/s ≈ 16 ms/frame** (verify against the shipped v2.1 config; all numbers scale linearly if hop differs).

---

## 2. Per-component streaming treatment (blocker analysis)

| Component | Streaming blocker? | Treatment | Causal / chunk / lookahead / state |
|---|---|---|---|
| **g2pw + g2p_en frontend** | Soft. g2pw is a BERT polyphone disambiguator that wants sentence context. | **Chunk at punctuation / prosodic breaks only.** The boundary *is* the natural context edge, so no cross-boundary lookahead is needed. Never split a word/compound. Cap very long clauses at ~10-12 chars with a soft break at a word boundary (accepts minor polyphone risk to protect first-audio latency). | Chunk = one prosodic phrase. Lookahead = 0 (wait for the boundary). State = none. Runs on A57 CPU. |
| **Text encoder** (windowed rel-pos attn, window=4) | No, in phrase mode. | Whole phrase fits inside one call — attention window (4) is already local, so a phrase is ample context. For true token-streaming (Phase 3 only), use an **Incremental-FastPitch-style receptive-field-constrained chunk-attention mask + fixed-size past K/V** (their optimal past = 5 frames). | Phrase mode: no state. Token mode: past-K/V = 5 frames, right lookahead ≈ window/2 ≈ 2 phones. |
| **Duration predictor + length regulation** | No. Deterministic, per-phoneme. | Emit each phoneme's frames as its duration is predicted → intrinsically incremental. Runs per phrase. | Fully causal, no state, no lookahead. |
| **Flow** (4× ResidualCouplingLayer + WN) | Low. WN dilated convs have a bounded receptive field, but it is small (kernel 5, shallow) → nearly frame-local. | Phrase mode: run flow over the whole phrase at once (short → cheap). Streaming mode: run chunk-wise with a **left-context cache = WN receptive field** (a few frames of conv history per layer), à la CSSinger's causal posterior. Coupling layers mix per-frame, so this is well-behaved. | Left cache ≈ WN RF (~2-4 frames). No right lookahead needed if the encoder already produced the phrase. |
| **Vocoder** (MB-iSTFT GAN + PQMF + iSTFT) | Yes — but the *easy, highest-value* one. Conv net with limited RF → cleanly streamable. | **Causal cached-conv rewrite** (ring-buffer conv history per layer) + **overlap-add** for iSTFT and PQMF synthesis. Use CSSinger "natural padding" (pad the first chunk with actual z, not zeros) to kill the train/infer padding mismatch. Multi-band cuts per-chunk compute (each subband is 1/4 rate). | Chunk = **24 frames** (≈ 384 ms audio). Right lookahead = **4 frames** (≈ 64 ms). State = per-layer conv tails + PQMF filter-tap tail + iSTFT overlap tail (n_fft−hop ≈ 1 frame; gen_istft_n_fft=16 ⇒ tiny). |

**Net:** only the **vocoder** requires a real arch/finetune change to unlock true intra-phrase incremental output. Everything upstream is either already local (encoder/duration) or trivially chunkable (flow), and in the MVP they all just run per phrase, unchanged.

---

## 3. Architecture-family decision

| Option | Verdict | Why |
|---|---|---|
| **(a) Chunked non-streaming (reuse v2.1, phrase-split, crossfade)** | **Adopt as MVP** | Zero retrain, reuses every weight. Each phrase is short so "whole-utterance internally" is cheap and watchdog-safe. Weakness = prosody seams — mitigated by held speaker embed + crossfade + optional 1-word lookahead context. |
| **(b) Streaming-native VITS (chunk encoder + streaming flow + streaming vocoder, retrained)** | Partial adopt (target), from-scratch part rejected | CSSinger shows this works and *improves* quality, but it trains **from scratch (500k steps)**. Too costly to jump to wholesale. We take only the pieces that warm-start cheaply (vocoder, then flow/encoder finetune). |
| **(c) AR frame-level acoustic + streaming vocoder** | **Reject** | Per-frame autoregression = tens of thousands of tiny kernel launches on a launch-overhead-bound Maxwell GPU with no CUDA-graph replay. Directly antagonistic to the Nano. Natural streaming, wrong device. |
| **(d) Hybrid: phrase-incremental input + streaming causal vocoder** | **Recommend** | Best latency/quality/effort trade for the Nano. Input side reuses v2.1 as-is (no retrain); output side gets one warm-start vocoder finetune. Fixed small chunks respect the watchdog and cap launch overhead. Cleanly phase-able: (a) → (d) → optional (b). |

**Recommendation: (d)**, delivered as MVP=(a) then the streaming vocoder. Rationale for the Nano specifically: it keeps the model small and fp32 (no int8/fp16 dependence — matches the A57 reality), turns the one uncapped long-sequence kernel (the whole-utterance watchdog risk) into bounded 24-frame kernels, and needs at most a *vocoder finetune* rather than a from-scratch retrain, so it composes with the existing v2.1 checkpoints and the RapidSpeech.cpp `mbistft-vits` graph we already have.

---

## 4. Latency + RTF budget (Nano)

Anchors: whole-utterance RTF **0.42 GPU-ggml-CUDA**, **0.52 CPU-ORT fp32 @4thr**; 16 ms/frame; 2 s watchdog.

### Phase-1 MVP (chunked-reuse, no retrain)
First audio = frontend(P0) + whole-phrase synth(P0). Make **P0 deliberately short** (first 2-3 phones, ~0.25-0.5 s audio = 15-30 frames):
- g2pw on a short phrase (A57 CPU, small BERT): ~40-80 ms (Phase-0 must measure this — it is the real floor).
- synth 15-30 frames = 0.24-0.48 s audio × 0.42 GPU ≈ **100-200 ms** (× 0.52 CPU ≈ 125-250 ms).
- crossfade/buffer: ~20-40 ms.
- **First-audio ≈ 180-320 ms** — hits the ~200-300 ms target if P0 is kept to the first few phones and grows later phrases. Longer phrases stream behind it at RTF 0.42 (< 1), so no starvation.
- Watchdog: cap phrase ≤ ~90 frames (1.44 s audio ⇒ ~600 ms compute) → safe.

### Phase-2 target (streaming causal vocoder)
First audio = frontend(P0) + enough encoder/dur/flow to fill one vocoder chunk + vocode(24 frames):
- encoder/dur/flow only needs ~4-6 phones (24 + 4 lookahead frames).
- produce+vocode 28 frames = 0.45 s audio × ~0.45 (chunked, +~10% cache overhead) ≈ **~200 ms GPU**; encoder/flow are cheaper than the vocoder so the effective figure is lower.
- **First-audio ≈ 150-220 ms**; steady state emits a 24-frame (384 ms) chunk every ~170 ms ⇒ **per-chunk RTF ≈ 0.45 GPU / 0.60 CPU (< 1)**.
- Watchdog: a 24-frame chunk processes ≤ 24×256 = 6144 samples per layer → single-digit-ms kernels, zero watchdog risk, fixed shape.

**Conclusion:** the target ~200-300 ms first-audio is reachable already in the MVP with a short opening phrase, and comfortably so with the streaming vocoder. Per-chunk RTF stays < 1 on both runtimes.

---

## 5. Reuse-vs-retrain plan

| Phase | Weights | Training |
|---|---|---|
| **1 (MVP, option a)** | **100% reuse of v2.1** (encoder, duration, flow, MB-iSTFT GAN, speaker embeds). | **None.** Pure inference orchestration: phrase chunker + z/audio queues + crossfade. Hold the speaker embedding constant across chunks; optionally carry 1-word lookahead context into the next phrase to smooth prosody. |
| **2 (streaming vocoder)** | **Warm-start** the MB-iSTFT decoder from v2.1 GAN. | **Short finetune** with causal convs + fixed receptive-field chunk mask + "natural padding". Losses unchanged (mel + adversarial + feature-matching); **add a 2-frame lookahead / chunk-boundary consistency loss** (from streaming-VC lit) so chunk seams match the full-context output. Same corpus; add phrase-boundary segmentation metadata. Flow either kept phrase-level (no change) or finetuned with causal WN padding. Expect near-parity (Incremental FastPitch: ~8% mel-dist ↑, MOS parity; CSSinger: quality *up*). **Gate on resynth CER, not spectra** (per the aligner lesson). |
| **3 (optional, streaming encoder/flow)** | Warm-start encoder + flow. | Finetune encoder with an Incremental-FastPitch static chunk-attention mask (past=5) and causal flow. Only pursue if Phase-1/2 phrase-boundary latency proves insufficient for token-by-token LLM input. |

Corpus/loss changes are minimal: **no new data**, just boundary metadata and one auxiliary consistency loss. The v2.1 multi-speaker path is unaffected — the speaker embedding is global and simply held constant while streaming.

---

## 6. Deployment path (RapidSpeech.cpp vs ORT)

**Primary: RapidSpeech.cpp ggml-CUDA for the streaming decoder.** It is the better streaming fit because:
- We own the graph → we can build a **fixed-shape 24-frame per-chunk graph** and re-run it per chunk. sm_53 has **no CUDA-graph replay**, but fixed shapes still eliminate per-run shape-inference/allocation churn and, crucially, **bound every kernel's sequence length** so nothing approaches the 2 s watchdog (this is exactly the lever that removes the long-sequence kernel which tripped the watchdog in whole-utterance v2).
- Stateful streaming is explicit: keep the ring-buffer conv tails, PQMF filter tails, iSTFT overlap tails, and (Phase 3) encoder past-K/V as **persistent ggml tensors** carried across chunk invocations.

**Secondary / fallback: ORT-CPU fp32.** Competitive for the vocoder (0.52 whole-utterance) and fully portable, but stateful streaming in ONNX means threading state tensors through graph inputs/outputs (a stateful export) — more painful than owning the ggml graph. Use ORT-CPU for the **g2pw/g2p_en frontend** (stays on CPU regardless) and as the portable path where CUDA isn't available.

**Split of work:** frontend on A57 CPU (ORT or native); acoustic + streaming vocoder on the Maxwell GPU via RapidSpeech.cpp with per-chunk state. All fp32 — no reliance on int8 dot-product or fp16 arithmetic the A57/Maxwell lack for fast paths.

---

## 7. Risks + open questions

1. **Prosody discontinuity at phrase seams (MVP)** — F0/energy jumps across chunks. Mitigate: hold speaker/prosody embedding constant, 20-40 ms crossfade, 1-word lookahead context. *Open:* is it audible enough (CMOS) to force the streaming vocoder sooner?
2. **g2pw latency on A57** — the polyphone BERT per phrase sets the first-audio floor. *Open:* is it fast enough per short phrase, or does it need quantization / a lighter polyphone head / caching?
3. **Cross-boundary polyphone errors** — splitting near a polyphone whose disambiguating context sits across the boundary. Mitigate: split only at punctuation/strong breaks; never mid-word/compound.
4. **Causal-vocoder finetune warm-start** — train/infer padding mismatch. Mitigate: "natural padding" + lookahead-consistency loss. *Open:* does it hold at 16 kHz with PQMF, verified by resynth CER (not MCD/spectra)?
5. **iSTFT/PQMF overlap-add correctness** — PQMF near-perfect-reconstruction filters have boundary transients; overlap-add must be exact. Carry filter-tap and iSTFT tails; validate bit-level continuity + resynth CER.
6. **Chunked flow quality** — small left cache may degrade the coupling bijector at seams. Low risk (small WN RF); confirm in the Phase-2 experiment; fall back to phrase-level flow if needed.
7. **Watchdog on any remaining uncapped kernel** — audit that no per-chunk kernel exceeds the 24-frame bound (e.g. a global attention/normalization slipping in).

---

## 8. Phased build plan

- **Phase 0 — measure & scaffold (no model changes).** Profile on the Nano: g2pw latency per short phrase, and whole-phrase synth latency for 15/30/60/90-frame phrases (GPU + CPU). Establish the real first-audio floor. Build the phrase chunker, z/audio ring buffers, and crossfade harness. Deliverable: measured latency table + streaming harness.
- **Phase 1 — MVP chunked-reuse (option a), zero retrain.** Phrase splitter (punctuation + ≤90-frame cap, word-safe) → synth each phrase with v2.1 → 20-40 ms crossfade → stream. Hold speaker embed; optional 1-word lookahead context. Ship. Deliverable: streaming demo; first-audio + CMOS-vs-non-streaming numbers. **This likely already meets ~200-300 ms first-audio for conversational (LLM emits clause-by-clause).**
- **Phase 2 — streaming causal vocoder (warm-start finetune) → hybrid (d).** Causal cached-conv MB-iSTFT + PQMF/iSTFT overlap-add, chunk=24 / lookahead=4, natural padding, lookahead-consistency loss; warm-start from v2.1 GAN. Implement ring-buffer state in a fixed-24-frame RapidSpeech.cpp graph. Now true intra-phrase incremental output; first-audio ~150-220 ms; watchdog-proof. Deliverable: streaming vocoder checkpoint + resynth-CER gate + on-device RTF.
- **Phase 3 — optional streaming encoder/flow (option b pieces).** Chunk-attention encoder (Incremental-FastPitch static mask, past=5) + causal flow, warm-start finetune → enables token-by-token LLM input without waiting for a phrase boundary. Only if Phase-1/2 boundary latency is insufficient.

---

## Sources

- Incremental FastPitch: Chunk-based High Quality Text to Speech — arXiv:2401.01755 (chunk-based FFT, receptive-field-constrained chunk attention mask, fixed-size past K/V; 30-frame chunk, optimal past=5, ~30 ms first-chunk, MOS parity with parallel).
- CSSinger: End-to-End Chunkwise Streaming SVS based on Conditional VAE — arXiv:2412.08918 (ChunkStream decoder 20 frames / 10 left / 4 right; causal HiFiGAN + "natural padding"; causal posterior; trains from scratch; quality ≥ parallel; CPU RTF 0.635).
- Comparative Analysis of Fast and High-Fidelity Neural Vocoders for Low-Latency Streaming in Resource-Constrained Environments — arXiv:2506.03554 (HiFiGAN / iSTFTNet / multi-band streaming; cached conv state; RTF < 1 on edge; multi-band lowers per-chunk compute).
- VoXtream: Full-Stream TTS with Extremely Low Latency — arXiv:2509.15969 (102 ms first-packet, output starts after first word).
- SpeakStream: Streaming TTS with Interleaved Data — arXiv:2505.19206 (decoder-only, 45 ms first-token, interleaved speech-text).
- SyncSpeech: Low-Latency Dual-Stream TTS with Temporal Masked Transformer — arXiv:2502.11094 (chunk-aware decoder).
- Streaming Voice Conversion through Chunk-wise Training and Lookahead Loss — Univ. Rochester ECE477 2024 (2-frame lookahead loss for chunk consistency).
- S5-TTS / Streaming T5-based TTS with Limited Lookahead — arXiv:2606.21882 (lookahead-causal masking, word-by-word).
- NVIDIA Riva TTS streaming docs — chunked FastPitch + HiFi-GAN, time-to-first-audio streaming (docs.nvidia.com Riva TTS).
