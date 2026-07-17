# zh/en Code-Mix TTS Architectures — Survey & Nano Design Recommendation (2026-07)

> Scope: on-device zh-TW + English **code-mix** TTS (single unified frontend, no language
> routing; entity-heavy input: numbers/emails/serials) for **Jetson Nano gen-1**.
> This document surveys small/on-device TTS architecture families (~5–100M params,
> 2022–2026), analyzes code-mix frontends, characterizes the Nano performance frontier,
> investigates the measured **Matcha RTF anomaly**, and commits to a keep/evolve/replace
> verdict for our flagship **MB-iSTFT-VITS (PrimeTTS v2/v2.1, 34.7M, 16 kHz)**.
> A separate agent owns the streaming variant; streaming is noted here only as a per-arch property.

---

## Executive summary (5 lines)

1. **Keep MB-iSTFT-VITS as the shipped flagship** — it is user-approved, hits X-ASR CER 0.027 (beats its 7B teacher), is already ported to RapidSpeech.cpp with CPU parity, and is a defensible fit for a launch-overhead-bound GPU (parallel, single-pass, iSTFT vocoder). No survey candidate dominates it on quality at our size.
2. **The Matcha anomaly is real and explainable, not noise**: the sherpa `matcha-icefall-zh-en` model that ran **RTF 0.18** vs our **0.42** is a **3-step CFM acoustic + a Vocos-16 kHz vocoder** — an architecture with *far fewer, larger kernel launches* than VITS's normalizing-flow + multiband-decoder stack, which is exactly what a no-CUDA-graph Maxwell GPU rewards.
3. **Recommendation = EVOLVE (add a fast lane), do not replace yet**: prototype a **Matcha-class CFM acoustic + Vocos/iSTFT-16 kHz vocoder** reusing our existing g2pw 3-embedding frontend and Xinran/VibeVoice corpus. Target: land the measured ~0.18 RTF (2.3× latency win) and feed the streaming effort (CFM chunks cleanly).
4. **Biggest lever is the acoustic model, not the vocoder**: VITS's launch cost is dominated by the inverse **flow** (many small WaveNet coupling ops), so swapping only the vocoder gives partial gain; replacing the flow-based acoustic with a dense CFM U-Net is the structural fix.
5. **Frontend stays**: g2pw + phone/tone/lang 3-embedding is state-of-the-practice for zh-TW code-mix; no surveyed system offers a clearly better *small-model* frontend. Keep it; the one migration cost of moving to Matcha is re-deriving durations via MAS (Matcha has no built-in stochastic duration predictor).

---

## 1. Taxonomy of small/on-device TTS architecture families

Five families are relevant at 5–100M params. The axis that matters most for Nano is **inference
pattern** (single-pass parallel vs iterative-few-step vs autoregressive-many-step), because the
Maxwell GPU is **kernel-launch-overhead-bound** (no CUDA-graph replay on sm_53) and the A57 CPU is
**fp32-only** (no int8 dot-product, no fp16 arithmetic).

### 1.1 VITS family (end-to-end VAE + flow + GAN vocoder) — *where we are*
- **VITS** (Kim et al. 2021): text encoder → **stochastic** duration predictor → normalizing **flow** (prior↔latent) → HiFiGAN decoder. Single-pass, distributional (does *not* mean-regress prosody). Heavy: many small flow/WaveNet ops.
- **VITS2** (2023): transformer-in-flow, adversarial duration, better mono alignment; quality up, cost similar.
- **MB-iSTFT-VITS** (Kawamura et al., ICASSP 2023, arXiv 2210.15975): replaces the most expensive decoder convs with **multi-band generation + inverse STFT**. Paper: **3.4–4.1× faster than VITS**, RTF **0.066 on an Intel i7 CPU**, naturalness on par with VITS. iSTFT alone = 1.8×; +multiband = 1.9–2.3× on top. `Mini-MB-iSTFT-VITS` beats Nix-TTS. **← our PrimeTTS v2/v2.1 arch.**
- **MeloTTS** (MyShell): VITS/VITS2/**Bert-VITS2** lineage — text encoder + **stochastic** duration predictor + BERT linguistic features + HiFiGAN. The **Chinese speaker natively supports zh+en code-mix**. Ships at **44.1 kHz**, which is its on-device weakness (see §3).
- **Piper** (rhasspy): straight VITS configs (x-low/low/medium/high), 22.05 kHz, MIT. The de-facto edge VITS baseline in sherpa-onnx.
- **Bert-VITS2 / GPT-SoVITS**: VITS backbone + large BERT/LLM conditioning; strong quality but frontend/params balloon past our budget.

### 1.2 Flow-matching / ODE (CFM) — *the challenger*
- **Matcha-TTS** (Mehta et al., ICASSP 2024, arXiv 2309.03199): text encoder + **OT-CFM** decoder whose backbone is a **1D-conv U-Net with a Transformer block per residual stage** (Grad-TTS lineage). First-order Euler ODE; **NFE ≤ 10** (the sherpa zh-en export uses **3 steps**). Deterministic duration (MAS-derived, like FastSpeech) but a **distributional** CFM decoder → avoids the FastSpeech mean-regression wall. No built-in stochastic duration predictor.
- **matcha-icefall-zh-en** (k2-fsa): the concrete zh/en model — `model-steps-3.onnx` (3-step ODE) + **pinyin lexicon** + **`vocos-16khz-univ.onnx`** vocoder, 16 kHz, with zh TN rule-FSTs. **This is the model that measured RTF 0.18 on our Nano.**
- **Supertonic / Supertonic-2** (Supertone, 2025): **66M**, **ConvNeXt** backbone, speech-autoencoder + **flow-matching** text-to-latent + duration predictor, LARoPE alignment, **~2 inference steps**, ONNX-first, multilingual, on-device focus (claims up to 167× RT on M4 Pro); an int8 sherpa-onnx export exists (2026-03). Apache-ish, ONNX weights public.
- **F5-TTS / E2-TTS** (2024): DiT flow-matching, ~330M, zero-shot; SEED-TTS-eval **CER 1.56% test-zh**. LLM-scale, not a Nano candidate but the reference for code-switch quality.
- **VoiceFlow / ReFlow-TTS / StableTTS / RapFlow-TTS (2025)**: rectified-flow / consistency variants pushing NFE→1–2. Direction of travel, not yet a shipped zh/en edge model.

### 1.3 FastSpeech-class (deterministic) + external vocoder — *our v1 lineage*
- FastSpeech2 / LightSpeech / EfficientSpeech: fully parallel, cheapest, but **deterministic → mean-regresses F0/prosody** (our own prior finding: a capacity-independent wall at small scale). Retired for us in favor of distributional models. Listed for completeness / streaming-cheapness.

### 1.4 Small autoregressive / LLM-token TTS — *unsuitable for Nano*
- **MOSS-TTS-Nano (~100M)**, KittenTTS (~15–25M, en, CPU-fast), Parler-mini, OuteTTS, Kani-TTS, VUI, tiny VALL-E derivatives, CosyVoice-1/2/3 (LLM+FM). AR = **one forward pass per codec token = thousands of sequential kernel launches**. On a launch-bound Maxwell with a 2 s kernel watchdog and no CUDA graphs this is the worst-case pattern. Naturally streaming, but latency/RTF on this device is prohibitive. Excluded from the Nano shortlist.

### 1.5 Diffusion / style-diffusion — *mixed*
- **StyleTTS2** (~148M): diffusion only for the *style vector* (cheap), decoder is iSTFTNet — but total size and LSTM/duration stack are heavy; weak zh.
- **Kokoro-82M** (StyleTTS2-derived): **decoder-only, iSTFTNet vocoder, no diffusion at inference**, Apache-2.0, <100 h training data, punches above weight — but primarily en/British-en (zh added late and weaker) and its LSTM path is a **launch risk on Maxwell** (prior rejection stands).
- NaturalSpeech 2/3: on-device infeasible.

### 1.6 Comparison table

| Model | Params | SR | Inference pattern | Vocoder | Quality (reported) | On-device fit (Nano) | zh/en mix | Streaming | License |
|---|---|---|---|---|---|---|---|---|---|
| **MB-iSTFT-VITS (ours, v2.1)** | 34.7M | 16k | single-pass, distributional | multi-band iSTFT | **X-ASR CER 0.027** (ours); ≈VITS MOS | **Measured RTF 0.42 GPU** / 0.52 CPU@4thr | yes (ours) | yes (agent) | MIT (code) |
| VITS / VITS2 | 30–40M | 22k | single-pass | HiFiGAN | high MOS | flow = many small ops | via retrain | moderate | MIT |
| MeloTTS (zh) | ~50–60M | 44.1k | single-pass | HiFiGAN | high MOS | **RTF 2.5@4thr RPi4** (44k kills it) | **yes native** | moderate | MIT |
| Piper (medium) | ~20–30M | 22k | single-pass | HiFiGAN | good | RTF ~0.35@4thr RPi4 | en-centric | moderate | MIT |
| vits-icefall-zh-aishell3 | ~29MB | 8k | single-pass | HiFiGAN | ok | **RTF 0.156@4thr RPi4** (8k) | zh | moderate | Apache |
| **Matcha-TTS (zh-en, 3-step)** | ~18–26M | 16k | **iterative, 3-step CFM** | **Vocos-16k** | ≈VITS MOS; good | **Measured RTF 0.18 GPU** | yes (pinyin) | chunkable | MIT |
| Supertonic-2 | 66M | 24k+ | 2-step flow, ConvNeXt | speech-AE | high; very fast | ONNX/int8; promising | multilingual | chunkable | open |
| Kokoro-82M | 82M | 24k | single-pass (no diff) | iSTFTNet | SOTA-small MOS | LSTM launch risk | weak zh | moderate | Apache-2.0 |
| F5-TTS | ~330M | 24k | many-step DiT flow | Vocos | **CER 1.56 test-zh** | too big | yes | no | MIT |
| CosyVoice-2/3 | 0.5–1.5B | 24k | AR LLM + FM | — | best code-switch | infeasible | best | yes | Apache |
| MOSS-TTS-Nano | ~100M | — | AR codec-token | codec dec | — | AR = launch-bound death | — | yes | open |
| FastSpeech2-class | 5–30M | var | single-pass **deterministic** | any | **prosody mean-regresses** | cheapest but flat | via retrain | yes | MIT |

*(RPi4 = Cortex-A72, a step above our A57; relative ordering transfers. GPU RTFs are our own Nano measurements.)*

---

## 2. Code-mix (zh/en) frontend analysis

**Two frontend philosophies in the field:**
- **Explicit G2P + phone tokens** (VITS/Matcha/MeloTTS/icefall): a lexicon or G2P maps text to
  phones; language is disambiguated at the *phone* level or via a language tag. Small, deterministic,
  cheap on-device. **← our path.**
- **Raw text / BPE into an LM** (CosyVoice, F5, Fish/OpenAudio, IndexTTS): the model learns G2P
  implicitly; best code-switch quality but requires LLM-scale params. Not viable at Nano size.

**Phone set choices among small open zh/en systems:**
- `matcha-icefall-zh-en`: **pinyin** lexicon + espeak-ng for English, language handled by lexicon
  entries, tones baked into pinyin tokens.
- MeloTTS-Chinese: pinyin + **BERT** (`bert-base-multilingual`) linguistic features; en handled by
  the same zh model for code-mix.
- **Ours**: **g2pw (bopomofo) + g2p_en**, unified into an **88-symbol set with three parallel
  embeddings — phone + tone + language**. This is the cleaner design for zh-TW because bopomofo is
  the native Taiwanese notation and separating tone into its own embedding lets the encoder share
  phone identity across tones (better data efficiency at small scale).

**Polyphone disambiguation:**
- **g2pW** (Chen et al., arXiv 2203.10430): conditional weighted-softmax BERT, SOTA on the CPP
  dataset; Taiwan-origin, integrates naturally with bopomofo. **Our choice — still SOTA-competitive
  in 2026.**
- Alternatives (g2pM; 2025 end-to-end BERT G2P, arXiv 2501.01102; G2PL lexicon-adapter): marginal
  accuracy deltas, all still BERT-based. No compelling *smaller/BERT-free* winner has displaced g2pW.
  Note: g2pW's BERT runs **once at frontend time** (host-side in our pipeline), not on the Nano audio
  hot path, so its cost is not a deployment constraint for us.

**Tone / language embedding:** best practice at small scale is exactly what we do — **separate tone
embedding** (not tone-tagged phones) + a **language embedding** for accent/consistency control. This
also gives a knob for the "one voice across languages" accent-consistency goal.

**Code-switch prosody & accent:** the literature (SEED-TTS-eval code-switch subset; CosyVoice3
cross-lingual zh2en/en2zh) shows the hard problems are (a) prosodic continuity across the switch
boundary and (b) accent leakage. LLM-scale models win here; at our scale the language embedding +
a single consistent teacher voice (our VibeVoice/Xinran distillation) is the right lever, and our
measured CER 0.027 says it is working.

**Text normalization (entity-heavy):** the field standard is **rule-FST TN** — WeTextProcessing /
the `number-zh.fst`/`date-zh.fst` FSTs shipped with sherpa matcha/vits, or NeMo TN. Rule-based is
correct for numbers/emails/serials (deterministic, auditable). Keep our rule-based/FST TN; do not
hand entities to a neural frontend.

**Verdict on frontend:** **keep g2pw + 3-embedding.** It is state-of-the-practice for small zh-TW
code-mix and better-suited to zh-TW than the pinyin-only icefall frontend. The *only* frontend change
implied by a Matcha migration is duration sourcing (§4).

---

## 3. The Nano performance frontier (constraint-driven)

Two hard constraints define the sweet spot:

- **GPU (Maxwell sm_53, 472 GFLOPS, 2 s watchdog, no CUDA-graph replay):**
  time ≈ `Σ(kernel launches) × launch_overhead + compute`. With no graph capture, launch overhead is
  *paid per kernel every inference*. This **rewards fewer, larger, denser ops** (parallel convs,
  U-Net blocks) and **punishes many-small-op graphs** (normalizing flows with stacked WaveNet
  couplings, AR token loops, LSTMs). Our own measurement: the 34.7M conv model **floors at RTF 0.42
  regardless of precision** (F16 GEMM proven neutral) — i.e., we are launch-bound, not FLOP-bound.
- **CPU (Cortex-A57, ARMv8.0, fp32-only):** no int8 sdot, no fp16 arith → **fp32 is the only fast
  path** (ORT-MLAS fp32 RTF 0.52@4thr; int8 either breaks the voice or is slower). This **rewards
  small param counts** and penalizes anything relying on quantization for speed.
- **Sample rate is a first-order cost multiplier.** sherpa RPi4 numbers make this stark: the same
  VITS family runs **RTF 0.156 @ 8 kHz** (icefall-zh-aishell3) but **RTF 2.5 @ 44.1 kHz**
  (MeloTTS-zh) — a ~16× spread driven mostly by vocoder output length. Our **16 kHz** choice is the
  right middle: intelligible for code-mix + entities, without MeloTTS's 44.1 kHz tax.

**Where is the sweet spot?** The GPU wants **fewer-larger parallel ops**; the CPU wants **few
params**; both want **16 kHz** and a **low-launch vocoder (iSTFT/Vocos, not a HiFiGAN upsampling
stack)**. Our 20–40M conv-parallel iSTFT model sits in the right *region*. The open question the
survey surfaces is whether, *within* that region, a **dense few-step CFM U-Net** is a better
launch-profile match than a **normalizing-flow VITS** — which is exactly the Matcha anomaly.

---

## 4. The Matcha anomaly investigation

**Fact:** on this exact Nano, `matcha-icefall-zh-en` (~18–26M, 3-step) ran **RTF 0.18**; our
MB-iSTFT-VITS (34.7M) runs **RTF 0.42**. Both are **16 kHz**, so sample rate is *not* the cause.
Matcha is 2.3× faster despite being an *iterative* (3-pass) model. Why?

**Cause 1 — kernel-launch count (dominant).** On a no-CUDA-graph Maxwell, runtime tracks *number of
kernel launches*, not FLOPs. The two graphs differ structurally:
- **VITS inference path:** text encoder → duration → **inverse normalizing flow** (multiple affine
  coupling blocks, each a WaveNet stack of dilated conv + gate + residual = *many small kernels*) →
  multiband-iSTFT decoder (upsampling convs). The **flow is the launch multiplier** — dozens of tiny
  ops that each pay full launch overhead and barely use the 472 GFLOPS.
- **Matcha inference path:** text encoder → 3× **dense U-Net pass** (each pass: a handful of large
  1D-conv resblocks + transformer blocks over the whole sequence) → **Vocos** vocoder.

Even though Matcha runs the decoder **3×**, each pass is a *small number of large, dense kernels* —
high compute-per-launch, which is precisely what Maxwell can absorb (it has FLOPS to spare relative
to launch overhead). Net launches across a whole utterance are **fewer** than VITS's flow+decoder.
This is the core of the anomaly: **the 3-step ODE amortizes launches into big dense ops; VITS's flow
fragments them into small ops.**

**Cause 2 — the vocoder.** `matcha-icefall-zh-en` uses **Vocos-16 kHz** (arXiv 2306.00814): a
ConvNeXt backbone that does **all** work at *frame* resolution and upsamples **solely via inverse
STFT** — no temporal-upsampling conv stack. Vocos is reported **~13× faster than HiFiGAN** and ~70×
faster than BigVGAN. Our multi-band iSTFT vocoder is also efficient (that's why we chose it), but it
is embedded in the heavier VITS decoder/flow. So part of Matcha's win is a **cleaner, lower-launch
vocoder**, and part is the acoustic model.

**Cause 3 — sherpa-onnx runtime.** Both ran under sherpa-onnx/ORT, so the runtime is *not* the
differentiator here; it's a controlled comparison. (sherpa's graph is well-fused, which helps both.)

**Which cause dominates?** The acoustic flow, not the vocoder. Evidence: our own F16-GEMM-neutral
finding says we're launch-bound, and the flow contributes the majority of small ops in the VITS
graph. **Corollary: swapping only our vocoder to Vocos would give a partial win; replacing the
flow-based acoustic with a dense CFM U-Net is the structural fix.** This directly answers the
orchestrator's question — **yes, a flow-matching (CFM) acoustic + our/Vocos iSTFT vocoder is the
credible next-gen path**, and it is *measured*, not hypothesized: 0.18 vs 0.42 on our silicon.

**Streaming implication (for the parallel effort):** CFM/Matcha chunks naturally — the encoder output
and duration are known up front, and the U-Net can be run over sequence windows; Vocos is
frame-local. This is *more* streaming-friendly than unwinding a VITS flow. Worth flagging to the
streaming-design agent as a reason the fast lane and the streaming lane may converge.

**Caveats before crowning Matcha:**
- Matcha has **no stochastic duration predictor** — durations come from **MAS** at train time (a
  Glow-TTS-style aligner). Our current frontend feeds VITS's internal duration; a Matcha build needs
  an **MAS/alignment stage**. (We already have the MMS-aligner lesson on file: gate on **resynth
  CER**, not duration distribution.)
- **No warm-start** from our VITS weights into a Matcha U-Net (different arch) — the acoustic model
  retrains from scratch. Frontend, corpus, teacher, and TN **do** transfer.
- **Quality is unproven for our voice.** Our v2.1 is user-approved at CER 0.027; a Matcha rebuild
  risks a prosody/CER regression until tuned (ODE-step count vs quality is a knob: 3 steps is fast
  but 2 vs 4 changes MOS). This is why the verdict is *evolve/prototype*, not *replace*.

---

## 5. Verdict & costed plan

### 5.1 Verdict: **KEEP shipped, EVOLVE a fast lane** (do not replace v2.1 yet)

- **Keep MB-iSTFT-VITS (PrimeTTS v2/v2.1)** as the production flagship. Rationale: user-approved,
  CER 0.027 (beats its 7B teacher), already ported to RapidSpeech.cpp with **CPU parity (0.9998)**,
  and its RTF 0.42 GPU / 0.52 CPU is *acceptable*, just not best-in-class. No surveyed model at our
  size demonstrably beats it on **quality**. The A24 shrink (24.8M) remains a valid orthogonal win.
- **Evolve**: stand up a **Matcha-class CFM acoustic + Vocos-16 kHz vocoder** prototype as the
  next-gen **low-latency lane**, because the 0.18 RTF is *measured on our exact device* — a **2.3×
  latency win** that matters for interactivity/streaming and is the correct architectural match to a
  launch-bound Maxwell. Decide replace-vs-coexist **only after** the prototype passes the CER/CMOS
  gate against v2.1.

### 5.2 Prototype architecture (the fast lane)
- **Frontend:** unchanged — g2pw (bopomofo) + g2p_en, phone+tone+lang 3-embedding, rule-FST TN.
- **Aligner:** MAS/priorgrad to source durations (gate on resynth CER per our aligner lesson).
- **Acoustic:** Matcha-style CFM — text encoder + 1D-conv U-Net (transformer-per-stage) decoder,
  **3-step Euler ODE** (sweep 2/3/4 for the quality/latency knee).
- **Vocoder:** Vocos-16 kHz (proven low-launch) **or** re-use our multi-band iSTFT (already in
  RapidSpeech.cpp) — bench both; Vocos likely wins launches, our iSTFT wins integration.
- **Params/SR budget:** ~18–26M, **16 kHz** (unchanged), fits ~3.5 GB RAM trivially.

### 5.3 Corpus / warm-start reuse
- **Reusable:** the Xinran/VibeVoice distillation corpus, the g2pw frontend, the TN FSTs, the eval
  harness (X-ASR CER gate). **Not reusable:** VITS→Matcha weight warm-start (arch mismatch → train
  acoustic from scratch). This is the main *new* training cost.

### 5.4 Cost / benefit / risk
- **Benefit:** ~2.3× lower GPU latency (0.42→~0.18), better streaming fit, aligns with the streaming
  agent's work, modern arch trajectory (Supertonic/Matcha momentum in 2025–26).
- **Cost:** one from-scratch acoustic train + an MAS aligner stage + Vocos train/finetune; deployment
  work in RapidSpeech.cpp/ORT for a 3-step ODE loop (small).
- **Risk (ranked):** (1) prosody/CER regression vs a tuned, user-approved v2.1 — mitigate by keeping
  v2.1 shipped until the gate passes; (2) ODE-step/quality tradeoff eating the latency win; (3) MAS
  alignment quality (known failure mode — gate on resynth CER); (4) teacher-timbre transfer under a
  new acoustic. All are *contained* because v2.1 remains the fallback.

### 5.5 What NOT to do
- Do **not** adopt MeloTTS as-is (44.1 kHz = RTF 2.5 on ARM). Do **not** chase AR/LLM-token TTS
  (CosyVoice/F5/MOSS-Nano) — thousands of sequential launches are the anti-pattern for this GPU.
  Do **not** switch to Kokoro (LSTM launch risk on Maxwell, weak zh). Do **not** replace the frontend.

---

## Sources
- Matcha-TTS — arXiv 2309.03199 (ICASSP 2024): https://arxiv.org/abs/2309.03199 ; system arch: https://deepwiki.com/shivammehta25/Matcha-TTS/2-system-architecture
- matcha-icefall-zh-en config (3-step, pinyin lexicon, Vocos-16k vocoder) — sherpa-onnx pretrained models: https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html ; icefall matcha recipe: https://github.com/k2-fsa/icefall/blob/master/egs/ljspeech/TTS/matcha/export_onnx_hifigan.py
- MB-iSTFT-VITS — arXiv 2210.15975 (3.4–4.1× vs VITS, RTF 0.066 i7): https://arxiv.org/abs/2210.15975 ; ar5iv: https://ar5iv.labs.arxiv.org/html/2210.15975 ; repo: https://github.com/MasayaKawamura/MB-iSTFT-VITS
- iSTFTNet — arXiv 2203.02395: https://arxiv.org/pdf/2203.02395
- Vocos (ConvNeXt + iSTFT, ~13× faster than HiFiGAN) — arXiv 2306.00814 (ICLR 2024): https://arxiv.org/abs/2306.00814
- sherpa-onnx VITS RTF tables (RPi4: melo-zh_en 44.1k RTF 2.5@4thr; piper 22k ~0.35; icefall-zh-aishell3 8k RTF 0.156@4thr): https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html
- sherpa-onnx TTS overview / DeepWiki: https://deepwiki.com/k2-fsa/sherpa/3.2-tts-models ; repo: https://github.com/k2-fsa/sherpa-onnx
- MeloTTS (VITS/VITS2/Bert-VITS2, zh speaker does zh+en) — model card: https://huggingface.co/myshell-ai/MeloTTS-Chinese ; cpp port: https://github.com/apinge/MeloTTS.cpp
- g2pW (conditional weighted-softmax BERT, CPP dataset) — arXiv 2203.10430: https://arxiv.org/abs/2203.10430 ; 2025 end-to-end BERT G2P: https://arxiv.org/abs/2501.01102
- Supertonic (66M, ConvNeXt, flow-matching, 2-step, ONNX) — https://huggingface.co/Supertone/supertonic-2 ; sherpa int8 export: https://huggingface.co/csukuangfj2/sherpa-onnx-supertonic-tts-int8-2026-03-06
- Kokoro-82M (StyleTTS2 + iSTFTNet, no diffusion at inference, Apache) — https://huggingface.co/hexgrad/Kokoro-82M
- F5-TTS (flow matching, CER 1.56 test-zh) — arXiv 2410.06885: https://arxiv.org/html/2410.06885v1
- CosyVoice 3 (code-switch/cross-lingual SOTA, LLM+FM) — arXiv 2505.17589: https://arxiv.org/pdf/2505.17589
- SEED-TTS-eval (zh/en + code-switch benchmark) — referenced via CosyVoice3/F5 papers above.
