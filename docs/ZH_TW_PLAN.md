# Plan: zh-CN/en → zh-TW/en TTS (Taiwan Mandarin + good English + code-mixing, on edge)

## The gap (nobody has all four)
| system | TW accent | good English | zh/en mixing | lightweight edge (8k CPU) |
|---|---|---|---|---|
| our matcha-8k / melo-8k | ✗ (Mainland Putonghua) | ✓ | ✓ | ✓ |
| BreezyVoice (MediaTek) | ✓ | ✗ | partial | ✗ (CosyVoice = ~0.5B LLM + flow-matching) |
| Breeze2VITS (sherpa VITS) | ✓ | ✗ (maps `hello`→`哈囉`; "英數表現不佳") | ✗ | ✓ |

Goal = TW accent **and** good English **and** code-mixing **and** edge-deployable. That combination doesn't exist; we have to build it.

## zh-TW decomposes into TWO independent layers
1. **Readings / lexicon (字音)** — Taiwan picks different syllables for some words:
   垃圾 lè sè (TW) vs lā jī (CN); 和(conj) hàn vs hé; 研究 yán jiù vs yán jiū; 企業 qì yè vs qǐ yè;
   法國 Fà vs Fǎ; plus polyphone defaults (期/質/暴…). **This is FRONT-END only** (pinyin selection) —
   the acoustic model can already *say* the right syllable; we just need to choose it. Cheap, no retrain.
2. **Accent / phonetics / prosody (腔調)** — retroflex reduction (zh/ch/sh→z/c/s), no 兒化, fewer neutral
   tones, ㄥ/ㄣ merging, Taiwan intonation & rhythm. **This is ACOUSTIC** — baked into the model's
   training audio. A CN-trained acoustic model sounds Mainland no matter the pinyin. Needs acoustic
   fine-tuning / TW training data.

**Key reframing for the product:** Taiwanese speakers code-switch English with near-native pronunciation.
So the ideal target is **TW-accent Chinese + our existing (good) English**, NOT TW-accented English. That
means we *keep* matcha's English and only shift the Chinese — which is exactly what fine-tuning-with-retention
gives. This makes the goal more achievable than it first looks.

## Reference roles
- **BreezyVoice** (Apache-2.0, zero-shot, clean TW Mandarin, bopomofo polyphone control): too heavy to
  deploy (LLM), but an excellent **offline TW-accent data generator / teacher**. Use it to synthesize a
  clean, consistent-voice TW-Mandarin corpus. Its weak English is irrelevant — we only use it for Chinese.
- **Breeze2VITS**: confirms the lightweight-TW-without-English failure mode; not a base for us.
- **Our matcha-icefall-zh-en**: the lightweight acoustic+vocoder base with great English/mixing — the thing
  we adapt. Matcha is fine-tunable (k2-fsa/icefall `egs/ljspeech/TTS#matcha` + Matcha-TTS fine-tune recipe).

## Data options for the accent layer
- **BreezyVoice-generated TW corpus** (preferred): clean, single reference voice, arbitrary text incl. our
  receptionist corpus. Apache-2.0 → synthetic derivatives OK.
- **Common Voice zh-TW**: ~80 h validated Taiwan Mandarin, 2317 speakers (real accent, but multi-speaker /
  crowd-noisy → better for accent coverage / ASR-style than clean single-voice TTS).
- **TAT / TAT-TTS**: large, but largely Taiwanese **Hokkien (台語)**, not Mandarin — mostly not what we want.
- **Code-mixed retention set**: generated from our matcha (English + zh/en) so fine-tuning doesn't forget English.

## Phased plan

### Phase 0 — feasibility spikes (1–2 days)
- Confirm we can obtain/repro a **trainable Matcha zh-en checkpoint** (the dengcunqin/icefall PyTorch ckpt,
  not just ONNX). If unavailable → fall back to retraining Matcha from the icefall recipe on combined data.
- Stand up BreezyVoice locally on the 5090s; generate ~20 TW-Mandarin clips to confirm quality + pick a voice.
- Quick lexicon PoC: patch ~30 TW readings into matcha's lexicon, render, sanity-check.

### Phase 1 — Taiwan-readings front-end (ship now, cheap, high ROI)
- Build a **TW-readings override lexicon** (curated list of CN→TW reading differences + polyphone defaults;
  borrow BreezyVoice's bopomofo-disambiguation idea) and apply to **matcha-8k and melo-8k** `lexicon.txt`.
- Traditional-input already works (verified earlier). On-device = swap lexicon only, **no retrain, keeps
  English/mixing**. Fixes the most jarring "Mainland tells" (lèsè, hàn, …). Ships as a model-dir update.
- Deliverable: updated lexicons + A/B samples; ~60–70% of perceived "Taiwanese-ness" at near-zero cost.

### Phase 2 — TW accent via BreezyVoice-distilled data (the real accent; ~weeks)
1. Generate a large **TW-Mandarin corpus** with BreezyVoice (offline, 5090s), consistent reference voice,
   text = receptionist + general + numbers/names.
2. Build a **code-mixed/English retention set** from matcha.
3. **Fine-tune the Matcha acoustic model** on (BreezyVoice TW Chinese + retention) — continual-learning mix
   to shift Chinese→TW accent while preserving English. (Or retrain Matcha from the icefall recipe if no ckpt.)
4. **Re-distill the 8k vocoder**: re-dump mels from the fine-tuned acoustic model, light-tune our VocosMel8k
   (reuse the whole matcha8k pipeline — vocoder is largely accent-agnostic, so this is cheap).
5. Export drop-in (sample_rate=8000), verify in sherpa, evaluate (PESQ/MCD + native-speaker A/B).

### Phase 3 — polish & ship
- Native-speaker listening eval (TW accent correctness + English intactness), iterate.
- Publish `Luigi/matcha-zh-tw-en-8k` drop-in + DEVICE_ACCEPTANCE; update comparison dataset.

## Risks / decisions
- **Trainable-checkpoint dependency** (Phase 2 gate): have ONNX, need PyTorch ckpt or recipe repro.
- **English retention** (catastrophic forgetting): mitigate with retention data + low LR + mixing ratio.
- **Voice identity shifts** to the BreezyVoice reference voice — acceptable for the attendant product (confirm).
- **Compute win unchanged**: this is about *accent quality*, not speed; RTF stays ~matcha-8k's ~0.18 A57.
- Licensing: BreezyVoice & sherpa Apache-2.0, Matcha MIT → synthetic data + derivatives fine, keep attribution.

## Recommendation
Do **Phase 1 immediately** (cheap, ships a real improvement, low risk) and run **Phase 0 spikes in parallel**
to de-risk Phase 2. Decide on full Phase 2 once the trainable-checkpoint question and a BreezyVoice data
sample are in hand.
