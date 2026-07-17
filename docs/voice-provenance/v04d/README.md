# Voice provenance: "v04d" (PrimeTTS stage-3 flagship voice)

**Asset:** `mossnano/zhtw8k/voice_bakeoff/v04d_design.wav` (48 kHz, ~20 s zh-TW, young female)

## Generation chain (fully synthetic — no real person's voice anywhere)

1. **Voice design (2026-07-02):** VoxCPM2 (`openbmb/VoxCPM2`, Apache-2.0, "free for commercial use",
   snapshot `bffb3df5a29440629464e5e839f4d214c8714c3d`) VoiceDesign mode — unconditioned generation
   from the text description, no reference audio:
   - Description: 「年輕女生，聲音高而甜，帶台灣腔調，講話輕快」
   - Script: `mossnano/zhtw8k/voice_bakeoff.py` (candidate v04 of 16)
   - Output: `voice_bakeoff/v04_design.wav`
2. **Denoise + self-clone (2026-07-02):** v04_design had an elevated noise floor (−55.7 dBFS).
   Regenerated the same design text with `reference_wav_path=v04_design.wav, denoise=True`
   (VoxCPM2 built-in ZipEnhancer denoiser, `iic/speech_zipenhancer_ans_multiloss_16k_base`):
   - Script: `mossnano/zhtw8k/voice_bakeoff/fix_v04.py`
   - Output: `v04d_design.wav` (floor −67.2 dBFS; speaker cos to v04 = 0.867; F0 median 339 Hz)
3. **Selection:** user ear-picked v04 (2026-07-02), then v04d over re-roll alternatives
   (v04r4/v04r1) and the other 15 designs. Alternatives kept on disk for audit.

The clip's spoken text (used as `prompt_text` in corpus generation) is in `design_text.txt`.

## Legal basis

- **Copyright:** VoxCPM2 weights are Apache-2.0 with no output-use restrictions; ASF guidance
  treats model outputs as not copyrightable. Model run locally — no service ToS attaches.
- **Personality rights:** the voice is model-invented (no reference audio, no real speaker);
  no identifiable individual → no publicity/personality-rights holder (Taiwan Civil Code
  18/195 analysis in memory note `voice-provenance-legal`).
- **Replaces:** the prior corpus reference derived from Microsoft Edge-TTS `zh-TW-HsiaoChen`,
  which violated the Microsoft Services Agreement AI clause (no training on AI-service output).
  Models distilled from that voice should not ship.
- **Disclosure:** mark the product voice as AI-generated (Taiwan AI Basic Act transparency
  principle, in force 2026-01-14).

## Residual-risk mitigations

- Coincidental sound-alike: voice was screened against our own speaker pool only; if the
  product becomes prominent, run a speaker-verification screen against known zh-TW public voices.
- Corpus generation QC: per-utterance speaker-similarity gate (≥0.85 to v04d centroid) +
  noise-floor gate with retry, so fluke rolls (observed: one en clone at cos 0.767) are resynthesized.

## File hashes (sha256)

See `hashes.txt` (generated alongside this file).
