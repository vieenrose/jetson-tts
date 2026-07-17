# Voice provenance: "Xinran" (PrimeTTS stage-3 teacher voice)

**Decision (2026-07-02):** user selected VibeVoice-Large + `zh-Xinran_woman` preset as the
corpus teacher after A/B against fully-synthetic candidates (VoxCPM2 VoiceDesign lineage,
see `../v04d/` — superseded but archived).

## Generation chain

1. **Model:** VibeVoice-Large (~7B, Microsoft). Official HF repo was pulled by Microsoft;
   weights obtained from mirror `aoi-ot/VibeVoice-Large`
   (snapshot `1b81fecc784a076dcd935678db551871f4598ebf`). License: MIT.
   Also used: `microsoft/VibeVoice-1.5B` (official, snapshot
   `c00898d257e6b46004e3e2866a47534085fb685a`) for comparison only.
2. **Inference code:** `vibevoice-community/VibeVoice` (MIT), cloned at
   `third_party/VibeVoice`; installed in isolated `/home/luigi/vibe-venv`
   (user-authorized). Attention: SDPA (flash-attn not installed) — matches the
   samples the user approved.
3. **Voice preset:** `third_party/VibeVoice/demo/voices/zh-Xinran_woman.wav`, shipped in
   the MIT repo. Speech style: mature female, standard-Mandarin register.
4. **Corpus:** `mossnano/zhtw8k/vibe_gen_corpus.py` — cfg_scale 1.3, ddpm 10 steps,
   per-utterance seed = sha1(id), 24 kHz; texts = the 29,359-row cc0 text set;
   per-utterance QC gate (speaker-sim ≥ 0.86 to Xinran centroid, seeded retries).

## Legal basis & caveats (weaker than the fully-synthetic chain — flagged to user)

- **Copyright/ToS:** MIT weights + MIT preset, run locally — no service ToS in the chain
  (unlike Edge-TTS). MIT imposes no output/training restrictions.
- **Caveats:** (a) the preset is a recorded/produced voice of UNKNOWN speaker provenance —
  if it derives from a real voice talent, her consent chain runs to Microsoft, not us;
  (b) Microsoft pulled the Large model from distribution (concerns unstated) — we rely on
  a third-party mirror of MIT-licensed weights, which is lawful but worth documenting;
  (c) mark product voice as AI-generated (Taiwan AI Basic Act transparency).
- Full analysis: memory note `voice-provenance-legal` + session research 2026-07-02.

## Hashes

See `hashes.txt`.
