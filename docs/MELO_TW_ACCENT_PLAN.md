# Plan — MeloTTS 8 kHz zh-TW/en ONNX model (Jetson Nano gen1)

**Objective:** a `melotts-zh_tw_en-8k` sherpa-onnx drop-in (sample_rate=8000, fp32, opset≤17, no custom
ops) with **Taiwan readings + Taiwan accent** and retained zh/en code-mixing, running real-time on the
Jetson Nano gen1 CPU.

**Why melo (vs the Matcha struggle):** MeloTTS is VITS — **end-to-end** text→waveform, one jointly-trained
stack, so there is **no separate-vocoder coupling** (the exact failure that drops words in Matcha). And our
8 kHz distillation is **self-consistent**: we distill on melo's *own* latent `z`, so train==inference
distribution by construction → no dropped/unclear words. Cost: melo's zh/en code-mix baseline is weaker
than Matcha, and 8 kHz fidelity ceiling is ~PESQ 2.9 (telephony-fine; melo-8k is already device-accepted).

## Assets already in repo (reuse)
- Melo training stack: `third_party/MeloTTS/melo/{train.py,data_utils.py,preprocess_text.py,models.py,losses.py}`.
- Validated 8 k distillation: `scripts/dump_teacher.py` (`--zero-bert`), `student/train.py` (Vocos8k),
  `scripts/export_full_onnx.py` (drop-in ONNX, sets sample_rate=8000). Released baseline `Luigi/vits-melo-tts-zh_en-8k`
  (device-accepted, Nano RTF 0.34 @4thr).
- TW readings: `data/tw_readings/{char,word}_overrides.tsv` (垃圾 lèsè · 期 qí · 究 jiù · 質 zhí …),
  `scripts/apply_tw_lexicon.py`. Melo Chinese g2p: `third_party/MeloTTS/melo/text/chinese*.py`.
- TW-accent corpus: `matcha_eval/tw_qwen_corpus` (14,752 clips / 18 h, Qwen3-cloned young-TW-female).
  English/code-mix retention text: `data/text/en_retention*.tsv`. Eval: `matcha8k/{zh_eval_lines,en_eval_lines}.txt`,
  X-ASR + opencc CER gate (`matcha8k/xasr_verify.py`).

## Phases

### Phase 0 — Setup & de-risk (host, GPU)
- Stand up melo **training** env (training deps beyond inference: discriminators, mel/KL losses,
  `mono-align`, webrtcvad). Confirm `melo/train.py` runs one step on a tiny shard from the base zh_en ckpt.
- Confirm we can resume from the official `myshell-ai/MeloTTS-Chinese` (zh_en) checkpoint (config + G/D/optim).
- Decide native fine-tune SR = melo's 44100 (decoder fixed); 8 k is produced only by re-distillation (Phase 3).

### Phase 1 — TW readings front-end (cheap, no training)
- Port the `data/tw_readings` overrides into melo's Chinese g2p (pypinyin tone3 lookup in `chinese.py`/
  `chinese_mix.py`): override per-char/word pinyin before phoneme conversion. Same readings layer already
  validated for Matcha; this makes melo say lèsè/qí/jiù/zhí without retraining.
- Gate: render the readings A/B (same 8 lines as the matcha readings set) → confirm TW readings.

### Phase 2 — Accent fine-tune (end-to-end VITS, 44.1 kHz)
- Preprocess the TW corpus to melo format (`preprocess_text.py`): resample 24 k→44.1 k, build the
  train/val lists with melo's bert/phoneme features (BERT **on** for training quality; deployment zeroes it).
- **Mix in English/code-mix retention** (pure-EN + EN-heavy lines, in the same TW voice or base-EN audio)
  at ~30–40 % to fight catastrophic forgetting (the Matcha lesson).
- Fine-tune full VITS from the base zh_en ckpt, **low LR**, modest steps; checkpoints frequent.
  Anti-forgetting options if needed (research-backed): freeze the **text encoder** (preserves content/
  pronunciation), train flow+decoder+duration; or L2-to-base. Gate on the ASR CER/recall set, not by step count.
- Gate (44.1 k output): X-ASR Chinese CER ≈ base + English recall ≈ base + audible TW accent. Pick best ckpt.

### Phase 3 — Re-distill the 8 k vocoder from the ACCENTED melo (self-consistent)
- `scripts/dump_teacher.py --zero-bert` on the **fine-tuned** melo → dump (`z[192,T]`, `g[256]`, 44.1 k audio→8 k).
  (Critical: `--zero-bert` so the distill matches the deployed BERT-less graph — established constraint.)
- `student/train.py --arch vocos` → 8 k Vocos decoder on the accented melo's own `z`. Self-consistent
  (train==inference `z`) → **no coupling gap, no dropped words**. Target PESQ ~melo-8k baseline (~2.9).
- `scripts/export_full_onnx.py` → `model.onnx` (accented melo enc/flow + 8 k decoder, sample_rate=8000,
  opset 17, fp32, no custom ops). Same 7-input/1-output contract as melo-8k.

### Phase 4 — Verify, device-gate, ship
- sherpa-onnx end-to-end: clean code-mixed 8 k audio; X-ASR CER/recall ≥ targets; PESQ; G.711 phone A/B.
- Host x86 ORT-CPU RTF @1/2/4 thr → Nano RTF via ×13 factor. **Arch identical to melo-8k → Nano-compatible
  by construction** (accent fine-tune changes weights, not graph/op-set/size).
- Upload A/B to `Luigi/zh-en-tts-8k-comparison`; on user approval publish `Luigi/melotts-zh_tw_en-8k`
  with DEVICE_ACCEPTANCE.md.

## Risks / mitigations
- **English forgetting** — retention mix + low LR + freeze text encoder; ASR-gated.
- **Melo zh/en weaker than Matcha** — accept lower baseline; this path's win is *clean, no dropped words*.
- **Teacher-voice quality** — melo distills its *own* audio (not qwen directly), so 8 k fidelity is bounded by
  melo (~2.9), independent of the qwen ceiling that capped the Matcha vocoder.
- **Melo training env** — first real setup cost; Phase 0 de-risks it before committing the full run.

## Decision gate
Run only if (a) the Matcha joint co-training (task #19) does **not** clear the bar, or (b) user wants the
melo variant regardless. Melo = easier/cleaner path, lower zh/en ceiling; Matcha-joint = higher ceiling, harder.

## Progress log
- **Phase 0 DONE (2026-06-15):** melo training env de-risked. Base = `MeloTTS-Chinese/checkpoint.pth`
  (G-only, 256-spk, 44.1k, spk2id ZH:1). Fixes (see `scripts/melo_train_compat.patch`): merge full
  train/data hyperparams into the stripped inference config; patch dead-S3 `load_pretrain_model` (pass
  `--pretrain_G` explicitly, D/dur fresh); patch matplotlib `tostring_rgb`->`buffer_rgba`. Smoke: trained
  22+ epochs on 200 clips, saved G/D/DUR_200.pth, sane losses [disc 2.9, gen 1.5, mel 14, kl 2.0].
  Data prep = `scripts/melo_prep_corpus.py` (resample 44.1k + metadata) -> melo `preprocess_text.py` g2p.
- **NEXT — Phase 1 g2p robustness:** code-mix g2p drops ~70% of lines (phone numbers/punct/English).
  Must fix melo Chinese g2p (`text/chinese_mix.py`) or pre-normalize metadata before the full preprocess.
