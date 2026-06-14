# Training — jetson-tts zh-TW accent model (Matcha-TTS)

## Overview

The deployed model is a **task-vector–interpolated** Matcha-TTS zh-en acoustic model with α=0.3, paired with a freshly distilled 8 kHz Vocos vocoder. Three stages:

1. **Full fine-tune** on Qwen3-TTS zh-TW synthetic corpus → θ_ft
2. **Accent vector interpolation** θ = θ_base + α·(θ_ft − θ_base) at α=0.3
3. **Vocoder re-distillation** from the blended acoustic (teacher-forced mels → 8 kHz audio)

## 1. Full fine-tune (produces θ_ft)

```bash
python -m matcha8k.finetune \
  --root matcha_eval/tw_qwen_corpus \
  --out matcha8k/ft_runs/tw_v2 \
  --device cuda:0 --steps 3000 \
  --batch 16 --lr 1e-5 --save-every 1000
```

- **Model**: MatchaTTS (n_vocab=2190, n_spks=1, spk_emb_dim=64, RoPE encoder 6L×192d, CFM decoder 1-block+2-mid, out_size=128)
- **Corpus**: Qwen3-TTS zh-TW voice-cloned code-mixed (~1400 clips, 16 kHz), plus English retention set
- **Loss**: duration + prior + CFM diffusion (standard Matcha forward pass, out_size=128 random crop)
- **Collate**: `max_yl` padded to multiple of 4 (CFM U-Net requirement)
- **LR**: 1e-5 (low, to shift accent without destroying English)
- **Result**: accent OK, EN recall only 0.65 → θ_ft alone isn't deployable, but serves as source for task vector

Checkpoint used: `matcha8k/ft_runs/tw_v2/ft_step3000.bin` (303 params changed vs base, max diff 0.015)

## 2. Accent vector interpolation (θ_base + α·τ)

```bash
python -m matcha8k.accent_vector \
  --base-ckpt models/matcha-src/pytorch_model.bin \
  --ft-ckpt matcha8k/ft_runs/tw_v2/ft_step3000.bin \
  --alphas 0.1,0.2,0.3,0.5,0.7,1.0
```

This computes the task vector τ = θ_ft − θ_base, then blends:

> θ_α = θ_base + α · τ

Only the 303 changed parameters are interpolated; everything else stays at base values.

Blended checkpoints saved to `matcha8k/ft_runs/accent_vectors/`.

ASR evaluation at each α:

| α | CER | EN recall |
|---|-----|-----------|
| 0 (base) | 0.409 | 0.733 |
| 0.1 | 0.409 | 0.733 |
| 0.2 | 0.409 | 0.800 |
| **0.3** | **0.421** | **0.733** |
| 0.5 | 0.478 | 0.733 |
| 0.7 | 0.496 | 0.915 |
| 1.0 | 1.000 | 0.415 |

**Chosen: α=0.3** — best zh-TW accent while preserving base CER and English quality.

Checkpoint: `matcha8k/ft_runs/accent_vectors/accent_ft_step3000_a0.3.bin`

## 3. Vocoder re-distillation

The blended acoustic model produces different mels from the base, so the 8 kHz vocoder must be re-distilled to match. Two sub-steps:

### 3a. Teacher-forced mel dump

```bash
python -m matcha8k.dump_tf_mels \
  --ckpt matcha8k/ft_runs/accent_vectors/accent_ft_step3000_a0.3.bin \
  --corpus matcha_eval/cv_combined \
  --ids matcha_eval/cv_combined_ids.json \
  --out matcha_eval/pairs_accent_a0.3 \
  --device cuda:0
```

For each clip: encoder(text) → MAS alignment against real mel → CFM decoder samples mel at ground-truth length → save (TTS_mel, 8kHz_audio) pairs. The vocoder trains on these TTS-conditioned features, matching what the acoustic model actually produces at runtime.

### 3b. Train 8 kHz Vocos vocoder

```bash
python -m matcha8k.train \
  --root matcha_eval/pairs_accent_a0.3 \
  --out matcha8k/runs/vocos8k_a0.3 \
  --device cuda:0 --steps 120000 \
  --batch 64 --seg-frames 48
```

- **Architecture**: VocosMel8k — Conv1d(80→384) + 8×ConvNeXtBlock(384) + Linear(384→513×2) → ISTFT head (n_fft=512, hop=128)
- **Loss**: λ_mel·TelephonyMelLoss + λ_stft·MultiResSTFTLoss + λ_adv·GAN + λ_fm·feature-match (warmup 1000 steps)
- **Mel**: 80 bins, 62.5 Hz frame rate (hop=128 @8kHz), fmax=4kHz (Nyquist)
- **Output**: ONNX graph produces (mag, cos_phase, sin_phase); sherpa does iSTFT to waveform

## 4. Export

### Acoustic model ONNX

```bash
python -m matcha8k.export_acoustic \
  --ckpt matcha8k/ft_runs/accent_vectors/accent_ft_step3000_a0.3.bin \
  --out export/matcha-zh-tw-en-8k-accent-a0.3/model-steps-3.onnx
```

Copies metadata (sample_rate etc.) from the reference `matcha-icefall-zh-en` ONNX.

### Vocoder ONNX

(Export script wraps VocosMel8k → ExportWrapper → mag/x/y per sherpa VocosVocoder contract.)

### Full sherpa-onnx drop-in

The ONNX model dir contains:

```
model-steps-3.onnx    # acoustic (enc+CFM, 3 ODE steps)
vocos-8k.onnx         # 8 kHz vocoder (mag/x/y iSTFT)
tokens.txt             # from base model
lexicon.txt            # TW-reading overrides applied
```

`sample_rate=8000` in metadata; sherpa-onnx runs it with no code changes.

## 5. Evaluation

```bash
# Quality (PESQ-NB + MCD)
python scripts/eval_quality.py --ckpt matcha8k/runs/vocos8k_a0.3/best.pt --root data/eval_pairs

# ASR intelligibility (CER + EN recall)
python matcha8k/asr_verify.py --dir matcha_eval/accent_eval_a0.3

# RTF (ORT CPU)
python scripts/bench_full_onnx.py --onnx export/matcha-zh-tw-en-8k-accent-a0.3/model.onnx
```

## Key design decisions

- **Task vector over LoRA**: LoRA on CFM decoder attention (even 0.4% params) catastrophically degrades flow-matching quality (CER 0.37→0.97). Task vector interpolation preserves the base distribution — only the accent direction is dialed in.
- **out_size=None during fine-tune**: Random 128-frame crops break with variable-length mels. The finetune script uses `out_size=128` with padded collate (`max_yl` rounded to mult of 4).
- **BERT zeroed**: The sherpa-deployed graph produces zero BERT embeddings. Training data must be dumped with `--zero-bert` to match runtime conditions.
- **speaker_id=1**: MeloTTS spk2id={'ZH': 1}. sherpa hardcodes the graph to use this embedding; the `--sid` flag is ignored.
- **Vocoder must match acoustic**: Re-distilling the vocoder from TTS-forced mels (not ground-truth mels) is essential — the base 8k vocoder was trained on base-model mels and produces audible artifacts with the blended acoustic.