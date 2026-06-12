# MeloTTS zh_en — Teacher decoder config (verified against live checkpoint)

Source: `myshell-ai/MeloTTS-Chinese` (language `ZH` → internal `ZH_MIX_EN`, the zh/en
code-mixed model). The old S3 base-speaker bucket is decommissioned; HF is the live source.
Config fetched from `…/MeloTTS-Chinese/resolve/main/config.json`.

## Audio / framing
| param | value |
|---|---|
| sampling_rate | 44100 Hz |
| filter_length (n_fft) | 2048 |
| hop_length | **512** |
| win_length | 2048 |
| **z frame rate** | 44100 / 512 = **86.1328125 Hz** |

## Latent interface (the distillation contract)
Decoder call (models.py:1019): `o = self.dec((z * y_mask)[:, :, :max_len], g=g)`
| tensor | shape | meaning |
|---|---|---|
| z | `[B, 192, T]` | post-flow latent, decoder input (inter_channels=192) |
| g | `[B, 256, 1]` | speaker embedding (gin_channels=256) |
| o | `[B, 1, T*512]` | 44.1 kHz waveform |

- hidden_channels = 192, inter_channels = 192.
- **n_speakers = 256 but spk2id = {'ZH': 1} → effectively single speaker.** g is a single
  fixed 256-d vector at inference. Big simplification: g is a constant for our student; we
  keep it as a graph input for drop-in compatibility but it never varies.
- n symbols = 112.

## Teacher HiFi-GAN decoder (Generator)
| param | value |
|---|---|
| upsample_rates | [8, 8, 2, 2, 2]  (∏ = **512** = hop ✓) |
| upsample_kernel_sizes | [16, 16, 8, 2, 2] |
| upsample_initial_channel | 512 |
| resblock | type "1" |
| resblock_kernel_sizes | [3, 7, 11] |
| resblock_dilation_sizes | [[1,3,5], [1,3,5], [1,3,5]] |

So the teacher maps 86.13 Hz frames → 44100 Hz by an **integer ×512** upsample stack.

## ⚠️ The central design fact for the 8 kHz student
The z frame grid is **fixed at 86.1328125 Hz** (set by enc/flow, which we must keep
byte-identical). The teacher hits 44100 because 44100 = 86.1328125 × 512 (clean integer).

**8000 is NOT an integer multiple of 86.1328125** → 8000 / 86.1328 = **92.8798…**
Equivalently 8000/44100 = 80/441, and 441 = 21² shares no factor that yields a clean
per-frame integer upsample. There is therefore **no clean integer ConvTranspose/iSTFT stack
that lands on exactly 8000 Hz** from this frame grid. The reachable-by-integer-stack rates
are 86.1328 × N for integer N:

| N (∏ upsample) | output rate | factorization | Δ vs 8000 |
|---|---|---|---|
| 128 | **11025 Hz** (= 44100/4) | [8,8,2] / [8,4,4] | +37.8% |
| 100 | 8613.3 Hz | [5,5,4] / [10,10] | +7.7% |
| 96 | 8268.75 Hz | [8,4,3] / [4,4,3,2] | +3.4% |
| 93 | 8010.4 Hz | 3×31 (ugly) | +0.13% |
| 92 | 7924.2 Hz | 4×23 (ugly) | −0.95% |
| 90 | 7751.9 Hz | [5,3,3,2] | −3.1% |

This is a real decision that also touches the **device drop-in contract** (what sample_rate
the device app trusts), which cannot be tested on this host. See the question raised to the
user. Nyquist for the telephony band (≤3400 Hz) is satisfied by any N ≥ ~90.
