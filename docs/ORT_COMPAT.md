# ORT CPU export-contract probe (run BEFORE training)

`scripts/ort_compat_probe.py` validates that the student decoder's critical-path ops export
and run in the oldest installable ONNX Runtime, matching the device's sherpa-onnx-pinned ORT.

## Result (PASS)
- Export: **opset 17**, fp32, legacy TorchScript exporter (`torch.onnx.export(..., dynamo=False)`).
  torch 2.10 defaults to the dynamo exporter (needs onnxscript); legacy is more predictable
  for ORT 1.17 and is what we use for the real export too.
- Runtime: **onnxruntime 1.17.0** (oldest on PyPI for py3.12), **CPUExecutionProvider**.
- Ops emitted: `Add Cast Clip Concat Constant Conv ConvTranspose Cos Exp Mul Resize Shape Sin
  Slice Tanh Unsqueeze` — **all standard, no custom ops**, all in ORT 1.17.
- Dynamic time axis works (T = 200 / 380 / 711 z-frames).
- Numerical agreement torch vs ORT CPU: **max abs err ~1e-6** (fp32 noise).

## Ops de-risked for the two students
| op | used by | status |
|---|---|---|
| `Conv1d` / `ConvTranspose1d` | HiFi-GAN upsample (student A), conv body | ✓ |
| `Resize` (linear) | latent time-resample 86.13→125 Hz (exact-8000 path) | ✓ |
| iSTFT as fixed `ConvTranspose1d` overlap-add | Vocos/iSTFTNet head (student B) | ✓ |
| `Cos/Sin/Exp/Clip/Tanh` | STFT-domain mag/phase reconstruction | ✓ |

## Exact-8000 framing confirmed
z @ 86.1328 Hz → Resize to 125 Hz frames → iSTFT(n_fft=256, hop=64). Output samples =
out_frames × 64 (after trimming the n_fft−hop iSTFT tail) → 8000.0 Hz exactly. Nyquist 4000 Hz
covers the ≤3400 Hz telephony band.

## Pins
torch 2.10.0+cu128, onnx 1.16.0, onnxruntime 1.17.0. Export uses opset 17.
