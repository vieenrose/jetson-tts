# TensorRT vocoder path for jetson-tts (Jetson Nano gen1, sm_53)

The repo's models run on the **Nano CPU** via ONNX Runtime. This adds an optional
**TensorRT** path for the heaviest component — the **neural vocoder** — to offload it
to the Nano's Maxwell GPU and free CPU cores (useful in the multi-call phone-attendant
scenario). Measured on a real Jetson Nano gen1 (JetPack 4.5, **TensorRT 7.1.3**,
CUDA 10.2, sm_53, `jetson_clocks` pinned).

## Results (FP16, clocks pinned)

| model · vocoder | input | ORT-CPU 4thr | **TRT-FP16 GPU** | speedup | accuracy vs ORT | engine RSS |
|---|---|---:|---:|---:|---|---:|
| **MeloTTS-8k** (`/dec/student` Vocos8k + iSTFT) | 256 latent frames (≈2.96 s) | 154.6 ms | **22.1 ms** | **7.0×** | FP32 corr 1.000 · FP16 corr 0.996 | 1370 MB |
| **Matcha-8k** (`vocos-8khz-univ`, → STFT bins) | 300 mel frames | 208.5 ms | **32.2 ms** | **6.5×** | FP32 corr 1.000 · FP16 corr 0.99997 | 879 MB |

Both fit well under the Nano's 4 GB (the ~0.9–1.4 GB is mostly the fixed cuDNN/TRT
context; the engines are 8–16 MB). FP16 is numerically faithful.

## Usage

```bash
# MeloTTS-8k: vocoder is fused in the VITS model.onnx -> extract + rewrite
python3 scripts/trt/melo_vocoder_to_trt.py  path/to/melo8k/model.onnx  melo_voc_trt.onnx

# Matcha-8k: Vocos is already standalone
python3 scripts/trt/matcha_vocoder_to_trt.py  path/to/matcha8k/vocos-8khz-univ.onnx  matcha_voc_trt.onnx

# Build + benchmark on the Nano (TensorRT 7.1.3 from JetPack 4.5):
trtexec --onnx=melo_voc_trt.onnx --fp16 --workspace=1200 --saveEngine=melo_voc.plan --iterations=30
```

The conversion handles the three things TRT 7.1's old ONNX parser needs:
`LayerNormalization` → primitives (no LN importer; unique node names so ORT also loads
it), opset-13 `Unsqueeze`/`Reduce*` axes-as-input → attribute, and `Resize` empty `roi`
→ empty tensor. Shapes are pinned static (batch 1, fixed frame count) since TRT 7.1 has
no data-dependent shape support; IR is set to 7 so ONNX Runtime 1.6 (the Nano's build)
can also load the rewritten model for the accuracy check.

## Why only the vocoder (not the full TTS model)

The full MeloTTS / Matcha / Kokoro ONNX graphs are **not** TRT-7.1-buildable, and
custom plugins don't change that:

- **Custom plugins fix unsupported *operators*** (`Erf`, `LessOrEqual`, `RandomNormalLike`,
  `CumSum`, `GatherElements`, …) — doable, if tedious, on the 7.1 `IPluginV2` API.
- **They cannot fix data-dependent *shapes*.** A TRT plugin reports its output shape from
  the input *dimensions* only — never from the input *values*. The TTS frontends have
  tensors whose *size* depends on runtime data: `NonZero` (melo), and the
  duration→length regulator (melo + matcha) where the output mel length = sum of predicted
  durations. TRT 7.1 builds a static engine and can't represent those; data-dependent
  output shapes didn't land until TRT 8.5+ (never for sm_53).
- **Kokoro** additionally has `Loop`/`If` control flow and fuses the neural source-filter
  (CumSum phase + RandomNormal/Uniform noise + ScatterND) into its generator with no clean
  static boundary — the hardest of the three; it would need a purpose-built standalone
  generator re-export from the PyTorch model.

A fully-fused TRT engine would require **re-exporting** the model with fixed/bounded
shapes (max length + mask, noise-as-input, regulated alignment) — a model change, not
plugin patching. Since the distilled Vocos8k vocoder is the dominant *GPU-friendly*
compute and the frontend is light + dynamic, the **hybrid CPU-frontend + TRT-vocoder**
path here is the pragmatic win.

## Notes

- `jetson_clocks` matters: the Nano GPU idles at 76 MHz and takes ~5.7 s to ramp; pin it
  for steady latency.
- `nvprof` GPU profiling needs root on Tegra (`ERR_NVGPUCTRPERM`).
- When copying large ONNX to the device, use `rsync` and verify the byte size — a
  truncated `scp` makes `trtexec` fall back to text-format protobuf parsing and fail
  with a confusing "no field named pytorch".
