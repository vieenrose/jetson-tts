# ggml-CUDA offload for vits-melo-tts-zh_en-8k — flow+dec on the Maxwell, 1 CPU thread total

Goal (Luigi, 2026-06-12): during a live call the gen1's 4 A57 cores are spoken
for (X-ASR -t 2, livekit/Asterisk/agent ≈ 1 core) — TTS must run on **≤ 1 CPU
thread**, offloading the heavy compute to the otherwise-idle Maxwell GPU,
**without ONNX Runtime CUDA / TensorRT memory overhead** (both banned; no
cuDNN on device anyway).

## Why this split (measured basis)
Post-distill profile of `Luigi/vits-melo-tts-zh_en-8k` (device-accepted,
Nano RTF 0.79 @ t=1 / 0.34 @ t=4): flow ≈ 70–75 % of compute, enc+SDP ≈ 15 %,
Vocos8k dec ≈ 12.6 %. The GPU-portable 85 % (flow + dec) is **conv-only** —
no attention, no SDP splines. The hard-to-port 15 % stays on CPU.

```
text → [ORT CPU, 1 thread: enc + duration/SDP]  ──z──►
       [ggml-CUDA: flow convs + Vocos8k dec]    ──8 kHz audio
```

## Predicted result
CPU leg ≈ 0.79 × 0.15 ≈ 0.12; GPU leg ≈ 0.79 × 0.85 / (3–6×, Maxwell
im2col+GEMM) ≈ 0.11–0.22; partially overlapped ⇒ **RTF ≈ 0.25–0.35 on one
core** (today's t=4 class, three cores freed). Memory: ~45 MB f16 weights +
activations — tens of MB on the UMA.

## Build on (all proven in-repo)
- **Toolchain**: user/sensevoice_cpp ggml-CUDA cross-build (CUDA 10.2, sm_53,
  CUDA_STANDARD=14, NO_VMM, SDK container). Modern ggml is blocked by the
  nvcc-10.2/C++17 wall — use the sensevoice-era ggml.
- **Graph/weights**: vieenrose/jetson-tts has the exact decoder config,
  z[192,T]/g[256] interface, framing (86.13→125 Hz resample, ×64 iSTFT to
  8 kHz), and parity test vectors (`scripts/`, `docs/TEACHER_DECODER_CONFIG`).
- **Reference ggml audio-conv code**: maxilevi/vits.cpp (custom conv ops,
  bench_conv1d), encodec.cpp, sensevoice SAN-M convs.
- **ONNX split point**: cut model.onnx at the z→flow boundary (node names in
  the jetson-tts export script); enc+SDP sub-onnx runs in the existing
  sherpa/ORT CPU stack or bare ORT.

## Work items
1. Weights → GGUF f16 (flow + dec; converter pattern from sensevoice).
2. ggml graph: flow = WaveNet dilated conv1d stacks + affine coupling/flip
   (compose from conv1d + elementwise); dec = ConvNeXt blocks @125 Hz +
   iSTFT head (zero-insertion + conv for transpose; iSTFT as matmul with
   precomputed DFT basis + overlap-add — encodec.cpp pattern).
3. CUDA kernels: reuse im2col conv path; new: zero-insert upsample,
   overlap-add. All C++14-safe for nvcc 10.2.
4. Parity harness vs ORT reference (byte-level on jetson-tts test vectors,
   then PESQ-NB spot-check) — sensevoice-style.
5. Integration: small C++ lib + CLI (`tts8k-gpu`) consumed by the agent the
   same way sense-voice-server is; later optional sherpa-onnx plugin shim.
6. Device bench: RTF + RSS at 1 CPU thread, GPU busy %, concurrent-with-ASR
   test (X-ASR -t 2 running simultaneously — the real acceptance gate).

## Gates
- RTF ≤ 0.4 with exactly 1 CPU thread, X-ASR -t 2 running concurrently.
- Peak extra RAM ≤ 150 MB. No cuDNN/TRT/ORT-CUDA dependencies.
- Parity: PESQ-NB vs ORT output ≥ 4.0 (same model, should be ~identical).
