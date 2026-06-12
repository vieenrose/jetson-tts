#!/usr/bin/env python3
"""Host speed gate for a dec-only ONNX: x86 ORT CPU RTF at 1/2/4 threads.

RTF = compute_seconds / audio_seconds. Predicts Jetson A57 RTF via the project's measured
x6-8 host->A57 factor. Reports per-thread RTF over several utterance lengths.
"""
import argparse, time, numpy as np, onnxruntime as ort
from collections import defaultdict

Z_CH = 192
G_CH = 256
ZFR = 86.1328125
SR = 8000


def bench(path, threads, durations, reps=30, warmup=5):
    res = {}
    for th in threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        per = {}
        for dur in durations:
            T = int(round(dur * ZFR))
            z = np.random.randn(1, Z_CH, T).astype(np.float32)
            g = np.random.randn(1, G_CH, 1).astype(np.float32)
            out = sess.run(None, {"z": z, "g": g})[0]
            audio_s = out.shape[-1] / SR
            for _ in range(warmup):
                sess.run(None, {"z": z, "g": g})
            t0 = time.perf_counter()
            for _ in range(reps):
                sess.run(None, {"z": z, "g": g})
            comp = (time.perf_counter() - t0) / reps
            per[dur] = comp / audio_s
        res[th] = per
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--durations", default="2,4,8")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    threads = [int(x) for x in args.threads.split(",")]
    durations = [float(x) for x in args.durations.split(",")]
    print(f"ORT {ort.__version__} CPU | {args.onnx}")
    res = bench(args.onnx, threads, durations, reps=args.reps)
    print(f"\n{'threads':>8} " + " ".join(f"{d:>6.0f}s" for d in durations) + "   (RTF, lower=better)")
    for th in threads:
        row = " ".join(f"{res[th][d]:6.3f}" for d in durations)
        print(f"{th:>8} {row}")
    best4 = min(res[max(threads)].values())
    print(f"\n4-thread best RTF {best4:.4f}  -> predicted A57 RTF {best4*6:.2f}-{best4*8:.2f} (x6-8)")
    print(f"gate: host 4-thread RTF <= 0.05  =>  {'PASS' if best4 <= 0.05 else 'CHECK'} ({best4:.4f})")


if __name__ == "__main__":
    main()
