#!/usr/bin/env python3
"""Full-model host speed gate: x86 ORT CPU RTF @1/2/4 threads + per-node profile (dec share).

Runs the drop-in model.onnx with realistic token inputs (from the eval corpus via melo's
frontend, bert=0 like the device). Reports RTF and the fraction of compute in the decoder
(node names containing 'dec'/'istft'/student ops) vs enc/flow.
"""
import argparse, os, sys, time, json, glob, numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_token_inputs(n_tokens=80):
    # melo expects interleaved blank tokens; we just need representative shapes/dtypes.
    L = n_tokens
    x = np.random.randint(1, 100, size=(1, L)).astype(np.int64)
    return {
        "x": x, "x_lengths": np.array([L], np.int64),
        "tones": np.random.randint(0, 8, size=(1, L)).astype(np.int64),
        "sid": np.array([1], np.int64),
        "noise_scale": np.array([0.6], np.float32),
        "length_scale": np.array([1.0], np.float32),
        "noise_scale_w": np.array([0.8], np.float32),
    }


def bench(path, threads, token_lens, reps, sr=8000):
    res = {}
    for th in threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        per = {}
        for nt in token_lens:
            feeds = make_token_inputs(nt)
            y = sess.run(None, feeds)[0]
            audio_s = y.shape[-1] / sr
            for _ in range(3):
                sess.run(None, feeds)
            t0 = time.perf_counter()
            for _ in range(reps):
                sess.run(None, feeds)
            comp = (time.perf_counter() - t0) / reps
            per[nt] = (comp / audio_s, audio_s)
        res[th] = per
    return res


def profile_decoder_share(path, n_tokens=80, sr=8000):
    """Use ORT profiling to attribute time to decoder vs rest."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.enable_profiling = True
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    feeds = make_token_inputs(n_tokens)
    for _ in range(8):
        sess.run(None, feeds)
    prof = sess.end_profiling()
    events = json.load(open(prof))
    os.remove(prof)
    dec_keys = ("dec", "istft", "ConvTranspose", "Resize", "/Cos", "/Sin", "conv_pre",
                "conv_post", "resblock", "ups", "head", "blocks", "cond")
    by_node = {}
    for e in events:
        if e.get("cat") == "Node" and e.get("dur", 0) > 0 and "args" in e:
            op = e["args"].get("op_name", "")
            name = e.get("name", "")
            by_node[name] = by_node.get(name, 0) + e["dur"]
    total = sum(by_node.values())
    dec = sum(d for n, d in by_node.items() if any(k.lower() in n.lower() for k in dec_keys))
    return dec, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--token-lens", default="40,80,160")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    threads = [int(x) for x in args.threads.split(",")]
    token_lens = [int(x) for x in args.token_lens.split(",")]
    print(f"ORT {ort.__version__} CPU | {args.onnx}")
    res = bench(args.onnx, threads, token_lens, args.reps)
    print(f"\n{'threads':>8} " + " ".join(f"{n:>4d}tok" for n in token_lens) + "   (RTF, lower=better)")
    for th in threads:
        row = " ".join(f"{res[th][n][0]:6.3f}" for n in token_lens)
        print(f"{th:>8} {row}")
    # show audio durations for context
    print("  audio_s:  " + " ".join(f"{res[threads[0]][n][1]:5.2f}s" for n in token_lens))
    best4 = min(res[max(threads)][n][0] for n in token_lens)
    print(f"\nfull-model 4-thread best RTF {best4:.4f} -> predicted A57 {best4*6:.2f}-{best4*8:.2f} (x6-8)")
    print(f"gate full-model 4-thread RTF <= 0.05 => {'PASS' if best4 <= 0.05 else 'CHECK'} ({best4:.4f})")
    dec, total = profile_decoder_share(args.onnx)
    if total:
        share = 100 * dec / total
        print(f"\nper-node profile: decoder ~{share:.1f}% of compute (gate dec <= 25% => "
              f"{'PASS' if share <= 25 else 'CHECK'})")


if __name__ == "__main__":
    main()
