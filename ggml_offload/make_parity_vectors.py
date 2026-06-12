#!/usr/bin/env python3
"""Generate parity test vectors for the ggml flow+dec offload runtime.

Adds the flow/dec boundary tensors as graph outputs, runs the full model once
under ORT-CPU, and saves everything the C++ side needs to prove byte-level
(well, fp-tolerance) parity:

  x / tones / sid / scales      the original graph inputs used
  flow_in_*                     every non-initializer input the flow consumes
                                (z_p, mask, g, ...) with its tensor name
  flow_out                      the flow's boundary output (what dec-side sees)
  y                             final 8 kHz audio

The in-graph RandomNormalLike makes runs nondeterministic — irrelevant here,
because the ggml runtime consumes the *dumped* flow inputs, not fresh noise.

Usage:
  python ggml_offload/make_parity_vectors.py \
      --onnx export/vits-melo-tts-zh_en-8k/model.onnx \
      --out ggml_offload/testdata/parity.npz
"""
import argparse
import pathlib

import numpy as np
import onnx
import onnxruntime as ort


def boundary(graph):
    prod = {o: n for n in graph.node for o in n.output}
    inits = {i.name for i in graph.initializer}
    fnodes = [n for n in graph.node if n.name.startswith("/flow")]
    fset = {o for n in fnodes for o in n.output}
    f_in = sorted({i for n in fnodes for i in n.input
                   if i and i not in fset and i not in inits
                   and prod.get(i) is not None
                   and "Constant" not in prod[i].op_type})
    # flow outputs consumed outside the flow
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    f_out = sorted({o for o in fset
                    if any(not c.name.startswith("/flow")
                           for c in consumers.get(o, []))})
    return f_in, f_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = onnx.load(args.onnx, load_external_data=True)
    f_in, f_out = boundary(m.graph)
    print("flow inputs :", f_in)
    print("flow outputs:", f_out)
    for t in f_in + f_out:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))

    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    # short fixed token sequence (token realism is irrelevant for parity —
    # we only need real tensors AT the boundary)
    L = 32
    rng = np.random.default_rng(42)
    feed = {
        "x": rng.integers(10, 80, size=(1, L)).astype(np.int64),
        "x_lengths": np.array([L], dtype=np.int64),
        "tones": rng.integers(0, 4, size=(1, L)).astype(np.int64),
        "sid": np.array([1], dtype=np.int64),
        "noise_scale": np.array([0.6], dtype=np.float32),
        "length_scale": np.array([1.0], dtype=np.float32),
        "noise_scale_w": np.array([0.8], dtype=np.float32),
    }
    names = [o.name for o in sess.get_outputs()]
    vals = sess.run(None, feed)
    out = {}
    for k, v in feed.items():
        out[f"in.{k}"] = v
    for name, v in zip(names, vals):
        key = ("flow_in." + name) if name in f_in else \
              ("flow_out." + name) if name in f_out else name
        out[key] = v
        print(f"{key:48s} {v.shape} {v.dtype}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
