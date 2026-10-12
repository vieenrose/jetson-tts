#!/usr/bin/env python3
"""Make the matcha8k Vocos vocoder (vocos-8khz-univ.onnx) TensorRT-7.1-buildable.

Unlike melo's vocoder, matcha's Vocos is already a STANDALONE ONNX (mels -> STFT bins
mag/x/y; the iSTFT is host-side in sherpa), so no subgraph extraction is needed — just
the op rewrites TRT 7.1's old parser needs.

Verified on a real Jetson Nano gen1 (TRT 7.1.3, CUDA 10.2, sm_53, clocks pinned):
  300 mel frames
  ORT-CPU 4thr 208.5 ms  vs  TRT-FP16 32.2 ms  = 6.5x ; RSS 879 MB ; no OOM
  accuracy vs ORT (per output): FP32 corr 1.000, FP16 corr 0.99997 (MAE 0.003)

Usage:
  python3 matcha_vocoder_to_trt.py vocos-8khz-univ.onnx matcha_voc_trt.onnx [--frames 300]
Then on the Nano:
  trtexec --onnx=matcha_voc_trt.onnx --fp16 --workspace=1200 --iterations=30
"""
import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper

MEL_BINS = 80


def decompose_layernorm(g):
    """TRT 7.1 has no LayerNormalization importer -> primitives, with UNIQUE node names
    (a layer has two ReduceMean etc.; duplicate names make ORT reject the model)."""
    new, n_ln, ctr = [], 0, [0]
    def uname(b):
        ctr[0] += 1; return f"{b}_{ctr[0]}"
    for n in g.node:
        if n.op_type != "LayerNormalization":
            new.append(n); continue
        n_ln += 1
        x, scale = n.input[0], n.input[1]
        bias = n.input[2] if len(n.input) > 2 else None
        out, nm = n.output[0], (n.name or f"ln{n_ln}")
        axis, eps = -1, 1e-5
        for a in n.attribute:
            if a.name == "axis": axis = a.i
            if a.name == "epsilon": eps = a.f
        ep = nm + "_eps"
        g.initializer.append(numpy_helper.from_array(np.array(eps, np.float32), ep))
        t = lambda s: nm + s
        A = lambda op, i, o, **k: new.append(helper.make_node(op, i, o, name=uname(nm+"_"+op), **k))
        A("ReduceMean", [x], [t("_m")], axes=[axis], keepdims=1)
        A("Sub", [x, t("_m")], [t("_c")])
        A("Mul", [t("_c"), t("_c")], [t("_s")])
        A("ReduceMean", [t("_s")], [t("_v")], axes=[axis], keepdims=1)
        A("Add", [t("_v"), ep], [t("_ve")])
        A("Sqrt", [t("_ve")], [t("_st")])
        A("Div", [t("_c"), t("_st")], [t("_n")])
        if bias is not None:
            A("Mul", [t("_n"), scale], [t("_sc")])
            new.append(helper.make_node("Add", [t("_sc"), bias], [out], name=uname(nm+"_badd")))
        else:
            new.append(helper.make_node("Mul", [t("_n"), scale], [out], name=uname(nm+"_smul")))
    del g.node[:]; g.node.extend(new)
    return n_ln


def fix_axes_as_input(g):
    """opset-13 Unsqueeze/Squeeze/Reduce* take axes as INPUT; TRT 7.1 wants attribute."""
    vals = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value": vals[n.output[0]] = numpy_helper.to_array(a.t)
    fixed = 0
    for n in g.node:
        if n.op_type in ("Unsqueeze", "Squeeze") and len(n.input) == 2 and n.input[1] in vals:
            ax = [int(v) for v in vals[n.input[1]].reshape(-1)]
            del n.input[1]; n.attribute.append(helper.make_attribute("axes", ax)); fixed += 1
        elif n.op_type in ("ReduceSum", "ReduceMean", "ReduceMax") and len(n.input) == 2 and n.input[1] in vals:
            ax = [int(v) for v in vals[n.input[1]].reshape(-1)]
            del n.input[1]
            if not any(a.name == "axes" for a in n.attribute):
                n.attribute.append(helper.make_attribute("axes", ax))
            fixed += 1
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--ir", type=int, default=7)
    a = ap.parse_args()
    m = onnx.load(a.src); g = m.graph
    print("decomposed LayerNorms:", decompose_layernorm(g))
    del m.opset_import[:]; m.opset_import.append(helper.make_opsetid("", 11))
    print("axes-as-input fixed:", fix_axes_as_input(g))
    for i in g.input:
        if i.name == "mels":
            del i.type.tensor_type.shape.dim[:]
            for d in [1, MEL_BINS, a.frames]:
                i.type.tensor_type.shape.dim.add().dim_value = d
    m.ir_version = a.ir
    onnx.checker.check_model(m)
    onnx.save(m, a.dst)
    print(f"saved {a.dst} (opset 11, ir {a.ir}, {a.frames} frames) — checker OK")


if __name__ == "__main__":
    main()
