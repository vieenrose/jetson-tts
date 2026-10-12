#!/usr/bin/env python3
"""Make the melo8k (VITS) vocoder sub-network TensorRT-7.1-buildable on Jetson Nano.

The full melo8k VITS model.onnx is NOT TRT-portable: its frontend uses NonZero /
ScatterND / RandomNormalLike (data-dependent shapes — TRT can't build these). But the
vocoder tail (`/dec/student/` = Vocos8k backbone + iSTFT) is pure conv/matmul and ports
cleanly. This script extracts it and rewrites the few ops TRT 7.1's old ONNX parser
rejects, producing a static-shape ONNX you can feed to trtexec.

Verified on a real Jetson Nano gen1 (TRT 7.1.3, CUDA 10.2, sm_53, clocks pinned):
  256 latent frames -> 2.96 s audio @8kHz
  ORT-CPU 4thr 154.6 ms (RTF 0.052)  vs  TRT-FP16 22.1 ms (RTF 0.0075)  = 7.0x
  accuracy vs ORT: FP32 corr 1.000, FP16 corr 0.996 (MAE ~0)

Usage:
  python3 melo_vocoder_to_trt.py model.onnx melo_voc_trt.onnx [--frames 256]
Then on the Nano:
  trtexec --onnx=melo_voc_trt.onnx --fp16 --workspace=1200 --iterations=30
"""
import sys, argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper

# vocoder boundary in the VITS graph: latent z and speaker-emb g -> audio y
IN_Z, IN_G, OUT_Y = "/Mul_10_output_0", "/Unsqueeze_6_output_0", "y"
INTER_CH, GIN_CH = 192, 256


def decompose_layernorm(g):
    """TRT 7.1 has no LayerNormalization importer -> expand to primitives."""
    new, n_ln, ctr = [], 0, [0]
    def uname(base):
        ctr[0] += 1; return f"{base}_{ctr[0]}"   # unique: a layer has 2 ReduceMean etc.
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
        epsname = nm + "_eps"
        g.initializer.append(numpy_helper.from_array(np.array(eps, np.float32), epsname))
        t = lambda s: nm + s
        A = lambda op, ins, outs, **kw: new.append(helper.make_node(op, ins, outs, name=uname(nm+"_"+op), **kw))
        A("ReduceMean", [x], [t("_mean")], axes=[axis], keepdims=1)
        A("Sub", [x, t("_mean")], [t("_xc")])
        A("Mul", [t("_xc"), t("_xc")], [t("_sq")])
        A("ReduceMean", [t("_sq")], [t("_var")], axes=[axis], keepdims=1)
        A("Add", [t("_var"), epsname], [t("_ve")])
        A("Sqrt", [t("_ve")], [t("_std")])
        A("Div", [t("_xc"), t("_std")], [t("_norm")])
        if bias is not None:
            A("Mul", [t("_norm"), scale], [t("_sc")])
            new.append(helper.make_node("Add", [t("_sc"), bias], [out], name=uname(nm+"_badd")))
        else:
            new.append(helper.make_node("Mul", [t("_norm"), scale], [out], name=uname(nm+"_smul")))
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


def fix_resize_empty_roi(g):
    """Resize with empty-string roi: ORT 1.6 rejects it; wire to a real empty tensor."""
    g.initializer.append(numpy_helper.from_array(np.array([], np.float32), "voc_roi_empty"))
    for n in g.node:
        if n.op_type == "Resize":
            for k, inp in enumerate(n.input):
                if inp == "": n.input[k] = "voc_roi_empty"


def set_static(g, frames):
    def setshape(inp, dims):
        del inp.type.tensor_type.shape.dim[:]
        for d in dims:
            inp.type.tensor_type.shape.dim.add().dim_value = d
    for i in g.input:
        if i.name == IN_Z: setshape(i, [1, INTER_CH, frames])
        elif i.name == IN_G: setshape(i, [1, GIN_CH, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--ir", type=int, default=7, help="ONNX IR version (7 for ORT 1.6 on Nano)")
    a = ap.parse_args()
    onnx.utils.extract_model(a.src, "/tmp/_voc_raw.onnx", input_names=[IN_Z, IN_G], output_names=[OUT_Y])
    m = onnx.load("/tmp/_voc_raw.onnx"); g = m.graph
    print("extracted vocoder nodes:", len(g.node))
    print("decomposed LayerNorms:", decompose_layernorm(g))
    del m.opset_import[:]; m.opset_import.append(helper.make_opsetid("", 11))
    print("axes-as-input fixed:", fix_axes_as_input(g))
    fix_resize_empty_roi(g)
    set_static(g, a.frames)
    m.ir_version = a.ir
    onnx.save(m, a.dst)
    print(f"saved {a.dst} (opset 11, ir {a.ir}, {a.frames} frames)")


if __name__ == "__main__":
    main()
