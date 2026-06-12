#!/usr/bin/env python3
"""Extract the flow + dec(student) weights from the exported full model.onnx
into a GGUF (f32) for the ggml offload runtime.

Two kinds of initializers feed those subgraphs:
  1. `model.model.flow.*` / `model.model.dec.*` — keep, minus the prefix.
  2. torch `nn.Linear` weights that the exporter renamed to `onnx::MatMul_*` —
     recovered by walking the subgraph nodes and renaming from the node's
     scope path, e.g. node `/flow/flows.0/enc/spk_emb_linear/MatMul` with a
     renamed weight becomes `flow.flows.0.enc.spk_emb_linear.weight`.
     NOTE: these are stored as ONNX MatMul B-matrices, i.e. ALREADY
     TRANSPOSED vs the torch Linear weight ([in, out] not [out, in]).
     The metadata flag `melo8k.linear_is_in_out=true` records this.

Everything is stored f32 (40-80 MB — fine; quantization is a later, measured
decision: int8 conv is a proven anti-remedy on this hardware family).

Usage:
  python ggml_offload/convert_onnx_to_gguf.py \
      --onnx export/vits-melo-tts-zh_en-8k/model.onnx \
      --out ggml_offload/flowdec-f32.gguf
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper

try:
    import gguf
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install gguf") from e

SEGS = ("/flow", "/dec")
PREFIX = "model.model."


def canonical_from_node(node_name: str) -> str:
    # '/flow/flows.0/enc/spk_emb_linear/MatMul' -> 'flow.flows.0.enc.spk_emb_linear.weight'
    parts = [p for p in node_name.split("/") if p]
    assert parts[-1].startswith("MatMul"), node_name
    return ".".join(parts[:-1]) + ".weight"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.onnx, load_external_data=True)
    g = model.graph
    inits = {i.name: i for i in g.initializer}

    tensors: dict[str, np.ndarray] = {}
    for name, init in inits.items():
        if name.startswith((PREFIX + "flow.", PREFIX + "dec.")):
            tensors[name[len(PREFIX):]] = numpy_helper.to_array(init)
    renamed = 0
    for n in g.node:
        if not n.name.startswith(SEGS):
            continue
        for i in n.input:
            if i in inits and not i.startswith("model."):
                cname = canonical_from_node(n.name)
                arr = numpy_helper.to_array(inits[i])
                assert cname not in tensors or np.array_equal(tensors[cname], arr), cname
                tensors[cname] = arr
                renamed += 1

    w = gguf.GGUFWriter(args.out, "melo8k-flowdec")
    w.add_bool("melo8k.linear_is_in_out", True)
    total = 0
    for name in sorted(tensors):
        arr = tensors[name]
        # ggml's conv_1d/conv_transpose_1d CPU kernels require F16 kernels;
        # everything else stays f32 (int8 conv is a measured anti-remedy on
        # this hardware family, and f16 weights keep full conv throughput).
        dt = np.float16 if arr.ndim >= 3 else np.float32
        arr = np.ascontiguousarray(arr.astype(dt))
        w.add_tensor(name, arr)
        total += arr.nbytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {len(tensors)} tensors ({renamed} recovered MatLuls), "
          f"{total/1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
