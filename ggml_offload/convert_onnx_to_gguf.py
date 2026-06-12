#!/usr/bin/env python3
"""Extract the flow + dec(student) weights from the exported full model.onnx
into a GGUF (f16) for the ggml-CUDA offload runtime.

The cut follows the measured compute split on the Jetson Nano gen1 (see
PLAN.md): enc_p/sdp/dp stay in ORT-CPU at one thread; everything under
`model.model.flow.*` (VITS2 transformer-coupling flow) and
`model.model.dec.*` (Vocos8k student) runs in ggml-CUDA.

Usage:
  python ggml_offload/convert_onnx_to_gguf.py \
      --onnx export/vits-melo-tts-zh_en-8k/model.onnx \
      --out ggml_offload/flowdec-f16.gguf

Tensor names are kept verbatim (minus the `model.model.` prefix) so the C++
loader can address them mechanically; shapes/dtypes recorded as-is, data
stored f16 except 1-D norm/bias tensors which stay f32 (precision-sensitive,
negligible size).
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper

try:
    import gguf
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install gguf") from e

PREFIXES = ("model.model.flow.", "model.model.dec.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.onnx, load_external_data=True)
    inits = {i.name: i for i in model.graph.initializer}

    w = gguf.GGUFWriter(args.out, "melo8k-flowdec")
    n = 0
    total = 0
    for name, init in sorted(inits.items()):
        if not name.startswith(PREFIXES):
            continue
        arr = numpy_helper.to_array(init)
        short = name[len("model.model."):]
        if arr.ndim <= 1 or arr.dtype != np.float32:
            data = arr  # keep small/bias/norm tensors at native precision
        else:
            data = arr.astype(np.float16)
        w.add_tensor(short, np.ascontiguousarray(data))
        n += 1
        total += data.nbytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {n} tensors, {total/1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
