#!/usr/bin/env python3
"""Convert parity.npz into flat little-endian bins + a manifest the C++ parity
harness reads without any numpy dependency.

Usage:
  python ggml_offload/export_parity_bins.py \
      --npz ggml_offload/testdata/parity.npz --out ggml_offload/testdata/bins
"""
import argparse
import pathlib

import numpy as np

# canonical short names for the boundary tensors (ONNX names are unwieldy)
RENAME = {
    "flow_in./Add_2_output_0": "z_p",                 # [1,192,T]
    "flow_in./Cast_2_output_0": "y_mask",             # [1,1,T]
    "flow_in./Unsqueeze_10_output_0": "attn_mask",    # [1,1,T,1]
    "flow_in./enc_p/encoder/Transpose_output_0": "g",  # [1,1,256]
    "flow_out./flow/flows.0/Concat_output_0": "z",    # [1,192,T]
    "y": "y",                                          # [1,1,S]
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz)
    lines = []
    for key, short in RENAME.items():
        arr = np.ascontiguousarray(data[key].astype(np.float32))
        arr.tofile(out / f"{short}.f32")
        lines.append(f"{short} {' '.join(map(str, arr.shape))}")
        print(short, arr.shape)
    (out / "manifest.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
