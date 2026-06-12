#!/usr/bin/env python3
"""Export a trained student vocoder to a dec-only ONNX graph (opset 17, fp32, no custom ops).

Graph signature matches melo's decoder: inputs (z[B,192,T], g[B,256,1]) -> wav[B,1,S] @ 8 kHz.
Verifies the export runs in ORT 1.17 CPU and matches torch numerically.
"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from student.models import build_generator
from student.audio_config import Z_CHANNELS, G_CHANNELS


def load_student(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    G = build_generator(ck["arch"])
    G.load_state_dict(ck["G"])
    G.remove_wn()                       # fuse weight_norm for inference
    G.eval().to(device)
    return G, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    G, ck = load_student(args.ckpt)
    print(f"loaded {ck['arch']} step {ck.get('step')} pesq {ck.get('pesq')}")

    T = 345                              # ~4s of z frames
    z = torch.randn(1, Z_CHANNELS, T)
    g = torch.randn(1, G_CHANNELS, 1)
    with torch.no_grad():
        y_ref = G(z, g).numpy()

    torch.onnx.export(
        G, (z, g), args.out, opset_version=args.opset, dynamo=False,
        input_names=["z", "g"], output_names=["wav"],
        dynamic_axes={"z": {0: "B", 2: "T"}, "g": {0: "B"}, "wav": {0: "B", 2: "S"}},
    )
    import onnx, onnxruntime as ort
    m = onnx.load(args.out); onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    print("opset", m.opset_import[0].version, "ops", ops)

    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])
    for Tt in (T, 120, 700):
        zt = np.random.randn(1, Z_CHANNELS, Tt).astype(np.float32)
        gt = np.random.randn(1, G_CHANNELS, 1).astype(np.float32)
        yo = sess.run(None, {"z": zt, "g": gt})[0]
        with torch.no_grad():
            yt = G(torch.from_numpy(zt), torch.from_numpy(gt)).numpy()
        n = min(yo.shape[-1], yt.shape[-1])
        err = np.abs(yo[..., :n] - yt[..., :n]).max()
        print(f"  T={Tt:4d} wav {yo.shape} max|ort-torch|={err:.2e}")
    print(f"OK exported {args.out} ({ck['arch']}), ORT {ort.__version__} CPU, opset {args.opset}")


if __name__ == "__main__":
    main()
