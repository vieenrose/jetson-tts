"""Export a fine-tuned Matcha acoustic checkpoint to the dengcunqin/sherpa ONNX contract:
inputs (x[N,L], x_length[N], noise_scale[1], length_scale[1]) -> mel[N,80,T], opset 14,
num_ode_steps baked in, metadata sample_rate (kept; vocoder sets the real 8k iSTFT).
"""
import argparse, sys, torch
sys.path.insert(0, "third_party/Matcha-TTS")
import onnx


class AcousticWrapper(torch.nn.Module):
    def __init__(self, m, n_steps):
        super().__init__()
        self.m = m
        self.n_steps = n_steps

    def forward(self, x, x_length, noise_scale, length_scale):
        out = self.m.synthesise(x, x_length, self.n_steps, temperature=noise_scale[0],
                                spks=None, length_scale=length_scale[0])
        return out["mel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--ref-onnx", default="matcha_eval/matcha-icefall-zh-en/model-steps-3.onnx")
    args = ap.parse_args()

    from matcha8k.finetune import build_model
    import os
    # build_model loads the BASE; then overlay the fine-tuned weights
    m = build_model()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    m.eval()
    wrap = AcousticWrapper(m, args.n_steps).eval()

    x = torch.randint(1, 2000, (1, 40), dtype=torch.long)
    x_length = torch.tensor([40], dtype=torch.long)
    noise_scale = torch.tensor([0.667], dtype=torch.float32)
    length_scale = torch.tensor([1.0], dtype=torch.float32)
    torch.onnx.export(
        wrap, (x, x_length, noise_scale, length_scale), args.out, opset_version=args.opset,
        input_names=["x", "x_length", "noise_scale", "length_scale"], output_names=["mel"],
        dynamic_axes={"x": {0: "N", 1: "L"}, "x_length": {0: "N"}, "mel": {0: "N", 2: "T"}},
        dynamo=False)

    # copy metadata from the reference onnx (sample_rate etc.)
    ref = onnx.load(args.ref_onnx)
    refmeta = {p.key: p.value for p in ref.metadata_props}
    g = onnx.load(args.out)
    while len(g.metadata_props): g.metadata_props.pop()
    for k, v in refmeta.items():
        p = g.metadata_props.add(); p.key = k; p.value = str(v)
    onnx.save(g, args.out)
    print(f"exported {args.out} (n_steps={args.n_steps}, opset {args.opset}); meta sample_rate={refmeta.get('sample_rate')}")


if __name__ == "__main__":
    main()
