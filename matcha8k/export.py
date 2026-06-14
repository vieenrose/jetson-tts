"""Export the trained 8 kHz Matcha vocoder to ONNX (sherpa VocosVocoder contract) and build the
drop-in model dir: 8k vocoder + acoustic model copy with sample_rate=8000.

Vocoder ONNX: input `mels`[B,80,T] -> outputs `mag`,`x`,`y` [B,257,T]; metadata n_fft=512,
hop_length=128, win_length=512, center=1 so sherpa's iSTFT yields exactly 8 kHz.
"""
import argparse, os, shutil, numpy as np, torch, onnx, onnxruntime as ort
from matcha8k.models import VocosMel8k, ExportWrapper, NFFT8, HOP8, WIN8


def add_meta(path, meta):
    m = onnx.load(path)
    while len(m.metadata_props):
        m.metadata_props.pop()
    for k, v in meta.items():
        p = m.metadata_props.add(); p.key = k; p.value = str(v)
    onnx.save(m, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src-dir", default="matcha_eval/matcha-icefall-zh-en")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    G = VocosMel8k(); G.load_state_dict(ck["G"]); G.eval()
    wrap = ExportWrapper(G).eval()
    voc_path = os.path.join(args.out_dir, "vocos-8khz-univ.onnx")

    mel = torch.randn(1, 80, 120)
    with torch.no_grad():
        mag, x, y = wrap(mel)
    torch.onnx.export(
        wrap, (mel,), voc_path, opset_version=args.opset, dynamo=False,
        input_names=["mels"], output_names=["mag", "x", "y"],
        dynamic_axes={"mels": {0: "batch_size", 2: "time"},
                      "mag": {0: "batch_size", 2: "time"},
                      "x": {0: "batch_size", 2: "time"}, "y": {0: "batch_size", 2: "time"}})
    add_meta(voc_path, {"model_type": "matcha-tts vocos", "n_fft": NFFT8, "hop_length": HOP8,
                        "win_length": WIN8, "center": 1, "window_type": "hann", "pad_mode": "reflect",
                        "sample_rate": 8000, "comment": "8khz distilled vocoder for matcha-icefall-zh-en"})
    # verify ORT
    sess = ort.InferenceSession(voc_path, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"mels": mel.numpy().astype(np.float32)})
    err = max(np.abs(o[0] - mag.detach().numpy()).max(),
              np.abs(o[1] - x.detach().numpy()).max(), np.abs(o[2] - y.detach().numpy()).max())
    print(f"vocoder ONNX out shapes mag{o[0].shape} x{o[1].shape} y{o[2].shape} | max|ort-torch|={err:.2e}")

    # acoustic model copy with sample_rate=8000
    ac_src = os.path.join(args.src_dir, "model-steps-3.onnx")
    ac_dst = os.path.join(args.out_dir, "model-steps-3.onnx")
    m = onnx.load(ac_src)
    meta = {p.key: p.value for p in m.metadata_props}
    meta["sample_rate"] = "8000"
    while len(m.metadata_props): m.metadata_props.pop()
    for k, v in meta.items():
        p = m.metadata_props.add(); p.key = k; p.value = str(v)
    onnx.save(m, ac_dst)
    print(f"acoustic copy sample_rate -> {meta['sample_rate']}")

    # copy the rest of the model dir (tokens, lexicon, fsts, espeak-ng-data)
    for name in ["tokens.txt", "lexicon.txt", "date-zh.fst", "number-zh.fst", "phone-zh.fst"]:
        src = os.path.join(args.src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out_dir, name))
    esp = os.path.join(args.src_dir, "espeak-ng-data")
    if os.path.isdir(esp) and not os.path.exists(os.path.join(args.out_dir, "espeak-ng-data")):
        shutil.copytree(esp, os.path.join(args.out_dir, "espeak-ng-data"))
    print(f"wrote drop-in dir {args.out_dir} (vocoder + 8k acoustic + tokens/lexicon/fsts/espeak)")


if __name__ == "__main__":
    main()
