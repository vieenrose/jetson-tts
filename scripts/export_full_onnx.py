#!/usr/bin/env python3
"""Export the FULL drop-in sherpa-onnx model.onnx: melo enc/flow + the trained 8 kHz student
decoder, with sample_rate=8000. The stock sherpa-onnx-offline-tts binary runs it unmodified.

Strategy: reuse the official sherpa-onnx melo export contract EXACTLY (same ModelWrapper I/O,
same tokens.txt / lexicon.txt, same metadata schema, bert=0) but monkeypatch model.model.dec
with our student generator so the exported graph's decoder outputs 8 kHz. Only metadata
sample_rate changes (44100 -> 8000); every other input/output name and dtype is identical, so
nothing on the device side changes.

Run from a clean work dir (it writes tokens.txt, lexicon.txt, model.onnx there).
"""
import argparse, os, sys, shutil, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party"))

from scripts.export_dec_onnx import load_student
from student.audio_config import TARGET_SR


class StudentDecAdapter(torch.nn.Module):
    """Wrap the student so it presents melo's decoder signature dec(z, g=...)."""
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, x, g=None):
        return self.student(x, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True, help="output model dir (drop-in)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # import the official reference module for its exact helpers/wrapper
    import importlib.util
    ref_path = os.path.join(os.path.dirname(__file__), "..", "third_party", "sherpa_melo_export.py")
    spec = importlib.util.spec_from_file_location("sherpa_melo_export", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    from melo.api import TTS
    from melo.text import language_id_map, language_tone_start_map

    model = TTS(language="ZH", device="cpu")
    # --- swap in the trained 8 kHz student decoder ---
    student, ck = load_student(args.ckpt, device="cpu")
    model.model.dec = StudentDecAdapter(student)
    print(f"swapped melo dec with {ck['arch']} student (step {ck.get('step')}, pesq {ck.get('pesq')})")

    cwd = os.getcwd()
    os.chdir(args.out_dir)
    try:
        ref.generate_lexicon()                     # lexicon.txt
        ref.generate_tokens(model.hps["symbols"])  # tokens.txt
        wrapper = ref.ModelWrapper(model)

        x = torch.randint(0, 10, (1, 60), dtype=torch.int64)
        x_lengths = torch.tensor([x.size(1)], dtype=torch.int64)
        tones = torch.zeros_like(x)
        sid = torch.tensor([1], dtype=torch.int64)
        ns = torch.tensor([1.0]); ls = torch.tensor([1.0]); nsw = torch.tensor([1.0])

        torch.onnx.export(
            wrapper, (x, x_lengths, tones, sid, ns, ls, nsw), "model.onnx",
            opset_version=args.opset, dynamo=False,
            input_names=["x", "x_lengths", "tones", "sid", "noise_scale", "length_scale", "noise_scale_w"],
            output_names=["y"],
            dynamic_axes={"x": {0: "N", 1: "L"}, "x_lengths": {0: "N"},
                          "tones": {0: "N", 1: "L"}, "y": {0: "N", 1: "S", 2: "T"}},
        )
        meta = {
            "model_type": "melo-vits", "comment": "melo-8khz-distilled", "version": 2,
            "language": "Chinese + English", "add_blank": int(model.hps.data.add_blank),
            "n_speakers": 1, "jieba": 1,
            "sample_rate": TARGET_SR,                       # <-- only functional change
            "bert_dim": 1024, "ja_bert_dim": 768,
            "speaker_id": list(model.hps.data.spk2id.values())[0],
            "lang_id": language_id_map[model.language],
            "tone_start": language_tone_start_map[model.language],
            "url": "https://github.com/myshell-ai/MeloTTS",
            "license": "MIT license",
            "description": f"MeloTTS zh_en with distilled 8kHz {ck['arch']} vocoder (Jetson Nano CPU)",
        }
        ref.add_meta_data("model.onnx", meta)
    finally:
        os.chdir(cwd)
    print(f"wrote {args.out_dir}/{{model.onnx, tokens.txt, lexicon.txt}}  sample_rate={TARGET_SR}")


if __name__ == "__main__":
    main()
