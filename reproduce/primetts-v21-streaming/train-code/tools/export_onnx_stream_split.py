#!/usr/bin/env python3
"""Split-export the v2-Stream model into TWO ONNX graphs for incremental
streaming through sherpa-onnx (the ONNX analog of RapidSpeech --stream-chunks):

  enc.onnx : (x, tone, lang, x_lengths, noise_scale, length_scale) -> z [1,192,T]
             = text encoder (band-attn) + duration predictor + reverse flow.
             Run ONCE per utterance/phrase.
  dec.onnx : z [1,192,Tc] -> wav [1,1,Tc*256]
             = causal MB-iSTFT vocoder. Run PER CHUNK with overlap-save
             (feed z[:, :, a-LEFT : b+RIGHT], keep the middle (b-a)*256 samples).

Run with the streaming env so the encoder is band-limited:
  MBVITS_STREAM_ENC=1 MBVITS_ENC_LOOKAHEAD=5 python tools/export_onnx_stream_split.py \
    --ckpt .../keep_streamenc_10k_G.pth --config .../xinran_streamenc.json --outdir <dir>
"""
import argparse, json, math, os, sys
import numpy as np, torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from models import SynthesizerTrn
import models as models_mod
from tools.export_onnx_primetts_v2 import OnnxISTFT  # reuse the exact iSTFT


def install_onnx_dsp(m):
    """Swap the decoder's inline TorchSTFT + PQMF for ONNX-exportable versions."""
    class PatchedTorchSTFT(torch.nn.Module):
        def __init__(self, filter_length=16, hop_length=4, win_length=16, window="hann"):
            super().__init__()
            self._oi = OnnxISTFT(filter_length, hop_length, torch.hann_window(win_length))
        def inverse(self, mag, ph): return self._oi.inverse(mag, ph)
        def to(self, *a, **k): return self
    models_mod.TorchSTFT = PatchedTorchSTFT
    import stft as stft_mod; stft_mod.TorchSTFT = PatchedTorchSTFT
    from pqmf import design_prototype_filter
    class CpuPQMF(torch.nn.Module):
        def __init__(self, device=None, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
            super().__init__()
            h = design_prototype_filter(taps, cutoff_ratio, beta)
            hs = np.zeros((subbands, len(h)))
            for k in range(subbands):
                hs[k] = 2 * h * np.cos((2*k+1)*(np.pi/(2*subbands))*(np.arange(taps+1)-((taps-1)/2)) - (-1)**k*np.pi/4)
            self.register_buffer("synthesis_filter", torch.from_numpy(hs).float().unsqueeze(0))
            ud = torch.zeros((subbands, subbands, subbands))
            for k in range(subbands): ud[k, k, 0] = 1.0
            self.register_buffer("updown_filter", ud); self.subbands = subbands
            self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)
        def synthesis(self, x):
            x = torch.nn.functional.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
            return torch.nn.functional.conv1d(self.pad_fn(x), self.synthesis_filter)
        def to(self, *a, **k): return self
    models_mod.PQMF = CpuPQMF


class EncWrap(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x, tone, lang, x_lengths, noise_scale, length_scale):
        o, o_mb, attn, y_mask, (z, z_p, m_p, logs_p) = self.net.infer(
            x, tone, lang, x_lengths, noise_scale=noise_scale, length_scale=length_scale)
        return z * y_mask                      # [1,192,T] — decoder input


class DecWrap(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, z):
        return self.net.dec(z)[0]              # [1,1,T*256] wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/luigi/mbvits_run/keep_streamenc_10k_G.pth")
    ap.add_argument("--config", default=os.path.join(_ROOT, "configs", "zhtw_mbistft_16k_xinran_streamenc.json"))
    ap.add_argument("--outdir", default="/home/luigi/mbvits_run/v2stream_split")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cfg = json.load(open(a.config)); m, d = cfg["model"], cfg["data"]
    net = SynthesizerTrn(88, d["filter_length"] // 2 + 1,
                         cfg["train"]["segment_size"] // d["hop_length"], **m)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    net.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}, strict=True)
    net.eval(); net.dec.remove_weight_norm()
    assert os.environ.get("MBVITS_STREAM_ENC") == "1", "set MBVITS_STREAM_ENC=1 (band encoder)"
    from causal_patch import causalize_generator
    causalize_generator(net.dec, verbose=True)
    install_onnx_dsp(m)

    # ---- enc.onnx ----
    T = 33
    ex = (torch.randint(1, 87, (1, T)), torch.randint(0, 6, (1, T)), torch.randint(0, 2, (1, T)),
          torch.tensor([T], dtype=torch.long), torch.tensor([0.0], dtype=torch.float32),
          torch.tensor([1.0], dtype=torch.float32))
    enc_path = os.path.join(a.outdir, "v2stream_enc.onnx")
    torch.onnx.export(EncWrap(net), ex, enc_path, opset_version=17, dynamo=False,
        input_names=["x", "tone", "lang", "x_lengths", "noise_scale", "length_scale"],
        output_names=["z"], dynamic_axes={"x": {1: "T"}, "tone": {1: "T"}, "lang": {1: "T"}, "z": {2: "F"}})
    with torch.no_grad():
        z = EncWrap(net)(*ex)
    print(f"[enc] wrote {enc_path}  z={tuple(z.shape)}")

    # ---- dec.onnx ----  (z chunk -> wav)
    dec_path = os.path.join(a.outdir, "v2stream_dec.onnx")
    zex = torch.randn(1, m["inter_channels"], 92)
    torch.onnx.export(DecWrap(net), (zex,), dec_path, opset_version=17, dynamo=False,
        input_names=["z"], output_names=["wav"], dynamic_axes={"z": {2: "F"}, "wav": {2: "L"}})
    with torch.no_grad():
        w = DecWrap(net)(zex)
    print(f"[dec] wrote {dec_path}  wav={tuple(w.shape)} (z F={zex.shape[2]} -> {w.shape[2]} = {w.shape[2]//zex.shape[2]}/frame)")


if __name__ == "__main__":
    main()
