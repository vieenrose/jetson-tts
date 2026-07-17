#!/usr/bin/env python3
"""Export PrimeTTS v2 (MB-iSTFT-VITS, Xinran G_400000) to ONNX for the demo Space
(ORT-CPU). opset17, dynamo=False per the project's validated export contract.

torch.istft has no ONNX op, so the tiny gen-head iSTFT (n_fft=16, hop=4) is replaced
by an exact equivalent: irFFT as a fixed matrix product + windowed overlap-add via
ConvTranspose1d + window-envelope normalization (verified vs torch.istft before export).

Inputs : x[1,T] int64, tone[1,T] int64, lang[1,T] int64, x_lengths[1] int64,
         noise_scale[1] f32, length_scale[1] f32
Output : wav[1,1,L] f32 @16kHz
"""
import argparse, json, math, os, sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from models import SynthesizerTrn
import models as models_mod


class OnnxISTFT(torch.nn.Module):
    """Drop-in for TorchSTFT.inverse (center=True), ONNX-exportable, exact."""

    def __init__(self, n_fft, hop, window):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        n_bins = n_fft // 2 + 1
        k = torch.arange(n_bins).unsqueeze(1).float()
        n = torch.arange(n_fft).unsqueeze(0).float()
        coef = torch.full((n_bins, 1), 2.0)
        coef[0, 0] = 1.0
        if n_fft % 2 == 0:
            coef[-1, 0] = 1.0
        ang = 2 * math.pi * k * n / n_fft
        self.register_buffer("C", (coef * torch.cos(ang)) / n_fft)   # [bins, n_fft]
        self.register_buffer("S", (-coef * torch.sin(ang)) / n_fft)  # [bins, n_fft]
        self.register_buffer("win", window.reshape(1, -1, 1))        # [1, n_fft, 1]
        ola_k = torch.eye(n_fft).unsqueeze(1)                        # [n_fft,1,n_fft]
        self.register_buffer("ola_kernel", ola_k)
        self.register_buffer("env_kernel", (window ** 2).reshape(1, 1, -1))

    def inverse(self, magnitude, phase):
        real = magnitude * torch.cos(phase)          # [B, bins, T]
        imag = magnitude * torch.sin(phase)
        # frames[b, n, t] = sum_k real[b,k,t]*C[k,n] + imag[b,k,t]*S[k,n]
        frames = torch.einsum("bkt,kn->bnt", real, self.C) + \
                 torch.einsum("bkt,kn->bnt", imag, self.S)
        frames = frames * self.win                                   # analysis window
        y = torch.nn.functional.conv_transpose1d(frames, self.ola_kernel, stride=self.hop)
        ones = torch.ones_like(frames[:, :1, :])
        env = torch.nn.functional.conv_transpose1d(ones, self.env_kernel, stride=self.hop)
        y = y / torch.clamp(env, min=1e-9)
        half = self.n_fft // 2
        y = y[:, :, half:-half]                                      # center=True trim
        return y  # [B,1,L] (matches TorchSTFT.inverse's unsqueeze(-2))


class ExportWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, tone, lang, x_lengths, sid, noise_scale, length_scale):
        o, *_ = self.net.infer(x, tone, lang, x_lengths, sid=sid,
                               noise_scale=noise_scale, length_scale=length_scale)
        return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/luigi/mbvits_run/keep_v21b_12500_G.pth")
    ap.add_argument("--config", default=os.path.join(_ROOT, "configs", "zhtw_mb_istft_16k_v21b.json"))
    ap.add_argument("--out", default="/home/luigi/mbvits_run/primetts_v21_3voice.onnx")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    m, d = cfg["model"], cfg["data"]
    net = SynthesizerTrn(88, d["filter_length"] // 2 + 1,
                         cfg["train"]["segment_size"] // d["hop_length"], **m)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    net.eval()
    net.dec.remove_weight_norm()

    # numeric check of OnnxISTFT vs torch.istft BEFORE swapping it in
    ts = net.dec.stft if hasattr(net.dec, "stft") else None
    # Multiband generator constructs TorchSTFT inline in forward via module-level import;
    # check models.py: it uses `stft.inverse(...)` where stft is built in forward? Inspect:
    oi = OnnxISTFT(m["gen_istft_n_fft"], m["gen_istft_hop_size"],
                   torch.hann_window(m["gen_istft_n_fft"]))
    from stft import TorchSTFT
    ref = TorchSTFT(filter_length=m["gen_istft_n_fft"], hop_length=m["gen_istft_hop_size"],
                    win_length=m["gen_istft_n_fft"])
    mag = torch.rand(4, m["gen_istft_n_fft"] // 2 + 1, 57) + 0.1
    ph = (torch.rand(4, m["gen_istft_n_fft"] // 2 + 1, 57) - 0.5) * 2 * math.pi
    a = ref.inverse(mag, ph)
    b = oi.inverse(mag, ph)
    err = (a - b).abs().max().item()
    print(f"[istft-check] torch vs onnx-istft max abs err = {err:.3e}  shapes {tuple(a.shape)} {tuple(b.shape)}")
    assert err < 1e-4, "OnnxISTFT mismatch"

    # swap: the MB generator calls `stft.inverse(spec, phase)` on a TorchSTFT instance
    # created in its forward (models.py line ~330: stft = TorchSTFT(...).to(x.device)).
    # Patch the class used by models.py so the instance built in forward IS ours.
    class PatchedTorchSTFT(torch.nn.Module):
        def __init__(self, filter_length=16, hop_length=4, win_length=16, window="hann"):
            super().__init__()
            self._oi = OnnxISTFT(filter_length, hop_length, torch.hann_window(win_length))
        def inverse(self, magnitude, phase):
            return self._oi.inverse(magnitude, phase)
        def to(self, *a, **k):
            return self
    models_mod.TorchSTFT = PatchedTorchSTFT
    import stft as stft_mod
    stft_mod.TorchSTFT = PatchedTorchSTFT

    # PQMF hardcodes .cuda(); rebuild it CPU-safe with identical filters
    from pqmf import design_prototype_filter

    class CpuPQMF(torch.nn.Module):
        def __init__(self, device=None, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
            super().__init__()
            h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
            h_synthesis = np.zeros((subbands, len(h_proto)))
            for k in range(subbands):
                h_synthesis[k] = 2 * h_proto * np.cos(
                    (2 * k + 1) * (np.pi / (2 * subbands)) *
                    (np.arange(taps + 1) - ((taps - 1) / 2)) - (-1) ** k * np.pi / 4)
            self.register_buffer("synthesis_filter",
                                 torch.from_numpy(h_synthesis).float().unsqueeze(0))
            updown = torch.zeros((subbands, subbands, subbands))
            for k in range(subbands):
                updown[k, k, 0] = 1.0
            self.register_buffer("updown_filter", updown)
            self.subbands = subbands
            self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

        def synthesis(self, x):
            x = torch.nn.functional.conv_transpose1d(
                x, self.updown_filter * self.subbands, stride=self.subbands)
            return torch.nn.functional.conv1d(self.pad_fn(x), self.synthesis_filter)

        def to(self, *a, **k):
            return self

    models_mod.PQMF = CpuPQMF

    wrap = ExportWrapper(net)
    T = 33
    ex = (torch.randint(1, 87, (1, T)), torch.randint(0, 6, (1, T)),
          torch.randint(0, 2, (1, T)), torch.tensor([T], dtype=torch.long),
          torch.tensor([0], dtype=torch.long), torch.tensor([0.667], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32))
    with torch.no_grad():
        wav = wrap(*ex)
    print(f"[trace-check] eager wav {tuple(wav.shape)}")

    torch.onnx.export(
        wrap, ex, args.out, opset_version=17, dynamo=False,
        input_names=["x", "tone", "lang", "x_lengths", "sid", "noise_scale", "length_scale"],
        output_names=["wav"],
        dynamic_axes={"x": {1: "T"}, "tone": {1: "T"}, "lang": {1: "T"}, "wav": {2: "L"}},
    )
    print(f"[export] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
