"""ORT 1.17.0 CPU compatibility probe for the 8 kHz student decoder ops.

Validates, at opset 17, fp32, no custom ops:
  1. Conv1d + weight_norm-style + ConvTranspose1d upsampling (HiFi-GAN head)
  2. F.interpolate(mode='linear') latent time-resample  -> ONNX Resize
  3. iSTFT implemented as a fixed ConvTranspose1d (Vocos-style) overlap-add
  4. Dynamic time axis (variable number of z frames)
All checked for torch-vs-ORT numerical agreement on CPU.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import onnxruntime as ort, onnx, math, tempfile, os

torch.manual_seed(0)
Z = 192          # melo z channels
ZFR = 86.1328125 # z frame rate
INTER = 125.0    # intermediate frame rate -> 8000/64
NFFT, HOP = 256, 64   # iSTFT: 8000/64 = 125 Hz frames, Nyquist 4000 Hz


def istft_basis(n_fft, hop):
    """Fixed ConvTranspose1d weights implementing iSTFT (overlap-add) for a
    real STFT with magnitude+phase -> (real, imag). No custom op."""
    # irfft basis: for each of (n_fft//2+1) complex bins -> n_fft time samples
    win = torch.hann_window(n_fft)
    n_bins = n_fft // 2 + 1
    # time index t, bin k: contribution of real & imag parts to sample t
    t = torch.arange(n_fft).float()
    k = torch.arange(n_bins).float().unsqueeze(1)
    ang = 2 * math.pi * k * t / n_fft
    # irfft: x[t] = (1/N) * sum_k scale_k * (Re*cos + (-Im)*sin)... build real/imag bases
    scale = torch.ones(n_bins); scale[1:-1 if n_fft % 2 == 0 else None] = 2.0
    cos_basis = (scale.unsqueeze(1) * torch.cos(ang)) / n_fft * win  # [n_bins, n_fft]
    sin_basis = (-scale.unsqueeze(1) * torch.sin(ang)) / n_fft * win
    return cos_basis, sin_basis, win


class StudentProbe(nn.Module):
    """Mimics the critical-path graph: z -> (Resize resample) -> conv stack ->
    STFT-domain head -> iSTFT(ConvTranspose1d) -> 8 kHz waveform."""
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv1d(Z, 128, 7, padding=3)
        self.body = nn.Conv1d(128, 128, 3, padding=1)
        n_bins = NFFT // 2 + 1
        self.to_spec = nn.Conv1d(128, 2 * n_bins, 1)  # magnitude + phase
        cos_b, sin_b, win = istft_basis(NFFT, HOP)
        # ConvTranspose1d weight shape [in_ch=n_bins, out_ch=1, kernel=n_fft] per part
        self.register_buffer("cos_w", cos_b.unsqueeze(1))  # [n_bins,1,n_fft]
        self.register_buffer("sin_w", sin_b.unsqueeze(1))
        self.register_buffer("win", win)
        self.n_bins = n_bins

    def forward(self, z, out_frames):
        # 1) latent time-resample 86.13 -> 125 Hz via linear interpolate (ONNX Resize)
        z = F.interpolate(z, size=out_frames, mode="linear", align_corners=False)
        x = torch.tanh(self.conv_pre(z))
        x = torch.tanh(self.body(x))
        spec = self.to_spec(x)                       # [B, 2*n_bins, Tf]
        mag = torch.exp(spec[:, : self.n_bins].clamp(max=10))
        phase = spec[:, self.n_bins :]
        real = mag * torch.cos(phase)                # [B, n_bins, Tf]
        imag = mag * torch.sin(phase)
        # 2) iSTFT as ConvTranspose1d overlap-add (stride = hop)
        y = F.conv_transpose1d(real, self.cos_w, stride=HOP) + \
            F.conv_transpose1d(imag, self.sin_w, stride=HOP)
        return y                                     # [B,1, Tf*hop approx]


def main():
    m = StudentProbe().eval()
    T = 380                                  # ~4.4s of z frames at 86.13 Hz
    out_frames = int(round(T * INTER / ZFR)) # -> 125 Hz frames
    z = torch.randn(1, Z, T)
    with torch.no_grad():
        y_torch = m(z, out_frames).numpy()

    f = os.path.join(tempfile.gettempdir(), "student_probe.onnx")
    torch.onnx.export(
        m, (z, torch.tensor(out_frames)), f, opset_version=17,
        input_names=["z", "out_frames"], output_names=["wav"],
        dynamic_axes={"z": {0: "B", 2: "T"}, "wav": {0: "B", 2: "S"}},
        dynamo=False,   # legacy TorchScript exporter: predictable for ORT 1.17 / opset 17
    )
    model = onnx.load(f); onnx.checker.check_model(model)
    ops = sorted({n.op_type for n in model.graph.node})
    print("opset:", model.opset_import[0].version)
    print("ops in graph:", ops)

    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
    # dynamic time test: different T
    for Ttest in (T, 200, 711):
        oframes = int(round(Ttest * INTER / ZFR))
        zt = torch.randn(1, Z, Ttest).numpy()
        yo = sess.run(None, {"z": zt, "out_frames": np.array(oframes, dtype=np.int64)})[0]
        with torch.no_grad():
            yt = m(torch.from_numpy(zt), oframes).numpy()
        err = np.abs(yo - yt).max()
        print(f"  T={Ttest:4d} -> wav {yo.shape}  out_sr_frames={oframes}  max|ort-torch|={err:.2e}")
    print("PROBE OK: opset17 graph runs in ORT", ort.__version__, "CPU, dynamic T, matches torch.")


if __name__ == "__main__":
    main()
