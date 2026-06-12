"""Export-friendly iSTFT as a fixed ConvTranspose1d (no custom ops; runs in ORT 1.17 CPU).

Given STFT magnitude and phase at INTER_FRAME_RATE, reconstruct the 8 kHz waveform via
windowed overlap-add. The irfft + windowing + overlap-add collapse into one ConvTranspose1d
with fixed (non-learned) weights, plus a fixed window-normalisation also done by a second
ConvTranspose1d of ones. Validated numerically against torch.istft in scripts/.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISTFTHead(nn.Module):
    def __init__(self, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        n_bins = n_fft // 2 + 1
        self.n_bins = n_bins
        win = torch.hann_window(n_fft)
        t = torch.arange(n_fft).float()
        k = torch.arange(n_bins).float().unsqueeze(1)
        ang = 2 * math.pi * k * t / n_fft
        scale = torch.ones(n_bins)
        scale[1 : (n_bins - 1 if n_fft % 2 == 0 else n_bins)] = 2.0
        cos_basis = (scale.unsqueeze(1) * torch.cos(ang)) / n_fft * win   # [n_bins, n_fft]
        sin_basis = (-scale.unsqueeze(1) * torch.sin(ang)) / n_fft * win
        self.register_buffer("cos_w", cos_basis.unsqueeze(1))             # [n_bins,1,n_fft]
        self.register_buffer("sin_w", sin_basis.unsqueeze(1))
        # window normalisation (overlap-add of win^2), as ConvTranspose1d of a ones input
        self.register_buffer("win_sq", (win ** 2).view(1, 1, n_fft))

    def forward(self, mag, phase):
        # mag, phase: [B, n_bins, Tf] -> waveform [B, 1, (Tf-1)*hop]  (center=True convention)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        y = F.conv_transpose1d(real, self.cos_w, stride=self.hop) + \
            F.conv_transpose1d(imag, self.sin_w, stride=self.hop)         # [B,1, (Tf-1)*hop + n_fft]
        ones = torch.ones(mag.shape[0], 1, mag.shape[2], device=mag.device, dtype=mag.dtype)
        norm = F.conv_transpose1d(ones, self.win_sq, stride=self.hop)
        y = y / (norm + 1e-8)
        # trim n_fft//2 each side: discards the low-overlap (ill-normalised) edge regions and
        # matches torch.istft(center=True). y[0] then aligns to z-resampled frame 0.
        p = self.n_fft // 2
        return y[..., p:-p]


if __name__ == "__main__":
    # numerical self-test vs torch.istft on random complex spectra
    torch.manual_seed(0)
    n_fft, hop, Tf, B = 256, 64, 40, 2
    head = ISTFTHead(n_fft, hop)
    n_bins = n_fft // 2 + 1
    mag = torch.rand(B, n_bins, Tf) + 0.1
    phase = (torch.rand(B, n_bins, Tf) - 0.5) * 6.28
    spec = mag * torch.exp(1j * phase)                                    # [B,n_bins,Tf]
    ref = torch.istft(spec, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=torch.hann_window(n_fft), center=True, normalized=False)
    ours = head(mag, phase)[:, 0]                # already center-trimmed to (Tf-1)*hop
    n = min(ours.shape[1], ref.shape[1])
    err = (ours[:, :n] - ref[:, :n]).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"ISTFTHead vs torch.istft: max abs err {err:.3e}  rel {rel:.3e}  "
          f"({'OK' if rel < 1e-4 else 'MISMATCH'})")
