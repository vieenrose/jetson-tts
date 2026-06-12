"""Student vocoders: z (192ch @ 86.13 Hz) + g (256) -> 8 kHz waveform.

Two architectures, identical I/O contract `forward(z, g) -> wav[B,1,S]`, both export to
opset-17 ONNX with only the ops validated in scripts/ort_compat_probe.py:
  A. HiFiGAN8k  -- re-derived HiFi-GAN upsample stack (x64), safest quality.
  B. Vocos8k    -- ConvNeXt backbone at 125 Hz + ISTFTHead, expected speed winner.

Shared front end: F.interpolate(scale_factor=RESAMPLE_SCALE, linear) resamples z from the
fixed 86.13 Hz grid to 125 Hz so an integer x64 (iSTFT hop / upsample product) lands on
exactly 8000 Hz. g is a constant (single speaker) but kept as an input for drop-in parity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from .audio_config import Z_CHANNELS, G_CHANNELS, RESAMPLE_SCALE, NFFT, HOP
from .istft import ISTFTHead

LRELU = 0.1


def _resample_z(z):
    # z: [B,192,T] @ 86.13 Hz -> [B,192, floor(T*scale)] @ 125 Hz (ONNX Resize w/ scales)
    return F.interpolate(z, scale_factor=RESAMPLE_SCALE, mode="linear",
                         align_corners=False, recompute_scale_factor=False)


# ----------------------------------------------------------------------------- HiFi-GAN
def get_padding(k, d=1):
    return int((k * d - d) / 2)


class ResBlock1(nn.Module):
    def __init__(self, ch, k=3, dil=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=get_padding(k, d)))
            for d in dil])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=1, padding=get_padding(k, 1)))
            for _ in dil])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
            x = xt + x
        return x

    def remove_wn(self):
        for c in list(self.convs1) + list(self.convs2):
            remove_weight_norm(c)


class HiFiGAN8k(nn.Module):
    """x64 upsample: 125 Hz frames -> 8000 Hz. Channels shrunk vs teacher (512) for A57."""
    def __init__(self, init_ch=128, up_rates=(8, 8), up_kernels=(16, 16),
                 rk=(3, 7, 11), rd=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        self.conv_pre = weight_norm(nn.Conv1d(Z_CHANNELS, init_ch, 7, 1, padding=3))
        self.cond = nn.Conv1d(G_CHANNELS, init_ch, 1)
        self.ups = nn.ModuleList()
        ch = init_ch
        for u, k in zip(up_rates, up_kernels):
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)))
            ch //= 2
        self.resblocks = nn.ModuleList()
        chs = []
        c = init_ch
        for _ in up_rates:
            c //= 2
            chs.append(c)
            for k, d in zip(rk, rd):
                self.resblocks.append(ResBlock1(c, k, d))
        self.num_kernels = len(rk)
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False))

    def forward(self, z, g):
        x = _resample_z(z)
        x = self.conv_pre(x) + self.cond(g)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU))
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x, LRELU))
        return torch.tanh(x)

    def remove_wn(self):
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for u in self.ups:
            remove_weight_norm(u)
        for r in self.resblocks:
            r.remove_wn()


# ----------------------------------------------------------------------------- Vocos
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mult=3):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x):                      # x: [B, dim, T]
        res = x
        x = self.dw(x).transpose(1, 2)         # [B,T,dim]
        x = self.norm(x)
        x = self.pw2(F.gelu(self.pw1(x)))
        x = (self.gamma * x).transpose(1, 2)
        return res + x


class Vocos8k(nn.Module):
    """ConvNeXt backbone at 125 Hz -> STFT mag/phase -> ISTFTHead -> 8000 Hz."""
    def __init__(self, dim=256, n_layers=8, n_fft=NFFT, hop=HOP):
        super().__init__()
        n_bins = n_fft // 2 + 1
        self.conv_pre = nn.Conv1d(Z_CHANNELS, dim, 7, 1, padding=3)
        self.cond = nn.Conv1d(G_CHANNELS, dim, 1)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_bins * 2)
        self.istft = ISTFTHead(n_fft, hop)
        self.n_bins = n_bins

    def forward(self, z, g):
        x = _resample_z(z)
        x = self.conv_pre(x) + self.cond(g)              # [B,dim,T]
        x = self.norm_in(x.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        x = self.norm_out(x.transpose(1, 2))             # [B,T,dim]
        h = self.head(x).transpose(1, 2)                 # [B, 2*n_bins, T]
        mag = torch.exp(h[:, : self.n_bins].clamp(max=9.0))
        phase = h[:, self.n_bins :]
        return self.istft(mag, phase)

    def remove_wn(self):
        pass


# ----------------------------------------------------------------------------- Discriminators
class DiscriminatorP(nn.Module):
    def __init__(self, period, k=5, s=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (k, 1), (s, 1), padding=(get_padding(k), 0))),
            weight_norm(nn.Conv2d(32, 128, (k, 1), (s, 1), padding=(get_padding(k), 0))),
            weight_norm(nn.Conv2d(128, 512, (k, 1), (s, 1), padding=(get_padding(k), 0))),
            weight_norm(nn.Conv2d(512, 1024, (k, 1), (s, 1), padding=(get_padding(k), 0))),
            weight_norm(nn.Conv2d(1024, 1024, (k, 1), 1, padding=(get_padding(k), 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect")
        x = x.view(b, c, x.shape[-1] // self.period, self.period)
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiDiscriminator(nn.Module):
    """MPD (periods tuned for 8 kHz) + MSD (3 scales). Used for BOTH students."""
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.mpd = nn.ModuleList([DiscriminatorP(p) for p in periods])
        self.msd = nn.ModuleList([DiscriminatorS() for _ in range(3)])
        self.pools = nn.ModuleList([nn.Identity(), nn.AvgPool1d(4, 2, padding=2),
                                    nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        outs = []  # (yd_real, yd_fake, fmap_real, fmap_fake) per sub-disc
        for d in self.mpd:
            yr, fr = d(y); yf, ff = d(y_hat)
            outs.append((yr, yf, fr, ff))
        yr_s, yf_s = y, y_hat
        for i, d in enumerate(self.msd):
            yr_s = self.pools[i](yr_s) if i else yr_s
            yf_s = self.pools[i](yf_s) if i else yf_s
            yr, fr = d(yr_s); yf, ff = d(yf_s)
            outs.append((yr, yf, fr, ff))
        return outs


def build_generator(arch):
    if arch == "hifigan":
        return HiFiGAN8k()
    if arch == "vocos":
        return Vocos8k()
    raise ValueError(arch)
