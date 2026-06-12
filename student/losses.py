"""Losses for 8 kHz vocoder distillation.

- Multi-resolution STFT loss (spectral convergence + log-magnitude).
- Telephony-band mel L1 (mel filters limited to 300-3400 Hz, where G.711 lives).
- LSGAN adversarial + feature-matching (HiFi-GAN style).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .audio_config import TARGET_SR


def _stft_mag(x, n_fft, hop, win):
    w = torch.hann_window(win, device=x.device)
    s = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w,
                   center=True, return_complex=True)
    return s.abs().clamp_min(1e-7)


class MultiResSTFTLoss(nn.Module):
    def __init__(self, ffts=(128, 256, 512), hops=(32, 64, 128), wins=(128, 256, 512)):
        super().__init__()
        self.cfg = list(zip(ffts, hops, wins))

    def forward(self, y_hat, y):
        sc, lm = 0.0, 0.0
        for n_fft, hop, win in self.cfg:
            Yh = _stft_mag(y_hat, n_fft, hop, win)
            Y = _stft_mag(y, n_fft, hop, win)
            sc = sc + torch.norm(Y - Yh, p="fro") / (torch.norm(Y, p="fro") + 1e-7)
            lm = lm + F.l1_loss(torch.log(Yh), torch.log(Y))
        n = len(self.cfg)
        return sc / n, lm / n


class TelephonyMelLoss(nn.Module):
    """Mel L1 restricted to the 300-3400 Hz telephony band."""
    def __init__(self, n_fft=512, hop=128, n_mels=64, fmin=300.0, fmax=3400.0):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=1.0, center=True)

    def forward(self, y_hat, y):
        self.mel = self.mel.to(y.device)
        mh = torch.log(self.mel(y_hat).clamp_min(1e-5))
        mt = torch.log(self.mel(y).clamp_min(1e-5))
        return F.l1_loss(mh, mt)


def generator_adv_loss(disc_outs):
    """LSGAN generator loss + feature matching over all sub-discriminators."""
    adv, fm = 0.0, 0.0
    for yr, yf, fr, ff in disc_outs:
        adv = adv + torch.mean((yf - 1.0) ** 2)
        for a, b in zip(fr, ff):
            fm = fm + F.l1_loss(b, a.detach())
    return adv, fm


def discriminator_loss(disc_outs):
    loss = 0.0
    for yr, yf, fr, ff in disc_outs:
        loss = loss + torch.mean((yr - 1.0) ** 2) + torch.mean(yf ** 2)
    return loss
