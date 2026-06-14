"""8 kHz vocoder for Matcha-icefall-zh-en: mel[80,T] @62.5Hz -> 8 kHz waveform.

Drop-in replacement for vocos-16khz-univ.onnx but at 8 kHz. Keeps the EXACT sherpa VocosVocoder
contract: ONNX input `mels`[B,80,T], ONNX outputs `mag`,`x`,`y` ([B,n_fft/2+1,T], x=cos(phase),
y=sin(phase)); sherpa does the iSTFT with n_fft=512/hop=128/win=512 (set in ONNX metadata) -> 8 kHz.

Training applies the same iSTFT (student.istft.ISTFTHead) to compute the waveform for the losses;
the exported graph stops at mag/x/y. mel rate 62.5Hz x hop128 = exactly 8000 Hz (no resample).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from student.istft import ISTFTHead
from student.models import ConvNeXtBlock

N_MELS = 80
NFFT8 = 512
HOP8 = 128
WIN8 = 512


class VocosMel8k(nn.Module):
    def __init__(self, n_mels=N_MELS, dim=384, n_layers=8, n_fft=NFFT8, hop=HOP8):
        super().__init__()
        n_bins = n_fft // 2 + 1
        self.n_bins = n_bins
        self.conv_pre = nn.Conv1d(n_mels, dim, 7, padding=3)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_bins * 2)
        self.istft = ISTFTHead(n_fft, hop)

    def _spec(self, mel):
        x = self.conv_pre(mel)                                  # [B,dim,T]
        x = self.norm_in(x.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        x = self.norm_out(x.transpose(1, 2))                   # [B,T,dim]
        h = self.head(x).transpose(1, 2)                       # [B,2*bins,T]
        mag = torch.exp(h[:, : self.n_bins].clamp(max=9.0))
        phase = h[:, self.n_bins :]
        return mag, phase

    def forward(self, mel):                                    # training -> waveform [B,1,S]
        mag, phase = self._spec(mel)
        return self.istft(mag, phase)

    def export_spec(self, mel):                                # ONNX -> mag, x=cos, y=sin
        mag, phase = self._spec(mel)
        return mag, torch.cos(phase), torch.sin(phase)

    def remove_wn(self):
        pass


class ExportWrapper(nn.Module):
    """Presents the sherpa VocosVocoder ONNX interface: mels -> mag, x, y."""
    def __init__(self, voc):
        super().__init__()
        self.voc = voc

    def forward(self, mels):
        return self.voc.export_spec(mels)
