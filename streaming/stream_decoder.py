"""Streaming MB-iSTFT vocoder: emit audio in fixed frame-chunks as latent z frames
arrive, with bounded state. Validated math (streaming/../MB-iSTFT-VITS/lr_test.py):
CHUNK=24 + LEFT=64 + RIGHT=4 frames reproduces the full-utterance waveform bit-exactly.

This uses overlap-save: to emit chunk frames [a,b) we run the causal decoder on
[a-LEFT : b+RIGHT] and keep the middle (b-a)*HOP samples. It carries only the last
LEFT+CHUNK+RIGHT frames of z, so state is bounded (O(LEFT) recompute per chunk — the
zero-recompute ring-buffer version is the RapidSpeech.cpp Phase-4 optimization; this is
the exact reference every runtime is checked against).

The vocoder is the streaming part; enc/flow/dp run per-utterance upstream (frozen v2),
producing z which is fed here incrementally.
"""
from __future__ import annotations
import numpy as np, torch

HOP = 256          # samples per 16 kHz frame (upsample 16 * iSTFT-hop 4 * subbands 4)
CHUNK, LEFT, RIGHT = 24, 64, 4


class StreamingVocoder:
    def __init__(self, dec, chunk=CHUNK, left=LEFT, right=RIGHT, device="cpu"):
        self.dec, self.chunk, self.left, self.right = dec, chunk, left, right
        self.device = device
        self.buf = None      # [1, ch, T] rolling z buffer (frames not yet emitted + left cache)
        self.emitted = 0     # frames already emitted (relative to buffer start)
        self.base = 0        # absolute frame index of buffer[...,0]

    @torch.no_grad()
    def _decode(self, z):
        return self.dec(z.to(self.device))[0].reshape(-1).cpu().numpy()

    @torch.no_grad()
    def push(self, z_frames, final=False):
        """Feed z frames [1, ch, n]; yield finished audio chunks (np.float32).
        Call with final=True (any tail) to flush the last, possibly-short chunk."""
        z = z_frames if isinstance(z_frames, torch.Tensor) else torch.as_tensor(z_frames)
        if z.dim() == 2: z = z.unsqueeze(0)
        self.buf = z if self.buf is None else torch.cat([self.buf, z], dim=-1)
        T = self.buf.shape[-1]
        # emit while a full chunk + right-lookahead is available (or flushing)
        while True:
            a = self.emitted
            need_right = 0 if final else self.right
            if a + self.chunk + need_right > T and not final:
                break
            b = min(a + self.chunk, T)
            if a >= b:
                break
            s = max(0, a - self.left); e = min(T, b + self.right)
            seg = self._decode(self.buf[:, :, s:e])
            off = (a - s) * HOP; keep = (b - a) * HOP
            yield seg[off:off + keep]
            self.emitted = b
            if final and b >= T:
                break
        # trim consumed-but-not-needed left history to keep the buffer bounded
        keep_from = max(0, self.emitted - self.left)
        if keep_from > 0:
            self.buf = self.buf[:, :, keep_from:]
            self.emitted -= keep_from
            self.base += keep_from
