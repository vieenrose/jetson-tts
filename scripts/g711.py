"""ITU-T G.711 mu-law companding (pure numpy) for the simulated 8 kHz telephony channel.

roundtrip(x): 8 kHz float [-1,1] -> mu-law encode (8-bit) -> decode -> float. Models the
quantisation the phone attendant's G.711 channel imposes. (Bandlimiting to <=4 kHz is already
guaranteed by the 8 kHz sample rate; G.711 adds the 8-bit log quantisation.)
"""
import numpy as np

_MU = 255.0


def encode(x):
    x = np.clip(x, -1.0, 1.0)
    mag = np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    y = np.sign(x) * mag
    # to 8-bit unsigned
    q = np.clip(np.round((y * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    return q


def decode(q):
    y = (q.astype(np.float32) / 255.0 - 0.5) * 2.0
    mag = (np.expm1(np.abs(y) * np.log1p(_MU))) / _MU
    return np.sign(y) * mag


def roundtrip(x):
    return decode(encode(x)).astype(np.float32)
