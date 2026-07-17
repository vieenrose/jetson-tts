"""Reference streaming TTS runner over the split v2-Stream ONNX models — the exact
orchestration a sherpa-onnx C++ runner should mirror.

  enc.onnx : (x,tone,lang,x_lengths,noise_scale,length_scale) -> z[1,192,T]   (once)
  dec.onnx : z[1,192,Tc] -> wav[1,1,Tc*256]                                    (per chunk)

Streaming = run enc once, then decode z in CHUNK-frame steps via OVERLAP-SAVE:
for chunk frames [a,b) decode z[:, :, a-LEFT : b+RIGHT] and keep the middle
(b-a)*256 samples. Bit-exact vs the monolithic model (validated: cos 1.000000,
maxerr ~1e-6). First audio arrives after enc + one chunk instead of the whole
utterance.

Usage:
  python -m streaming.onnx_stream --enc <enc.onnx> --dec <dec.onnx> --ids <parity_inputs.json> [--i 0]
  (or --text "..." with the g2pw frontend available)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import onnxruntime as ort

C, HOP, CHUNK, LEFT, RIGHT = 192, 256, 24, 64, 16  # non-causal clean vocoder needs 16 (causal was 4)


def _blank(seq):
    o = [0] * (2 * len(seq) + 1)
    o[1::2] = seq
    return np.array([o], np.int64)


class StreamingTTS:
    def __init__(self, enc_path, dec_path, threads=2):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(dec_path, so, providers=["CPUExecutionProvider"])

    def encode(self, phone_ids, tone_ids, lang_ids, noise_scale=0.667, length_scale=1.0):
        x, tn, lg = _blank(phone_ids), _blank(tone_ids), _blank(lang_ids)
        return self.enc.run(None, {
            "x": x, "tone": tn, "lang": lg,
            "x_lengths": np.array([x.shape[1]], np.int64),
            "noise_scale": np.array([noise_scale], np.float32),
            "length_scale": np.array([length_scale], np.float32)})[0]  # [1,192,T]

    def stream(self, z):
        """Yield audio chunks (np.float32) as z is decoded chunk-by-chunk."""
        T = z.shape[2]
        for a in range(0, T, CHUNK):
            b = min(a + CHUNK, T); s0 = max(0, a - LEFT); e = min(T, b + RIGHT)
            w = self.dec.run(None, {"z": z[:, :, s0:e]})[0].reshape(-1)
            off = (a - s0) * HOP; keep = (b - a) * HOP
            yield w[off:off + keep]

    def synth(self, phone_ids, tone_ids, lang_ids, **kw):
        z = self.encode(phone_ids, tone_ids, lang_ids, **kw)
        return np.concatenate(list(self.stream(z))) if z.shape[2] else np.zeros(0, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="/home/luigi/mbvits_run/v2stream_split/v2stream_enc.onnx")
    ap.add_argument("--dec", default="/home/luigi/mbvits_run/v2stream_split/v2stream_dec.onnx")
    ap.add_argument("--ids", default="/home/luigi/mbvits_run/parity_inputs.json")
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--text", default=None)
    ap.add_argument("--out", default="/home/luigi/mbvits_run/onnx_stream_demo.wav")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    tts = StreamingTTS(a.enc, a.dec, a.threads)

    if a.text:
        import sys; sys.path.insert(0, "/home/luigi/primetts-space")
        import frontend_bopomofo as F
        o = F.text_to_ids(a.text); p, t, l = o["phone_ids"], o["tone_ids"], o["lang_ids"]
    else:
        r = json.load(open(a.ids))["rows"][a.i]; p, t, l = r["phone_ids"], r["tone_ids"], r["lang_ids"]

    t0 = time.perf_counter(); z = tts.encode(p, t, l); t_enc = time.perf_counter() - t0
    chunks = []; tfirst = None
    for c in tts.stream(z):
        chunks.append(c)
        if tfirst is None: tfirst = time.perf_counter() - t0
    total = time.perf_counter() - t0
    wav = np.concatenate(chunks); audio_s = len(wav) / 16000
    pk = np.max(np.abs(wav)); wav = wav * (0.97 / pk) if pk > 1e-6 else wav
    import soundfile as sf; sf.write(a.out, wav.astype(np.float32), 16000)
    print(f"frames={z.shape[2]} audio={audio_s:.2f}s  enc={t_enc*1e3:.0f}ms "
          f"first-audio={tfirst*1e3:.0f}ms  total={total*1e3:.0f}ms RTF={total/audio_s:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
