#!/usr/bin/env python3
"""Definitive drop-in proof: run model.onnx through the actual sherpa_onnx OfflineTts engine
(the same C++ core as the device's sherpa-onnx-offline-tts) and save 8 kHz WAVs.

Run with a python that has sherpa_onnx installed, e.g.:
  /home/luigi/realtime-bot/venv/bin/python scripts/verify_sherpa_dropin.py <model_dir> <out_dir>

For MeloTTS (n_speakers=1) sherpa hardcodes the graph sid to metadata.speaker_id (=1), so the
deployed voice == melo's own emb_g(1) == the distillation target. We pass sid=0 (the valid
external id) to avoid the validation warning; the internal sid is overridden regardless.
"""
import sys, os, wave, struct

LINES = [
    "您好,這裡是宏達電子,很高興為您服務。",
    "幫您轉接給 Kevin 陳經理,他的分機是 533。",
    "Amy 林不在位子上,需要幫您留言嗎?",
    "您的 Wi-Fi 設定已完成,如有問題請撥分機 218。",
    "麻煩您撥打 0912-345-678 聯絡 Jason 王襄理,謝謝。",
]


def save_wav(path, samples, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        frames = b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 32767)) for s in samples)
        w.writeframes(frames)


def main():
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    import sherpa_onnx, numpy as np
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{model_dir}/model.onnx", tokens=f"{model_dir}/tokens.txt",
                lexicon=f"{model_dir}/lexicon.txt"),
            num_threads=4, provider="cpu"),
        max_num_sentences=1)
    tts = sherpa_onnx.OfflineTts(cfg)
    print(f"sherpa_onnx {sherpa_onnx.__version__} | OfflineTts loaded | sample_rate {tts.sample_rate}")
    assert tts.sample_rate == 8000, tts.sample_rate
    for i, t in enumerate(LINES):
        a = tts.generate(t, sid=0, speed=1.0)
        x = np.array(a.samples)
        out = f"{out_dir}/sherpa_{i:02d}.wav"
        save_wav(out, x, a.sample_rate)
        print(f"  [{i}] {a.sample_rate}Hz {len(x)/a.sample_rate:.2f}s peak {abs(x).max():.3f}  {t}")
    print(f"OK: stock sherpa-onnx engine ran the drop-in unmodified -> 8 kHz WAVs in {out_dir}")


if __name__ == "__main__":
    main()
