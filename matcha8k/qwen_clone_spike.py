"""Phase-2 spike: does Qwen3-TTS-Base transfer a Taiwan accent via voice cloning,
and does it code-switch zh-TW/en cleanly? Clone a young-TW-female reference, synth test lines.
"""
import argparse, os, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

REF_TEXT = "你好,很高興為您服務。今天天氣很好,希望您有美好的一天。我們會盡力協助您解決問題。"

# Mix: TW-Chinese-only (accent test) + zh-TW/en code-mixed (code-switch test)
LINES = [
    ("zh", "您好,這個星期的研究進度,請您過目一下。"),
    ("zh", "記得攜帶證件,這是基本常識,謝謝您的配合。"),
    ("mix", "幫您轉接給 Kevin 陳經理,他的分機是二一八。"),
    ("mix", "這款 iPhone 支援 Wi-Fi,品質非常好,目前很受歡迎。"),
    ("mix", "Amy 林正在開會,您的 email 已經 reset,請查收。"),
    ("mix", "中英文合成測試。It supports both English 和中文合成。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-TTS-1.7B-Base")
    ap.add_argument("--ref", default="models/tw_ref/hsiaochen_16k.wav")
    ap.add_argument("--out", default="matcha_eval/qwen_spike")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation=args.attn)
    # reuse the same reference prompt across generations
    try:
        prompt = model.create_voice_clone_prompt(ref_audio=args.ref, ref_text=REF_TEXT)
        use_prompt = True
    except Exception:
        use_prompt = False
    for i, (kind, text) in enumerate(LINES, 1):
        kwargs = dict(text=text, language="Chinese")
        if use_prompt:
            kwargs["voice_clone_prompt"] = prompt
        else:
            kwargs["ref_audio"] = args.ref; kwargs["ref_text"] = REF_TEXT
        wavs, sr = model.generate_voice_clone(**kwargs)
        out = os.path.join(args.out, f"qwen_{i:02d}_{kind}.wav")
        sf.write(out, wavs[0], sr)
        print(f"[{i:02d}|{kind}] sr {sr} -> {out}  | {text[:34]}")
    print("DONE qwen clone spike ->", args.out)


if __name__ == "__main__":
    main()
