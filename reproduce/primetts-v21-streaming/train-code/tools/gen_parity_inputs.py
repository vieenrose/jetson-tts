#!/usr/bin/env python
"""Generate FIXED (phone,tone,lang) inputs for the MB-iSTFT-VITS parity refs by
running the real zh-TW/en frontend (frontend_bopomofo.py) on a fixed sentence
list incl code-mix. Streams are blank-interleaved (intersperse id 0) to match
training (add_blank=True), which is how SynthesizerTrn.infer must be fed.

Must run in an interpreter that has g2pw + g2p_en (e.g. .venv-breezy), CPU-only:
  CUDA_VISIBLE_DEVICES="" NLTK_DATA=/home/luigi/nltk_data \
  /home/luigi/jetson-tts/.venv-breezy/bin/python tools/gen_parity_inputs.py

Writes /home/luigi/mbvits_run/parity_inputs.json
"""
import os, sys, json

FRONTEND_DIR = "/home/luigi/jetson-tts/mossnano/zhtw8k"
OUT = "/home/luigi/mbvits_run/parity_inputs.json"

SENTENCES = [
    "今天天氣很好。",
    "你好，很高興認識你。",
    "我想用 VIP 帳號登入系統。",
    "請輸入你的 password 然後按 enter。",
    "Widevine 和 Irdeto 都是 DRM 系統。",
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test.",
    "序號是 AB1234CD，請確認。",
    "會議將在下午三點半開始。",
    "我的電話號碼是 0912345678。",
    "價格是 1980 元，含稅。",
    "OTP 驗證碼已經寄到你的信箱。",
    "台北今天的氣溫是攝氏二十八度。",
    "請問這附近有沒有捷運站？",
    "謝謝你的幫忙，我很感激。",
    "This product uses LTE and WiMAX networks.",
    "系統更新完成，請重新啟動。",
    "下載進度百分之九十五。",
    "歡迎使用我們的語音助理。",
    "台灣的夜市非常有名，例如士林夜市。",
]


def intersperse(lst, item=0):
    out = [item] * (len(lst) * 2 + 1)
    out[1::2] = lst
    return out


def main():
    sys.path.insert(0, FRONTEND_DIR)
    os.environ.setdefault("NLTK_DATA", "/home/luigi/nltk_data")
    import frontend_bopomofo as fe
    assert len(fe.SYMBOLS) == 88, f"symbol table size {len(fe.SYMBOLS)} != 88"

    rows = []
    for idx, text in enumerate(SENTENCES):
        r = fe.text_to_ids(text)
        p, t, l = r["phone_ids"], r["tone_ids"], r["lang_ids"]
        assert len(p) == len(t) == len(l), text
        pb, tb, lb = intersperse(p, 0), intersperse(t, 0), intersperse(l, 0)
        rows.append({
            "idx": idx, "text": text,
            "phones": r["phones"],
            "phone_ids_raw": p, "tone_ids_raw": t, "lang_ids_raw": l,
            "phone_ids": pb, "tone_ids": tb, "lang_ids": lb,   # blank-interleaved (fed to model)
            "len_raw": len(p), "len": len(pb),
        })
        print(f"[{idx:02d}] len_raw={len(p):3d} len_blanked={len(pb):3d}  {text}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "note": "blank-interleaved (id 0) on all 3 streams; matches training add_blank=True",
            "n_symbols": 88, "num_tones": 6, "num_langs": 2, "blank_id": 0,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[write] {OUT}  ({len(rows)} utterances)")


if __name__ == "__main__":
    main()
