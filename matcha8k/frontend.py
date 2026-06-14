"""Matcha zh-en text frontend (matches dengcunqin's training frontend), full-line (no sentence split).

Chinese -> pypinyin TONE3 -> vocab id. English -> espeak IPA -> convert_to_gruut_en_us_strict ->
per-char vocab id. Punctuation normalised. OOV -> 1. Produces the exact token-id sequence the
matcha_tts_zh_en checkpoint expects (validated by feeding the official ONNX am+vocoder).

espeak provided via espeakng-loader (no system espeak-ng needed).
"""
import os
from pypinyin import pinyin, Style

# --- espeak setup (bundled lib) ---
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(espeakng_loader.get_library_path())
os.environ.setdefault("ESPEAK_DATA_PATH", os.path.dirname(espeakng_loader.get_data_path()))
from phonemizer import phonemize as _phonemize


def _load_vocab(path):
    vocab = [x.rstrip("\n") for x in open(path, encoding="utf-8")]
    return {tok: i + 1 for i, tok in enumerate(vocab)}   # id = line_index + 1


_PUNCT = {"，": ",", "。": ".", "！": "!", "？": "?"}

_GRUUT_REPL = [
    ("ɝ", "ɜɹ"), ("ɚ", "əɹ"),
    ("eɪ", "A"), ("aɪ", "I"), ("ɔɪ", "Y"), ("oʊ", "O"), ("əʊ", "O"), ("aʊ", "W"),
    ("tʃ", "ʧ"), ("dʒ", "ʤ"), ("ː", ""), ("g", "ɡ"), ("r", "ɹ"), ("e", "ɛ"),
]


def _to_gruut(ipa):
    t = "".join(ipa) if isinstance(ipa, list) else ipa
    for a, b in _GRUUT_REPL:
        t = t.replace(a, b)
    return t


def _espeak_ipa(text):
    return _phonemize(text, language="en-us", backend="espeak", strip=True,
                      with_stress=True, preserve_punctuation=False)


def _zh_to_ids(text, v):
    py = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    out = []
    for item in py:
        p = item[0]
        for k, r in _PUNCT.items():
            p = p.replace(k, r)
        out.append(v.get(p, 1))
    return out


class MatchaFrontend:
    def __init__(self, vocab_path="models/matcha-src/vocab_tts.txt"):
        self.v = _load_vocab(vocab_path)

    def __call__(self, text):
        return self.text_to_ids(text)

    def text_to_ids(self, s):
        result, i = [], 0
        while i < len(s):
            c = s[i]
            if "一" <= c <= "鿿":
                part = ""
                while i < len(s) and "一" <= s[i] <= "鿿":
                    part += s[i]; i += 1
                result.extend(_zh_to_ids(part, self.v))
            elif c.isalpha():
                part = ""
                while i < len(s) and s[i].isalpha():
                    part += s[i]; i += 1
                ipa = _to_gruut(_espeak_ipa(part))
                result.extend(self.v.get(ch, 1) for ch in ipa)
            else:
                result.append(self.v.get(_PUNCT.get(c, c), 1))
                i += 1
        return result


if __name__ == "__main__":
    fe = MatchaFrontend()
    for t in ["你好,世界。", "幫您轉接給 Kevin 陳經理。", "中英文 hello 測試"]:
        ids = fe.text_to_ids(t)
        print(f"{t!r} -> {len(ids)} ids: {ids}")
