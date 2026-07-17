"""zh-TW/en unified frontend for the Inflect-Nano retrain.
zh chars -> bopomofo (g2pw, Taiwan readings) -> zhuyin symbol units + tone (1-5);
en words -> arpabet (g2p_en) + stress; one sequence, per-phone language id (ZH/EN).
"""
from __future__ import annotations
import re
from g2pw import G2PWConverter
from g2p_en import G2p

# 37 standard zhuyin symbols (U+3105..U+3129)
ZHUYIN = [chr(c) for c in range(0x3105, 0x312A)]
ARPABET = ['AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH',
           'IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T','TH',
           'UH','UW','V','W','Y','Z','ZH']
PUNCT = [',', '.', '?', '!', '…', '-', "'"]
SPECIAL = ['_blank', '_pad', 'UNK', 'SP']            # SP = inter-word/space pause
# ㄭ = syllabic-vowel symbol for empty-rime syllables (是/十/日/司...). U+312D is outside the
# U+3105..U+3129 ZHUYIN range so it was missing. APPENDED AT END to preserve all existing phone ids
# (warm-start compatibility); new id, embedding row trained during the re-align retrain.
SYLLABIC = ['ㄭ']
SYMBOLS = SPECIAL + ZHUYIN + ARPABET + PUNCT + SYLLABIC
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
LANG = {'ZH': 0, 'EN': 1}                            # per-phone language id

# Letter-name arpabet for spelled-out single UPPERCASE letters (serials, codes: AB1234CD).
# g2p_en mispronounces a lone "A" as the article schwa (AH) instead of the letter name (EY);
# uppercase single letters in this text only ever appear in spelled codes, so map them by name.
LETTER_ARP = {
    'A': 'EY', 'B': 'B IY', 'C': 'S IY', 'D': 'D IY', 'E': 'IY', 'F': 'EH F', 'G': 'JH IY',
    'H': 'EY CH', 'I': 'AY', 'J': 'JH EY', 'K': 'K EY', 'L': 'EH L', 'M': 'EH M', 'N': 'EH N',
    'O': 'OW', 'P': 'P IY', 'Q': 'K Y UW', 'R': 'AA R', 'S': 'EH S', 'T': 'T IY', 'U': 'Y UW',
    'V': 'V IY', 'W': 'D AH B AH L Y UW', 'X': 'EH K S', 'Y': 'W AY', 'Z': 'Z IY',
}

# OOV brand proper nouns g2p_en mispronounces (Widevine->"white wine", etc.). Hand-written
# arpabet so they're spoken correctly. Keys matched case-insensitively (see EN branch).
BRAND_LEX = {
    'widevine': 'W AY D V AY N', 'irdeto': 'IH R D EH T OW', 'conax': 'K OW N AE K S',
    'verimatrix': 'V EH R IH M EY T R IH K S', 'wimax': 'W AY M AE K S',
}

_g2pw = None
_g2pen = None
import text_norm                                       # entity-aware normalizer (phone/email/price/date/…)

def _lazy():
    global _g2pw, _g2pen
    if _g2pw is None:
        _g2pw = G2PWConverter()
        _g2pen = G2p()

def _split_syllable(syl: str):
    """'ㄓㄨㄢ3' -> (['ㄓ','ㄨ','ㄢ'], tone 3)."""
    tone = 0
    if syl and syl[-1].isdigit():
        tone = int(syl[-1]); syl = syl[:-1]
    units = [c for c in syl if c in SYM2ID]
    # Empty-rime syllables (zhi/chi/shi/ri/zi/ci/si: 是/十/日/司/思/資...) are written in bopomofo as
    # the bare retroflex/dental sibilant with an IMPLICIT syllabic vowel. Without an explicit vowel
    # phone the model renders a clipped fricative that merges into the next syllable. Append the
    # syllabic-vowel symbol ㄭ so these carry a proper rime. (Requires training data with ㄭ.)
    if len(units) == 1 and units[0] in "ㄓㄔㄕㄖㄗㄘㄙ" and "ㄭ" in SYM2ID:
        units = [units[0], "ㄭ"]
    return units, tone

def text_to_phones(text: str):
    _lazy()
    text = text_norm.normalize(text)                  # entities -> spoken form, normalize punct
    bopo = _g2pw(text)[0]                             # per-char bopomofo or None
    chars = list(text)
    phones, tones, langs = [], [], []
    i = 0
    while i < len(chars):
        b = bopo[i] if i < len(bopo) else None
        ch = chars[i]
        if b is not None:                             # zh char
            units, tone = _split_syllable(b)
            for u in units:
                phones.append(u); tones.append(min(tone, 5)); langs.append(LANG['ZH'])
            i += 1
        elif re.match(r'[A-Za-z]', ch):               # English run -> g2p_en
            j = i
            while j < len(chars) and re.match(r"[A-Za-z']", chars[j]):
                j += 1
            word = ''.join(chars[i:j])
            # Pronunciation source, in priority order:
            #  1. known OOV brand -> hand-written arpabet (g2p_en mangles them)
            #  2. single UPPERCASE letter (serial/code) -> letter name (not g2p_en's article schwa)
            #  3. ALL-CAPS 2-6 letter acronym (VIP/OTP/PDF/LTE/RDK/BUC/LNB) -> spell letter-by-letter
            #     (g2p_en treats them as words and truncates VIP->"p")
            #  4. otherwise -> g2p_en
            if word.lower() in BRAND_LEX:
                _src = BRAND_LEX[word.lower()].split()
            elif len(word) == 1 and word in LETTER_ARP:
                _src = LETTER_ARP[word].split()
            elif word.isupper() and 2 <= len(word) <= 6 and all(c in LETTER_ARP for c in word):
                _src = ' '.join(LETTER_ARP[c] for c in word).split()
            else:
                _src = _g2pen(word)
            for p in _src:
                p = p.strip()
                if not p:
                    continue
                stress = 0
                if p[-1].isdigit():
                    stress = int(p[-1]); p = p[:-1]
                if p in SYM2ID:
                    phones.append(p); tones.append(stress); langs.append(LANG['EN'])
            phones.append('SP'); tones.append(0); langs.append(LANG['EN'])
            i = j
        else:                                         # punctuation / space / other
            if ch in PUNCT:
                phones.append(ch); tones.append(0); langs.append(LANG['ZH'])
            elif ch.strip() == '':
                if phones and phones[-1] != 'SP':
                    phones.append('SP'); tones.append(0); langs.append(LANG['ZH'])
            i += 1
    return phones, tones, langs

def text_to_ids(text: str):
    phones, tones, langs = text_to_phones(text)
    ids = [SYM2ID.get(p, SYM2ID['UNK']) for p in phones]
    return {"phones": phones, "phone_ids": ids, "tone_ids": tones, "lang_ids": langs}
