"""Phrase chunker for streaming TTS (Phase 0/1).

Buffers streamed text and emits prosodic phrases at punctuation / strong breaks.
The phrase boundary is the natural context edge for g2pw polyphone disambiguation,
so we never split inside a word / zh compound. Very long clauses get a soft break
at a word boundary under a char cap to protect first-audio latency.

Design ref: docs/streaming-arch-design.md section 2 (frontend row) + section 8 Phase 1.
"""
from __future__ import annotations
import re

# Strong break chars: zh full-width + EN. Split AFTER the delimiter (keep it with the
# preceding clause for prosody), matching _SPLIT_RE in the demo app.
_STRONG = "。！？；;!?\n"
_SOFT = "，、,:：… "
_BREAK_RE = re.compile(r"(?<=[" + re.escape(_STRONG) + r"])")
_SOFT_RE = re.compile(r"(?<=[" + re.escape(_STRONG + _SOFT) + r"])")

# ASCII word char: don't split inside an English word or a run of digits.
_WORDCHAR = re.compile(r"[A-Za-z0-9@._%+\-]")


def _is_wordish(a: str, b: str) -> bool:
    """True if splitting between chars a and b would cut an English word / number."""
    return bool(a and b and _WORDCHAR.match(a) and _WORDCHAR.match(b))


def split_phrases(text: str, soft_cap: int = 12, first_cap: int | None = 6):
    """Split text into word-safe prosodic phrases.

    soft_cap: max chars before forcing a soft break at a word boundary.
    first_cap: if set, the FIRST phrase is capped shorter (fast first-audio); the
        rest use soft_cap. Set None to disable the short-opener behaviour.
    Returns list[str]; whitespace-only fragments are dropped, delimiters kept.
    """
    text = (text or "").strip()
    if not text:
        return []
    # 1. hard split at strong breaks
    hard = [c for c in _BREAK_RE.split(text) if c.strip()]
    out: list[str] = []
    for clause in hard:
        out.extend(_soft_split(clause, soft_cap))
    # 2. shorten the opener for latency, if it exceeds first_cap
    if first_cap and out and _visible_len(out[0]) > first_cap:
        head, tail = _cut_at(out[0], first_cap)
        if head.strip():
            out = [head] + ([tail] if tail.strip() else []) + out[1:]
    return [c for c in out if c.strip()]


def _soft_split(clause: str, cap: int):
    """Break an over-long clause at soft punctuation, then at word-safe positions."""
    if _visible_len(clause) <= cap:
        return [clause]
    # prefer soft-punctuation boundaries
    pieces = [c for c in _SOFT_RE.split(clause) if c]
    out, cur = [], ""
    for p in pieces:
        if cur and _visible_len(cur) + _visible_len(p) > cap:
            out.append(cur); cur = ""
        cur += p
        while _visible_len(cur) > cap:  # still too long -> force a word-safe cut
            head, cur = _cut_at(cur, cap)
            out.append(head)
    if cur.strip():
        out.append(cur)
    return out


def _visible_len(s: str) -> int:
    return len(s.strip())


def _cut_at(s: str, cap: int):
    """Cut s near `cap` chars without splitting an English word / number."""
    if len(s) <= cap:
        return s, ""
    i = cap
    # walk left off a word boundary
    while 0 < i < len(s) and _is_wordish(s[i - 1], s[i]):
        i -= 1
    if i == 0:  # whole prefix is one long word -> walk right instead
        i = cap
        while i < len(s) and _is_wordish(s[i - 1], s[i]):
            i += 1
    return s[:i], s[i:]


if __name__ == "__main__":
    import sys, json
    t = sys.argv[1] if len(sys.argv) > 1 else \
        "Anderson 先生您好,您 2024年3月15日 訂的 3 件商品總共 NT$1,299,序號 AB1234CD,發票已寄到 anderson.wang@gmail.com,謝謝。"
    for i, p in enumerate(split_phrases(t)):
        print(f"[{i}] ({_visible_len(p):2d}) {p!r}")
