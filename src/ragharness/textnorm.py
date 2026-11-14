"""Language-aware answer normalization for the multilingual check (research/2026-11-15-multilingual-plan.md, section 5).

Only questions that carry a language tag (`Question.lang` is "hi", "en" or "es", set by scripts/prepare_xquad.py) use this module.
Every other suite keeps its original code path in `metrics`, `rescore`, `checker` and `scripts/build_verifier_cases.py`, byte for byte.

The rule follows the official XQuAD / MLQA evaluation script: lower-case, remove every Unicode punctuation character and every ASCII
punctuation character, remove the language's articles, collapse white space, then compare white-space-separated tokens. Two additions
made before any model ran on these questions (plan, "Amendments"): Unicode NFC, and Devanagari digits read as ASCII digits.
"""

from __future__ import annotations

import collections
import re
import string
import unicodedata

ARTICLES = {
    "en": frozenset({"a", "an", "the"}),
    "es": frozenset({"el", "la", "los", "las", "un", "una", "unos", "unas"}),
    "hi": frozenset(),
}
_ASCII_PUNCT = frozenset(string.punctuation)
_DIGITS = {ord(c): str(i) for i, c in enumerate("\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f")}
_DEVANAGARI_LETTER = re.compile(r"[\u0900-\u0963\u0971-\u097f]")


def _drop(ch: str) -> bool:
    return ch in _ASCII_PUNCT or unicodedata.category(ch).startswith("P")


def normalize(text: str, lang: str) -> str:
    s = unicodedata.normalize("NFC", text or "").lower().translate(_DIGITS)
    s = "".join(ch for ch in s if not _drop(ch))
    articles = ARTICLES.get(lang, frozenset())
    return " ".join(w for w in s.split() if w not in articles)


def tokens(text: str, lang: str, stop: frozenset[str] = frozenset()) -> list[str]:
    return [t for t in normalize(text, lang).split() if t not in stop]


def f1_em(pred: str, gold: str, lang: str) -> tuple[float, float]:
    """(token F1, exact match) of one prediction against one gold string, on the normalized text."""
    p, g = normalize(pred, lang).split(), normalize(gold, lang).split()
    if not p or not g:
        same = float(p == g)
        return same, same
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not common:
        return 0.0, float(p == g)
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec), float(p == g)


def has_script_letters(text: str, lang: str) -> bool | None:
    """True when the text holds a letter of the question's own script. Hindi only; None for the Latin-script languages."""
    if lang != "hi":
        return None
    return bool(_DEVANAGARI_LETTER.search(text or ""))
