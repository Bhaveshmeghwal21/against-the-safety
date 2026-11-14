"""The scripted oracle query policy shared by the retrieval ceiling and the episode builder.

It only ever emits text that is visible at that turn: the question, a title mention found in the question or in an
already observed (or pre-supplied) gold passage, or a keyword reformulation of the question. Gold labels pick WHICH
visible mention to try; they never put unseen text into a query (plan section 5.2).
"""

from __future__ import annotations

import re

from .data import Question
from .retrieval import Hit

_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-]*")
STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but is are was were be been being do does did has have had "
    "which what who whom whose when where why how that this these those it its as into than then there their "
    "he she they his her him them both either neither also not no yes more most other another same one two".split()
)
# Multilingual suites (research/2026-11-15-multilingual-plan.md): question words, postpositions, auxiliaries and articles to leave out of
# the scripted reformulation. The danda (U+0964, U+0965) is punctuation, not part of a word.
_WORD_ML = re.compile(r"[\w\u0900-\u0963\u0966-\u097f][\w\u0900-\u0963\u0966-\u097f'\u2019\-]*")
_STOP_ML = {
    "en": STOPWORDS,
    "es": frozenset(
        "el la los las un una unos unas de del al en y e o u a por para con sin sobre entre que qu\u00e9 qui\u00e9n qui\u00e9nes cu\u00e1l cu\u00e1les "
        "cu\u00e1ndo d\u00f3nde c\u00f3mo cu\u00e1nto cu\u00e1nta cu\u00e1ntos cu\u00e1ntas es son fue fueron era eran ser est\u00e1 est\u00e1n se su sus lo le les "
        "como m\u00e1s muy no s\u00ed ha han hay fue ha sido".split()),
    "hi": frozenset(
        "\u0915\u093e \u0915\u0940 \u0915\u0947 \u0915\u094b \u0938\u0947 \u092e\u0947\u0902 \u092a\u0930 \u0914\u0930 \u092f\u093e \u0939\u0948 \u0939\u0948\u0902 \u0925\u093e \u0925\u0940 \u0925\u0947 \u0939\u094b "
        "\u0939\u0941\u0906 \u0939\u0941\u0908 \u0939\u0941\u090f \u0915\u094d\u092f\u093e \u0915\u094c\u0928 \u0915\u092c \u0915\u0939\u093e\u0901 \u0915\u0939\u093e\u0902 \u0915\u0948\u0938\u0947 \u0915\u094d\u092f\u094b\u0902 "
        "\u0915\u093f\u0924\u0928\u093e \u0915\u093f\u0924\u0928\u0947 \u0915\u093f\u0924\u0928\u0940 \u0915\u093f\u0938 \u0915\u093f\u0938\u0928\u0947 \u0915\u093f\u0938\u0947 \u0915\u093f\u0938\u0915\u093e \u0915\u093f\u0938\u0915\u0940 "
        "\u0915\u093f\u0938\u0915\u0947 \u0915\u093f \u091c\u094b \u092f\u0939 \u0935\u0939 \u0907\u0938 \u0909\u0938 \u0928\u0947 \u092d\u0940 \u0924\u094b \u090f\u0915 \u0939\u0940 \u0930\u0939\u093e \u0930\u0939\u0940 \u0930\u0939\u0947 "
        "\u0917\u092f\u093e \u0917\u0908 \u0917\u090f \u0932\u093f\u090f \u0926\u094d\u0935\u093e\u0930\u093e \u0915\u0930".split()),
}


def mention(title: str) -> str:
    return _PAREN.sub("", title).strip()


def contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE) is not None


def keyword_query(question: str, max_words: int = 8, lang: str = "") -> str:
    if lang:  # multilingual suites: Unicode words (Devanagari vowel signs stay inside the word) and that language's function words
        words = [w.strip("'\u2019-") for w in _WORD_ML.findall(question)]
        stop = _STOP_ML.get(lang, STOPWORDS)
        return " ".join([w for w in words if w and w.lower() not in stop][:max_words])
    words = [w.strip("'\u2019-") for w in _WORD.findall(question)]
    words = [w for w in words if w and w.lower() not in STOPWORDS]
    return " ".join(words[:max_words])


def _norm(query: str) -> str:
    return " ".join(query.lower().split())


def next_query(
    q: Question,
    support: dict[str, str],
    observed: dict[str, Hit],
    used: set[str],
    n_searches: int,
    prefilled: bool = False,
    keywords: bool = True,
) -> tuple[str | None, str | None]:
    """Return (query, stop_reason). `used` holds normalized queries already issued; they are never repeated.

    stop_reason is "complete" (every support paragraph observed), or "no_new_query" (no unused visible query left).
    """
    missing = [t for t, c in support.items() if c not in observed]
    if not missing:
        return None, "complete"
    if n_searches == 0 and not prefilled:
        candidates = [q.question]
    else:
        visible = [q.question] + [observed[c].text for c in support.values() if c in observed]
        candidates = [mention(t) for t in missing if any(contains(v, mention(t)) for v in visible)]
        candidates.append(q.question)
        if keywords:
            candidates.append(keyword_query(q.question, lang=q.lang))
    for cand in candidates:
        if cand and _norm(cand) not in used:
            return cand, None
    return None, "no_new_query"
