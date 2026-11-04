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


def mention(title: str) -> str:
    return _PAREN.sub("", title).strip()


def contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE) is not None


def keyword_query(question: str, max_words: int = 8) -> str:
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
            candidates.append(keyword_query(q.question))
    for cand in candidates:
        if cand and _norm(cand) not in used:
            return cand, None
    return None, "no_new_query"
