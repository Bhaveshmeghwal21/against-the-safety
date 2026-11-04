"""The one action format shared by AgentTune rollouts, training rows and the AuditKIT adapter (plan section 4.1).

Search      -> AgentTune's Tiny Aya fallback list: [{"tool_name": "search_corpus", "parameters": {"query": ...}}]
Final turn  -> <answer>...</answer><citations>id,id</citations>   or   <abstain>what is missing</abstain>

AgentTune's rollout parser (`_extract_tool_calls`) reads the search list; `answer_correctness_reward`
reads the <answer> tag. The <citations> and <abstain> tags are ours and are parsed only here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agenttune.agentic.rollout_engines.rollout_factory import DEFAULT_TOOL_RESULT_FORMAT, DEFAULT_TOOLS_FALLBACK_PROMPT

TOOL_NAME = "search_corpus"

ANSWER_PROTOCOL = (
    "Answer only from passages returned by search_corpus. "
    "When the passages contain the answer, reply with only "
    "<answer>the answer</answer><citations>chunk ids separated by commas</citations>. "
    "If the passages do not contain the answer, search again with a different query, "
    "and when you cannot find it reply with only <abstain>what is missing</abstain>."
)

_ANSWER = re.compile(r"^\s*<answer>(.*?)</answer>\s*<citations>(.*?)</citations>\s*$", re.DOTALL | re.IGNORECASE)
_ABSTAIN = re.compile(r"^\s*<abstain>(.*?)</abstain>\s*$", re.DOTALL | re.IGNORECASE)
_CHUNK_ID = re.compile(r"^c\d{5}$")


def system_prompt(tool) -> str:
    """AgentTune's fallback tool prompt with the tool schema, followed by our answer protocol."""
    tools = json.dumps([tool.to_schema()], indent=2)
    return DEFAULT_TOOLS_FALLBACK_PROMPT.format(tools=tools) + "\n\n" + ANSWER_PROTOCOL


def format_search(query: str) -> str:
    return json.dumps([{"tool_name": TOOL_NAME, "parameters": {"query": query}}], ensure_ascii=False)


def format_answer(answer: str, chunk_ids: list[str]) -> str:
    return f"<answer>{answer}</answer><citations>{','.join(chunk_ids)}</citations>"


def format_abstain(missing: str) -> str:
    return f"<abstain>{missing}</abstain>"


def format_tool_result(output: str) -> str:
    """How a search result re-enters the conversation: AgentTune's own user-turn fold."""
    return DEFAULT_TOOL_RESULT_FORMAT.format(name=TOOL_NAME, content=output)


@dataclass
class Turn:
    kind: str  # "search" | "answer" | "abstain" | "invalid"
    query: str | None = None
    answer: str | None = None
    citations: list[str] = field(default_factory=list)
    missing: str | None = None
    error: str | None = None


def parse_turn_lenient(text: str) -> tuple[Turn, str]:
    """First-action parser, as a tool-calling harness reads a model that does not stop where it should.

    Returns the turn and the text that stands as the assistant's message. Strictly valid turns pass through unchanged.
    Otherwise the earliest of these wins: a valid `search_corpus` call anywhere in the text (everything after it is
    dropped, as if generation had stopped at the end of the call; the message becomes the normalized call), an
    `<abstain>` tag, or an `<answer>` tag whose chunk ids are whatever `c12345` ids appear in `<citations>`
    (possibly none). Used for the fair baseline row; the strict parser stays the protocol-compliance measure.
    """
    strict = parse_turn(text)
    if strict.kind != "invalid":
        return strict, text
    found: list[tuple[int, Turn, str]] = []
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\[\s*\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except ValueError:
            continue
        calls = obj if isinstance(obj, list) else None
        if calls and len(calls) >= 1 and isinstance(calls[0], dict) and calls[0].get("tool_name") == TOOL_NAME:
            params = calls[0].get("parameters")
            query = params.get("query") if isinstance(params, dict) else None
            if isinstance(query, str) and query.strip():
                found.append((m.start(), Turn("search", query=query.strip()), format_search(query.strip())))
                break
    m = re.search(r"<abstain>(.*?)</abstain>", text, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        found.append((m.start(), Turn("abstain", missing=m.group(1).strip()), text))
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        cites = re.search(r"<citations>(.*?)</citations>", text, re.DOTALL | re.IGNORECASE)
        ids = re.findall(r"\bc\d{5}\b", cites.group(1)) if cites else []
        found.append((m.start(), Turn("answer", answer=m.group(1).strip(), citations=ids), text))
    if not found:
        return strict, text
    _, turn, message = min(found, key=lambda f: f[0])
    return turn, message


def parse_turn(text: str) -> Turn:
    """Strict parser. Anything that is not exactly one valid action is `invalid` (never an abstention)."""
    m = _ANSWER.match(text)
    if m:
        ids = [c.strip() for c in m.group(2).split(",") if c.strip()]
        if not m.group(1).strip():
            return Turn("invalid", error="empty answer")
        if not ids or not all(_CHUNK_ID.match(c) for c in ids):
            return Turn("invalid", error="citations must be non-empty chunk ids like c00012")
        return Turn("answer", answer=m.group(1).strip(), citations=ids)
    m = _ABSTAIN.match(text)
    if m:
        if not m.group(1).strip():
            return Turn("invalid", error="abstention must say what is missing")
        return Turn("abstain", missing=m.group(1).strip())
    try:
        calls = json.loads(text)
    except ValueError:
        return Turn("invalid", error="not a tagged final turn or a JSON search list")
    if (
        isinstance(calls, list)
        and len(calls) == 1
        and isinstance(calls[0], dict)
        and calls[0].get("tool_name") == TOOL_NAME
        and isinstance(calls[0].get("parameters"), dict)
        and isinstance(calls[0]["parameters"].get("query"), str)
        and calls[0]["parameters"]["query"].strip()
        and set(calls[0]) == {"tool_name", "parameters"}
    ):
        return Turn("search", query=calls[0]["parameters"]["query"].strip())
    return Turn("invalid", error="JSON is not exactly one search_corpus call with a query")
