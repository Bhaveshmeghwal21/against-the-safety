"""Training rows for Tiny Aya: one prompt/completion row per supervised assistant decision (plan section 6.1).

Input is what AgentTune hands a `trainer_factory`: `EventLog.as_dataset_rows("sft")` rows, i.e. `messages` plus
`segment_weights` (0.0 for user and tool text, above 0 for assistant text). Two repairs are needed for Tiny Aya:
- AgentTune renders a search as `<tool_call>{"name", "arguments"}</tool_call>`; Tiny Aya's fallback format is the
  `[{"tool_name", "parameters"}]` list, so searches are restored to `protocol.format_search`.
- The `tool` role renders as an empty turn in Tiny Aya's template, so results become user turns with
  AgentTune's own `fold_tool_messages_into_user` wording.
The template has no generation markers, so assistant-only loss is not available; prompt/completion rows give the same
effect because only the completion is trained.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import protocol

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_aya_global_chat_template.jinja"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
PROMPT_END = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
COMPLETION_END = "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"


def fetch_template(path: Path = FIXTURE) -> Path:
    """Download the pinned Tiny Aya Global chat template from the public Hub API (no gated files, no tokenizer)."""
    import json
    import urllib.request

    url = f"https://huggingface.co/api/models/CohereLabs/tiny-aya-global/revision/{REVISION}"
    with urllib.request.urlopen(url, timeout=30) as r:
        info = json.load(r)
    if info["sha"] != REVISION:
        raise RuntimeError(f"Hub returned revision {info['sha']}, expected {REVISION}")
    tmpl = info["config"]["tokenizer_config"]["chat_template"]
    if isinstance(tmpl, list):
        tmpl = next(t["template"] for t in tmpl if t["name"] == "default")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tmpl, encoding="utf-8", newline="")
    return path


def restore_assistant(content: str) -> str:
    """Undo AgentTune's `<tool_call>` rendering of a search; leave final turns untouched."""
    c = content.strip()
    if c.startswith("<tool_call>") and c.endswith("</tool_call>"):
        call = json.loads(c[len("<tool_call>") : -len("</tool_call>")])
        return protocol.format_search(call["arguments"]["query"])
    return content


def decisions(row: dict, system: str) -> list[dict]:
    """One conversational prompt/completion row per assistant message with positive weight."""
    out: list[dict] = []
    history: list[dict] = [{"role": "system", "content": system}]
    for msg, weight in zip(row["messages"], row["segment_weights"], strict=True):
        role, content = msg["role"], msg["content"]
        if role == "assistant":
            text = restore_assistant(content)
            if weight > 0:
                out.append({"prompt": list(history), "completion": [{"role": "assistant", "content": text}]})
            history.append({"role": "assistant", "content": text})
            continue
        if weight != 0:
            raise ValueError(f"non-assistant message carries loss weight {weight}")
        if role == "tool":
            content = protocol.format_tool_result(content)
        elif role != "user":
            raise ValueError(f"unexpected role {role!r}")
        if history[-1]["role"] == "user":  # strict alternation: merge consecutive user text
            history[-1] = {"role": "user", "content": f"{history[-1]['content']}\n\n{content}"}
        else:
            history.append({"role": "user", "content": content})
    return out


@lru_cache(maxsize=1)
def _template():
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    if not FIXTURE.exists():
        fetch_template()
    return env.from_string(FIXTURE.read_text(encoding="utf-8"))


def render(messages: list[dict], add_generation_prompt: bool) -> str:
    """Render with the pinned Tiny Aya Global chat template (no tokenizer or gated download needed)."""
    return _template().render(messages=messages, bos_token="<BOS_TOKEN>", add_generation_prompt=add_generation_prompt)


def check_row(row: dict) -> dict:
    """Template assertions from plan section 6.1. Returns the rendered prompt and completion strings."""
    prompt, completion = row["prompt"], row["completion"]
    prompt_text = render(prompt, add_generation_prompt=True)
    full_text = render(prompt + completion, add_generation_prompt=False)
    assert prompt_text.endswith(PROMPT_END), "prompt must end at the assistant response start"
    assert full_text.startswith(prompt_text), "prompt is not a prefix of the full render"
    completion_text = full_text[len(prompt_text) :]
    assert completion_text == completion[0]["content"] + COMPLETION_END, "completion must be content + END_RESPONSE + END_OF_TURN"
    # every tool result must survive rendering (the `tool` role would not)
    for m in prompt:
        if m["role"] == "user":
            assert m["content"] in prompt_text, "a user/observation turn was dropped by the template"
    kind = protocol.parse_turn(completion[0]["content"]).kind
    assert kind in {"search", "answer", "abstain"}, f"completion is not a valid action: {kind}"
    return {"prompt_text": prompt_text, "completion_text": completion_text, "kind": kind}


CHARS_PER_TOKEN = 3.5  # conservative English estimate; replace with real Tiny Aya token counts once the tokenizer is available


def rendered_chars(row: dict) -> tuple[int, int]:
    """(prompt chars, completion chars) of the exact string the model is trained on."""
    prompt = render(row["prompt"], add_generation_prompt=True)
    return len(prompt), len(row["completion"][0]["content"]) + len(COMPLETION_END)


def est_tokens(row: dict) -> tuple[int, int]:
    """(processed tokens, supervised tokens) estimated from characters. Processed = prompt + completion."""
    p, c = rendered_chars(row)
    return round((p + c) / CHARS_PER_TOKEN), round(c / CHARS_PER_TOKEN)


def real_tokens(row: dict, tokenizer, strict: bool = True) -> tuple[int, int]:
    """(processed tokens, supervised tokens) with the real tokenizer and the chat template it ships.

    Uses the same two renders as TRL: the prompt with the generation marker, then prompt + completion. The completion
    tokens are the difference, so the count matches what the loss will cover. `strict` raises if the prompt tokens are
    not a prefix of the full tokens (TRL only warns); keep it on with the real tokenizer, where the markers are single
    special tokens.
    """
    prompt_ids = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    full_ids = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True, return_dict=True)["input_ids"]
    if strict and list(full_ids[: len(prompt_ids)]) != list(prompt_ids):
        raise ValueError("prompt tokens are not a prefix of prompt+completion tokens")
    return len(full_ids), len(full_ids) - len(prompt_ids)
