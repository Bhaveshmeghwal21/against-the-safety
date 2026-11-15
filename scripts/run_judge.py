"""Run AuditKIT's judge-based RAG metrics on the cases from scripts/build_judge_cases.py with the Azure gpt-6-luna deployment as the judge.

  python scripts/run_judge.py --cases data/judge/cases_r5t.jsonl --out data/judge/run1 [--limit 12] [--budget 2.5]

Metrics (AuditKIT classes, unmodified prompts, prompt version v2):
  every answered episode          faithfulness, hallucination, response_groundedness, answer_relevancy
  every episode with passages     context_relevance (question only)
  answerable episodes             context_precision and context_recall (reference = the gold answer)
Hidden-evidence episodes get no reference answer: nothing in what the system could see contains it.

Rules (research/2026-11-16-judge-latency-preregistration.md): an unreadable judge reply or a provider block leaves that metric empty for that episode
and is counted; it is never scored as 0 or 1. A network or API failure that survives the retries stops the run (results so far are kept and the run
resumes). Every judge call is cached by the hash of its prompt, so a rerun reads the same replies, and the token usage of every call is logged so the
cost is metered rather than guessed. The run stops when the metered cost reaches `--budget`. The key is read at run time and never written.
"""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

URL = "https://pawaac.services.ai.azure.com/openai/v1/chat/completions"
VERSION = "gpt-6-luna-2026-09-22"
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.10 / 1e6, 0.01 / 1e6, 0.50 / 1e6  # dollars per token, Global Standard short context (reports/azure_roi.md)


class ProviderFiltered(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def api_key() -> str:
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not key and os.name == "nt":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
            key = winreg.QueryValueEx(h, "AZURE_OPENAI_API_KEY")[0]
    if not key:
        raise SystemExit("AZURE_OPENAI_API_KEY is unavailable")
    return key.strip().strip('"')


def cost_of(usage: dict) -> float:
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return (usage.get("prompt_tokens", 0) - cached) * PRICE_IN + cached * PRICE_CACHED + usage.get("completion_tokens", 0) * PRICE_OUT


class AzureJudge:
    """A text-in, text-out judge in the shape AuditKIT's metrics expect (`generate` and `identity`)."""

    def __init__(self, key: str, calls_path: Path, budget: float, max_completion_tokens: int = 3000, post=None):
        self.key, self.budget, self.max_completion_tokens = key, budget, max_completion_tokens
        self.calls_path = calls_path
        self.lock = threading.Lock()
        self.cache, self.spent, self.new_calls, self.cache_hits = {}, 0.0, 0, 0
        self._post = post or self._http
        if calls_path.exists():
            for line in calls_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["h"]] = row["reply"]
                    self.spent += cost_of(row["usage"])

    def identity(self) -> dict:
        return {"provider": "azure", "model": VERSION, "sampling": "deployment default"}

    def _http(self, prompt: str) -> dict:
        body = json.dumps({"model": "gpt-6-luna", "messages": [{"role": "user", "content": prompt}],
                           "max_completion_tokens": self.max_completion_tokens}).encode("utf-8")
        for attempt in range(9):
            req = urllib.request.Request(URL, data=body, method="POST", headers={"api-key": self.key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").replace(self.key, "[REDACTED]")
                if exc.code == 400 and "content_filter" in detail:
                    raise ProviderFiltered(detail[:200]) from None
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 8:
                    raise RuntimeError(f"Azure HTTP {exc.code}: {detail[:600]}") from None
                delay = min(55.0, max(float(exc.headers.get("Retry-After", 0) or 0), 2 ** attempt) + random.random())
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
                if attempt == 8:
                    raise RuntimeError(f"Azure network failure: {str(exc).replace(self.key, '[REDACTED]')}") from None
                delay = min(55.0, 2 ** attempt + random.random())
            time.sleep(delay)
        raise RuntimeError("Azure retry limit exhausted")

    def complete(self, prompt: str) -> str:
        h = hashlib.sha256((VERSION + "\0" + prompt).encode("utf-8")).hexdigest()
        with self.lock:
            if h in self.cache:
                self.cache_hits += 1
                return self.cache[h]
            if self.spent >= self.budget:
                raise BudgetExceeded(f"metered cost ${self.spent:.3f} reached the ${self.budget:.2f} limit")
        obj = self._post(prompt)
        if obj.get("model") != VERSION:
            raise RuntimeError(f"Unexpected judge model version: {obj.get('model')!r}")
        choice = obj["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError("judge reply was cut off by the completion limit")
        reply = choice["message"].get("content") or ""
        usage = obj.get("usage", {})
        with self.lock:
            self.cache[h] = reply
            self.spent += cost_of(usage)
            self.new_calls += 1
            with self.calls_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"h": h, "reply": reply, "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details")}}) + "\n")
        return reply

    def generate(self, requests):
        from auditkit.model import Generated, Result_

        return [Result_(completions=[Generated(text=self.complete(r.prompt))]) for r in requests]


def plan(case: dict) -> list[str]:
    metrics = []
    if case["final_kind"] == "answer" and (case["answer"] or "").strip():
        metrics += ["faithfulness", "hallucination", "response_groundedness", "answer_relevancy"]
    if case["contexts"]:
        metrics.append("context_relevance")
        if case["answerable"] and case["gold"]:
            metrics += ["context_precision", "context_recall"]
    return metrics


def score_case(case: dict, judge, metrics_by_name: dict | None = None) -> list[dict]:
    from auditkit.metrics.rag_judge import (AnswerRelevancy, ContextPrecision, ContextRecall, ContextRelevance, Faithfulness, Hallucination,
                                            ResponseGroundedness)
    from auditkit.sample import Sample

    classes = {"faithfulness": Faithfulness, "hallucination": Hallucination, "response_groundedness": ResponseGroundedness,
               "answer_relevancy": AnswerRelevancy, "context_relevance": ContextRelevance, "context_precision": ContextPrecision,
               "context_recall": ContextRecall}
    sample = Sample(input=case["question"], target=case["gold"] if case["answerable"] else None, retrieval_context=list(case["contexts"]),
                    id=f"{case['part']}|{case['qid']}|{case['variant']}")
    base = {k: case[k] for k in ("system", "part", "qid", "variant", "comp", "answerable")}
    rows = []
    for name in plan(case):
        metric = classes[name](judge_model=judge, temperature=None)
        row = {**base, "metric": name, "value": None, "status": "scored", "reason": None, "detail": None}
        if name in ("faithfulness", "hallucination", "response_groundedness") and not case["contexts"]:
            row["status"] = "no_context"  # the system answered without having been shown any passage: nothing to judge the answer against
            rows.append(row)
            continue
        try:
            scores = metric.score(sample, case["answer"] or "", None)
        except ProviderFiltered as exc:
            row.update(status="provider_blocked", reason=str(exc)[:200])
        except ValueError as exc:
            row.update(status="unreadable_reply", reason=str(exc)[:300])
        else:
            wanted = [s for s in scores if s.name == name]
            if not wanted:
                row["status"] = "no_claims" if name in ("faithfulness", "hallucination", "answer_relevancy") else "not_applicable"
            else:
                s = wanted[0]
                row.update(value=s.value, reason=s.reason, detail=s.metadata or None)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (pilot)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=2.5, help="stop when the metered cost reaches this many dollars")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise SystemExit("output must stay inside the workspace")
    out.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in (ROOT / args.cases).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        first = list(dict.fromkeys((c["part"], c["qid"], c["variant"]) for c in cases))[: args.limit]
        cases = [c for c in cases if (c["part"], c["qid"], c["variant"]) in set(first)]
    scores_path = out / "scores.jsonl"
    done = set()
    if scores_path.exists():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["system"], r["part"], r["qid"], r["variant"]))
    todo = [c for c in cases if (c["system"], c["part"], c["qid"], c["variant"]) not in done]
    judge = AzureJudge(api_key(), out.parent / "judge_calls.jsonl", args.budget)
    print(f"{len(cases)} cases, {len(todo)} to do, metered spend so far ${judge.spent:.3f}", flush=True)
    write_lock, finished, started = threading.Lock(), [0], time.time()

    def work(case):
        rows = score_case(case, judge)
        with write_lock, scores_path.open("a", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            finished[0] += 1
            if finished[0] % 25 == 0:
                print(f"  {finished[0]}/{len(todo)} cases, {time.time() - started:.0f}s, ${judge.spent:.3f}", flush=True)

    stop = None
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, c) for c in todo]
        for fut in concurrent.futures.as_completed(futures):
            exc = fut.exception()
            if exc and stop is None:
                stop = exc
                for other in futures:
                    other.cancel()
    summary = {"cases_done": finished[0], "new_judge_calls": judge.new_calls, "cache_hits": judge.cache_hits, "metered_spend_usd": round(judge.spent, 4),
               "stopped_by": repr(stop) if stop else None}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if stop:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
