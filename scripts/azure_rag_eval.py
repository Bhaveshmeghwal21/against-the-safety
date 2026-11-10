"""Evaluate the specified Azure deployment through the existing RAG harness.

Results are checkpointed after every episode. API and infrastructure errors stop
the run; they are never scored as model failures. The key is read at runtime.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import http.client
import json
import os
import random
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ragharness import corpus as C, data, episodes, evaluation, metrics, protocol, rollout
from ragharness.retrieval import MaskedSearch, open_verified_backend
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-6-luna'
VERSION = 'gpt-6-luna-2026-09-22'
NATIVE_TOOL = {'type': 'function', 'function': {'name': 'search_corpus',
    'description': 'Search the supplied document corpus with a query.',
    'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                   'required': ['query'], 'additionalProperties': False}}}
R5 = ROOT / 'runs/account2_round5/ats-eval-jobs-r5test-ref/out/eval'
OLD = ROOT / 'runs/kaggle/ats-eval-bench-a/out/eval'
SUITES = ('r5t_squad', 'r5t_hotpot', 'r5t_rgb', '2wiki', 'musique', 'squad2', 'rgb', 'rgb_int')


class ProviderFiltered(Exception):
    """An Azure policy block is not a model answer or a scored abstention."""

    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage


def key_from_environment():
    key = os.environ.get('AZURE_OPENAI_API_KEY')
    if not key and os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as h:
            key = winreg.QueryValueEx(h, 'AZURE_OPENAI_API_KEY')[0]
    if not key:
        raise RuntimeError('AZURE_OPENAI_API_KEY is unavailable')
    return key.strip().strip('"')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def build_jobs(name):
    bench = ROOT / 'data/bench' / name
    corp = C.load_corpus(bench / 'corpus.json')
    backend = open_verified_backend(bench / 'corpus.sqlite', corp)
    qs = data.read_questions(bench / 'questions.jsonl')
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    jobs = evaluation.make_jobs(qs, 'bench_' + name, corp, tool, search, 0)
    variants = bench / 'prompt_variants.json'
    if variants.exists():
        jobs += evaluation.prompt_jobs(qs, corp, tool, json.loads(variants.read_text(encoding='utf-8')))
    reference = (R5 if name.startswith('r5t_') else OLD) / f'main_bench_{name}/lenient/scores.jsonl'
    refs = read_jsonl(reference)
    lookup = {(j.q.qid, j.variant): j for j in jobs}
    assert len(lookup) == len(jobs) == len(refs), (name, len(jobs), len(refs))
    keys = [(r['qid'], r['variant']) for r in refs]
    assert len(set(keys)) == len(keys) and set(keys) == set(lookup), name
    for row in refs:
        assert bool(row['answerable']) == bool(lookup[(row['qid'], row['variant'])].spec.answerable)
    return corp, backend, [lookup[k] for k in keys], protocol.system_prompt(tool)


class Azure:
    def __init__(self, api_key, max_tokens, url, request_interval=0):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('AZURE_OPENAI_CHAT_URL must be an HTTPS chat-completions URL')
        self.key = api_key
        self.max_tokens = max_tokens
        self.url = url
        self.endpoint_host = parsed.hostname
        self.request_interval = request_interval
        if self.request_interval < 0:
            raise ValueError('request interval must be nonnegative')

    def wait_for_request_slot(self):
        """Share a conservative request rate across local worker processes."""
        if not self.request_interval:
            return
        path = ROOT / '.cache/tmp/azure_rate_slots.sqlite3'
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(sqlite3.connect(path, timeout=30)) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS slots (model TEXT PRIMARY KEY, next_at REAL NOT NULL)')
            conn.execute('BEGIN IMMEDIATE')
            previous = conn.execute('SELECT next_at FROM slots WHERE model = ?', (MODEL,)).fetchone()
            now = time.time()
            slot = max(now, previous[0] if previous else now)
            conn.execute('INSERT OR REPLACE INTO slots (model, next_at) VALUES (?, ?)',
                         (MODEL, slot + self.request_interval))
            conn.commit()
        time.sleep(max(0, slot - time.time()))

    def request(self, messages, native=False):
        payload = {'model': MODEL, 'messages': messages, 'max_completion_tokens': self.max_tokens}
        if native:
            payload['tools'] = [NATIVE_TOOL]
            payload['tool_choice'] = 'auto'
            payload['reasoning_effort'] = 'none'  # required by this deployment's chat-completions API for function tools
        body = json.dumps(payload).encode('utf-8')
        for attempt in range(9):
            self.wait_for_request_slot()
            req = urllib.request.Request(self.url, data=body, method='POST',
                                         headers={'api-key': self.key, 'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    obj = json.load(response)
                if obj.get('model') != VERSION:
                    raise RuntimeError(f"Unexpected Azure model version: {obj.get('model')!r}")
                choice = obj['choices'][0]
                raw = choice['message'].get('content') or ''
                return raw, {'usage': obj.get('usage', {}), 'content_raw': raw,
                    'tool_calls': choice['message'].get('tool_calls') or [],
                    'refusal': choice['message'].get('refusal'), 'finish_reason': choice.get('finish_reason'),
                    'model': obj.get('model'), 'id': obj.get('id')}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode('utf-8', errors='replace').replace(self.key, '[REDACTED]')
                try:
                    error_body = json.loads(detail)
                    code = error_body.get('error', {}).get('code')
                except (ValueError, AttributeError):
                    error_body = {}
                    code = None
                filtered_choice = any(c.get('finish_reason') == 'content_filter' for c in error_body.get('choices', []))
                if exc.code == 400 and (code == 'content_filter' or filtered_choice):
                    raise ProviderFiltered('Azure content filter blocked the unmodified request or response', error_body.get('usage')) from None
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 8:
                    raise RuntimeError(f'Azure HTTP {exc.code}: {detail[:1200]}') from None
                delay = min(55.0, max(float(exc.headers.get('Retry-After', 0) or 0), 2 ** attempt) + random.random())
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
                if attempt == 8:
                    raise RuntimeError(f'Azure network failure: {str(exc).replace(self.key, "[REDACTED]")}') from None
                delay = min(55.0, 2 ** attempt + random.random())
            time.sleep(delay)
        raise RuntimeError('Azure retry limit exhausted')


def canonical_text(raw):
    """Losslessly wrap a plain refusal or plain answer with explicit citations."""
    turn, _ = protocol.parse_turn_lenient(raw)
    if turn.kind != 'invalid':
        return raw
    stripped = raw.strip()
    if re.match(r"^(?:sorry[, ]+)?(?:i\s+)?(?:can(?:not|'t)|am unable to|do not have enough|don't have enough)\b", stripped, re.I):
        return protocol.format_abstain(stripped)
    m = re.fullmatch(r'\s*([^<>]+?)\s*<citations>(c\d{5}(?:\s*,\s*c\d{5})*)</citations>\s*', raw, re.I | re.S)
    return protocol.format_answer(m.group(1).strip(), [c.strip() for c in m.group(2).split(',')]) if m else raw


def one(job, corp, backend, system, api, mode='exact'):
    tool = CorpusSearchTool(backend, top_k=3)
    co = rollout.rollout_steps(job.q, corp, tool, system, job.variant, job.spec.masked,
                               job.spec.first_search_masked, job.spec.prefill, 3, lenient=True)
    messages = next(co)
    native_messages = [{'role': 'system', 'content': protocol.ANSWER_PROTOCOL +
                        ' Use the native search_corpus tool for searches. Follow the final-answer tags exactly.'}, messages[1]]
    calls, start = [], time.time()
    while True:
        try:
            text, meta = api.request(native_messages if mode == 'native' else messages, native=(mode == 'native'))
        except ProviderFiltered as exc:
            blocked_call = len(calls) + 1
            if exc.usage is not None:
                calls.append({'usage': exc.usage, 'content_raw': '', 'tool_calls': [], 'refusal': None,
                              'finish_reason': 'content_filter', 'model': None, 'id': None})
            return {'score': {'qid': job.q.qid, 'variant': job.variant, 'answerable': job.spec.answerable,
                              'qtype': job.q.qtype, 'provider_blocked': True, 'model_calls': len(calls),
                              'input_tokens': sum(c['usage'].get('prompt_tokens', 0) for c in calls),
                              'output_tokens': sum(c['usage'].get('completion_tokens', 0) for c in calls)},
                    'transcript': None, 'api_calls': calls, 'blocked': str(exc), 'blocked_call': blocked_call}
        calls.append(meta)
        if meta['finish_reason'] == 'content_filter':
            return {'score': {'qid': job.q.qid, 'variant': job.variant, 'answerable': job.spec.answerable,
                              'qtype': job.q.qtype, 'provider_blocked': True, 'model_calls': len(calls),
                              'input_tokens': sum(c['usage'].get('prompt_tokens', 0) for c in calls),
                              'output_tokens': sum(c['usage'].get('completion_tokens', 0) for c in calls)},
                    'transcript': None, 'api_calls': calls, 'blocked': 'Azure filtered the generated response',
                    'blocked_call': len(calls)}
        tool_call = None
        if mode == 'native' and meta['tool_calls']:
            tool_call = meta['tool_calls'][0]
            fun = tool_call.get('function', {})
            try:
                query = json.loads(fun.get('arguments', '{}')).get('query')
            except (ValueError, AttributeError):
                query = None
            if fun.get('name') == 'search_corpus' and isinstance(query, str) and query.strip():
                text = protocol.format_search(query.strip())
            else:
                text = ''
        elif mode == 'native':
            text = canonical_text(text)
        try:
            messages = co.send(text)
            if mode == 'native':
                if tool_call is not None:
                    native_messages += [{'role': 'assistant', 'content': None, 'tool_calls': [tool_call]},
                                        {'role': 'tool', 'tool_call_id': tool_call['id'], 'content': messages[-1]['content']}]
                else:
                    native_messages += [{'role': 'assistant', 'content': text}, messages[-1]]
        except StopIteration as stop:
            ep = stop.value
            break
    score = metrics.score_episode(ep, job.q, corp, job.spec.answerable)
    from ragharness.round4 import aspect_coverage
    score.update(aspect_coverage(job.q, corp, ep))
    score['evidence_supported_good'] = bool(score['good'] and (not job.spec.answerable or score['all_aspects_cited']))
    score.update(qtype=job.q.qtype, seconds=round(time.time()-start, 3), model_calls=len(calls),
                 input_tokens=sum(c['usage'].get('prompt_tokens', 0) for c in calls),
                 output_tokens=sum(c['usage'].get('completion_tokens', 0) for c in calls))
    return {'score': score, 'transcript': episodes.to_transcript(ep, job.q, score), 'api_calls': calls}


def main():
    global MODEL, VERSION
    ap = argparse.ArgumentParser()
    ap.add_argument('--suites', default=','.join(SUITES))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--mode', choices=('exact', 'native'), default='exact')
    ap.add_argument('--model', default=MODEL, help='Azure deployment name')
    ap.add_argument('--response-model', default=VERSION, help='Expected model name returned by the API')
    ap.add_argument('--temperature-label', default='deployment default (1)', help='Recorded description; does not set temperature')
    ap.add_argument('--request-interval', type=float, default=0, help='Minimum seconds between requests across local workers')
    ap.add_argument('--limit', type=int, default=0, help='per-suite pilot limit; never mix with full outputs')
    ap.add_argument('--out', default='runs/azure_round5')
    args = ap.parse_args()
    MODEL, VERSION = args.model, args.response_model
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise SystemExit('Output must stay within the workspace')
    url = os.environ.get('AZURE_OPENAI_CHAT_URL')
    if not url:
        raise SystemExit('Set AZURE_OPENAI_CHAT_URL to your Azure chat-completions endpoint')
    api = Azure(key_from_environment(), args.max_tokens, url, args.request_interval)
    for name in args.suites.split(','):
        if name not in SUITES:
            raise ValueError(f'unknown suite {name}')
        corp, backend, jobs, system = build_jobs(name)
        if args.limit:
            jobs = jobs[:args.limit]
        folder = out / f'azure_bench_{name}' / 'lenient'
        folder.mkdir(parents=True, exist_ok=True)
        config = {'model': VERSION, 'endpoint_host': api.endpoint_host, 'max_completion_tokens': args.max_tokens,
                  'temperature': args.temperature_label, 'episodes': len(jobs), 'corpus_hash': corp.index_hash,
                  'search_budget': 3, 'top_k': 3, 'passage_chars': 800, 'parser': 'lenient', 'mode': args.mode}
        if args.mode == 'native':
            config['reasoning_effort'] = 'none'
        config_path = folder / 'run_config.json'
        if config_path.exists():
            stored_config = json.loads(config_path.read_text(encoding='utf-8'))
            if args.mode == 'exact':
                stored_config.setdefault('mode', 'exact')  # earlier exact runs used the same protocol before mode was named
            if stored_config != config:
                raise RuntimeError(f'existing run has different settings: {config_path}')
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        partial = folder / 'episodes.partial.jsonl'
        previous = read_jsonl(partial) if partial.exists() else []
        done = {(r['score']['qid'], r['score']['variant']): r for r in previous}
        if len(done) != len(previous):
            raise RuntimeError('duplicate persisted episodes')
        expected = {(j.q.qid, j.variant) for j in jobs}
        if not set(done) <= expected:
            raise RuntimeError('persisted episodes do not match the requested run')
        remaining = [j for j in jobs if (j.q.qid, j.variant) not in done]
        print(f'{name}: {len(done)}/{len(jobs)} complete, {len(remaining)} to request', flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool, partial.open('a', encoding='utf-8', newline='\n') as f:
            todo = iter(remaining)
            active = {pool.submit(one, j, corp, backend, system, api, args.mode) for j in [next(todo, None) for _ in range(args.workers)] if j is not None}
            while active:
                finished, active = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in finished:
                    row = fut.result()
                    k = (row['score']['qid'], row['score']['variant'])
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    f.flush()
                    done[k] = row
                    if len(done) % 25 == 0 or len(done) == len(jobs):
                        print(f'{name}: {len(done)}/{len(jobs)} complete', flush=True)
                for _ in finished:
                    j = next(todo, None)
                    if j is not None:
                        active.add(pool.submit(one, j, corp, backend, system, api, args.mode))
        rows = [done[(j.q.qid, j.variant)] for j in jobs]
        available = [r for r in rows if not r['score'].get('provider_blocked')]
        blocked = [r for r in rows if r['score'].get('provider_blocked')]
        for field, file in [('score','scores.jsonl'), ('transcript','transcripts.jsonl')]:
            (folder / file).write_text(''.join(json.dumps(r[field], ensure_ascii=False) + '\n' for r in available), encoding='utf-8')
        (folder / 'provider_blocked.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in blocked), encoding='utf-8')
        totals = {'episodes': len(rows), 'model_calls': sum(len(r['api_calls']) for r in rows),
                  'provider_blocked': len(blocked), 'scorable': len(available),
                  'prompt_tokens': sum(r['score']['input_tokens'] for r in rows),
                  'completion_tokens': sum(r['score']['output_tokens'] for r in rows),
                  'cached_prompt_tokens': sum(c['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0) for r in rows for c in r['api_calls']),
                  'reasoning_tokens': sum(c['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in rows for c in r['api_calls']),
                  'length_finishes': sum(c['finish_reason']=='length' for r in rows for c in r['api_calls'])}
        (folder/'usage.json').write_text(json.dumps(totals,indent=2),encoding='utf-8')
        print(json.dumps({'suite':name,**totals}),flush=True)


if __name__ == '__main__':
    main()
