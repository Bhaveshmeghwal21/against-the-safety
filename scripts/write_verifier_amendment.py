"""Record the built verifier dataset hashes before any model training."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report = json.loads((root / 'reports/round6_verifier_data.json').read_text(encoding='utf-8'))
target = root / 'research/2026-11-14-round6-preregistration.md'
text = target.read_text(encoding='utf-8')
marker = '\n## Amendment 1: verifier data\n'
if marker in text:
    raise SystemExit('Amendment already exists; do not silently revise preregistration')
train, val = report['splits']['train'], report['splits']['val']
text += (marker + '\nRecorded before GPU training. The builder removes held-out matches, deterministic failures, '
         'duplicate cases and conflicting labels. Swap negatives are weak labels and may contain remaining noise.\n\n'
         f"- Train: {train['rows']} cases, {train['questions']} questions; SHA256 `{train['sha256']}`.\n"
         f"- Validation: {val['rows']} cases, {val['questions']} questions; SHA256 `{val['sha256']}`.\n"
         f"- Yes/no cases: {train['yes_no']} train and {val['yes_no']} validation; the validation set cannot establish yes/no performance.\n")
target.write_text(text, encoding='utf-8')
