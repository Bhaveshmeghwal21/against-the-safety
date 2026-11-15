"""Build the general-capability check sets (research/2026-11-16-general-capability-preregistration.md).

  python scripts/prepare_general.py

Downloads public, ungated test sets from the Hugging Face hub and writes one JSONL per benchmark to data/general/ with a common shape:
{"id", "bench", "subject", "question", "choices": [...], "gold": index}. A manifest with counts, sources and SHA-256 hashes goes to
data/general/manifest.json (tracked; the JSONL files themselves are rebuilt from this script).

  mmlu          cais/mmlu, config "all", test split: 14,042 questions in 57 subjects (MIT)
  arc_c, arc_e  allenai/ai2_arc, ARC-Challenge and ARC-Easy, test split: 1,172 and 2,376 questions (CC BY-SA 4.0)
  truthfulqa    truthfulqa/truthful_qa, multiple_choice, validation split, MC1: 817 questions (Apache-2.0). The dataset lists the correct answer
                first, so the choices are shuffled with a fixed seed (61 plus the row index) to remove any position effect.
  gmmlu_en/hi/es CohereLabs/Global-MMLU-Lite, test split: the same 400 questions in English, Hindi and Spanish (Apache-2.0)
"""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "general"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def fetch(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def mmlu() -> list[dict]:
    out = []
    for i, r in enumerate(rows(fetch("cais/mmlu", "all/test-00000-of-00001.parquet"))):
        out.append({"id": f"mmlu-{i:05d}", "bench": "mmlu", "subject": r["subject"], "question": r["question"], "choices": list(r["choices"]), "gold": int(r["answer"])})
    return out


def arc(config: str, name: str) -> list[dict]:
    out = []
    for r in rows(fetch("allenai/ai2_arc", f"{config}/test-00000-of-00001.parquet")):
        labels = list(r["choices"]["label"])
        out.append({"id": r["id"], "bench": name, "subject": "science", "question": r["question"], "choices": list(r["choices"]["text"]), "gold": labels.index(r["answerKey"])})
    return out


def truthfulqa() -> list[dict]:
    out = []
    for i, r in enumerate(rows(fetch("truthfulqa/truthful_qa", "multiple_choice/validation-00000-of-00001.parquet"))):
        choices, labels = list(r["mc1_targets"]["choices"]), list(r["mc1_targets"]["labels"])
        assert sum(labels) == 1, "MC1 has exactly one correct choice"
        order = list(range(len(choices)))
        random.Random(61 + i).shuffle(order)
        out.append({"id": f"truthfulqa-{i:04d}", "bench": "truthfulqa", "subject": "truthfulness", "question": r["question"],
                    "choices": [choices[j] for j in order], "gold": order.index(labels.index(1))})
    return out


def global_mmlu_lite(lang: str) -> list[dict]:
    out = []
    for r in rows(fetch("CohereLabs/Global-MMLU-Lite", f"{lang}/test-00000-of-00001.parquet")):
        out.append({"id": r["sample_id"], "bench": f"gmmlu_{lang}", "subject": r["subject"], "category": r.get("subject_category"), "question": r["question"],
                    "choices": [r["option_a"], r["option_b"], r["option_c"], r["option_d"]], "gold": LETTERS.index(r["answer"])})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {"mmlu": (mmlu, "cais/mmlu (all, test)", "MIT"), "arc_c": (lambda: arc("ARC-Challenge", "arc_c"), "allenai/ai2_arc (ARC-Challenge, test)", "CC BY-SA 4.0"),
            "arc_e": (lambda: arc("ARC-Easy", "arc_e"), "allenai/ai2_arc (ARC-Easy, test)", "CC BY-SA 4.0"),
            "truthfulqa": (truthfulqa, "truthfulqa/truthful_qa (multiple_choice, validation, MC1)", "Apache-2.0")}
    for lang in ("en", "hi", "es"):
        sets[f"gmmlu_{lang}"] = ((lambda l=lang: global_mmlu_lite(l)), f"CohereLabs/Global-MMLU-Lite ({lang}, test)", "Apache-2.0")
    manifest = {}
    for name, (build, source, licence) in sets.items():
        data = build()
        assert all(0 <= d["gold"] < len(d["choices"]) <= len(LETTERS) for d in data), name
        path = OUT / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in data) + "\n", encoding="utf-8", newline="\n")
        manifest[name] = {"source": source, "license": licence, "questions": len(data), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name, len(data), manifest[name]["file_sha256"][:12], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
