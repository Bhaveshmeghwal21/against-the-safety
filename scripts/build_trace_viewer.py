"""Build the static trace viewer (a Hugging Face static Space) from saved evaluation runs.

  python scripts/build_trace_viewer.py \
      --suite "Nimbus Home support suite (fictional, 24 tasks)" --system "Base=runs/eval/base_product" --system "Tuned=runs/eval/main_product" \
      --system "Control=runs/eval/control_product" --suite "HotpotQA test sample" --system ... (systems apply to the preceding --suite)

Each --system directory holds `lenient/transcripts.jsonl` and `strict/transcripts.jsonl` written by scripts/run_eval.py.
Writes demo/space/{index.html,data.js,README.md}. The page loads no external code or fonts and renders all text with textContent.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "space"


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append", required=True)
    ap.add_argument("--system", action="append", required=True, help="Name=directory; belongs to the most recent --suite")
    ap.add_argument("--limit", type=int, default=60, help="episodes shown per suite (the first N in the run order)")
    ap.add_argument("--footer", default="")
    ap.add_argument("--titles-from", default=None, help="docs.jsonl mapping chunk ids to titles (product suite); others resolve from the transcripts")
    args = ap.parse_args()

    # argparse keeps the two lists separate; walk argv in order to attach each --system to the most recent --suite
    import sys

    suites = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--suite":
            suites.append({"title": argv[i + 1], "systems": []})
            i += 2
        elif argv[i] == "--system":
            name, path = argv[i + 1].split("=", 1)
            suites[-1]["systems"].append((name, Path(path)))
            i += 2
        else:
            i += 1

    titles: dict[str, str] = {}
    out_suites = []
    for s in suites:
        systems, labels, order = [], {}, []
        for name, d in s["systems"]:
            runs = {}
            for mode in ("lenient", "strict"):
                rows = []
                for t in read(d / mode / "transcripts.jsonl"):
                    key = f"{t['qid']}|{t['variant']}"
                    t["key"] = key
                    for st in t["steps"]:
                        for h in st["retrieved"]:
                            titles[h["chunk_id"]] = h["title"]
                    for h in t["provided_evidence"]:
                        titles[h["chunk_id"]] = h["title"]
                    rows.append(t)
                    if key not in labels:
                        labels[key] = f"{t['qid']} [{t['variant']}] {t['question'][:90]}"
                        order.append(key)
                runs[mode] = rows
            systems.append({"name": name, "runs": runs})
        order = order[: args.limit]
        out_suites.append({"title": s["title"], "systems": systems, "labels": labels, "order": order})
    if args.titles_from:
        for l in Path(args.titles_from).read_text(encoding="utf-8").splitlines():
            d = json.loads(l)
            titles[d["chunk_id"]] = d["title"]

    OUT.mkdir(parents=True, exist_ok=True)
    data = {"suites": out_suites, "titles": titles, "footer": args.footer}
    (OUT / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8", newline="\n")
    (OUT / "index.html").write_text((ROOT / "demo" / "template.html").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    (OUT / "README.md").write_text(
        "---\ntitle: Against-the-safety saved episodes\nemoji: 🔎\ncolorFrom: gray\ncolorTo: blue\nsdk: static\npinned: false\n---\n\n"
        "Static viewer of recorded evaluation episodes (base vs fine-tuned Tiny Aya on retrieval-augmented question answering). "
        "No model runs here. Code and results: the project repository linked from the model card.\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}  suites={len(out_suites)}  episodes={sum(len(s['order']) for s in out_suites)}")


if __name__ == "__main__":
    main()
