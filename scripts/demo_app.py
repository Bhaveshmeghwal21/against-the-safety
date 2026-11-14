"""A web demo of the submitted system: paste documents, ask a question, see the decision, the citations and the search trace.

  python scripts/demo_app.py --adapter BMEscanor/against-the-safety-tiny-aya-rag-main            # local, needs a CUDA GPU
  python scripts/demo_app.py --adapter <adapter folder> --share --minutes 50                      # public link for a while (what the Kaggle `demo` stage runs)
  python scripts/demo_app.py --adapter <adapter folder> --smoke --out runs/demo_smoke.json         # no web page: answers the example questions and saves them

It needs a CUDA GPU and access to the gated base model (set TINY_AYA_PATH for a local copy). `pip install gradio` for the web page.
The first answer takes longer while the model warms up. The system is the one scored in the README: main adapter, fail-closed rule, self-check.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def example_documents() -> str:
    rows = [json.loads(l) for l in (EXAMPLES / "nimbus_home.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return "\n".join(f"{r['title']}: {r['text']}" for r in rows)


def example_questions() -> list[dict]:
    return json.loads((EXAMPLES / "nimbus_questions.json").read_text(encoding="utf-8"))


def build_ui(session):
    import gradio as gr

    from ragharness import demo

    def run(docs_text: str, question: str):
        try:
            result = session.ask(docs_text, question)
        except demo.DocumentError as error:
            return f"**{error}**", ""
        return demo.render_verdict(result), demo.render_trace(result)

    questions = example_questions()
    with gr.Blocks(title="Against-the-safety: answers from your documents") as ui:
        gr.Markdown(
            "# Answers from your documents, or a refusal\n"
            "Tiny Aya Global with a LoRA adapter searches the passages (up to three searches), then answers with citations or says the documents don't hold the answer. "
            "A fail-closed rule and a self-check withhold answers the passages do not support. Research demo: English, short factual questions."
        )
        with gr.Row():
            with gr.Column(scale=5):
                docs = gr.Textbox(label="Documents: one passage per line, as `Title: text`", lines=18, value=example_documents())
            with gr.Column(scale=4):
                question = gr.Textbox(label="Question", lines=2, placeholder="Ask something the documents answer, or something they don't")
                gr.Examples(examples=[[q["question"]] for q in questions], inputs=[question], label="Example questions about the fictional Nimbus Home documents")
                ask = gr.Button("Ask", variant="primary")
                verdict = gr.Markdown()
                with gr.Accordion("Search trace", open=False):
                    trace = gr.Markdown()
        ask.click(run, [docs, question], [verdict, trace])
        question.submit(run, [docs, question], [verdict, trace])
    return ui


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Hugging Face repository id or a local adapter folder (the main adapter)")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--minutes", type=float, default=0, help="with --share: how long to keep the page up before exiting (0 = until interrupted)")
    ap.add_argument("--smoke", action="store_true", help="no web page: answer the example questions and write --out")
    ap.add_argument("--out", default="runs/demo_smoke.json")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from ask import load_model_for_system

    from ragharness import demo

    model, tokenizer = load_model_for_system(args.adapter)
    session = demo.DemoSession(model, tokenizer)

    if args.smoke:
        docs = example_documents()
        records = []
        for q in example_questions():
            start = time.time()
            result = session.ask(docs, q["question"])
            result.update(kind=q["kind"], expected=q["expected"], seconds=round(time.time() - start, 1))
            records.append(result)
            print(f"[{q['kind']:>8}] {result['decision']:>7} | {q['question']} -> {result['answer'] or result['why']}", flush=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print("wrote", out)
        return

    ui = build_ui(session)
    _, local_url, share_url = ui.launch(share=args.share, prevent_thread_lock=bool(args.minutes))
    if args.minutes:
        link, end = share_url or local_url, time.time() + args.minutes * 60
        while time.time() < end:  # repeated so the link stays in the tail of a notebook log
            print(f"DEMO LINK: {link}  ({max(0, int((end - time.time()) / 60))} min left)", flush=True)
            time.sleep(60)
        ui.close()


if __name__ == "__main__":
    main()
