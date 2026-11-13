"""Write the unexecuted, credential-free quickstart notebook."""
import json
from pathlib import Path


def md(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


def code(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source.splitlines(keepends=True)}


cells = [
    md('# Quickstart: ask over six passages\n\nRun from the repository root on a CUDA GPU. On Kaggle, attach the project bundle and the Tiny Aya Global model first. Outside Kaggle, accept the base-model terms on Hugging Face and authenticate using your own account. This notebook has not yet been GPU-validated.'),
    code("%pip install --no-deps -r requirements-gpu.lock.txt\n%pip install --no-deps -e ."),
    code("""import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from scripts.ask import load_model_for_system
from ragharness import system

docs = [
    {'title': 'Returns', 'text': 'Nimbus Home accepts returns within 30 days after delivery.'},
    {'title': 'Warranty', 'text': 'Nimbus Home devices have a two-year warranty.'},
    {'title': 'Shipping', 'text': 'Orders ship within three business days.'},
    {'title': 'Support', 'text': 'Support is available Monday through Friday.'},
    {'title': 'Packaging', 'text': 'Keep the original packaging when returning a device.'},
    {'title': 'Delivery', 'text': 'A tracking link is sent when an order ships.'},
]
model, tokenizer = load_model_for_system('BMEscanor/against-the-safety-tiny-aya-rag-main')
# Fail loudly if this Python build loses FTS5 after the model libraries load.
with sqlite3.connect(':memory:') as connection:
    connection.execute('CREATE VIRTUAL TABLE probe USING fts5(text)')
backend, _ = system.index_documents(docs, Path('runs/quickstart/docs.sqlite'))
rag = system.RagSystem(model, tokenizer, backend)
"""),
    md('The next three cells ask an answerable question, one missing from the documents, and a two-part question. The actual decision depends on the model; a refusal on the two-part question is a valid result to inspect.'),
    *[code(f"decision = rag.ask({question!r})\nprint(json.dumps(asdict(decision), ensure_ascii=False, indent=2))") for question in (
        'What is the return window for Nimbus Home devices?',
        'What is the price of a Nimbus Home device?',
        'What are the return window and the warranty length for Nimbus Home devices?',
    )],
]
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                        'language_info': {'name': 'python', 'version': '3.12'}},
            'nbformat': 4, 'nbformat_minor': 5}
target = Path(__file__).resolve().parents[1] / 'notebooks/quickstart.ipynb'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
