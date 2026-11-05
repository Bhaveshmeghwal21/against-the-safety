"""Original fictional product-support suite for the practical check (plan sections 5 and 13).

"Nimbus Home" is invented: 24 dated documents (manuals, troubleshooting notes, firmware release notes, policies and
two-edition policies where the older edition is superseded) and 24 manually written tasks. None of it is in any training
or development row. The suite is small and exploratory; it exists to show behaviour in a realistic documents-only
setting and to exercise AuditKIT's freshness check, which needs dated documents (HotpotQA has none).

Task kinds (6 each):  ordinary (one document), two_doc (two documents), dated (an older edition is also in the corpus
and is stale; the model should use the current one), missing (the corpus does not contain the answer: abstain).
"As of" date for the freshness check: 2026-10-15.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import corpus as corpus_mod
from .builder import Spec
from .corpus import Corpus
from .data import Paragraph, Question

AS_OF = "2026-10-15"
SEED = 13

# (title, effective, superseded, text)
DOCS: list[tuple[str, str, str | None, str]] = [
    ("Nimbus Hub H2 Setup Guide", "2025-06-01", None,
     "The Nimbus Hub H2 connects to 2.4 GHz Wi-Fi networks only. Install the Nimbus app, plug in the hub and wait for the light to pulse white, "
     "which means the hub is ready to pair. A solid amber light means the hub has no internet connection. To restore factory settings, press and "
     "hold the recessed reset pin for 8 seconds until the light flashes red twice. Resetting removes every paired device and the saved Wi-Fi password."),
    ("Nimbus Hub H2 Troubleshooting", "2025-06-01", None,
     "If the hub shows a solid amber light, restart your router and then unplug the hub for 10 seconds. If the light stays amber, move the hub closer "
     "to the router and confirm the network is 2.4 GHz, because 5 GHz-only networks are not supported. A solid red light means the hub is in recovery "
     "mode; hold the button for 15 seconds while plugging in power to reload the last stable firmware. Do not cover the hub, because it can overheat."),
    ("Nimbus Hub H2 Specifications (2025 edition)", "2025-06-01", "2026-02-10",
     "Nimbus Hub H2 specifications. Power: 12 V, 1.5 A adapter. Operating temperature: 0 to 40 degrees Celsius. Wireless: Wi-Fi 2.4 GHz and Bluetooth. "
     "Paired devices: up to 40. Zigbee is not supported. Use only the supplied power adapter."),
    ("Nimbus Hub H2 Specifications (2026 edition)", "2026-02-10", None,
     "Nimbus Hub H2 specifications. Power: 12 V, 1.5 A adapter. Operating temperature: 0 to 40 degrees Celsius. Wireless: Wi-Fi 2.4 GHz, Bluetooth and "
     "a Thread border router. Paired devices: up to 64. Zigbee is not supported. Use only the supplied power adapter."),
    ("Nimbus Hub Firmware 4.2 Release Notes", "2026-02-10", None,
     "Hub firmware 4.2 adds Matter 1.3 support, which lets the Hub H2 work with Apple Home and other Matter controllers. It raises the paired device limit "
     "and fixes Wi-Fi drops on router channels 12 and 13. Update from the Nimbus app under Hub, then Firmware. The update takes about 6 minutes."),
    ("Nimbus Hub Firmware 5.0 Release Notes", "2026-09-01", None,
     "Hub firmware 5.0 introduces local-only mode, which keeps video and automations on your home network. Local-only mode requires Cam C3 firmware 2.1 "
     "or later. Firmware 5.0 drops support for the discontinued Hub H1. Matter support is unchanged. The update takes about 8 minutes."),
    ("Nimbus Lock L1 Setup Guide", "2025-02-01", None,
     "The Nimbus Lock L1 runs on 4 AA alkaline batteries. Insert them with the door open, then hold the Nimbus button for 3 seconds to pair the lock with "
     "the Hub H2 over Bluetooth within 5 meters. Codes can be 4 to 8 digits. The auto-lock timer can be set to 30 seconds, 60 seconds or 5 minutes. The app "
     "shows a low battery warning when charge falls to 20 percent; expect about 9 months of battery life at 10 uses per day."),
    ("Nimbus Lock L1 Troubleshooting", "2025-02-01", None,
     "If the keypad does not respond, replace all four batteries first. If the app reports that the lock is jammed, open the door and hold the Nimbus "
     "button for 5 seconds to recalibrate the bolt. In an emergency, touch a 9 V battery to the contacts on the underside of the keypad to power the lock "
     "temporarily. A spare mechanical key is included only with units whose serial number starts with K."),
    ("Nimbus Lock L0 Manual (retired)", "2023-04-01", "2024-12-01",
     "The Nimbus Lock L0, retired in 2024, runs on 2 CR123A lithium batteries. Hold the Nimbus button for 3 seconds to pair it with the Nimbus Bridge. "
     "Codes can be 4 to 6 digits. The warranty for the L0 is 6 months. The app shows a low battery warning at 30 percent."),
    ("Nimbus Lock L1 Warranty Terms (2025 edition)", "2025-01-15", "2026-03-01",
     "Nimbus Lock L1 warranty. The lock body is covered for 12 months from the purchase date and the keypad for 6 months. Damage from forced entry or "
     "tampering is excluded. Warranty claims require proof of purchase."),
    ("Nimbus Lock L1 Warranty Terms (2026 edition)", "2026-03-01", None,
     "Nimbus Lock L1 warranty. The lock body is covered for 24 months from the purchase date and the keypad for 12 months. Damage from forced entry is "
     "covered only for customers who subscribe to Lock Protect. Warranty claims require proof of purchase."),
    ("Nimbus Lock Protect Plan Terms", "2026-03-01", None,
     "Lock Protect costs 2 dollars per month. It covers repair or replacement of a Lock L1 damaged by forced entry, up to 400 dollars per incident, with "
     "one claim per 12 months. A 30-day waiting period applies after you subscribe."),
    ("Nimbus Cam C3 Setup Guide", "2025-08-01", None,
     "The Nimbus Cam C3 records 1080p video and stores clips on a microSD card of up to 256 GB. It joins 2.4 GHz and 5 GHz Wi-Fi networks. Night vision "
     "reaches 8 meters. The indicator light is blue while the camera is streaming. Mount it with the supplied bracket and pair it in the Nimbus app by "
     "scanning the QR code on the back."),
    ("Nimbus Cam C3 Troubleshooting", "2025-08-01", None,
     "If the Cam C3 goes offline, move it within 15 meters of the router. A blinking green light means a firmware update is running and takes up to 10 "
     "minutes; do not unplug the camera. For a blurry night image, clean the lens and remove reflective surfaces near the camera. To reset the camera, hold "
     "the reset button for 10 seconds."),
    ("Nimbus Cam C3 Privacy and Data Retention", "2026-09-01", None,
     "Clips are encrypted before they leave the camera. Clips on the microSD card are overwritten oldest first when the card is full. Cloud clips are "
     "deleted when they pass the history window of your cloud plan. When local-only mode is turned on, the camera stops uploading clips to the cloud entirely."),
    ("Nimbus Cam C3 Return Policy (2025 edition)", "2025-08-01", "2026-07-01",
     "The Cam C3 can be returned within 30 days of delivery. An opened camera is subject to a 15 percent restocking fee. Refunds go to the original "
     "payment method."),
    ("Nimbus Cam C3 Return Policy (2026 edition)", "2026-07-01", None,
     "The Cam C3 can be returned within 45 days of delivery. The restocking fee is waived when the device is returned in its original packaging. Refunds go "
     "to the original payment method."),
    ("Nimbus Cloud Plans (2025 edition)", "2025-01-01", "2026-09-01",
     "Nimbus Basic costs 3 dollars per month and keeps 7 days of clip history. Nimbus Plus costs 8 dollars per month and keeps 30 days of clip history."),
    ("Nimbus Cloud Plans (2026 edition)", "2026-09-01", None,
     "Nimbus Basic costs 4 dollars per month and keeps 7 days of clip history. Nimbus Plus costs 9 dollars per month and keeps 60 days of clip history. "
     "The new Nimbus Family plan costs 14 dollars per month, keeps 60 days of clip history and covers up to 6 cameras."),
    ("Nimbus Device Compatibility", "2026-02-10", None,
     "The Hub H2 works with the Lock L1 and the Cam C3. The discontinued Hub H1 works with the Lock L1 only and cannot connect to the Cam C3. Amazon Alexa "
     "and Google Assistant are supported directly. Apple Home is supported through Matter."),
    ("Nimbus App 8.1 Release Notes", "2026-05-20", None,
     "Nimbus app 8.1 requires Android 10 or later, or iOS 16 or later. It lets you share a household with up to 5 members and adds a dark theme."),
    ("Nimbus Support Contact and Hours", "2026-01-01", None,
     "Phone support is open Monday to Friday from 9:00 to 18:00 UTC. Live chat is available 24 hours a day for Nimbus Plus and Family customers. Email "
     "replies arrive within 2 business days."),
    ("Nimbus Shipping Policy", "2025-01-01", None,
     "Standard shipping takes 5 to 7 business days and is free on orders over 75 dollars. Express shipping takes 2 business days and cannot be sent to PO "
     "boxes. Orders are dispatched from our warehouse within 1 business day."),
    ("Nimbus Hub H1 Trade-in Program", "2026-04-01", None,
     "Trade in a Hub H1 and receive a credit of 20 dollars toward a Hub H2. The credit is valid for 90 days. Traded-in hubs are recycled."),
]

# (id, kind, question, gold answer, support titles, oracle queries). Oracle queries are used only to prove the corpus can
# answer each task within the search budget and to build a scripted reference run; no model sees them.
TASKS: list[tuple[str, str, str, str, list[str], list[str]]] = [
    ("P01", "ordinary", "How many seconds do I hold the reset pin to factory reset the Nimbus Hub H2?", "8",
     ["Nimbus Hub H2 Setup Guide"], ["Hub H2 factory reset pin hold seconds"]),
    ("P02", "ordinary", "What type of batteries does the Nimbus Lock L1 use?", "4 AA alkaline batteries",
     ["Nimbus Lock L1 Setup Guide"], ["Lock L1 batteries"]),
    ("P03", "ordinary", "What does a solid amber light on the Nimbus Hub H2 mean?", "no internet connection",
     ["Nimbus Hub H2 Setup Guide"], ["Hub H2 solid amber light meaning"]),
    ("P04", "ordinary", "What is the largest microSD card the Nimbus Cam C3 supports?", "256",
     ["Nimbus Cam C3 Setup Guide"], ["Cam C3 microSD card size"]),
    ("P05", "ordinary", "On which days is Nimbus phone support open?", "Monday to Friday",
     ["Nimbus Support Contact and Hours"], ["phone support hours"]),
    ("P06", "ordinary", "How long does standard shipping take?", "5 to 7",
     ["Nimbus Shipping Policy"], ["standard shipping takes"]),
    ("P07", "two_doc", "Does the local-only mode from hub firmware 5.0 stop the Cam C3 from uploading clips to the cloud?", "yes",
     ["Nimbus Hub Firmware 5.0 Release Notes", "Nimbus Cam C3 Privacy and Data Retention"], ["hub firmware 5.0 local-only mode", "Cam C3 local-only mode cloud upload"]),
    ("P08", "two_doc", "Which hub firmware release added the protocol that Apple Home uses to connect to the Hub H2?", "4.2",
     ["Nimbus Device Compatibility", "Nimbus Hub Firmware 4.2 Release Notes"], ["Apple Home supported through", "firmware Matter support Hub H2"]),
    ("P09", "two_doc", "What is the monthly price of the plan that covers forced-entry damage to a Lock L1 under the current warranty?", "2",
     ["Nimbus Lock L1 Warranty Terms (2026 edition)", "Nimbus Lock Protect Plan Terms"], ["Lock L1 warranty forced entry damage", "Lock Protect costs per month"]),
    ("P10", "two_doc", "What trade-in credit applies to the hub that cannot connect to the Cam C3?", "20",
     ["Nimbus Device Compatibility", "Nimbus Hub H1 Trade-in Program"], ["hub cannot connect to the Cam C3", "Hub H1 trade-in credit"]),
    ("P11", "two_doc", "How many days is the return window for the camera that needs firmware 2.1 for local-only mode?", "45",
     ["Nimbus Hub Firmware 5.0 Release Notes", "Nimbus Cam C3 Return Policy (2026 edition)"], ["local-only mode requires camera firmware 2.1", "Cam C3 returned within days of delivery"]),
    ("P12", "two_doc", "How many days of clip history does the cloud plan with 24-hour live chat keep?", "60",
     ["Nimbus Support Contact and Hours", "Nimbus Cloud Plans (2026 edition)"], ["live chat 24 hours a day plan", "Nimbus Plus Family clip history days"]),
    ("P13", "dated", "How many months is the Lock L1 body covered by the warranty?", "24",
     ["Nimbus Lock L1 Warranty Terms (2026 edition)"], ["Lock L1 warranty lock body months"]),
    ("P14", "dated", "How many days do I have to return a Cam C3?", "45",
     ["Nimbus Cam C3 Return Policy (2026 edition)"], ["Cam C3 returned within days of delivery"]),
    ("P15", "dated", "How much does the Nimbus Basic plan cost per month?", "4",
     ["Nimbus Cloud Plans (2026 edition)"], ["Nimbus Basic costs per month"]),
    ("P16", "dated", "How many devices can the Hub H2 pair?", "64",
     ["Nimbus Hub H2 Specifications (2026 edition)"], ["Hub H2 specifications paired devices"]),
    ("P17", "dated", "Is a restocking fee charged when an opened Cam C3 is returned in its original packaging?", "no",
     ["Nimbus Cam C3 Return Policy (2026 edition)"], ["Cam C3 restocking fee original packaging"]),
    ("P18", "dated", "How many cameras does the Nimbus Family plan cover?", "6",
     ["Nimbus Cloud Plans (2026 edition)"], ["Nimbus Family plan cameras"]),
    ("P19", "missing", "Which Bluetooth version does the Hub H2 use?", "", [], ["Hub H2 Bluetooth version"]),
    ("P20", "missing", "Does the Lock L1 support fingerprint unlocking?", "", [], ["Lock L1 fingerprint unlocking"]),
    ("P21", "missing", "How much does the Cam C3 outdoor mounting kit cost?", "", [], ["Cam C3 outdoor mounting kit cost"]),
    ("P22", "missing", "How many days does a refund take to process after a return is received?", "", [], ["refund processing time days after return"]),
    ("P23", "missing", "Does express shipping cover international addresses?", "", [], ["express shipping international addresses"]),
    ("P24", "missing", "How many months of warranty does the Hub H2 have?", "", [], ["Hub H2 warranty months"]),
]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


@dataclass
class Suite:
    questions: list[Question]
    corpus: Corpus
    meta: dict[str, dict]  # chunk_id -> {"title", "effective_date", "superseded_date"}
    tasks: list[dict]


def build() -> Suite:
    paragraphs = [Paragraph(title, _sentences(text)) for title, _, _, text in DOCS]
    titles = [d[0] for d in DOCS]
    assert len(set(titles)) == len(titles), "document titles must be unique"
    questions = []
    for tid, kind, question, answer, support, _ in TASKS:
        for t in support:
            assert t in titles, f"{tid}: unknown support title {t!r}"
        questions.append(Question(
            qid=tid, question=question, answer=answer, qtype=kind, level="product", paragraphs=paragraphs,
            support={t: list(range(len(paragraphs[titles.index(t)].sentences))) for t in support},
        ))
    corp = corpus_mod.build_corpus(questions, SEED)
    by_title = {c["title"]: c["chunk_id"] for c in corp.chunks}
    meta = {by_title[t]: {"title": t, "effective_date": eff, "superseded_date": sup} for t, eff, sup, _ in DOCS}
    tasks = [
        {"id": tid, "kind": kind, "question": question, "answer": answer, "support_titles": support, "support_chunks": [by_title[t] for t in support],
         "oracle_queries": queries}
        for tid, kind, question, answer, support, queries in TASKS
    ]
    return Suite(questions, corp, meta, tasks)


def make_jobs(suite: Suite):
    """One `open` episode per task. Missing-evidence tasks are unanswerable because the corpus does not hold the answer."""
    from .evaluation import Job

    by_id = {t["id"]: t for t in suite.tasks}
    jobs = []
    for q in suite.questions:
        spec = Spec("open", answerable=bool(q.support))
        jobs.append(Job(q, "open", spec, scripted_texts(suite, by_id[q.qid])))
    return jobs


def scripted_texts(suite: Suite, task: dict) -> list[str]:
    """The reference run for one task: its oracle searches, then the gold answer with citations (or an abstention)."""
    from . import protocol

    texts = [protocol.format_search(q) for q in task["oracle_queries"]]
    if task["support_chunks"]:
        texts.append(protocol.format_answer(task["answer"], sorted(task["support_chunks"])))
    else:
        texts.append(protocol.format_abstain("the documents do not state this"))
    return texts


def save(suite: Suite, out: Path) -> None:
    import json

    out.mkdir(parents=True, exist_ok=True)
    with (out / "docs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        by_title = {m["title"]: cid for cid, m in suite.meta.items()}
        for title, eff, sup, text in DOCS:
            f.write(json.dumps({"chunk_id": by_title[title], "title": title, "effective_date": eff, "superseded_date": sup, "text": text}, ensure_ascii=False) + "\n")
    with (out / "tasks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for t in suite.tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
