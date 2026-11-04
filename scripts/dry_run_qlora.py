"""CPU dry run of the whole training path with a tiny stand-in model (no GPU, no gated files).

AgentTune EventLogs -> create_distill_trainer -> Project.distill -> QLoraFactory -> TRL SFTTrainer -> a few steps.
Checks that (1) rows reach the trainer, (2) the loss is applied to completion tokens only, (3) the adapter saves and
reloads. Tiny Aya's tokenizer is gated, so a small public model stands in; its tokenizer treats Tiny Aya's special
tokens as plain text, which is fine for checking the plumbing but says nothing about Tiny Aya's quality or memory.

Run (after build_episodes.py):  python scripts/dry_run_qlora.py
"""

import json
import shutil
from pathlib import Path

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.distill_trainer import create_distill_trainer

from ragharness import protocol, qlora, rows

ROOT = Path(__file__).resolve().parents[1]
STAND_IN = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def load_logs(n: int) -> list[EventLog]:
    from ragharness import episodes  # noqa: F401  (EventLog JSON -> object below)

    logs = []
    for line in (ROOT / "data" / "processed" / "episodes" / "eventlogs_train.jsonl").read_text(encoding="utf-8").splitlines()[:n]:
        d = json.loads(line)
        log = EventLog(tier=d["tier"], id=d["id"])
        for e in d["events"]:
            log.append(Event(EventKind(e["kind"]), e["payload"], scope=e["scope"]))
        logs.append(log)
    return logs


def main() -> None:
    system = (ROOT / "data" / "processed" / "training" / "main.jsonl").read_text(encoding="utf-8").splitlines()[0]
    system = json.loads(system)["prompt"][0]["content"]
    logs = load_logs(24)

    factory = qlora.QLoraFactory(
        system, quantize=False, dtype="float32", lora_targets=["q_proj", "v_proj"], max_length=4096, use_cpu=True,
        gradient_checkpointing=False, chat_template=rows.FIXTURE.read_text(encoding="utf-8") if rows.FIXTURE.exists() else rows.fetch_template().read_text(encoding="utf-8"),
    )
    out = ROOT / "runs" / "dry_run"
    shutil.rmtree(out, ignore_errors=True)
    trainer = create_distill_trainer(
        STAND_IN, trajectories=logs, trainer_factory=factory, output_dir=str(out),
        max_steps=3, per_device_train_batch_size=1, gradient_accumulation_steps=2, learning_rate=1e-3,
    )
    preview = trainer.sft_dataset()
    print("AgentTune episodes:", len(preview))
    result = trainer.train()
    print("train result keys:", sorted(result.metrics) if hasattr(result, "metrics") else type(result))
    assert result.training_loss > 1.0, f"loss {result.training_loss}: no supervised tokens reached the loss (truncation?)"

    # loss mask: only completion tokens carry labels
    sft = factory.sft_trainer
    ds = sft.train_dataset
    ex = ds[0]
    labels, ids = ex["labels"] if "labels" in ex else None, ex["input_ids"]
    comp = ex["completion_mask"]
    print("example tokens:", len(ids), " completion tokens:", sum(comp))
    if labels is None:
        batch = sft.data_collator([ex])
        labels = batch["labels"][0].tolist()
    supervised = sum(1 for x in labels if x != -100)
    print("supervised label tokens:", supervised)
    assert supervised == sum(comp), "loss must cover exactly the completion tokens"
    decoded = factory.tokenizer.decode([t for t, l in zip(ids, labels) if l != -100])
    print("supervised text:", decoded[:160].replace("\n", " "), "...", decoded[-60:].replace("\n", " "))
    squashed = decoded.replace(" ", "")  # the stand-in tokenizer splits the special markers into plain pieces
    assert squashed.endswith(rows.COMPLETION_END), "supervised text must end with <|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
    assert rows.PROMPT_END.replace(" ", "") not in squashed, "the prompt's generation marker must not be supervised"

    # adapter save and reload
    adapter = out / "adapter"
    sft.model.save_pretrained(str(adapter))
    assert (adapter / "adapter_config.json").exists()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(STAND_IN)
    reloaded = PeftModel.from_pretrained(base, str(adapter))
    print("adapter reloaded:", type(reloaded).__name__)
    print("DRY RUN OK")


if __name__ == "__main__":
    main()
