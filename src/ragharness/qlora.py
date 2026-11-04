"""QLoRA `trainer_factory` for AgentTune's `create_distill_trainer` (plan section 6.1).

AgentTune's default SFT factory loads bf16 with `device_map="auto"` and trains on every `messages` token. This one
keeps the same `trainer_factory(model=, train_dataset=, **kwargs)` contract and instead:
- loads the student in 4-bit NF4 with FP16 compute (T4 has no bf16), one GPU, no `device_map="auto"`;
- turns the rows AgentTune produced (EventLog -> `as_dataset_rows`) into Tiny Aya prompt/completion rows
  (`rows.decisions`, checked against the pinned template) and trains on completions only;
- attaches a LoRA adapter to all linear layers and returns a TRL `SFTTrainer`.

`quantize=False` and `dtype="float32"` give a CPU dry run with a tiny stand-in model (see scripts/dry_run_qlora.py).
"""

from __future__ import annotations

from typing import Any

from . import rows as rowmod


def to_hf_rows(
    train_dataset,
    system: str,
    metas: list[dict] | None = None,
    keep: set | None = None,
    max_tokens_est: int | None = None,
) -> list[dict]:
    """AgentTune SFT rows -> [{prompt: [...], completion: [...]}] (template-checked).

    `metas[i]` describes `train_dataset[i]`; with `keep` (a set of (parent_qid, variant, step) keys) only the selected
    decisions of each episode are returned, so a row-level mixture can ride on AgentTune's episode-level input.
    """
    out: list[dict] = []
    for i, sft_row in enumerate(train_dataset):
        for j, d in enumerate(rowmod.decisions(sft_row, system)):
            rowmod.check_row(d)
            if keep is not None:
                if metas is None:
                    raise ValueError("`keep` needs `metas` aligned with the trajectories")
                if (metas[i]["parent_qid"], metas[i]["variant"], j) not in keep:
                    continue
            if max_tokens_est is not None and rowmod.est_tokens(d)[0] > max_tokens_est:
                continue
            out.append({"prompt": d["prompt"], "completion": d["completion"]})
    return out


class QLoraFactory:
    def __init__(
        self,
        system: str,
        *,
        quantize: bool = True,
        dtype: str = "float16",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: str | list[str] = "all-linear",
        revision: str | None = None,
        max_length: int = 2560,
        max_tokens_est: int | None = None,
        gradient_checkpointing: bool = True,
        use_cpu: bool = False,
        chat_template: str | None = None,
        metas: list[dict] | None = None,
        keep: set | None = None,
        eval_rows: list[dict] | None = None,
    ):
        self.system = system
        self.metas, self.keep, self.eval_rows = metas, keep, eval_rows
        self.n_train_rows = 0
        self.chat_template = chat_template  # only for stand-in models in dry runs; the real tokenizer ships the template
        self.quantize = quantize
        self.dtype = dtype
        self.lora_r, self.lora_alpha, self.lora_dropout, self.lora_targets = lora_r, lora_alpha, lora_dropout, lora_targets
        self.revision = revision
        self.max_length = max_length
        self.max_tokens_est = max_tokens_est
        self.gradient_checkpointing = gradient_checkpointing
        self.use_cpu = use_cpu
        self.model = self.tokenizer = self.sft_trainer = None

    def __call__(self, *, model: str, train_dataset, **kwargs: Any):
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer

        dtype = getattr(torch, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=self.revision)
        if self.chat_template:
            self.tokenizer.chat_template = self.chat_template
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"revision": self.revision, "dtype": dtype}
        if self.quantize:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True
            )
            load_kwargs["device_map"] = {"": 0}  # one GPU per run; DDP and device_map="auto" are not used
        self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        if self.quantize:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=self.gradient_checkpointing)

        rows_ = to_hf_rows(train_dataset, self.system, self.metas, self.keep, self.max_tokens_est)
        self.n_train_rows = len(rows_)
        data = Dataset.from_list(rows_)
        eval_data = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in self.eval_rows]) if self.eval_rows else None
        cfg_kwargs = {
            "max_length": self.max_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "fp16": dtype == torch.float16 and not self.use_cpu,
            "bf16": False,
            "completion_only_loss": True,
            "packing": False,
            "report_to": "none",
            "use_cpu": self.use_cpu,
            "save_strategy": "no",
            "logging_steps": 1,
        }
        cfg_kwargs.update(kwargs)  # AgentTune forwards output_dir and any SFT knobs
        self.sft_trainer = SFTTrainer(
            model=self.model,
            args=SFTConfig(**cfg_kwargs),
            train_dataset=data,
            eval_dataset=eval_data,
            processing_class=self.tokenizer,
            peft_config=LoraConfig(
                r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
                target_modules=self.lora_targets, task_type="CAUSAL_LM", bias="none",
            ),
        )
        return self.sft_trainer
