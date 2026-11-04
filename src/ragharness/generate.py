"""Hugging Face text generator for Tiny Aya (base or base + LoRA adapter), greedy and single sequence.

Stops on `<|END_RESPONSE|>` or `<|END_OF_TURN_TOKEN|>` and returns the response text between the markers. The same
code is used for the smoke test, the baselines and the evaluation, so every system sees identical decoding.
"""

from __future__ import annotations

from . import rows

STOP_MARKERS = ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>")


def clean(text: str) -> str:
    for m in STOP_MARKERS:
        text = text.split(m)[0]
    return text.replace("<|START_RESPONSE|>", "").strip()


class HFGenerator:
    def __init__(self, model, tokenizer, max_new_tokens: int = 160, max_input_tokens: int = 4096, chat_template: str | None = None):
        import torch

        self.torch = torch
        self.model, self.tokenizer = model, tokenizer
        if chat_template:
            self.tokenizer.chat_template = chat_template
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens
        stop_ids = []
        for m in STOP_MARKERS:
            ids = tokenizer.convert_tokens_to_ids(m)
            if isinstance(ids, int) and ids != tokenizer.unk_token_id:
                stop_ids.append(ids)
        self.stop_ids = stop_ids or [tokenizer.eos_token_id]
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def __call__(self, messages: list[dict]) -> str:
        torch = self.torch
        enc = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"][:, -self.max_input_tokens :].to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False,
                max_new_tokens=self.max_new_tokens, eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id or self.stop_ids[0],
            )
        new = out[0, input_ids.shape[1] :]
        self.n_calls += 1
        self.input_tokens += int(input_ids.shape[1])
        self.output_tokens += int(new.shape[0])
        return clean(self.tokenizer.decode(new, skip_special_tokens=False))


def load_model(model_id: str, revision: str | None = None, adapter: str | None = None, quantize: bool = True, dtype: str = "float16"):
    """Base model in 4-bit NF4 (or full precision for CPU/stand-in), with an optional PEFT adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tdtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(adapter or model_id, revision=None if adapter else revision)
    kwargs = {"revision": revision, "dtype": tdtype}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=tdtype, bnb_4bit_use_double_quant=True
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok
