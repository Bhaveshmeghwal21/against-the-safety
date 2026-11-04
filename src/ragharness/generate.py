"""Hugging Face text generator for Tiny Aya (base or base + LoRA adapter), greedy and single sequence.

Stops on `<|END_RESPONSE|>` or `<|END_OF_TURN_TOKEN|>` and returns the response text between the markers. The same
code is used for the smoke test, the baselines and the evaluation, so every system sees identical decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import rows

STOP_MARKERS = ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>")


@dataclass
class GenOut:
    text: str
    input_tokens: int
    output_tokens: int


def clean(text: str) -> str:
    for m in STOP_MARKERS:
        text = text.split(m)[0]
    return text.replace("<|START_RESPONSE|>", "").strip()


class HFGenerator:
    def __init__(self, model, tokenizer, max_new_tokens: int = 160, max_input_tokens: int = 4096, chat_template: str | None = None,
                 batch_size: int = 12, max_batch_tokens: int = 32768):
        import torch

        self.torch = torch
        self.model, self.tokenizer = model, tokenizer
        if chat_template:
            self.tokenizer.chat_template = chat_template
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens
        self.batch_size, self.max_batch_tokens = batch_size, max_batch_tokens
        stop_ids = []
        for m in STOP_MARKERS:
            ids = tokenizer.convert_tokens_to_ids(m)
            if isinstance(ids, int) and ids != tokenizer.unk_token_id:
                stop_ids.append(ids)
        self.stop_ids = stop_ids or [tokenizer.eos_token_id]
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.stop_ids[0]
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.truncated_inputs = 0  # prompts longer than max_input_tokens (the oldest tokens are dropped); should stay 0

    def _encode(self, messages: list[dict]) -> list[int]:
        enc = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
        ids = enc["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):  # a batch of one
            ids = ids[0]
        ids = list(ids)
        if len(ids) > self.max_input_tokens:
            self.truncated_inputs += 1
            ids = ids[-self.max_input_tokens :]
        return ids

    def _run(self, id_lists: list[list[int]]) -> list[GenOut]:
        torch = self.torch
        width = max(len(x) for x in id_lists)
        input_ids = torch.full((len(id_lists), width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(id_lists), width), dtype=torch.long)
        for i, ids in enumerate(id_lists):  # left padding: the new tokens of every row start at the same column
            input_ids[i, width - len(ids) :] = torch.tensor(ids)
            attention[i, width - len(ids) :] = 1
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids.to(self.model.device), attention_mask=attention.to(self.model.device), do_sample=False,
                max_new_tokens=self.max_new_tokens, eos_token_id=self.stop_ids, pad_token_id=self.pad_id,
            )
        results = []
        for i, ids in enumerate(id_lists):
            new = out[i, width:].tolist()
            n = next((k + 1 for k, t in enumerate(new) if t in self.stop_ids), len(new))  # tokens up to and including the stop marker
            text = clean(self.tokenizer.decode(new[:n], skip_special_tokens=False))
            results.append(GenOut(text, len(ids), n))
            self.n_calls += 1
            self.input_tokens += len(ids)
            self.output_tokens += n
        return results

    def __call__(self, messages: list[dict]) -> str:
        return self._run([self._encode(messages)])[0].text

    def generate_batch(self, batch: list[list[dict]]) -> list[GenOut]:
        """Greedy decoding of many prompts at once, left-padded, longest prompts grouped together to limit padding.

        Returns one `GenOut` per prompt in input order. Chunks are capped by `batch_size` rows and `max_batch_tokens`
        padded prompt tokens so a long prompt cannot exhaust GPU memory.
        """
        encoded = [self._encode(m) for m in batch]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        results: list[GenOut | None] = [None] * len(encoded)
        chunk: list[int] = []
        for i in order:
            widest = len(encoded[i])  # sorted ascending, so the newest row is the widest
            if chunk and (len(chunk) >= self.batch_size or (len(chunk) + 1) * widest > self.max_batch_tokens):
                for j, r in zip(chunk, self._run([encoded[k] for k in chunk])):
                    results[j] = r
                chunk = []
            chunk.append(i)
        if chunk:
            for j, r in zip(chunk, self._run([encoded[k] for k in chunk])):
                results[j] = r
        return results  # type: ignore[return-value]


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
