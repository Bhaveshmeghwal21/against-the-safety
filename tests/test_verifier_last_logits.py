"""The retry's memory-saving forward must preserve the verifier's final logits."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import Cohere2Config, Cohere2ForCausalLM


def test_cohere2_last_position_matches_full_logits_with_left_padding():
    cfg = Cohere2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=32,
                        eos_token_id=1, bos_token_id=2, pad_token_id=0)
    torch.manual_seed(47)
    model = get_peft_model(Cohere2ForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8,
                           target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    ids = torch.tensor([[2, 3, 4, 5], [0, 0, 6, 7]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    with torch.no_grad():
        full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :]
        last = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    assert torch.allclose(full, last, atol=1e-6, rtol=1e-5)
