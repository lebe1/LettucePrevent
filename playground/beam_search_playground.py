import torch
import json
import time
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from word2number import w2n
from tqdm import tqdm
import torch.nn.functional as F


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

seed = 42
torch.manual_seed(seed)

system_prompt = (
    ""
)

raw_prompt = "Once upon a time ..."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": raw_prompt}
]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_data = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=False)
input_data = {k: v.to(model.device) for k, v in input_data.items()}

gen_config = GenerationConfig(
    max_new_tokens=10,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_return_sequences=2,
    num_beams=2,
    return_dict_in_generate=True,
    output_scores=True,
)

output = model.generate(
    **input_data,
    generation_config = gen_config,
)


logits = torch.stack(output.scores, dim=1)  # Shape: (num_beams, steps, vocab_size)
log_probs = F.log_softmax(logits, dim=-1)

# Remove the prompt tokens from output.sequences
prompt_len = input_data["input_ids"].shape[1]
generated_tokens = output.sequences[:, prompt_len:]  # Shape: (num_beams, new_tokens)

for beam_idx, beam_tokens in enumerate(generated_tokens):
    print(f"\nBeam {beam_idx + 1}:")
    for step, token_id in enumerate(beam_tokens):
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        prob = log_probs[beam_idx, step, token_id].item()
        print(f"  Token: {token_str:<15} ID: {int(token_id):<5}  logp: {prob:.4f}")
        cumulative_logp = sum(log_probs[beam_idx, step, token_id].item() for step, token_id in enumerate(beam_tokens))
        print(f"  → Cumulative logp: {cumulative_logp:.4f}")
        print(f"  → HuggingFace score: {output.sequences_scores[beam_idx].item():.4f}")



for i, seq in enumerate(output.sequences):
    decoded = tokenizer.decode(seq, skip_special_tokens=True)
    print(f"Beam {i+1}: {decoded}")

# Important: Only working for summary tasks by looking for the last occurrence of "output:" 
if "[/INST]" in decoded:
    answer_only = decoded.split("[/INST]", 1)[-1].strip()
else:
    # Fallback: if for some reason "output:" isn't found
    answer_only = decoded.strip()

print(answer_only)