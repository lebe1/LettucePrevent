import torch
import json
import time
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from word2number import w2n
from tqdm import tqdm

BIAS_SCORE = 20.0

# -------------------------- Setup ------------------

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

# -------------------------- Prompt Formatting ------------------

system_prompt = (
    "You always respond very precise and clear. You never exceed the maximum number of words that is asked for."
    "Always end your answer with a complete sentence and a period! Do not hallucinate any numbers or other phrases!" 
    "Only stick to the information provided from the input!"
)

# -------------------------- Extract Number Logic ------------------

def extract_numbers_from_summary(text, return_logits):
    pattern = r'-?\d+(?:\.\d+)?'
    extracted_numbers = set(re.findall(pattern, text))

    number_word_pattern = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                                     r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                                     r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                                     r'eighty|ninety|hundred|thousand|million|billion|and|[-])+\b',
                                     re.IGNORECASE)
    matches = number_word_pattern.finditer(text)
    for match in matches:
        phrase = match.group().replace("-", " ").lower()
        try:
            number = str(w2n.word_to_num(phrase))
            extracted_numbers.add(number)
        except ValueError:
            continue

    if return_logits:
        sequence_bias = {}
        for num_str in extracted_numbers:
            token_ids = tokenizer.encode(num_str, add_special_tokens=False)
            if token_ids:
                sequence_bias[tuple(token_ids)] = BIAS_SCORE  
    
        return sequence_bias
    else:
        return list(extracted_numbers)
# -------------------------- Load Prompts ------------------

with open("./data/summary_prompt_counts.json", "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

results = []
num_generations = 0
start_dt = datetime.now()
start_time = time.time()

# -------------------------- Main Loop ------------------

def serialize_sequence_bias(sequence_bias):
    """Convert sequence_bias with tuple keys to list-of-dicts format."""
    return [
        {"token_ids": list(token_ids), "bias": bias}
        for token_ids, bias in sequence_bias.items()
    ]

for item in tqdm(prompt_data):
    start_dt_prompt = datetime.now()
    start_time_prompt = time.time()
    raw_prompt = item["prompt"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_data = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=False)
    input_data = {k: v.to(model.device) for k, v in input_data.items()}

    #sequence_bias = extract_numbers_from_summary(raw_prompt, True)

    gen_config = GenerationConfig(
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        min_length=150,
        num_beams=4,
    )

    output = model.generate(
        **input_data,
        generation_config = gen_config,
        # sequence_bias=sequence_bias
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Important: Only working for summary tasks by looking for the last occurrence of "output:" 
    if "[/INST]" in decoded:
        answer_only = decoded.split("[/INST]", 1)[-1].strip()
    else:
        # Fallback: if for some reason "output:" isn't found
        answer_only = decoded.strip()
    
    allowed_numbers = extract_numbers_from_summary(raw_prompt, False)

    end_dt_prompt = datetime.now()
    duration_prompt = round(time.time() - start_time_prompt, 2)

    results.append({
        "prompt": raw_prompt,
        "answer": answer_only,
        "allowed_numbers": allowed_numbers,
        #"sequence_bias": serialize_sequence_bias(sequence_bias),
        "task_type": "Summary",
        "dataset": "ragtruth",
        "language": "en",
        "start_time": start_dt_prompt.isoformat(),
        "end_time": end_dt_prompt.isoformat(),
        "duration_seconds": duration_prompt
    })

    num_generations += 1

# -------------------------- Metadata ------------------

end_dt = datetime.now()
duration = round(time.time() - start_time, 2)

results.append({
    "_meta": {
        "model": model_name,
        "system_prompt": system_prompt,
        "bias_score": BIAS_SCORE,
        "num_prompts": len(prompt_data),
        "total_generations": num_generations,
        "seed": seed,
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "duration_seconds": duration,
        "generation_config": gen_config.to_dict(),
    }
})

# -------------------------- Save ------------------

timestamp = start_dt.strftime("%Y%m%d_%H%M%S")
output_file = f"./data/summary_experiments_run_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {num_generations} generations to: {output_file}")
print(f"Total runtime: {duration} seconds")