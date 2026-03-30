import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    LogitsProcessorList, StoppingCriteria, StoppingCriteriaList,
)
from tqdm import tqdm
from datetime import datetime
import time
import json
from detectors.factory import DetectorFactory
from logits_processors.hallucination_logits_processor import HallucinationLogitsProcessor


class TokenPrintStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, _scores, **_kwargs):
        last_token_str = self.tokenizer.decode([input_ids[0, -1].item()])
        print(f">>> Actually generated token: {repr(last_token_str)}")
        return False


# -------------------------- Setup ------------------

# model_name = "Qwen/Qwen2.5-14B-Instruct"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer  = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

seed = 14
torch.manual_seed(seed)

# -------------------------- Prompt Formatting ------------------

system_prompt = (
    "You always respond very precise and clear. You never exceed the maximum number of words that is asked for. "
    "Always end your answer with a complete sentence and a period! "
    "Only stick to the information provided from the input!"
)

# -------------------------- Load Prompts ------------------

with open("./data/ragtruth_unique_summary_prompts.json", "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

print(f"Loaded {len(prompt_data)} prompts from RAGTruth dataset")

results        = []
num_generations = 0
start_dt       = datetime.now()
start_time     = time.time()

# -------------------------- Configuration ------------------

DETECTOR_TYPE             = 'ettin'   # 'tinylettuce' | 'ettin' | 'number' | 'none'
CONFIDENCE_THRESHOLD      = 0.9
LAST_K_TOKENS_TO_CONSIDER = 10        # ignored when USE_ALL_TOKENS=True
TOP_K_LOGITS              = 10
PENALTY_VALUE             = 0
USE_ALL_TOKENS            = True
LOGITS_SKIP_THRESHOLD     = 1.0
ETTIN_MODEL_PATH          = "lebe1/tinylettuce-ettin-decoder-68m-en"

print(f"Using detector: {DETECTOR_TYPE}")

# -------------------------- Main Loop ------------------

for item in tqdm(prompt_data[:1]):
    start_dt_prompt    = datetime.now()
    start_time_prompt  = time.time()
    raw_prompt         = item["prompt"]

    if DETECTOR_TYPE != 'none':
        detector = DetectorFactory.create_detector(
            detector_type=DETECTOR_TYPE,
            tokenizer=tokenizer,
            input_text=raw_prompt,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            model_path=ETTIN_MODEL_PATH,   # ignored by non-ettin detectors via **kwargs
        )

        logits_processor = HallucinationLogitsProcessor(
            hallucination_detector=detector,
            last_k_tokens_to_consider=LAST_K_TOKENS_TO_CONSIDER,
            top_k_logits=TOP_K_LOGITS,
            penalty_value=PENALTY_VALUE,
            use_all_tokens=USE_ALL_TOKENS,
            skip_threshold=LOGITS_SKIP_THRESHOLD,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": raw_prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_data = tokenizer(
        formatted_prompt, return_tensors="pt", padding=True, truncation=False
    )
    input_data = {k: v.to(model.device) for k, v in input_data.items()}

    gen_config = GenerationConfig(
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        min_length=150,
        num_beams=4,
    )

    token_print_criteria = StoppingCriteriaList(
        [TokenPrintStoppingCriteria(tokenizer)]
    )

    if DETECTOR_TYPE != 'none':
        output = model.generate(
            **input_data,
            generation_config=gen_config,
            logits_processor=LogitsProcessorList([logits_processor]),
            stopping_criteria=token_print_criteria,
        )
    else:
        output = model.generate(
            **input_data,
            generation_config=gen_config,
            stopping_criteria=token_print_criteria,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer_only = (
        decoded.split("[/INST]", 1)[-1].strip()
        if "[/INST]" in decoded
        else decoded.strip()
    )

    end_dt_prompt    = datetime.now()
    duration_prompt  = round(time.time() - start_time_prompt, 2)

    # ── Build result dict consistently for all detector types ──
    result_data = {
        "prompt":           raw_prompt,
        "answer":           answer_only,
        "original_counts":  item["counts"],
        "task_type":        "Summary",
        "dataset":          "ragtruth",
        "language":         "en",
        "start_time":       start_dt_prompt.isoformat(),
        "end_time":         end_dt_prompt.isoformat(),
        "duration_seconds": duration_prompt,
        "detector_type":    DETECTOR_TYPE,
    }

    if DETECTOR_TYPE == 'number':
        result_data["logits_modifications"] = logits_processor.modifications_count
        result_data["allowed_numbers"]      = list(detector.allowed_numbers)
    elif DETECTOR_TYPE == 'tinylettuce':
        result_data["logits_modifications"]  = logits_processor.modifications_count
        result_data["confidence_threshold"]  = CONFIDENCE_THRESHOLD
    elif DETECTOR_TYPE == 'ettin':
        result_data["logits_modifications"]  = logits_processor.modifications_count
        result_data["confidence_threshold"]  = CONFIDENCE_THRESHOLD
        result_data["model_path"]            = ETTIN_MODEL_PATH
    elif DETECTOR_TYPE == 'none':
        result_data["comparison_experiment"] = True

    results.append(result_data)
    num_generations += 1
    print(f"Processed {num_generations}/{len(prompt_data)} prompts")

# -------------------------- Metadata ------------------

end_dt   = datetime.now()
duration = round(time.time() - start_time, 2)

results.append({
    "_meta": {
        "model":             model_name,
        "system_prompt":     system_prompt,
        "num_prompts":       len(prompt_data),
        "total_generations": num_generations,
        "seed":              seed,
        "start_time":        start_dt.isoformat(),
        "end_time":          end_dt.isoformat(),
        "duration_seconds":  duration,
        "generation_config": gen_config.to_dict(),
        "detector_config": {
            "detector_type":             DETECTOR_TYPE,
            "last_k_tokens_to_consider": LAST_K_TOKENS_TO_CONSIDER,
            "top_k_logits":              TOP_K_LOGITS,
            "penalty_value":             str(PENALTY_VALUE),
            "confidence_threshold":      CONFIDENCE_THRESHOLD,
            "use_all_tokens":            USE_ALL_TOKENS,
            "logits_skip_threshold":     LOGITS_SKIP_THRESHOLD,
            "ettin_model_path":          ETTIN_MODEL_PATH if DETECTOR_TYPE == 'ettin' else None,
        },
    }
})

# -------------------------- Save ------------------

timestamp   = start_dt.strftime("%Y%m%d_%H%M%S")
output_file = f"./data/summary_experiments_{DETECTOR_TYPE}_run_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {num_generations} generations to: {output_file}")
print(f"Total runtime: {duration} seconds")
print(f"Average time per generation: {duration / num_generations:.2f} seconds")
print(f"Total logit modifications: {sum(r.get('logits_modifications', 0) for r in results if 'logits_modifications' in r)}")