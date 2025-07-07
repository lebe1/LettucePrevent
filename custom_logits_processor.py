import torch
from transformers import LogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from typing import Callable, Optional, List
from collections import Counter
import re

# -------------------------- Custom Processor ------------------

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        input_text: Optional[str] = None,
        input_extract_method: Optional[Callable[[str], List[str]]] = None,
        output_restrict_method: Optional[Callable[[str, Optional[List[str]]], bool]] = None,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.output_restrict_method = output_restrict_method
        self.verbose = verbose

        if input_text is not None and input_extract_method is not None:
            try:
                self.allowed_items = input_extract_method(input_text)
                if self.verbose:
                    print(f"[INIT] Allowed items extracted from input: {self.allowed_items}")
            except Exception as e:
                self.allowed_items = None
                if self.verbose:
                    print(f"[INIT ERROR] Failed to extract allowed items from input: {e}")
        else:
            self.allowed_items = None
            if self.verbose:
                print("[INIT] No extract method or input text provided. allowed_items = None")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.output_restrict_method is None:
            if self.verbose:
                print("[LOGITS] No output_restrict_method provided — skipping token filtering.")
            return scores

        batch_size, vocab_size = scores.shape
        if batch_size != 1:
            raise ValueError("[ERROR] This processor only supports batch size 1 for now.")

        for token_id in range(vocab_size):
            token_text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

            try:
                if self.output_restrict_method(token_text, self.allowed_items):
                    if self.verbose:
                        print(f"[BLOCKED] Blocking token '{token_text}' (ID {token_id})")
                    scores[0, token_id] = -float("inf")
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Restriction method failed for token '{token_text}': {e}")
        return scores

# -------------------------- Sample Extract / Restrict Functions ------------------

def extract_frequent_words(text):
    counts = Counter(re.findall(r"\b\w+\b", text.lower()))
    return [w for w, c in counts.items() if c >= 3]

def allow_only_frequent(token_text, allowed):
    return token_text.lower() in (allowed or [])

# -------------------------- Setup Model & Tokenizer ------------------

model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "the the the cat sat on the mat mat mat"
input = tokenizer(prompt, return_tensors="pt")

processor = CustomConstraintLogitsProcessor(
    tokenizer=tokenizer,
    input_text=prompt,
    input_extract_method=extract_frequent_words,
    output_restrict_method=allow_only_frequent,
    verbose=True
)

logits_processor = LogitsProcessorList()
logits_processor.append(processor)

pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

# -------------------------- Run Generation ------------------

output = model.generate(
    **input,
    max_length=20,
    logits_processor=logits_processor,
    do_sample=True,
    top_k=50,
    pad_token_id=pad_token_id
)

print("\nGenerated output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
