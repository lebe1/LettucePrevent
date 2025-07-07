import torch
import re
from transformers import LogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList


class RegexLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, pattern: str):
        """
        Args:
            tokenizer: The tokenizer used to decode token IDs.
            pattern (str): A regex pattern to match against token-level text.
        """
        self.tokenizer = tokenizer
        self.regex = re.compile(pattern)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape

        for batch_idx in range(batch_size):
            for token_id in range(vocab_size):
                token_text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                
                if self.regex.search(token_text):
                    # Log the blocked token
                    print(f"[Batch {batch_idx}] Blocking token '{token_text}' (ID {token_id}) due to regex match: {self.regex.pattern}")
                    
                    scores[batch_idx, token_id] = -float("inf")

        return scores



model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Regex pattern is defined here
regex_processor = RegexFilterLogitsProcessor(tokenizer, r"\d")  

logits_processor = LogitsProcessorList()
logits_processor.append(regex_processor)

input = tokenizer("The password is", return_tensors="pt")

output = model.generate(
    **input,
    max_length=20,
    logits_processor=logits_processor,
    do_sample=True,  # Use sampling so more tokens are evaluated
    top_k=50,
)

print("\nGenerated output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
