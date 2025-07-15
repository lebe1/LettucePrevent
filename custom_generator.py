import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict
from lettucedetect.models.inference import HallucinationDetector


# -------------------------- Custom Generator ------------------

class HallucinationFilteredGenerator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        detector,
        step_size: int = 3,
        top_k: int = 50,
        halluc_threshold: float = 0.9,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.detector = detector
        self.step_size = step_size
        self.top_k = top_k
        self.halluc_threshold = halluc_threshold
        self.device = device or model.device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated = input_ids.clone()
        past_key_values = None

        total_generated = 0
        bad_token_ids = set()

        while total_generated < max_new_tokens:
            step_buffer = []

            for _ in range(self.step_size):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=generated,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :] / temperature
                    past_key_values = outputs.past_key_values

                    # Clip to vocab size
                    vocab_size = self.tokenizer.vocab_size
                    logits = logits[:, :vocab_size]

                    # Mask known bad tokens
                    if bad_token_ids:
                        logits[:, list(bad_token_ids)] = float('-inf')

                    # Sample token
                    if do_sample:
                        topk = torch.topk(logits, self.top_k)
                        probs = torch.softmax(topk.values, dim=-1)
                        next_token_idx = torch.multinomial(probs, num_samples=1)
                        token_id = topk.indices.gather(1, next_token_idx)  # shape: [1, 1]
                    else:
                        token_id = torch.argmax(logits, dim=-1, keepdim=True)  # shape: [1, 1]

                    # Defensive check
                    if token_id.item() >= vocab_size:
                        raise ValueError(f"Sampled token ID {token_id.item()} exceeds vocab size {vocab_size}")

                    # Safe cat
                    generated = torch.cat([generated, token_id.view(1, 1)], dim=-1)

                    step_buffer.append(token_id.item())
                    total_generated += 1

                    if total_generated >= max_new_tokens:
                        break

            # Check hallucination on last `step_size` tokens
            new_text = self.tokenizer.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True)
            preds = self.detector.predict_prompt(prompt, new_text, output_format="tokens")

            # Evaluate only last k tokens
            recent_preds = preds[-len(step_buffer):]
            hallucinated = [
                (i, p["token"], p["prob"])
                for i, p in enumerate(recent_preds)
                if p["prob"] >= self.halluc_threshold
            ]

            if hallucinated:
                # Roll back
                generated = generated[:, :-len(step_buffer)]
                total_generated -= len(step_buffer)

                # Mask hallucinated token IDs
                for _, token, _ in hallucinated:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None:
                        bad_token_ids.add(token_id)

        final_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return final_text


# -------------------------- Run Generation ------------------


# Load model and tokenizer
model_name = "sshleifer/tiny-gpt2"  
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

detector = HallucinationDetector(method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")

# Instantiate the wrapper
gen = HallucinationFilteredGenerator(
    model=model,
    tokenizer=tokenizer,
    detector=detector,
    step_size=3,
    top_k=50,
    halluc_threshold=0.9
)

# Generate
output = gen.generate(prompt="The Eiffel Tower is located in", max_new_tokens=50)
print(output)


