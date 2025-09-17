import transformers
print(transformers.__file__)
import torch
from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from lettucedetect.models.inference import HallucinationDetector



class DebugLogitsProcessor(LogitsProcessor):
    """
    A minimal logits‑processor that:
      - prints the current ``input_ids`` (the tokens fed to the model);
      - prints the decoded text generated so far;
      - prints the three highest‑scoring logits at each generation step.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.detector = HallucinationDetector(
            method="transformer",
            model_path="KRLabsOrg/tinylettuce-ettin-68m-en",
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Show the raw token ids that are being fed back into the model
        print("\n=== INPUT IDS ===")
        print(input_ids.tolist())

        # Decode the tokens generated up to now (excluding the initial prompt)
        #    ``input_ids`` already contains the whole prefix, so we decode everything.
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print("\n=== GENERATED TEXT SO FAR ===")
        print(generated_text)

        # Find the top‑3 logits (and their token ids) for this step
        topk = torch.topk(scores, k=3, dim=-1)
        top_vals, top_ids = topk.values.squeeze(), topk.indices.squeeze()
        top_tokens = [self.tokenizer.convert_ids_to_tokens(int(t)) for t in top_ids]

        print("\n=== TOP‑3 LOGITS THIS STEP ===")
        for rank, (tok, val) in enumerate(zip(top_tokens, top_vals), start=1):
            print(f"{rank}. token='{tok}'  logit={val.item():.4f}")

        # We don’t modify the scores, so just return them unchanged
        return scores

# TODO 
# 1. Run experiment with TinyLettuce and special tokens using prompt as context but newly generated text as answer plus upcoming logit



model_name = "gpt2"            
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

debug_logits_processor = DebugLogitsProcessor(tokenizer)
logits_processors = LogitsProcessorList([debug_logits_processor])

prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Trigger generation – you should see your debug output
output_ids = model.generate(input_ids, max_length=30, logits_processor=logits_processors)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))