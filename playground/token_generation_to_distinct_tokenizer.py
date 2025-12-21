from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
#model_name = "meta-llama/Llama-2-7b-chat-hf"
#model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

# Input prompt
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

max_new_tokens = 20
current_ids = input_ids

for i in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(current_ids)
        next_token_id = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    
    token_id = next_token_id[0].item()
    
    # METHOD 1: Direct vocabulary lookup (RAW token string)
    raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    
    # METHOD 2: Using decode and skip special tokens
    decoded_str_true = tokenizer.batch_decode([token_id], clean_up_tokenization_spaces=True)

    # METHOD 3: Using decode and do not skip special tokens
    decoded_str_false = tokenizer.batch_decode([token_id], clean_up_tokenization_spaces=False)
    
    print(f"Token {i+1}:")
    print(f"  ID: {token_id}")
    print(f"  Raw vocabulary token:      '{raw_token}'")
    print(f"  Decoded string true:       '{decoded_str_true}'")
    print(f"  Decoded string false:      '{decoded_str_false}'")
    print(f"  Raw bytes: {raw_token.encode('utf-8')}")
    print()
    
    current_ids = torch.cat([current_ids, next_token_id], dim=1)
    
    if token_id == tokenizer.eos_token_id:
        print("EOS token reached!")
        break

print("=" * 70)
print("FULL SEQUENCE:")
print(tokenizer.decode(current_ids[0]))
print("=" * 70)