from transformers import AutoTokenizer


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample sentence to tokenize
sentence = (
    "Dagobert Duck owns 1,000,000 Dollars and Donald Duck has like 1.000 Dollars \n"
    "the disease they succumbed to, before February 7, 1945"
)


# `return_tensors="pt"` isn’t needed for pure tokenization,
# but you can keep it if you later feed the ids into a model.
encoded = tokenizer(sentence, return_attention_mask=False, return_token_type_ids=False)

token_ids   = encoded["input_ids"]          # List[int]
token_strs  = tokenizer.convert_ids_to_tokens(token_ids)

print("\nOriginal sentence:")
print(sentence)

print("\nToken IDs:")
print(token_ids)

print("\nCorresponding tokens:")
print(token_strs)