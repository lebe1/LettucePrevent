from transformers import AutoTokenizer


model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample sentence to tokenize
sentence = (
    "Mezza Thyme is a highly-rated Mediterranean restaurant and bar in Santa Barbara, California. Located at 20 E Cota St, the establishment is known for its variety of dishes and its nightlife. It is open from 4:00 PM to 8:00 PM, Tuesday to Saturday. Customers appreciate the freshly made pita bread,"
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

# Decode back to string
decoded = tokenizer.decode(token_ids)
print("\nDecoded text:")
print(decoded)

# Show if anything changed
if decoded == sentence:
    print("\n→ Round-trip is IDENTICAL (no change)")
else:
    print("\n→ Round-trip DIFFERS!")
    # Show character-level differences
    for i, (a, b) in enumerate(zip(sentence, decoded)):
        if a != b:
            print(f"  Position {i}: original='{a}' (U+{ord(a):04X}) → decoded='{b}' (U+{ord(b):04X})")
    if len(sentence) != len(decoded):
        print(f"  Length difference: original={len(sentence)}, decoded={len(decoded)}")

token_ids_no_special = tokenizer.encode(sentence, add_special_tokens=False)
decoded_clean = tokenizer.decode(token_ids_no_special)
print("\nDecoded (no special tokens):")
print(decoded_clean)
