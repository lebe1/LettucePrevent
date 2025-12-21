import torch
from transformers import AutoTokenizer
from lettucedetect.models.inference import HallucinationDetector
import json

# -------------------------- Setup ------------------

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
main_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load TinyLettuce
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/tinylettuce-ettin-68m-en"
)

# Get TinyLettuce's internal tokenizer
tinylettuce_tokenizer = AutoTokenizer.from_pretrained("KRLabsOrg/tinylettuce-ettin-68m-en")

# -------------------------- Load Prompt ------------------

system_prompt = (
    "You always respond very precise and clear. You never exceed the maximum number of words that is asked for. "
    "Always end your answer with a complete sentence and a period! "
    "Only stick to the information provided from the input!"
)

raw_prompt = """Summarize the following news within 116 words:\nSeventy years ago, Anne Frank died of typhus in a Nazi concentration camp at the age of 15. Just two weeks after her supposed death on March 31, 1945, the Bergen-Belsen concentration camp where she had been imprisoned was liberated -- timing that showed how close the Jewish diarist had been to surviving the Holocaust. But new research released by the Anne Frank House shows that Anne and her older sister, Margot Frank, died at least a month earlier than previously thought. Researchers re-examined archives of the Red Cross, the International Training Service and the Bergen-Belsen Memorial, along with testimonies of survivors. They concluded that Anne and Margot probably did not survive to March 1945 -- contradicting the date of death which had previously been determined by Dutch authorities. In 1944, Anne and seven others hiding in the Amsterdam secret annex were arrested and sent to the  Auschwitz-Birkenau concentration camp. Anne Frank's final entry. That same year, Anne and Margot were separated from their mother and sent away to work as slave labor at the Bergen-Belsen camp in Germany. Days at the camp were filled with terror and dread, witnesses said. The sisters stayed in a section of the overcrowded camp with no lighting, little water and no latrine. They slept on lice-ridden straw and violent storms shredded the tents, according to the researchers. Like the other prisoners, the sisters endured long hours at roll call. Her classmate, Nannette Blitz, recalled seeing Anne there in December 1944: \"She was no more than a skeleton by then. She was wrapped in a blanket; she couldn't bear to wear her clothes anymore because they were crawling with lice.\" Listen to Anne Frank's friends describe her concentration camp experience. As the Russians advanced further, the Bergen-Belsen concentration camp became even more crowded, bringing more disease. A deadly typhus outbreak caused thousands to die each day. Typhus is an infectious disease caused by lice that breaks out in places with poor hygiene. The disease causes high fever, chills and skin eruptions. \"Because of the lice infesting the bedstraw and her clothes, Anne was exposed to the main carrier of epidemic typhus for an extended period,\" museum researchers wrote. They concluded that it's unlikely the sisters survived until March, because witnesses at the camp said the sisters both had symptoms before February 7. \"Most deaths caused by typhus occur around twelve days after the first symptoms appear,\" wrote  authors Erika Prins and Gertjan Broek. The exact dates of death for Anne and Margot remain unclear. Margot died before Anne. \"Anne never gave up hope,\" said Blitz, her friend. \"She was absolutely convinced she would survive.\" Her diary endures as one of the world's most popular books. Read more about Anne Frank's cousin, a keeper of her legacy.\n\noutput:"""


# Simulate what happens during generation
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": raw_prompt}
]
formatted_prompt = main_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Simulate a partial answer (like what would happen during generation)
current_answer = "Anne Frank and her sister Margot died"
potential_answer = current_answer + " earlier"

print("="*80)
print("DEBUGGING TINYLETTUCE INPUT")
print("="*80)

# Test 1: Check what we're passing
print("\n1️⃣  INPUT TO TINYLETTUCE:")
print(f"   Context length: {len(raw_prompt)} chars")
print(f"   Answer length: {len(potential_answer)} chars")
print(f"   Context tokens (TinyLettuce): {len(tinylettuce_tokenizer.encode(raw_prompt))}")
print(f"   Answer tokens (TinyLettuce): {len(tinylettuce_tokenizer.encode(potential_answer))}")

print("Model name:", detector)
#print("Max position embeddings:", detector.model_path)

# Test 2: Try to reproduce the error
print("\n2️⃣  ATTEMPTING PREDICTION:")
try:
    predictions = detector.predict(
        context=raw_prompt[:1000], 
        answer=potential_answer, 
        output_format="spans"
    )
    print(f"   ✓ Success! Predictions: {predictions}")
except Exception as e:
    print(f"   ✗ Error occurred: {e}")
    print(f"   Error type: {type(e).__name__}")

# Test 3: Check if it's the formatted_prompt causing issues
print("\n3️⃣  TESTING WITH FORMATTED PROMPT AS CONTEXT:")
try:
    predictions = detector.predict(
        context=formatted_prompt[:2000],  # Using formatted prompt instead
        answer=potential_answer, 
        output_format="tokens"
    )
    print(f"   Context tokens: {len(tinylettuce_tokenizer.encode(formatted_prompt))}")
    print(f"   ✓ Success! Predictions: {predictions}")
except Exception as e:
    print(f"   ✗ Error occurred: {e}")
    print(f"   Error type: {type(e).__name__}")

