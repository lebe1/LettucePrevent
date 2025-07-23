import torch
from transformers import LogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from typing import Callable, Optional, List, Dict
import re
from word2number import w2n

# -------------------------- Number Extraction ------------------

def extract_cardinal_digits(text: str) -> List[str]:
    return re.findall(r'\b\d+\b', text)


def extract_number_words(text: str) -> List[str]:
    number_word_pattern = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                                     r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                                     r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                                     r'eighty|ninety|hundred|thousand|million|billion|and|[-])+\b',
                                     re.IGNORECASE)
    matches = number_word_pattern.finditer(text)
    number_strings = []
    for match in matches:
        phrase = match.group().replace("-", " ").lower()
        try:
            number = str(w2n.word_to_num(phrase))
            number_strings.append(number)
        except ValueError:
            continue
    return number_strings


def extract_allowed_numbers(text: str) -> List[str]:
    digits = extract_cardinal_digits(text)
    words = extract_number_words(text)
    all_numbers = list(set(digits + words))
    return all_numbers


# -------------------------- Output Restriction Method ------------------

def allow_only_input_numbers(token_text: str, allowed_numbers: List[str]) -> bool:
    token_text_clean = token_text.strip().lower()

    # Check if token is a numeric word or digit
    if re.fullmatch(r'\d+|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                    r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                    r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                    r'eighty|ninety|hundred|thousand|million|billion|[-\s]+)+',
                    token_text_clean):
        try:
            if re.fullmatch(r'\d+', token_text_clean):
                num_str = token_text_clean
            else:
                normalized = token_text_clean.replace("-", " ")
                num_str = str(w2n.word_to_num(normalized))

            return num_str not in allowed_numbers  # True = block
        except Exception:
            return False  # allow token if parsing fails
    else:
        return False  # allow non-number tokens


# -------------------------- Custom Logits Processor ------------------

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        input_text: Optional[str] = None,
        input_extract_method: Optional[Callable[[str], List[str]]] = None,
        output_restrict_method: Optional[Callable[[str, Optional[List[str]]], bool]] = None,
        top_k_filter: Optional[int] = None,
        top_p_filter: Optional[float] = None,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.output_restrict_method = output_restrict_method
        self.verbose = verbose
        self.top_k_filter = top_k_filter
        self.top_p_filter = top_p_filter
        self.decode_cache: Dict[int, str] = {}

        if input_text and input_extract_method:
            try:
                self.allowed_items = input_extract_method(input_text)
                if self.verbose:
                    print(f"[INIT] Allowed numbers extracted from input: {self.allowed_items}")
            except Exception as e:
                self.allowed_items = None
                if self.verbose:
                    print(f"[INIT ERROR] Failed to extract allowed items from input: {e}")
        else:
            self.allowed_items = None
            if self.verbose:
                print("[INIT] No input text or extract method provided. allowed_items = None")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.output_restrict_method is None or self.allowed_items is None:
            if self.verbose:
                print("[LOGITS] No output_restrict_method or allowed_items — skipping token filtering.")
            return scores

        batch_size, vocab_size = scores.shape
        if batch_size != 1:
            raise ValueError("[ERROR] This processor only supports batch size 1 for now.")

        logits = scores[0]
        probs = torch.softmax(logits, dim=-1)

        candidate_indices = set()

        if self.top_k_filter is not None:
            topk = torch.topk(logits, k=min(self.top_k_filter, vocab_size))
            candidate_indices.update(topk.indices.tolist())

        if self.top_p_filter is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            top_p_mask = cumulative_probs <= self.top_p_filter
            top_p_mask[0] = True  # Always include at least one
            top_p_indices = sorted_indices[top_p_mask].tolist()
            candidate_indices.update(top_p_indices)

        if self.top_k_filter is None and self.top_p_filter is None:
            candidate_indices = set(range(vocab_size))

        for token_id in candidate_indices:
            if token_id in self.decode_cache:
                token_text = self.decode_cache[token_id]
            else:
                token_text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                self.decode_cache[token_id] = token_text

            try:
                if self.output_restrict_method(token_text, self.allowed_items):
                    if self.verbose:
                        print(f"[BLOCKED] Blocking token '{token_text}' (ID {token_id})")
                    scores[0, token_id] = -float("inf")
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Restriction method failed for token '{token_text}': {e}")

        return scores


# -------------------------- Setup Model & Tokenizer ------------------

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# -------------------------- Input ------------------

prompt = "Summarize the following news within 116 words:\nSeventy years ago, Anne Frank died of typhus in a Nazi concentration camp at the age of 15. Just two weeks after her supposed death on March 31, 1945, the Bergen-Belsen concentration camp where she had been imprisoned was liberated -- timing that showed how close the Jewish diarist had been to surviving the Holocaust. But new research released by the Anne Frank House shows that Anne and her older sister, Margot Frank, died at least a month earlier than previously thought. Researchers re-examined archives of the Red Cross, the International Training Service and the Bergen-Belsen Memorial, along with testimonies of survivors. They concluded that Anne and Margot probably did not survive to March 1945 -- contradicting the date of death which had previously been determined by Dutch authorities. In 1944, Anne and seven others hiding in the Amsterdam secret annex were arrested and sent to the  Auschwitz-Birkenau concentration camp. Anne Frank's final entry. That same year, Anne and Margot were separated from their mother and sent away to work as slave labor at the Bergen-Belsen camp in Germany. Days at the camp were filled with terror and dread, witnesses said. The sisters stayed in a section of the overcrowded camp with no lighting, little water and no latrine. They slept on lice-ridden straw and violent storms shredded the tents, according to the researchers. Like the other prisoners, the sisters endured long hours at roll call. Her classmate, Nannette Blitz, recalled seeing Anne there in December 1944: \"She was no more than a skeleton by then. She was wrapped in a blanket; she couldn't bear to wear her clothes anymore because they were crawling with lice.\" Listen to Anne Frank's friends describe her concentration camp experience. As the Russians advanced further, the Bergen-Belsen concentration camp became even more crowded, bringing more disease. A deadly typhus outbreak caused thousands to die each day. Typhus is an infectious disease caused by lice that breaks out in places with poor hygiene. The disease causes high fever, chills and skin eruptions. \"Because of the lice infesting the bedstraw and her clothes, Anne was exposed to the main carrier of epidemic typhus for an extended period,\" museum researchers wrote. They concluded that it's unlikely the sisters survived until March, because witnesses at the camp said the sisters both had symptoms before February 7. \"Most deaths caused by typhus occur around twelve days after the first symptoms appear,\" wrote  authors Erika Prins and Gertjan Broek. The exact dates of death for Anne and Margot remain unclear. Margot died before Anne. \"Anne never gave up hope,\" said Blitz, her friend. \"She was absolutely convinced she would survive.\" Her diary endures as one of the world's most popular books. Read more about Anne Frank's cousin, a keeper of her legacy.\n\noutput:"
input_data = tokenizer(prompt, return_tensors="pt")

# -------------------------- Logits Processor ------------------

processor = CustomLogitsProcessor(
    tokenizer=tokenizer,
    input_text=prompt,
    input_extract_method=extract_allowed_numbers,
    output_restrict_method=allow_only_input_numbers,
    top_k_filter=1,
    top_p_filter=0.9,
    verbose=True
)

logits_processor = LogitsProcessorList([processor])

pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

# -------------------------- Run Generation ------------------

output = model.generate(
    **input_data,
    max_new_tokens=200,
    # logits_processor=logits_processor,
    do_sample=True,
    pad_token_id=pad_token_id,
    # encoder_repetition_penalty=1.2
)

print("\nGenerated output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
