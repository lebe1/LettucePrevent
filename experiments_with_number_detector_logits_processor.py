import torch
from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
from word2number import w2n
from tqdm import tqdm
import re
from typing import List, Set, Dict, Tuple, Optional 
from datetime import datetime
import time
import json
import string

# -------------------------- Setup ------------------

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

seed = 42
torch.manual_seed(seed)

# -------------------------- Prompt Formatting ------------------

system_prompt = (
    "You always respond very precise and clear. You never exceed the maximum number of words that is asked for. "
    "Always end your answer with a complete sentence and a period! "
    "Only stick to the information provided from the input!"
)

# -------------------------- NumberDetector ------------------
class NumberHallucinationDetector:
    """
    Detects potential number hallucinations by checking if generated digits
    are part of numbers that exist in the input text.
    """
    
    def __init__(self, tokenizer, input_text: str):
        self.tokenizer = tokenizer
        self.input_text = input_text
        self.allowed_numbers = self._extract_all_numbers_from_input()
        self.allowed_digit_sequences = self._generate_allowed_digit_sequences()
        self.digit_tokens = self._get_digit_tokens()
        
    def _extract_written_numbers(self, text: str) -> Set[str]:
        """Extract written numbers like 'twelve', 'twenty-three', etc."""
        written_numbers = set()
        
        # Common written number words
        number_words = [
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 
            'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
            'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
            'billion', 'trillion'
        ]
        
        # Find potential written numbers in text
        words = re.findall(r'\b\w+\b', text.lower())
        potential_number_phrases = []
        
        # Look for sequences that might be written numbers
        i = 0
        while i < len(words):
            if words[i] in number_words:
                phrase = [words[i]]
                j = i + 1
                # Continue collecting connected number words
                while j < len(words) and (words[j] in number_words or words[j] in ['-', 'and']):
                    if words[j] != 'and':  # Skip 'and' but continue
                        phrase.append(words[j])
                    j += 1
                
                if len(phrase) > 0:
                    phrase_str = ' '.join(phrase).replace('-', ' ')
                    potential_number_phrases.append(phrase_str)
                i = j
            else:
                i += 1
        
        # Convert written numbers to digits
        for phrase in potential_number_phrases:
            try:
                numeric_value = w2n.word_to_num(phrase)
                written_numbers.add(str(numeric_value))
            except ValueError:
                # If conversion fails, skip this phrase
                continue
                
        return written_numbers

    
    def _extract_numeric_numbers(self, text: str) -> Set[str]:
        """Extract numeric numbers and dates from text, preserving original formats."""
        results = set()
        
        # Date patterns - these will be included as-is
        date_patterns = [
            r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b',  # 12/12/1978, 12.12.1978
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b'         # 12-12-1978
        ]
        
        # Combined pattern to match all numbers in order of specificity
        # Most specific patterns first to avoid partial matches
        combined_pattern = r'\b(?:\d{1,3}(?:[,.]\d{3})+|\d+[.,]\d+|\d+)\b'
        
        # First, find and add all dates exactly as they appear
        for date_pattern in date_patterns:
            dates = re.findall(date_pattern, text)
            results.update(dates)
        
        # Then find all potential numbers using the combined pattern
        potential_numbers = re.findall(combined_pattern, text)
        
        # Filter out dates from numbers and add only original format
        for num_str in potential_numbers:
            is_date = any(re.match(date_pattern, num_str) for date_pattern in date_patterns)
            
            if not is_date:
                # Add only the original format - no normalization or alternatives
                results.add(num_str)
        
        return results
    
        
    def _extract_all_numbers_from_input(self) -> Set[str]:
        """Extract both numeric and written numbers from input text."""
        numeric_numbers = self._extract_numeric_numbers(self.input_text)
        written_numbers = self._extract_written_numbers(self.input_text)
        
        all_numbers = numeric_numbers.union(written_numbers)
        print(f"Found numbers in input: {sorted(all_numbers)}")
        return all_numbers
    
    def _generate_allowed_digit_sequences(self) -> Dict[int, Set[str]]:
        """Generate allowed digit sequences for each position."""
        sequences = {}
        
        for number in self.allowed_numbers:
            # Clean number string (remove non-digit characters except decimal points)
            clean_number = re.sub(r'[^\d.]', '', number)
            
            # Tokenize the number to see how it's split
            tokens = self.tokenizer.encode(clean_number, add_special_tokens=False)
            token_strs = [self.tokenizer.decode([token]) for token in tokens]
            
            # Store sequences of different lengths
            for i in range(len(token_strs)):
                seq_len = i + 1
                if seq_len not in sequences:
                    sequences[seq_len] = set()
                
                # Add partial sequences
                partial_seq = ''.join(token_strs[:seq_len])
                sequences[seq_len].add(partial_seq)
        
        return sequences
    
    def _get_digit_tokens(self) -> Set[int]:
        """Get all token IDs that represent single digits."""
        digit_tokens = set()
        for digit in '0123456789':
            token_ids = self.tokenizer.encode(digit, add_special_tokens=False)
            digit_tokens.update(token_ids)
        return digit_tokens
    
    def _is_digit_space_punctuation_token(self, token_id: int) -> bool:
        """Check if token represents a digit, space, or punctuation relevant to numbers."""
        token_str = self.tokenizer.decode([token_id]).strip()
        # Only consider tokens that could be part of a number (digits, comma, period)
        # But not standalone punctuation in other contexts
        return token_str.isdigit() or token_str in ',.'
    
    def _is_valid_number_format_prefix(self, number_str: str) -> bool:
        """Check if a partial number has valid formatting so far."""
        # Don't validate single punctuation marks
        if number_str in ',.':
            return True
            
        # Check for invalid patterns like consecutive separators, wrong separator positions, etc.
        
        # Invalid: multiple consecutive separators
        if re.search(r'[,.]{2,}', number_str):
            return False
        
        # Invalid: separator at the start
        if re.match(r'^[,.]', number_str):
            return False
        
        # Check comma placement for thousands separators
        if ',' in number_str:
            parts = number_str.split(',')
            # After a comma, we should have exactly 3 digits (or be building toward 3)
            for i, part in enumerate(parts[1:], 1):
                if i < len(parts) - 1:  # Not the last part
                    if len(part) != 3:
                        return False
                else:  # Last part - could be incomplete
                    if len(part) > 3:  # Too many digits after comma
                        return False
        
        return True
    
    def _could_reach_format(self, current: str, target: str) -> bool:
        """Check if current partial number could eventually match target format."""
        # If current is just punctuation, it could reach any target
        if current in ',.':
            return True
            
        current_digits = re.sub(r'[^\d]', '', current)
        target_digits = re.sub(r'[^\d]', '', target)
        
        # Digits must be a prefix
        if not target_digits.startswith(current_digits):
            return False
        
        # Check if the current format could lead to target
        if current == target[:len(current)]:
            return True
        
        # More sophisticated format checking could be added here
        return False

    def check_hallucination(self, current_sequence: str, next_token_id: int, k_tokens: int = 4) -> bool:
        """
        Check if the next token would create a hallucinated number.
        
        Args:
            current_sequence: Current sequence of generated tokens
            next_token_id: ID of the token being considered
            k_tokens: Number of recent tokens to consider
        
        Returns:
            True if hallucination detected, False if allowed
        """
        next_token_str = self.tokenizer.decode([next_token_id]).strip()
        
        # If next token is not a digit or number-related punctuation, allow it
        if not self._is_digit_space_punctuation_token(next_token_id):
            return False
        
        # Allow standalone punctuation marks (they're not number hallucinations)
        if next_token_str in ',.':
            return False
        
        # Get the last k tokens from current sequence
        tokens = self.tokenizer.encode(current_sequence, add_special_tokens=False)
        recent_tokens = tokens[-k_tokens:] if len(tokens) >= k_tokens else tokens
        recent_str = self.tokenizer.decode(recent_tokens).strip()
        
        # Check if adding this token would create a valid number sequence
        potential_sequence = recent_str + next_token_str
        
        # Extract the number-like pattern from the end of the potential sequence
        number_match = re.search(r'[\d,.]+$', potential_sequence)
        if not number_match:
            return False  # No number pattern found
        
        number_candidate = number_match.group()
        
        # Skip if it's just punctuation
        if number_candidate in ',.':
            return False
        
        # Check if this candidate matches exactly or is a valid prefix of any allowed number
        for allowed_number in self.allowed_numbers:
            # Exact match
            if number_candidate == allowed_number:
                return False  # Not a hallucination
            
            # Check if it's a valid prefix (both format and digits must align)
            if allowed_number.startswith(number_candidate):
                return False  # Valid prefix, not a hallucination
        
        # Additional check: validate format consistency
        if self._is_valid_number_format_prefix(number_candidate):
            # Even if it's not a prefix of allowed numbers, check if the digit sequence
            # could be building toward an allowed number
            digit_sequence = re.sub(r'[^\d]', '', number_candidate)
            
            for allowed_number in self.allowed_numbers:
                clean_allowed = re.sub(r'[^\d]', '', allowed_number)
                if clean_allowed.startswith(digit_sequence):
                    # The digits match, but format might be wrong
                    # Check if we can still reach the allowed format
                    if self._could_reach_format(number_candidate, allowed_number):
                        return False
        
        print("Hallucinating number sequence!! ", number_candidate)
        return True  # Potential hallucination detected

# -------------------------- LogitsProcessor ------------------
class NumberEnsuringLogitsProcessor(LogitsProcessor):
    def __init__(self, detector: NumberHallucinationDetector, last_k_tokens_to_consider: int = 4, top_k_logits: int = 10,
                    penalty_value: float = float('-inf')):
        self.detector = detector
        self.last_k_tokens_to_consider = last_k_tokens_to_consider
        self.penalty_value = penalty_value
        self.generation_history = ""
        self.top_k_logits = top_k_logits
        self.modifications_count = 0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to prevent number hallucinations.
        
        Args:
            input_ids: Current sequence of token IDs
            scores: Logit scores for next token prediction
            
        Returns:
            Modified logit scores
        """
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            # Get current sequence
            current_ids = input_ids[batch_idx]
            current_text = self.detector.tokenizer.decode(current_ids, skip_special_tokens=True)
            
            # Get top k token candidates
            top_k_scores, top_k_indices = torch.topk(scores[batch_idx], k=min(self.top_k_logits, scores.shape[-1]))
            
            # Important: We're assuming greedy logit selection here
            # Check highest candidate and quit for loop until logit with highest probability without hallucination is found
            for i, token_id in enumerate(top_k_indices):
                token_id_item = token_id.item()
                
                # Check if this token would create a hallucination
                if self.detector.check_hallucination(current_text, token_id_item, self.last_k_tokens_to_consider):
                    # Apply penalty to prevent this token
                    scores[batch_idx][token_id_item] = self.penalty_value
                    self.modifications_count += 1
                    print("----------Had to modify the highest probability logit----------")
                    continue
                
                # Once the if statement is passed, we get out of this loop
                break
        
        return scores
# -------------------------- Load Prompts ------------------

with open("./data/summary_prompt_counts.json", "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

print(f"Loaded {len(prompt_data)} prompts from RAGTruth dataset")

results = []
num_generations = 0
start_dt = datetime.now()
start_time = time.time()

# -------------------------- Main Loop ------------------

for item in tqdm(prompt_data):  
    start_dt_prompt = datetime.now()
    start_time_prompt = time.time()
    raw_prompt = item["prompt"]

    # Initialize NumberHallucinationDetector for this specific prompt
    detector = NumberHallucinationDetector(tokenizer, raw_prompt)
    
    # Get allowed numbers for this prompt
    allowed_numbers = list(detector.allowed_numbers)
    
    # Initialize LogitsProcessor with the detector
    LAST_K_TOKENS_TO_CONSIDER = 10
    TOP_K_LOGITS = 10
    logits_processor = NumberEnsuringLogitsProcessor(
        detector=detector,
        last_k_tokens_to_consider=LAST_K_TOKENS_TO_CONSIDER,
        top_k_logits=TOP_K_LOGITS,
        penalty_value=float('-inf')
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_data = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=False)
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

    # Create LogitsProcessorList with our number hallucination detector
    logits_processor_list = LogitsProcessorList([logits_processor])

    output = model.generate(
        **input_data,
        generation_config=gen_config,
        logits_processor=logits_processor_list
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Important: Only working for summary tasks by looking for the last occurrence of "[/INST]" 
    if "[/INST]" in decoded:
        answer_only = decoded.split("[/INST]", 1)[-1].strip()
    else:
        # Fallback: if for some reason "[/INST]" isn't found
        answer_only = decoded.strip()

    end_dt_prompt = datetime.now()
    duration_prompt = round(time.time() - start_time_prompt, 2)

    results.append({
        "prompt": raw_prompt,
        "answer": answer_only,
        "allowed_numbers": allowed_numbers,
        "logits_modifications": logits_processor.modifications_count,
        "original_counts": item["counts"],  # Original number of models that processed this prompt
        "task_type": "Summary",
        "dataset": "ragtruth",
        "language": "en",
        "start_time": start_dt_prompt.isoformat(),
        "end_time": end_dt_prompt.isoformat(),
        "duration_seconds": duration_prompt
    })

    num_generations += 1
    print(f"Processed {num_generations}/{len(prompt_data[:2])} prompts")

# -------------------------- Metadata ------------------

end_dt = datetime.now()
duration = round(time.time() - start_time, 2)

results.append({
    "_meta": {
        "model": model_name,
        "system_prompt": system_prompt,
        "num_prompts": len(prompt_data),  
        "total_generations": num_generations,
        "seed": seed,
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "duration_seconds": duration,
        "generation_config": gen_config.to_dict(),
        "number_hallucination_detector_config": {
            "last_k_tokens_to_consider": LAST_K_TOKENS_TO_CONSIDER,
            "top_k_logits": TOP_K_LOGITS,
            "penalty_value": "negative_infinity"
        }
    }
})

# -------------------------- Save ------------------

timestamp = start_dt.strftime("%Y%m%d_%H%M%S")
output_file = f"./data/summary_experiments_run_{timestamp}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {num_generations} generations to: {output_file}")
print(f"Total runtime: {duration} seconds")
print(f"Average time per generation: {duration/num_generations:.2f} seconds")