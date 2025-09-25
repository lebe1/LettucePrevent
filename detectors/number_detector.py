import torch
from transformers import LogitsProcessor, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
from word2number import w2n
from tqdm import tqdm
import re
from typing import List, Set, Dict, Tuple, Optional, Any, Union
from datetime import datetime
import time
import json
import string
from .base_detector import BaseHallucinationDetector

# -------------------------- Legacy Number Detector ------------------

class NumberHallucinationDetector(BaseHallucinationDetector):
    """
    Legacy detector for number hallucinations - kept for backward compatibility.
    """
    
    def __init__(self, tokenizer, input_text: str):
        super().__init__(tokenizer, input_text)
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
        return token_str.isdigit() or token_str in ',.'
    
    def _is_valid_number_format_prefix(self, number_str: str) -> bool:
        """Check if a partial number has valid formatting so far."""
        if number_str in ',.':
            return True
            
        if re.search(r'[,.]{2,}', number_str):
            return False
        
        if re.match(r'^[,.]', number_str):
            return False
        
        if ',' in number_str:
            parts = number_str.split(',')
            for i, part in enumerate(parts[1:], 1):
                if i < len(parts) - 1:
                    if len(part) != 3:
                        return False
                else:
                    if len(part) > 3:
                        return False
        
        return True
    
    def _could_reach_format(self, current: str, target: str) -> bool:
        """Check if current partial number could eventually match target format."""
        if current in ',.':
            return True
            
        current_digits = re.sub(r'[^\d]', '', current)
        target_digits = re.sub(r'[^\d]', '', target)
        
        if not target_digits.startswith(current_digits):
            return False
        
        if current == target[:len(current)]:
            return True
        
        return False

    def check_hallucination(self, current_sequence: str, next_token_id: int, k_tokens: int = 4) -> bool:
        """
        Check if the next token would create a hallucinated number.
        """
        next_token_str = self.tokenizer.decode([next_token_id]).strip()
        
        if not self._is_digit_space_punctuation_token(next_token_id):
            return False
        
        if next_token_str in ',.':
            return False
        
        tokens = self.tokenizer.encode(current_sequence, add_special_tokens=False)
        recent_tokens = tokens[-k_tokens:] if len(tokens) >= k_tokens else tokens
        recent_str = self.tokenizer.decode(recent_tokens).strip()
        
        potential_sequence = recent_str + next_token_str
        
        number_match = re.search(r'[\d,.]+$', potential_sequence)
        if not number_match:
            return False
        
        number_candidate = number_match.group()
        
        if number_candidate in ',.':
            return False
        
        for allowed_number in self.allowed_numbers:
            if number_candidate == allowed_number:
                return False
            
            if allowed_number.startswith(number_candidate):
                return False
        
        if self._is_valid_number_format_prefix(number_candidate):
            digit_sequence = re.sub(r'[^\d]', '', number_candidate)
            
            for allowed_number in self.allowed_numbers:
                clean_allowed = re.sub(r'[^\d]', '', allowed_number)
                if clean_allowed.startswith(digit_sequence):
                    if self._could_reach_format(number_candidate, allowed_number):
                        return False
        
        print("Hallucinating number sequence!! ", number_candidate)
        return True