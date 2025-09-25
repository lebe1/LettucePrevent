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
from lettucedetect import HallucinationDetector
from detectors.base_detector import BaseHallucinationDetector

# -------------------------- LogitsProcessor ------------------

class HallucinationLogitsProcessor(LogitsProcessor):
    """
    Generalized logits processor that works with any BaseHallucinationDetector implementation.
    """
    
    def __init__(self, 
                 detector: BaseHallucinationDetector, 
                 last_k_tokens_to_consider: int = 4, 
                 top_k_logits: int = 10,
                 penalty_value: float = float('-inf')):
        """
        Initialize the generalized logits processor.
        
        Args:
            detector: Any detector implementing BaseHallucinationDetector
            last_k_tokens_to_consider: Number of recent tokens to consider
            top_k_logits: Number of top logits to check for hallucinations
            penalty_value: Penalty value to apply to hallucinated tokens
        """
        self.detector = detector
        self.last_k_tokens_to_consider = last_k_tokens_to_consider
        self.penalty_value = penalty_value
        self.top_k_logits = top_k_logits
        self.modifications_count = 0
        
        print(f"Initialized LogitsProcessor with {type(detector).__name__}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to prevent hallucinations using the configured detector.
        
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
            
            # Check each candidate token for hallucinations
            for i, token_id in enumerate(top_k_indices):
                token_id_item = token_id.item()
                
                # Check if this token would create a hallucination
                if self.detector.check_hallucination(current_text, token_id_item, self.last_k_tokens_to_consider):
                    # Apply penalty to prevent this token
                    scores[batch_idx][token_id_item] = self.penalty_value
                    self.modifications_count += 1
                    print(f"----------Modified logit for token: {self.detector.tokenizer.decode([token_id_item])}----------")
                    continue
                
                # Break after finding the first non-hallucinating token (greedy approach)
                break
        
        return scores