import torch
from transformers import LogitsProcessor
from detectors.base_detector import BaseHallucinationDetector


class HallucinationLogitsProcessor(LogitsProcessor):
    """
    Generalized logits processor that works with any BaseHallucinationDetector implementation.
    """
    def __init__(self,
                 hallucination_detector: BaseHallucinationDetector,
                 last_k_tokens_to_consider: int = 4,
                 top_k_logits: int = 10,
                 penalty_value: float = float('-inf'),
                 use_all_tokens: bool = False):
        """
        Initialize the generalized logits processor.
        
        Args:
            hallucination_detector: Any detector implementing BaseHallucinationDetector
            last_k_tokens_to_consider: Number of recent tokens to consider (ignored if use_all_tokens=True)
            top_k_logits: Number of top logits to check for hallucinations
            penalty_value: Penalty value to apply to hallucinated tokens
            use_all_tokens: If True, consider all generated tokens instead of just last_k_tokens_to_consider
        """
        self.hallucination_detector = hallucination_detector
        self.last_k_tokens_to_consider = last_k_tokens_to_consider
        self.penalty_value = penalty_value
        self.top_k_logits = top_k_logits
        self.use_all_tokens = use_all_tokens
        self.modifications_count = 0
        
        tokens_context = "all tokens" if use_all_tokens else f"last {last_k_tokens_to_consider} tokens"
        print(f"Initialized LogitsProcessor with {type(hallucination_detector).__name__}")
        print(f"Context window: {tokens_context}")


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
        print("Batch size", batch_size)
        
        for batch_idx in range(batch_size):
            print("Batch_ids", batch_idx)

            # Get current sequence
            current_ids = input_ids[batch_idx]
            current_text = self.hallucination_detector.tokenizer.decode(
                current_ids, 
                skip_special_tokens=True
            )
            
            # Determine context window size dynamically either all or k tokens
            if self.use_all_tokens:
                context_window = len(current_ids)
                print("Entered all tokens, context window: ", context_window)
            else:
                context_window = min(self.last_k_tokens_to_consider, len(current_ids))
                print("Entered last k tokens, context window: ", context_window)

            
            # Get top k token candidates
            _, top_k_indices = torch.topk(
                scores[batch_idx], 
                k=min(self.top_k_logits, scores.shape[-1])
            )
            
            # Check each candidate token for hallucinations
            for i, token_id in enumerate(top_k_indices):
                token_id_item = token_id.item()
                
                # Check if this token would create a hallucination
                if self.hallucination_detector.check_hallucination(
                    current_text, 
                    token_id_item, 
                    context_window
                ):
                    # Apply penalty to prevent this token
                    scores[batch_idx][token_id_item] = self.penalty_value
                    self.modifications_count += 1
                    
                    token_str = self.hallucination_detector.tokenizer.decode([token_id_item])
                    print(f"----------Modified logit for token: {token_str}----------")
                    continue
                
                # Break after finding the first non-hallucinating token (greedy approach)
                break
        
        return scores