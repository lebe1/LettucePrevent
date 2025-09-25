from abc import ABC, abstractmethod


# -------------------------- Abstract Detector Base Class ------------------

class BaseHallucinationDetector(ABC):
    """
    Abstract base class for hallucination detectors.
    All detectors must implement the check_hallucination method.
    """
    
    def __init__(self, tokenizer, input_text: str):
        self.tokenizer = tokenizer
        self.input_text = input_text
    
    @abstractmethod
    def check_hallucination(self, current_sequence: str, next_token_id: int, k_tokens: int = 4) -> bool:
        """
        Check if the next token would create a hallucination.
        
        Args:
            current_sequence: Current sequence of generated tokens
            next_token_id: ID of the token being considered
            k_tokens: Number of recent tokens to consider
        
        Returns:
            True if hallucination detected, False if allowed
        """
        pass