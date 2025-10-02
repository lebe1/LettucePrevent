from lettucedetect.models.inference import HallucinationDetector
from .base_detector import BaseHallucinationDetector

# -------------------------- TinyLettuce Detector ------------------

class TinyLettuceDetector(BaseHallucinationDetector):
    """
    Hallucination detector using TinyLettuce model.
    """
    
    def __init__(self, tokenizer, input_text: str, confidence_threshold: float = 0.9):
        super().__init__(tokenizer, input_text)
        
        if HallucinationDetector is None:
            raise ImportError("lettucedetect package is required. Install with: pip install lettucedetect")
        
        self.detector = HallucinationDetector(
            method="transformer",
            model_path="KRLabsOrg/tinylettuce-ettin-68m-en",
        )
        self.confidence_threshold = confidence_threshold
        print(f"Initialized TinyLettuce detector with threshold: {confidence_threshold}")

    def check_hallucination(self, current_sequence: str, next_token_id: int, k_tokens: int = 4) -> bool:
        """
        Check if the next token would create a hallucination using TinyLettuce.
        
        Args:
            current_sequence: Current sequence of generated tokens
            next_token_id: ID of the token being considered
            k_tokens: Number of recent tokens to consider (not used in TinyLettuce)
        
        Returns:
            True if hallucination detected, False if allowed
        """
        next_token_str = self.tokenizer.decode([next_token_id]).strip()
        
        # Create the potential answer by adding the next token
        # Extract the generated part (everything after the input context)
        if self.input_text in current_sequence:
            # Find where the input ends and generation begins
            context_end = current_sequence.find(self.input_text) + len(self.input_text)
            current_answer = current_sequence[context_end:].strip()
        else:
            # Fallback: assume the entire sequence is the answer
            current_answer = current_sequence.strip()
        
        # Create the potential answer with the next token
        potential_answer = current_answer + next_token_str
        
        # Skip check for very short answers or single tokens
        if len(potential_answer.strip()) < 3:
            return False
        
        try:
            # Get predictions from TinyLettuce
            predictions = self.detector.predict(
                context=self.input_text, 
                answer=potential_answer, 
                output_format="spans"
            )
            
            # Check if any span indicates hallucination above threshold
            if predictions and isinstance(predictions, list):
                for span_info in predictions:
                    # Extract confidence score (assuming it's in the span_info structure)
                    # You might need to adjust this based on the actual TinyLettuce output format
                    if isinstance(span_info, dict):
                        confidence = span_info.get('confidence', 0.0)
                        span_text = span_info.get('text', '')
                    else:
                        # If predictions is just a list of spans without confidence
                        confidence = 1.0  # Assume high confidence if span is detected
                        span_text = str(span_info)
                    
                    # Check if confidence is above threshold and span includes the latest token
                    if confidence >= self.confidence_threshold:
                        # Check if the span overlaps with the end of the potential answer
                        # (where the new token would be added)
                        if span_text in potential_answer[-len(span_text)-10:]:  # Check near the end
                            print(f"TinyLettuce detected hallucination: '{span_text}' (confidence: {confidence:.3f})")
                            return True
            
            return False
            
        except Exception as e:
            print(f"Error in TinyLettuce detection: {e}")
            return False  # Don't block on errors