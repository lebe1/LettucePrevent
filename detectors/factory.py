from .base_detector import BaseHallucinationDetector
from .number_detector import NumberHallucinationDetector
from .tiny_lettuce_detector import TinyLettuceDetector

# -------------------------- Detector Factory ------------------

class DetectorFactory:
    """
    Factory class for creating different types of hallucination detectors.
    """
    
    @staticmethod
    def create_detector(detector_type: str, tokenizer, input_text: str, **kwargs) -> BaseHallucinationDetector:
        """
        Create a hallucination detector of the specified type.
        
        Args:
            detector_type: Type of detector ('tinylettuce', 'number')
            tokenizer: Tokenizer instance
            input_text: Input text context
            **kwargs: Additional arguments for specific detectors
            
        Returns:
            BaseHallucinationDetector instance
        """
        if detector_type.lower() == 'tinylettuce':
            confidence_threshold = kwargs.get('confidence_threshold', 0.9)
            return TinyLettuceDetector(tokenizer, input_text, confidence_threshold=confidence_threshold)
        
        elif detector_type.lower() == 'number':
            return NumberHallucinationDetector(tokenizer, input_text)
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Available: 'tinylettuce', 'number'")