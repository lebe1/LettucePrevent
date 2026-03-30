from .base_detector import BaseHallucinationDetector
from .number_detector import NumberHallucinationDetector
from .tiny_lettuce_detector import TinyLettuceDetector
from .ettin_decoder_detector import EttinDecoderDetector


# -------------------------- Detector Factory ------------------
class DetectorFactory:
    """
    Factory class for creating different types of hallucination detectors.
    """

    @staticmethod
    def create_detector(
        detector_type: str,
        tokenizer,
        input_text: str,
        **kwargs,
    ) -> BaseHallucinationDetector:
        """
        Create a hallucination detector of the specified type.

        Args:
            detector_type       : 'tinylettuce', 'ettin', 'number', or 'none'
            tokenizer           : Tokenizer instance of the generation model
            input_text          : Full input context passed to the generation model
            **kwargs            : Additional arguments forwarded to the detector:
                                  - confidence_threshold (float) for tinylettuce / ettin
                                  - model_path (str) required for ettin

        Returns:
            BaseHallucinationDetector instance, or None for 'none'
        """
        if detector_type.lower() == 'tinylettuce':
            confidence_threshold = kwargs.get('confidence_threshold', 0.9)
            return TinyLettuceDetector(
                tokenizer,
                input_text,
                confidence_threshold=confidence_threshold,
            )

        elif detector_type.lower() == 'ettin':
            confidence_threshold = kwargs.get('confidence_threshold', 0.9)
            model_path = kwargs.get('model_path')
            if model_path is None:
                raise ValueError(
                    "EttinDecoderDetector requires 'model_path' to be passed "
                    "as a keyword argument to create_detector()."
                )
            return EttinDecoderDetector(
                tokenizer,
                input_text,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
            )

        elif detector_type.lower() == 'number':
            return NumberHallucinationDetector(tokenizer, input_text)

        elif detector_type.lower() == 'none':
            return None

        else:
            raise ValueError(
                f"Unknown detector type: '{detector_type}'. "
                f"Available: 'tinylettuce', 'ettin', 'number', 'none'"
            )