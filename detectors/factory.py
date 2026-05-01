from .base_detector import BaseHallucinationDetector
from .number_detector import NumberHallucinationDetector
from .lettucedetect_detector import LettuceDetectDetector
from .lettuceprevent_detector import LettucePreventDetector


VALID_DETECTOR_TYPES = {
    "lettucedetect",
    "lettuceprevent",
    "number",
    "baseline-run-numbers",
    "baseline-run-facts",
}


class DetectorFactory:
    """Factory class for creating hallucination detectors."""

    @staticmethod
    def create_detector(
        detector_type: str,
        tokenizer,
        input_text: str,
        **kwargs,
    ):
        """
        Create a hallucination detector of the specified type.

        Args:
            detector_type:
                'lettucedetect'         -> any model from KRLabsOrg lettucedetect family
                                           (default: tinylettuce-ettin-68m-en)
                'lettuceprevent'        -> custom Ettin-decoder classifier
                'number'                -> rule-based number detector
                                           (uses local summary-only dataset)
                'baseline-run-numbers'  -> no detector, local summary dataset
                'baseline-run-facts'    -> no detector, HF RAGTruth dataset
            tokenizer    : generator's tokenizer
            input_text   : full input/context for this prompt
            **kwargs     : forwarded; relevant keys are
                           - confidence_threshold (float)
                           - model_path           (str)

        Returns:
            BaseHallucinationDetector or None for the two baselines.

        Raises:
            ValueError on unknown detector_type.
        """
        det = detector_type.lower()

        if det == "lettucedetect":
            return LettuceDetectDetector(
                tokenizer=tokenizer,
                input_text=input_text,
                confidence_threshold=kwargs.get("confidence_threshold", 0.9),
                model_path=kwargs.get("model_path"),
            )

        if det == "lettuceprevent":
            return LettucePreventDetector(
                tokenizer=tokenizer,
                input_text=input_text,
                confidence_threshold=kwargs.get("confidence_threshold", 0.9),
                model_path=kwargs.get("model_path"),
            )

        if det == "number":
            return NumberHallucinationDetector(tokenizer, input_text)

        if det in ("baseline-run-numbers", "baseline-run-facts"):
            return None

        raise ValueError(
            f"Unknown detector type: '{detector_type}'. "
            f"Available: {sorted(VALID_DETECTOR_TYPES)}"
        )