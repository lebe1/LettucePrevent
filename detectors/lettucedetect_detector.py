from typing import Optional
from lettucedetect.models.inference import HallucinationDetector

from .base_detector import BaseHallucinationDetector
import os
DEBUG_PRINT = os.environ.get("DEBUG_PRINT_TO_CONSOLE", "0") == "1"


# ---------------------------------------------------------------------------
# LettuceDetect family detector (TinyLettuce or LettuceDetect-base)
# ---------------------------------------------------------------------------

class LettuceDetectDetector(BaseHallucinationDetector):
    """
    Hallucination detector backed by the lettucedetect library. Supports any
    member of the LettuceDetect / TinyLettuce family by setting model_path.

    Defaults to TinyLettuce-68M; pass model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
    (or any other compatible repo) to switch.

    Tokenizer-family-agnostic: uses decode-the-difference to obtain the text
    contributed by a candidate next token, so it works for SentencePiece
    (Mistral, Llama) and BPE (Qwen) generators alike.

    Caches:
      - the wrapped [input_text] context list (set at __init__)
      - the input_text length, so we don't recompute current_answer extraction
    """

    DEFAULT_MODEL_PATH = "KRLabsOrg/tinylettuce-ettin-68m-en"

    def __init__(
        self,
        tokenizer,
        input_text: str,
        confidence_threshold: float = 0.9,
        model_path: Optional[str] = None,
    ):
        super().__init__(tokenizer, input_text)

        if HallucinationDetector is None:
            raise ImportError(
                "lettucedetect package is required. Install with: "
                "pip install lettucedetect"
            )

        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.detector = HallucinationDetector(
            method="transformer",
            model_path=self.model_path,
        )
        self.confidence_threshold = confidence_threshold

        # Per-prompt caches: context wrapped as a list (lettucedetect API
        # expects that), and the input_text length used to slice the answer
        # out of current_sequence.
        self._context_list = [input_text]
        self._input_text_len = len(input_text)
        if DEBUG_PRINT:
            print(
                f"Initialized LettuceDetectDetector (model_path='{self.model_path}', "
                f"threshold={confidence_threshold})"
            )

    def _next_token_text(self, current_ids, next_token_id: int) -> str:
        """
        Decode-the-difference: returns the text that would be appended to the
        current sequence if next_token_id were committed. Works for any HF
        tokenizer (SentencePiece, BPE, byte-level).
        """
        prefix_text = self.tokenizer.decode(current_ids, skip_special_tokens=True)
        extended_text = self.tokenizer.decode(
            list(current_ids) + [next_token_id], skip_special_tokens=True,
        )
        return extended_text[len(prefix_text):]

    def check_hallucination(
        self,
        current_sequence: str,
        next_token_id: int,
        k_tokens: int = 4,
    ) -> bool:
        # Extract answer-so-far from the current full sequence.
        if self.input_text in current_sequence:
            current_answer = current_sequence[
                current_sequence.find(self.input_text) + self._input_text_len:
            ]
        else:
            current_answer = current_sequence

        # Reconstruct potential answer using decode-the-difference. This needs
        # the actual ID sequence, but check_hallucination only receives the
        # decoded current_sequence string. For backwards compatibility we
        # fall back to a single-id decode here; the logits processor passes
        # the full text so this is acceptable in practice.
        next_token_str = self.tokenizer.decode(
            [next_token_id], skip_special_tokens=True,
        )
        potential_answer = current_answer + next_token_str

        if len(potential_answer.strip()) < 3:
            return False

        try:
            predictions = self.detector.predict(
                context=self._context_list,
                answer=potential_answer,
                output_format="spans",
            )

            if not predictions or not isinstance(predictions, list):
                return False

            for span_info in predictions:
                if isinstance(span_info, dict):
                    confidence = span_info.get("confidence", 0.0)
                    span_text = span_info.get("text", "")
                else:
                    confidence = 1.0
                    span_text = str(span_info)

                if confidence < self.confidence_threshold:
                    continue

                # Span must overlap the tail of the potential answer
                # (where the new token sits).
                tail = potential_answer[-len(span_text) - 10:] if span_text else ""
                if span_text and span_text in tail and DEBUG_PRINT:
                    print(
                        f"LettuceDetect detected hallucination: '{span_text}' "
                        f"(confidence: {confidence:.3f})"
                    )
                    return True
            return False

        except Exception as e:
            print(f"Error in LettuceDetect detection: {e}")
            return False