"""
Hallucination Detection Benchmark — Left-Encoder Evaluation

Evaluates hallucination detection models on their ability to detect hallucinations
in incrementally generated answers (simulating token-by-token LLM generation).

Supports:
- Multiple hallucination detection models (pluggable via abstract base class)
- Multiple triggering strategies: every token, every N tokens, smart POS-based
- Optional LLM tokenizer integration (Qwen, Llama, Mistral)
- Per-step and aggregate evaluation
- Configurable dataset size (first N samples or full dataset)

Metrics:
- Token-level Precision, Recall, F1
- Runtime per sample and total
- Per-step logging with carry-forward for smart trigger
"""

import json
import time
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


# ============================================================================
# Data Structures
# ============================================================================

class TriggerStrategy(Enum):
    EVERY_TOKEN = "every_token"
    EVERY_N = "every_n"
    SMART_POS = "smart_pos"


@dataclass
class TokenInfo:
    """Represents a single token with its metadata."""
    index: int
    text: str
    char_start: int  # character start in the reference string
    char_end: int    # character end in the reference string
    pos_tag: str = ""  # POS tag (filled when using spaCy)


@dataclass
class GroundTruthLabel:
    """A ground-truth hallucination span."""
    start: int
    end: int
    label: str = "hallucination"


@dataclass
class TokenStepResult:
    """Result for a single token at an evaluation step."""
    step: int
    token: str
    token_char_start: int
    token_char_end: int
    ground_truth: bool        # True = hallucinated
    predicted: bool           # True = model says hallucinated
    confidence: float         # model confidence for this token
    was_triggered: bool       # whether detection ran at this step
    cumulative_precision: float = 0.0
    cumulative_recall: float = 0.0
    cumulative_f1: float = 0.0


@dataclass
class SampleResult:
    """Full result for a single dataset sample."""
    sample_index: int
    prompt: str
    answer: str
    token_steps: List[TokenStepResult] = field(default_factory=list)
    aggregate_precision: float = 0.0
    aggregate_recall: float = 0.0
    aggregate_f1: float = 0.0
    runtime_seconds: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregate result across all samples."""
    model_name: str
    trigger_strategy: str
    step_size: int
    tokenizer_name: Optional[str]
    num_samples: int
    total_precision: float = 0.0
    total_recall: float = 0.0
    total_f1: float = 0.0
    total_runtime_seconds: float = 0.0
    sample_results: List[SampleResult] = field(default_factory=list)


# ============================================================================
# Abstract Base Class for Hallucination Detectors
# ============================================================================

class HallucinationDetectorBase(ABC):
    """
    Abstract base class for hallucination detection models.
    Implement this to plug in any hallucination detection model.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the model name for logging."""
        pass

    @abstractmethod
    def predict(
        self,
        context: List[str],
        question: str,
        answer: str,
    ) -> List[Dict[str, Any]]:
        """
        Run hallucination detection on the given answer.

        Args:
            context: List of context strings.
            question: The question/prompt.
            answer: The (partial) answer to evaluate.

        Returns:
            List of span dicts with keys: 'start', 'end', 'confidence', 'text'
            where start/end are character offsets in the answer string.
        """
        pass


# ============================================================================
# LettuceDetect Wrapper
# ============================================================================

class LettuceDetectWrapper(HallucinationDetectorBase):
    """Wrapper for the LettuceDetect hallucination detection model."""

    def __init__(self, model_path: str = "KRLabsOrg/tinylettuce-ettin-68m-en"):
        from lettucedetect.models.inference import HallucinationDetector
        self.model_path = model_path
        self.detector = HallucinationDetector(
            method="transformer",
            model_path=model_path,
        )

    def get_name(self) -> str:
        return f"LettuceDetect({self.model_path})"

    def predict(
        self,
        context: List[str],
        question: str,
        answer: str,
    ) -> List[Dict[str, Any]]:
        predictions = self.detector.predict(
            context=context,
            question=question,
            answer=answer,
            output_format="spans",
        )
        return predictions


# ============================================================================
# Tokenizer Abstraction
# ============================================================================

class TokenizerBase(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def tokenize_with_offsets(self, text: str) -> List[TokenInfo]:
        """
        Tokenize text and return tokens with character offsets
        in the detokenized string.
        """
        pass


class SpacyTokenizer(TokenizerBase):
    """
    Default tokenizer using spaCy. Provides POS tags for smart triggering.
    Character offsets are relative to the original text (no detokenization mismatch).
    """

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        try:
            self.nlp = spacy.load(model)
        except OSError:
            from spacy.cli import download
            download(model)
            self.nlp = spacy.load(model)

    def get_name(self) -> str:
        return "spaCy(en_core_web_sm)"

    def tokenize_with_offsets(self, text: str) -> List[TokenInfo]:
        doc = self.nlp(text)
        tokens = []
        for i, tok in enumerate(doc):
            tokens.append(TokenInfo(
                index=i,
                text=tok.text,
                char_start=tok.idx,
                char_end=tok.idx + len(tok.text),
                pos_tag=tok.pos_,
            ))
        return tokens


class LLMTokenizerWrapper(TokenizerBase):
    """
    Wrapper for HuggingFace LLM tokenizers (Qwen, Llama, Mistral).
    Handles the tokenize → detokenize flow and offset mapping.
    """

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # We also need spaCy for POS tags in smart trigger mode
        self._spacy_tokenizer = None

    def get_name(self) -> str:
        return f"LLMTokenizer({self.model_name})"

    def _get_spacy(self):
        if self._spacy_tokenizer is None:
            self._spacy_tokenizer = SpacyTokenizer()
        return self._spacy_tokenizer

    def tokenize_with_offsets(self, text: str) -> List[TokenInfo]:
        """
        Tokenize using the LLM tokenizer, then compute character offsets
        in the detokenized string. Also map each token back to the original
        text for ground-truth comparison.
        """
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = []
        current_char = 0

        for i, token_id in enumerate(encoded):
            # Detokenize from start up to and including this token
            prefix = self.tokenizer.decode(encoded[: i + 1], skip_special_tokens=True)
            # Detokenize from start up to (but not including) this token
            prev_prefix = self.tokenizer.decode(encoded[:i], skip_special_tokens=True) if i > 0 else ""

            char_start = len(prev_prefix)
            char_end = len(prefix)
            token_text = prefix[char_start:]

            tokens.append(TokenInfo(
                index=i,
                text=token_text,
                char_start=char_start,
                char_end=char_end,
                pos_tag="",  # will be filled by POS tagger if needed
            ))

        return tokens

    def add_pos_tags(self, tokens: List[TokenInfo], original_text: str) -> List[TokenInfo]:
        """Add POS tags to LLM tokens using spaCy alignment."""
        spacy_tokenizer = self._get_spacy()
        spacy_tokens = spacy_tokenizer.tokenize_with_offsets(original_text)

        # Build a character-level POS map from spaCy
        pos_map = {}
        for st in spacy_tokens:
            for c in range(st.char_start, st.char_end):
                pos_map[c] = st.pos_tag

        # Assign POS to each LLM token based on majority POS of its characters
        for token in tokens:
            pos_counts: Dict[str, int] = {}
            for c in range(token.char_start, min(token.char_end, len(original_text))):
                pos = pos_map.get(c, "X")
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            if pos_counts:
                token.pos_tag = max(pos_counts, key=pos_counts.get)
            else:
                token.pos_tag = "X"

        return tokens

    def get_detokenized_text(self, text: str) -> str:
        """Tokenize and detokenize to get the round-tripped string."""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.decode(encoded, skip_special_tokens=True)


# ============================================================================
# Offset Mapping (for LLM tokenizer mode)
# ============================================================================

class OffsetMapper:
    """
    Maps character offsets between the original answer and the detokenized answer.
    Used when an LLM tokenizer is active to correctly align ground-truth labels
    with detector predictions.
    """

    def __init__(self, original: str, detokenized: str):
        self.original = original
        self.detokenized = detokenized
        self._build_mapping()

    def _build_mapping(self):
        """
        Build a character-level alignment between original and detokenized strings
        using a simple longest-common-subsequence approach.
        """
        # For most tokenizers, the detokenized string is very close to the original.
        # We use a simple approach: align by matching characters sequentially.
        self.orig_to_detok = {}
        self.detok_to_orig = {}

        i, j = 0, 0
        while i < len(self.original) and j < len(self.detokenized):
            if self.original[i] == self.detokenized[j]:
                self.orig_to_detok[i] = j
                self.detok_to_orig[j] = i
                i += 1
                j += 1
            elif self.original[i] in (' ', '\n', '\t'):
                i += 1
            elif self.detokenized[j] in (' ', '\n', '\t'):
                j += 1
            else:
                # Skip both on mismatch
                i += 1
                j += 1

    def detok_range_to_orig(self, start: int, end: int) -> Tuple[int, int]:
        """Map a character range in detokenized text to original text."""
        orig_start = self.detok_to_orig.get(start)
        orig_end = self.detok_to_orig.get(end - 1)

        if orig_start is None:
            # Find nearest mapped character
            for s in range(start, end):
                if s in self.detok_to_orig:
                    orig_start = self.detok_to_orig[s]
                    break
            if orig_start is None:
                orig_start = start

        if orig_end is None:
            for e in range(end - 1, start - 1, -1):
                if e in self.detok_to_orig:
                    orig_end = self.detok_to_orig[e]
                    break
            if orig_end is None:
                orig_end = end - 1

        return orig_start, orig_end + 1

    def check_divergence(self) -> float:
        """Return the fraction of characters that could not be aligned."""
        if len(self.detokenized) == 0:
            return 0.0
        mapped = len(self.detok_to_orig)
        return 1.0 - (mapped / len(self.detokenized))


# ============================================================================
# Ground Truth Mapping
# ============================================================================

def is_token_hallucinated(token: TokenInfo, labels: List[GroundTruthLabel]) -> bool:
    """Check if a token's character range falls within any ground-truth label span."""
    for label in labels:
        # Token overlaps with label if token_start < label_end AND token_end > label_start
        if token.char_start < label.end and token.char_end > label.start:
            return True
    return False


def map_predictions_to_tokens(
    predictions: List[Dict[str, Any]],
    tokens: List[TokenInfo],
    confidence_threshold: float = 0.90,
) -> Dict[int, Tuple[bool, float]]:
    """
    Map detector span predictions back to individual tokens.

    Returns:
        Dict mapping token index -> (is_hallucinated, confidence)
    """
    result = {}
    for token in tokens:
        max_conf = 0.0
        is_hall = False
        for pred in predictions:
            pred_start = pred["start"]
            pred_end = pred["end"]
            pred_conf = pred["confidence"]
            # Check overlap
            if token.char_start < pred_end and token.char_end > pred_start:
                if pred_conf > max_conf:
                    max_conf = pred_conf
                    is_hall = pred_conf >= confidence_threshold
        result[token.index] = (is_hall, max_conf)
    return result


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(
    ground_truths: List[bool],
    predictions: List[bool],
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 from lists of booleans."""
    tp = sum(1 for g, p in zip(ground_truths, predictions) if g and p)
    fp = sum(1 for g, p in zip(ground_truths, predictions) if not g and p)
    fn = sum(1 for g, p in zip(ground_truths, predictions) if g and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


# ============================================================================
# Trigger Logic
# ============================================================================

def should_trigger(
    token: TokenInfo,
    step: int,
    strategy: TriggerStrategy,
    step_size: int = 1,
    pos_triggers: Optional[set] = None,
) -> bool:
    """Determine whether to run detection at this step."""
    if strategy == TriggerStrategy.EVERY_TOKEN:
        return True
    elif strategy == TriggerStrategy.EVERY_N:
        return (step % step_size == 0) or (step == 0)
    elif strategy == TriggerStrategy.SMART_POS:
        if pos_triggers is None:
            pos_triggers = {"NOUN", "VERB", "PROPN", "AUX"}
        return token.pos_tag in pos_triggers
    return True


# ============================================================================
# Core Evaluation Engine
# ============================================================================

class EvaluationEngine:
    """
    Main engine that runs the incremental hallucination detection benchmark.
    """

    def __init__(
        self,
        detector: HallucinationDetectorBase,
        tokenizer: Optional[TokenizerBase] = None,
        llm_tokenizer: Optional[LLMTokenizerWrapper] = None,
        trigger_strategy: TriggerStrategy = TriggerStrategy.EVERY_TOKEN,
        step_size: int = 1,
        confidence_threshold: float = 0.90,
        pos_triggers: Optional[set] = None,
    ):
        self.detector = detector
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.llm_tokenizer = llm_tokenizer
        self.trigger_strategy = trigger_strategy
        self.step_size = step_size
        self.confidence_threshold = confidence_threshold
        self.pos_triggers = pos_triggers or {"NOUN", "VERB", "PROPN", "AUX"}

    def evaluate_sample(
        self,
        prompt: str,
        context: str,
        answer: str,
        labels: List[Dict],
        sample_index: int = 0,
    ) -> SampleResult:
        """
        Evaluate a single sample by incrementally feeding tokens to the detector.
        """
        # Parse ground-truth labels
        gt_labels = [
            GroundTruthLabel(start=l["start"], end=l["end"], label=l.get("label", "hallucination"))
            for l in labels
        ]

        # Determine tokenization mode
        use_llm_tokenizer = self.llm_tokenizer is not None

        if use_llm_tokenizer:
            tokens = self.llm_tokenizer.tokenize_with_offsets(answer)
            # Add POS tags for smart trigger
            if self.trigger_strategy == TriggerStrategy.SMART_POS:
                tokens = self.llm_tokenizer.add_pos_tags(tokens, answer)
            # Compute offset mapping
            detokenized_full = self.llm_tokenizer.get_detokenized_text(answer)
            offset_mapper = OffsetMapper(answer, detokenized_full)
            divergence = offset_mapper.check_divergence()
            if divergence > 0.05:
                print(
                    f"  [WARNING] Sample {sample_index}: detokenization divergence "
                    f"= {divergence:.2%} for tokenizer {self.llm_tokenizer.get_name()}"
                )
        else:
            tokens = self.tokenizer.tokenize_with_offsets(answer)
            offset_mapper = None

        # Evaluate token by token
        token_steps: List[TokenStepResult] = []
        last_prediction: Dict[int, Tuple[bool, float]] = {}
        cumulative_gt = []
        cumulative_pred = []
        sample_start_time = time.time()

        for step, token in enumerate(tokens):
            triggered = should_trigger(
                token, step, self.trigger_strategy, self.step_size, self.pos_triggers
            )

            if triggered:
                # Build the prefix text
                if use_llm_tokenizer:
                    encoded = self.llm_tokenizer.tokenizer.encode(
                        answer, add_special_tokens=False
                    )
                    prefix_text = self.llm_tokenizer.tokenizer.decode(
                        encoded[: step + 1], skip_special_tokens=True
                    )
                else:
                    prefix_text = answer[: token.char_end]

                # Run detection
                try:
                    predictions = self.detector.predict(
                        context=[context],
                        question=prompt,
                        answer=prefix_text,
                    )
                except Exception as e:
                    print(f"  [ERROR] Detection failed at step {step}: {e}")
                    predictions = []

                # Map predictions to tokens seen so far
                tokens_so_far = tokens[: step + 1]
                pred_map = map_predictions_to_tokens(
                    predictions, tokens_so_far, self.confidence_threshold
                )
                last_prediction = pred_map

            # Get prediction for current token (carry-forward if not triggered)
            if token.index in last_prediction:
                is_pred_hall, pred_conf = last_prediction[token.index]
            else:
                is_pred_hall, pred_conf = False, 0.0

            # Get ground truth for current token
            if use_llm_tokenizer and offset_mapper:
                orig_start, orig_end = offset_mapper.detok_range_to_orig(
                    token.char_start, token.char_end
                )
                orig_token = TokenInfo(
                    index=token.index,
                    text=token.text,
                    char_start=orig_start,
                    char_end=orig_end,
                )
                is_gt_hall = is_token_hallucinated(orig_token, gt_labels)
            else:
                is_gt_hall = is_token_hallucinated(token, gt_labels)

            # Cumulative metrics
            cumulative_gt.append(is_gt_hall)
            cumulative_pred.append(is_pred_hall)
            cum_prec, cum_rec, cum_f1 = compute_metrics(cumulative_gt, cumulative_pred)

            token_steps.append(TokenStepResult(
                step=step,
                token=token.text,
                token_char_start=token.char_start,
                token_char_end=token.char_end,
                ground_truth=is_gt_hall,
                predicted=is_pred_hall,
                confidence=pred_conf,
                was_triggered=triggered,
                cumulative_precision=cum_prec,
                cumulative_recall=cum_rec,
                cumulative_f1=cum_f1,
            ))

        sample_runtime = time.time() - sample_start_time

        # Aggregate metrics for this sample
        all_gt = [ts.ground_truth for ts in token_steps]
        all_pred = [ts.predicted for ts in token_steps]
        agg_prec, agg_rec, agg_f1 = compute_metrics(all_gt, all_pred)

        return SampleResult(
            sample_index=sample_index,
            prompt=prompt,
            answer=answer,
            token_steps=token_steps,
            aggregate_precision=agg_prec,
            aggregate_recall=agg_rec,
            aggregate_f1=agg_f1,
            runtime_seconds=sample_runtime,
        )


# ============================================================================
# Dataset Loading
# ============================================================================

def load_ragtruth_dataset(filepath: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load the RAGTruth dataset from a JSON file.

    Args:
        filepath: Path to the JSON file.
        max_samples: If set, only load the first N samples. None = load all.

    Returns:
        List of sample dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    return data


def extract_context_from_prompt(prompt: str) -> Tuple[str, str]:
    """
    Extract the context and question from a RAGTruth prompt.
    The prompt typically contains instruction + context.
    Returns (context, question).
    """
    # For summary tasks, the context is the news article in the prompt
    # For data2txt tasks, the context is the structured data
    # We treat the full prompt as both context and question for simplicity
    return prompt, prompt


# ============================================================================
# Result Export
# ============================================================================

def export_per_step_csv(sample_results: List[SampleResult], filepath: str):
    """Export per-step results to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index", "step", "token", "token_char_start", "token_char_end",
            "ground_truth", "predicted", "confidence", "was_triggered",
            "cumulative_precision", "cumulative_recall", "cumulative_f1",
        ])
        for sr in sample_results:
            for ts in sr.token_steps:
                writer.writerow([
                    sr.sample_index, ts.step, ts.token, ts.token_char_start,
                    ts.token_char_end, ts.ground_truth, ts.predicted,
                    f"{ts.confidence:.6f}", ts.was_triggered,
                    f"{ts.cumulative_precision:.4f}",
                    f"{ts.cumulative_recall:.4f}",
                    f"{ts.cumulative_f1:.4f}",
                ])


def export_aggregate_csv(benchmark_result: BenchmarkResult, filepath: str):
    """Export aggregate results to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "trigger_strategy", "step_size", "tokenizer",
            "num_samples", "precision", "recall", "f1", "total_runtime_seconds",
        ])
        writer.writerow([
            benchmark_result.model_name,
            benchmark_result.trigger_strategy,
            benchmark_result.step_size,
            benchmark_result.tokenizer_name or "spaCy",
            benchmark_result.num_samples,
            f"{benchmark_result.total_precision:.4f}",
            f"{benchmark_result.total_recall:.4f}",
            f"{benchmark_result.total_f1:.4f}",
            f"{benchmark_result.total_runtime_seconds:.2f}",
        ])


def export_results_json(benchmark_result: BenchmarkResult, filepath: str):
    """Export full results to JSON."""
    output = {
        "model_name": benchmark_result.model_name,
        "trigger_strategy": benchmark_result.trigger_strategy,
        "step_size": benchmark_result.step_size,
        "tokenizer_name": benchmark_result.tokenizer_name,
        "num_samples": benchmark_result.num_samples,
        "total_precision": benchmark_result.total_precision,
        "total_recall": benchmark_result.total_recall,
        "total_f1": benchmark_result.total_f1,
        "total_runtime_seconds": benchmark_result.total_runtime_seconds,
        "samples": [],
    }
    for sr in benchmark_result.sample_results:
        sample_out = {
            "sample_index": sr.sample_index,
            "aggregate_precision": sr.aggregate_precision,
            "aggregate_recall": sr.aggregate_recall,
            "aggregate_f1": sr.aggregate_f1,
            "runtime_seconds": sr.runtime_seconds,
            "token_steps": [asdict(ts) for ts in sr.token_steps],
        }
        output["samples"].append(sample_out)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_benchmark(
    dataset_path: str,
    detector: HallucinationDetectorBase,
    trigger_strategy: TriggerStrategy = TriggerStrategy.EVERY_TOKEN,
    step_size: int = 1,
    max_samples: Optional[int] = None,
    llm_tokenizer_name: Optional[str] = None,
    confidence_threshold: float = 0.90,
    output_prefix: str = "benchmark",
) -> BenchmarkResult:
    """
    Run the full benchmark.

    Args:
        dataset_path: Path to the RAGTruth JSON file.
        detector: The hallucination detection model to evaluate.
        trigger_strategy: Triggering strategy to use.
        step_size: Step size for EVERY_N strategy.
        max_samples: Number of samples to evaluate (None = all).
        llm_tokenizer_name: Optional HuggingFace model name for LLM tokenizer.
        confidence_threshold: Confidence threshold for hallucination prediction.
        output_prefix: Prefix for output files.

    Returns:
        BenchmarkResult with all results.
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_ragtruth_dataset(dataset_path, max_samples)
    print(f"Loaded {len(data)} samples.")

    # Initialize tokenizer
    llm_tokenizer = None
    if llm_tokenizer_name:
        print(f"Loading LLM tokenizer: {llm_tokenizer_name}...")
        llm_tokenizer = LLMTokenizerWrapper(llm_tokenizer_name)

    # Initialize engine
    engine = EvaluationEngine(
        detector=detector,
        llm_tokenizer=llm_tokenizer,
        trigger_strategy=trigger_strategy,
        step_size=step_size,
        confidence_threshold=confidence_threshold,
    )

    # Run evaluation
    all_sample_results: List[SampleResult] = []
    all_gt = []
    all_pred = []
    total_start = time.time()

    for i, sample in enumerate(data):
        print(f"  Evaluating sample {i + 1}/{len(data)}...")
        context, question = extract_context_from_prompt(sample["prompt"])
        answer = sample["answer"]
        labels = sample.get("labels", [])

        result = engine.evaluate_sample(
            prompt=question,
            context=context,
            answer=answer,
            labels=labels,
            sample_index=i,
        )
        all_sample_results.append(result)

        # Collect all token-level gt/pred for global metrics
        for ts in result.token_steps:
            all_gt.append(ts.ground_truth)
            all_pred.append(ts.predicted)

        print(
            f"    P={result.aggregate_precision:.4f} "
            f"R={result.aggregate_recall:.4f} "
            f"F1={result.aggregate_f1:.4f} "
            f"Runtime={result.runtime_seconds:.2f}s"
        )

    total_runtime = time.time() - total_start
    total_prec, total_rec, total_f1 = compute_metrics(all_gt, all_pred)

    benchmark_result = BenchmarkResult(
        model_name=detector.get_name(),
        trigger_strategy=trigger_strategy.value,
        step_size=step_size,
        tokenizer_name=llm_tokenizer_name,
        num_samples=len(data),
        total_precision=total_prec,
        total_recall=total_rec,
        total_f1=total_f1,
        total_runtime_seconds=total_runtime,
        sample_results=all_sample_results,
    )

    # Export results
    print(f"\nExporting results...")
    export_per_step_csv(all_sample_results, f"{output_prefix}_per_step.csv")
    export_aggregate_csv(benchmark_result, f"{output_prefix}_aggregate.csv")
    export_results_json(benchmark_result, f"{output_prefix}_full_results.json")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model:             {detector.get_name()}")
    print(f"Trigger Strategy:  {trigger_strategy.value}")
    print(f"Step Size:         {step_size}")
    print(f"Tokenizer:         {llm_tokenizer_name or 'spaCy (default)'}")
    print(f"Samples:           {len(data)}")
    print(f"{'=' * 60}")
    print(f"Precision:         {total_prec:.4f}")
    print(f"Recall:            {total_rec:.4f}")
    print(f"F1:                {total_f1:.4f}")
    print(f"Total Runtime:     {total_runtime:.2f}s")
    print(f"{'=' * 60}")

    return benchmark_result


# ============================================================================
# Example Usage / CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hallucination Detection Benchmark — Left-Encoder Evaluation"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to the RAGTruth JSON dataset file."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Number of samples to evaluate (default: all)."
    )
    parser.add_argument(
        "--detector", type=str, default="lettuce",
        choices=["lettuce"],
        help="Hallucination detection model to use."
    )
    parser.add_argument(
        "--detector-model-path", type=str,
        default="KRLabsOrg/tinylettuce-ettin-68m-en",
        help="Model path for the detector."
    )
    parser.add_argument(
        "--trigger", type=str, default="every_token",
        choices=["every_token", "every_n", "smart_pos"],
        help="Triggering strategy."
    )
    parser.add_argument(
        "--step-size", type=int, default=1,
        help="Step size for every_n trigger strategy."
    )
    parser.add_argument(
        "--llm-tokenizer", type=str, default=None,
        help="HuggingFace model name for LLM tokenizer (optional)."
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.90,
        help="Confidence threshold for hallucination prediction."
    )
    parser.add_argument(
        "--output-prefix", type=str, default="benchmark",
        help="Prefix for output files."
    )

    args = parser.parse_args()

    # Initialize detector
    if args.detector == "lettuce":
        detector = LettuceDetectWrapper(model_path=args.detector_model_path)
    else:
        raise ValueError(f"Unknown detector: {args.detector}")

    # Map trigger strategy
    trigger_map = {
        "every_token": TriggerStrategy.EVERY_TOKEN,
        "every_n": TriggerStrategy.EVERY_N,
        "smart_pos": TriggerStrategy.SMART_POS,
    }

    run_benchmark(
        dataset_path=args.dataset,
        detector=detector,
        trigger_strategy=trigger_map[args.trigger],
        step_size=args.step_size,
        max_samples=args.max_samples,
        llm_tokenizer_name=args.llm_tokenizer,
        confidence_threshold=args.confidence_threshold,
        output_prefix=args.output_prefix,
    )
