"""
Hallucination Detection Benchmark — RQ3 Comparison

Runs both TinyLettuce and LettuceDetect-base in a single invocation,
simulating token-by-token incremental generation with a Llama-3 tokenizer.

Logs to W&B (two separate runs) and exports CSV/JSON with:
- Global micro metrics (precision, recall, F1)
- Per-class metrics (class 0 = supported, class 1 = hallucinated)
- Total runtime per model
"""

import json
import time
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

import wandb


# ============================================================================
# Fixed Configuration
# ============================================================================

WANDB_ENTITY  = "lebeccard-technical-university-wien"
WANDB_PROJECT = "hdm-benchmark-rq3"

LLM_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"

MODELS_TO_EVALUATE = [
    {
        "name":       "tinylettuce-ettin-68m-en",
        "model_path": "KRLabsOrg/tinylettuce-ettin-68m-en",
    },
    {
        "name":       "lettucedetect-base-modernbert-en-v1",
        "model_path": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    },
]


# ============================================================================
# Data Structures
# ============================================================================

class TriggerStrategy(Enum):
    EVERY_TOKEN = "every_token"


@dataclass
class TokenInfo:
    index: int
    text: str
    char_start: int
    char_end: int
    pos_tag: str = ""


@dataclass
class GroundTruthLabel:
    start: int
    end: int
    label: str = "hallucination"


@dataclass
class TokenStepResult:
    step: int
    token: str
    token_char_start: int
    token_char_end: int
    ground_truth: bool
    predicted: bool
    confidence: float
    was_triggered: bool
    cumulative_precision: float = 0.0
    cumulative_recall: float = 0.0
    cumulative_f1: float = 0.0


@dataclass
class SampleResult:
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
    model_name: str
    model_path: str
    trigger_strategy: str
    tokenizer_name: str
    num_samples: int

    # Global micro metrics
    precision_micro: float = 0.0
    recall_micro: float = 0.0
    f1_micro: float = 0.0

    # Per-class metrics — class 1 = hallucinated
    precision_class_1: float = 0.0
    recall_class_1: float = 0.0
    f1_binary_class_1: float = 0.0

    # Per-class metrics — class 0 = supported
    precision_class_0: float = 0.0
    recall_class_0: float = 0.0
    f1_binary_class_0: float = 0.0

    total_runtime_seconds: float = 0.0
    sample_results: List[SampleResult] = field(default_factory=list)


# ============================================================================
# Abstract Base Class for Hallucination Detectors
# ============================================================================

class HallucinationDetectorBase(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def predict(
        self,
        context: List[str],
        question: str,
        answer: str,
    ) -> List[Dict[str, Any]]:
        pass


class LettuceDetectWrapper(HallucinationDetectorBase):
    def __init__(self, name: str, model_path: str):
        from lettucedetect.models.inference import HallucinationDetector
        self.name = name
        self.model_path = model_path
        self.detector = HallucinationDetector(
            method="transformer",
            model_path=model_path,
        )

    def get_name(self) -> str:
        return self.name

    def predict(self, context, question, answer):
        return self.detector.predict(
            context=context,
            question=question,
            answer=answer,
            output_format="spans",
        )


# ============================================================================
# Tokenizer
# ============================================================================

class LLMTokenizerWrapper:
    """Wrapper for HuggingFace LLM tokenizers (Llama 3 in this case)."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_name(self) -> str:
        return f"LLMTokenizer({self.model_name})"

    def tokenize_with_offsets(self, text: str) -> List[TokenInfo]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = []

        for i, token_id in enumerate(encoded):
            prefix = self.tokenizer.decode(encoded[: i + 1], skip_special_tokens=True)
            prev_prefix = self.tokenizer.decode(encoded[:i], skip_special_tokens=True) if i > 0 else ""

            char_start = len(prev_prefix)
            char_end = len(prefix)
            token_text = prefix[char_start:]

            tokens.append(TokenInfo(
                index=i,
                text=token_text,
                char_start=char_start,
                char_end=char_end,
            ))

        return tokens

    def get_detokenized_text(self, text: str) -> str:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.decode(encoded, skip_special_tokens=True)


# ============================================================================
# Offset Mapping
# ============================================================================

class OffsetMapper:
    def __init__(self, original: str, detokenized: str):
        self.original = original
        self.detokenized = detokenized
        self._build_mapping()

    def _build_mapping(self):
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
                i += 1
                j += 1

    def detok_range_to_orig(self, start: int, end: int) -> Tuple[int, int]:
        orig_start = self.detok_to_orig.get(start)
        orig_end = self.detok_to_orig.get(end - 1)

        if orig_start is None:
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
        if len(self.detokenized) == 0:
            return 0.0
        mapped = len(self.detok_to_orig)
        return 1.0 - (mapped / len(self.detokenized))


# ============================================================================
# Ground Truth Mapping
# ============================================================================

def is_token_hallucinated(token: TokenInfo, labels: List[GroundTruthLabel]) -> bool:
    for label in labels:
        if token.char_start < label.end and token.char_end > label.start:
            return True
    return False


def map_predictions_to_tokens(
    predictions: List[Dict[str, Any]],
    tokens: List[TokenInfo],
    confidence_threshold: float = 0.90,
) -> Dict[int, Tuple[bool, float]]:
    result = {}
    for token in tokens:
        max_conf = 0.0
        is_hall = False
        for pred in predictions:
            pred_start = pred["start"]
            pred_end = pred["end"]
            pred_conf = pred["confidence"]
            if token.char_start < pred_end and token.char_end > pred_start:
                if pred_conf > max_conf:
                    max_conf = pred_conf
                    is_hall = pred_conf >= confidence_threshold
        result[token.index] = (is_hall, max_conf)
    return result


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics_class(
    ground_truths: List[bool],
    predictions: List[bool],
    positive_class: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 for a specific class.
    If positive_class=True → class 1 (hallucinated)
    If positive_class=False → class 0 (supported)
    """
    tp = sum(1 for g, p in zip(ground_truths, predictions)
             if g == positive_class and p == positive_class)
    fp = sum(1 for g, p in zip(ground_truths, predictions)
             if g != positive_class and p == positive_class)
    fn = sum(1 for g, p in zip(ground_truths, predictions)
             if g == positive_class and p != positive_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_metrics_micro(
    ground_truths: List[bool],
    predictions: List[bool],
) -> Tuple[float, float, float]:
    """Micro-averaged metrics (treating all tokens equally = overall accuracy for binary)."""
    if len(ground_truths) == 0:
        return 0.0, 0.0, 0.0
    correct = sum(1 for g, p in zip(ground_truths, predictions) if g == p)
    acc = correct / len(ground_truths)
    # For binary token-level, micro-precision = micro-recall = micro-F1 = accuracy
    return acc, acc, acc


# ============================================================================
# Trigger Logic
# ============================================================================

def should_trigger(step: int) -> bool:
    """Fixed to every-token triggering for RQ3."""
    return True


# ============================================================================
# Core Evaluation Engine
# ============================================================================

class EvaluationEngine:
    def __init__(
        self,
        detector: HallucinationDetectorBase,
        llm_tokenizer: LLMTokenizerWrapper,
        confidence_threshold: float = 0.90,
    ):
        self.detector = detector
        self.llm_tokenizer = llm_tokenizer
        self.confidence_threshold = confidence_threshold

    def evaluate_sample(
        self,
        prompt: str,
        context: str,
        answer: str,
        labels: List[Dict],
        sample_index: int = 0,
    ) -> SampleResult:
        gt_labels = [
            GroundTruthLabel(start=l["start"], end=l["end"],
                             label=l.get("label", "hallucination"))
            for l in labels
        ]

        tokens = self.llm_tokenizer.tokenize_with_offsets(answer)
        detokenized_full = self.llm_tokenizer.get_detokenized_text(answer)
        offset_mapper = OffsetMapper(answer, detokenized_full)
        divergence = offset_mapper.check_divergence()
        if divergence > 0.05:
            print(
                f"  [WARNING] Sample {sample_index}: detokenization divergence "
                f"= {divergence:.2%}"
            )

        token_steps: List[TokenStepResult] = []
        last_prediction: Dict[int, Tuple[bool, float]] = {}
        cumulative_gt = []
        cumulative_pred = []
        sample_start_time = time.time()

        encoded_full = self.llm_tokenizer.tokenizer.encode(answer, add_special_tokens=False)

        for step, token in enumerate(tokens):
            triggered = should_trigger(step)

            if triggered:
                prefix_text = self.llm_tokenizer.tokenizer.decode(
                    encoded_full[: step + 1], skip_special_tokens=True
                )

                try:
                    predictions = self.detector.predict(
                        context=[context],
                        question=prompt,
                        answer=prefix_text,
                    )
                except Exception as e:
                    print(f"  [ERROR] Detection failed at step {step}: {e}")
                    predictions = []

                tokens_so_far = tokens[: step + 1]
                pred_map = map_predictions_to_tokens(
                    predictions, tokens_so_far, self.confidence_threshold
                )
                last_prediction = pred_map

            if token.index in last_prediction:
                is_pred_hall, pred_conf = last_prediction[token.index]
            else:
                is_pred_hall, pred_conf = False, 0.0

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

            cumulative_gt.append(is_gt_hall)
            cumulative_pred.append(is_pred_hall)
            _, _, cum_f1 = compute_metrics_class(cumulative_gt, cumulative_pred, positive_class=True)
            cum_prec, cum_rec, _ = compute_metrics_class(cumulative_gt, cumulative_pred, positive_class=True)

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

        all_gt = [ts.ground_truth for ts in token_steps]
        all_pred = [ts.predicted for ts in token_steps]
        agg_prec, agg_rec, agg_f1 = compute_metrics_class(all_gt, all_pred, positive_class=True)

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
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return data


def extract_context_from_prompt(prompt: str) -> Tuple[str, str]:
    return prompt, prompt


# ============================================================================
# Result Export
# ============================================================================

def export_per_step_csv(sample_results: List[SampleResult], filepath: str):
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


def export_aggregate_csv(benchmark_results: List[BenchmarkResult], filepath: str):
    """Write all model aggregates into a single comparison CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "model_path", "trigger_strategy", "tokenizer", "num_samples",
            "precision_micro", "recall_micro", "f1_micro",
            "precision_class_1", "recall_class_1", "f1_binary_class_1",
            "precision_class_0", "recall_class_0", "f1_binary_class_0",
            "total_runtime_seconds",
        ])
        for br in benchmark_results:
            writer.writerow([
                br.model_name,
                br.model_path,
                br.trigger_strategy,
                br.tokenizer_name,
                br.num_samples,
                f"{br.precision_micro:.4f}",
                f"{br.recall_micro:.4f}",
                f"{br.f1_micro:.4f}",
                f"{br.precision_class_1:.4f}",
                f"{br.recall_class_1:.4f}",
                f"{br.f1_binary_class_1:.4f}",
                f"{br.precision_class_0:.4f}",
                f"{br.recall_class_0:.4f}",
                f"{br.f1_binary_class_0:.4f}",
                f"{br.total_runtime_seconds:.2f}",
            ])


def export_results_json(benchmark_result: BenchmarkResult, filepath: str):
    output = {
        "model_name":            benchmark_result.model_name,
        "model_path":            benchmark_result.model_path,
        "trigger_strategy":      benchmark_result.trigger_strategy,
        "tokenizer_name":        benchmark_result.tokenizer_name,
        "num_samples":           benchmark_result.num_samples,
        "precision_micro":       benchmark_result.precision_micro,
        "recall_micro":          benchmark_result.recall_micro,
        "f1_micro":              benchmark_result.f1_micro,
        "precision_class_1":     benchmark_result.precision_class_1,
        "recall_class_1":        benchmark_result.recall_class_1,
        "f1_binary_class_1":     benchmark_result.f1_binary_class_1,
        "precision_class_0":     benchmark_result.precision_class_0,
        "recall_class_0":        benchmark_result.recall_class_0,
        "f1_binary_class_0":     benchmark_result.f1_binary_class_0,
        "total_runtime_seconds": benchmark_result.total_runtime_seconds,
        "samples": [],
    }
    for sr in benchmark_result.sample_results:
        sample_out = {
            "sample_index":        sr.sample_index,
            "aggregate_precision": sr.aggregate_precision,
            "aggregate_recall":    sr.aggregate_recall,
            "aggregate_f1":        sr.aggregate_f1,
            "runtime_seconds":     sr.runtime_seconds,
            "token_steps":         [asdict(ts) for ts in sr.token_steps],
        }
        output["samples"].append(sample_out)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================================
# Benchmark Runner (per model)
# ============================================================================

def run_benchmark_for_model(
    dataset_path: str,
    model_name: str,
    model_path: str,
    llm_tokenizer: LLMTokenizerWrapper,
    max_samples: Optional[int],
    confidence_threshold: float,
    output_prefix: str,
) -> BenchmarkResult:
    print(f"\n{'#' * 70}")
    print(f"# Evaluating: {model_name}")
    print(f"# Path:       {model_path}")
    print(f"{'#' * 70}\n")

    # Start W&B run for this model
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=model_name,
        config={
            "model_name":           model_name,
            "model_path":           model_path,
            "trigger_strategy":     TriggerStrategy.EVERY_TOKEN.value,
            "tokenizer":            LLM_TOKENIZER_NAME,
            "confidence_threshold": confidence_threshold,
            "max_samples":          max_samples,
            "dataset_path":         dataset_path,
        },
        reinit=True,
    )

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_ragtruth_dataset(dataset_path, max_samples)
    print(f"Loaded {len(data)} samples.")

    # Init detector + engine
    detector = LettuceDetectWrapper(name=model_name, model_path=model_path)
    engine = EvaluationEngine(
        detector=detector,
        llm_tokenizer=llm_tokenizer,
        confidence_threshold=confidence_threshold,
    )

    all_sample_results: List[SampleResult] = []
    all_gt = []
    all_pred = []
    total_start = time.time()

    for i, sample in enumerate(data):
        print(f"  [{model_name}] Evaluating sample {i + 1}/{len(data)}...")
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

        for ts in result.token_steps:
            all_gt.append(ts.ground_truth)
            all_pred.append(ts.predicted)

        # Per-sample W&B log
        run.log({
            "sample_index":        i,
            "sample_precision":    result.aggregate_precision,
            "sample_recall":       result.aggregate_recall,
            "sample_f1":           result.aggregate_f1,
            "sample_runtime_s":    result.runtime_seconds,
        })

        print(
            f"    P={result.aggregate_precision:.4f} "
            f"R={result.aggregate_recall:.4f} "
            f"F1={result.aggregate_f1:.4f} "
            f"Runtime={result.runtime_seconds:.2f}s"
        )

    total_runtime = time.time() - total_start

    # Compute final metrics
    prec_micro, rec_micro, f1_micro = compute_metrics_micro(all_gt, all_pred)
    prec_c1, rec_c1, f1_c1 = compute_metrics_class(all_gt, all_pred, positive_class=True)
    prec_c0, rec_c0, f1_c0 = compute_metrics_class(all_gt, all_pred, positive_class=False)

    benchmark_result = BenchmarkResult(
        model_name=model_name,
        model_path=model_path,
        trigger_strategy=TriggerStrategy.EVERY_TOKEN.value,
        tokenizer_name=LLM_TOKENIZER_NAME,
        num_samples=len(data),
        precision_micro=prec_micro,
        recall_micro=rec_micro,
        f1_micro=f1_micro,
        precision_class_1=prec_c1,
        recall_class_1=rec_c1,
        f1_binary_class_1=f1_c1,
        precision_class_0=prec_c0,
        recall_class_0=rec_c0,
        f1_binary_class_0=f1_c0,
        total_runtime_seconds=total_runtime,
        sample_results=all_sample_results,
    )

    # Final W&B summary
    run.summary["precision_micro"]       = prec_micro
    run.summary["recall_micro"]          = rec_micro
    run.summary["f1_micro"]              = f1_micro
    run.summary["precision_class_1"]     = prec_c1
    run.summary["recall_class_1"]        = rec_c1
    run.summary["f1_binary_class_1"]     = f1_c1
    run.summary["precision_class_0"]     = prec_c0
    run.summary["recall_class_0"]        = rec_c0
    run.summary["f1_binary_class_0"]     = f1_c0
    run.summary["total_runtime_seconds"] = total_runtime

    # Export per-model results
    per_step_path = f"{output_prefix}_{model_name}_per_step.csv"
    json_path     = f"{output_prefix}_{model_name}_full_results.json"
    export_per_step_csv(all_sample_results, per_step_path)
    export_results_json(benchmark_result, json_path)

    # Log as W&B artifact
    artifact = wandb.Artifact(f"{model_name}_results", type="benchmark_results")
    artifact.add_file(per_step_path)
    artifact.add_file(json_path)
    run.log_artifact(artifact)

    # Summary print
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {model_name}")
    print(f"{'=' * 60}")
    print(f"Samples:              {len(data)}")
    print(f"Total runtime:        {total_runtime:.2f}s")
    print(f"--- Global (micro) ---")
    print(f"  Precision:          {prec_micro:.4f}")
    print(f"  Recall:             {rec_micro:.4f}")
    print(f"  F1:                 {f1_micro:.4f}")
    print(f"--- Class 1 (hallucinated) ---")
    print(f"  Precision:          {prec_c1:.4f}")
    print(f"  Recall:             {rec_c1:.4f}")
    print(f"  F1:                 {f1_c1:.4f}")
    print(f"--- Class 0 (supported) ---")
    print(f"  Precision:          {prec_c0:.4f}")
    print(f"  Recall:             {rec_c0:.4f}")
    print(f"  F1:                 {f1_c0:.4f}")
    print(f"{'=' * 60}\n")

    run.finish()

    return benchmark_result


# ============================================================================
# Main: run both models
# ============================================================================

def main(
    dataset_path: str,
    max_samples: Optional[int],
    confidence_threshold: float,
    output_prefix: str,
):
    # Shared tokenizer (loaded once)
    print(f"Loading Llama tokenizer: {LLM_TOKENIZER_NAME}...")
    llm_tokenizer = LLMTokenizerWrapper(LLM_TOKENIZER_NAME)

    all_results: List[BenchmarkResult] = []

    for model_cfg in MODELS_TO_EVALUATE:
        result = run_benchmark_for_model(
            dataset_path=dataset_path,
            model_name=model_cfg["name"],
            model_path=model_cfg["model_path"],
            llm_tokenizer=llm_tokenizer,
            max_samples=max_samples,
            confidence_threshold=confidence_threshold,
            output_prefix=output_prefix,
        )
        all_results.append(result)

    # Combined aggregate CSV
    aggregate_path = f"{output_prefix}_aggregate.csv"
    export_aggregate_csv(all_results, aggregate_path)
    print(f"\nCombined aggregate CSV written to: {aggregate_path}")

    # Final comparison print
    print(f"\n{'=' * 70}")
    print(f"FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<40} {'F1_c1':>8} {'F1_c0':>8} {'Runtime':>10}")
    print("-" * 70)
    for br in all_results:
        print(f"{br.model_name:<40} "
              f"{br.f1_binary_class_1:>8.4f} "
              f"{br.f1_binary_class_0:>8.4f} "
              f"{br.total_runtime_seconds:>9.2f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RQ3 Benchmark: TinyLettuce vs. LettuceDetect-base"
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
        "--confidence-threshold", type=float, default=0.90,
        help="Confidence threshold for hallucination prediction."
    )
    parser.add_argument(
        "--output-prefix", type=str, default="rq3_benchmark",
        help="Prefix for output files."
    )

    args = parser.parse_args()

    main(
        dataset_path=args.dataset,
        max_samples=args.max_samples,
        confidence_threshold=args.confidence_threshold,
        output_prefix=args.output_prefix,
    )