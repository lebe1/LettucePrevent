"""
Hallucination Detection Benchmark — RQ3 Comparison

Runs both TinyLettuce and LettuceDetect-base in a single invocation,
simulating token-by-token incremental generation with a Llama-3 tokenizer.

Dataset: wandb/RAGTruth-processed (test split), 150 unique (context, query)
combinations per task_type — 450 samples total.

Outputs:
- {prefix}_{model_name}_per_step.csv    (per model: token-level details)
- {prefix}_{model_name}_full_results.json (per model: full nested results)
- {prefix}_aggregate.csv                (combined: model-level comparison)
- {prefix}_per_sample_comparison.json   (combined: sample-centric comparison)

Logs to W&B (two separate runs). The comparison JSON is logged as an
artifact to the second W&B run.
"""

import json
import random
import time
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

import numpy as np
import torch
import wandb
from datasets import load_dataset


# ============================================================================
# Fixed Configuration
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WANDB_ENTITY  = "lebeccard-technical-university-wien"
WANDB_PROJECT = "hdm-benchmark-rq3"

LLM_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"

HF_DATASET_NAME        = "wandb/RAGTruth-processed"
HF_DATASET_SPLIT       = "test"
UNIQUE_PAIRS_PER_TASK  = 150

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
    task_type: str
    context: str
    query: str
    answer: str
    token_steps: List[TokenStepResult] = field(default_factory=list)
    aggregate_precision: float = 0.0
    aggregate_recall: float = 0.0
    aggregate_f1: float = 0.0
    runtime_seconds: float = 0.0

    # Per-sample 9 metrics (set by runner after evaluation)
    precision_micro: float = 0.0
    recall_micro: float = 0.0
    f1_micro: float = 0.0
    precision_class_1: float = 0.0
    recall_class_1: float = 0.0
    f1_binary_class_1: float = 0.0
    precision_class_0: float = 0.0
    recall_class_0: float = 0.0
    f1_binary_class_0: float = 0.0


@dataclass
class BenchmarkResult:
    model_name: str
    model_path: str
    trigger_strategy: str
    tokenizer_name: str
    num_samples: int

    precision_micro: float = 0.0
    recall_micro: float = 0.0
    f1_micro: float = 0.0

    precision_class_1: float = 0.0
    recall_class_1: float = 0.0
    f1_binary_class_1: float = 0.0

    precision_class_0: float = 0.0
    recall_class_0: float = 0.0
    f1_binary_class_0: float = 0.0

    total_runtime_seconds: float = 0.0
    sample_results: List[SampleResult] = field(default_factory=list)


# ============================================================================
# Hallucination Detectors
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
# Label Parsing
# ============================================================================

def parse_hallucination_labels(raw_labels):
    if isinstance(raw_labels, str):
        try:
            raw_labels = json.loads(raw_labels)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw_labels, list):
        return []
    return [lbl for lbl in raw_labels
            if isinstance(lbl, dict) and "start" in lbl and "end" in lbl]


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
    if len(ground_truths) == 0:
        return 0.0, 0.0, 0.0
    correct = sum(1 for g, p in zip(ground_truths, predictions) if g == p)
    acc = correct / len(ground_truths)
    return acc, acc, acc


def compute_nine_metrics(gts: List[bool], preds: List[bool]) -> Dict[str, float]:
    """Compute all 9 metrics (micro + both classes) at once."""
    p_mic, r_mic, f_mic = compute_metrics_micro(gts, preds)
    p_c1, r_c1, f_c1    = compute_metrics_class(gts, preds, positive_class=True)
    p_c0, r_c0, f_c0    = compute_metrics_class(gts, preds, positive_class=False)
    return {
        "precision_micro":   p_mic,
        "recall_micro":      r_mic,
        "f1_micro":          f_mic,
        "precision_class_1": p_c1,
        "recall_class_1":    r_c1,
        "f1_binary_class_1": f_c1,
        "precision_class_0": p_c0,
        "recall_class_0":    r_c0,
        "f1_binary_class_0": f_c0,
    }


# ============================================================================
# Trigger Logic
# ============================================================================

def should_trigger(step: int) -> bool:
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
        context: str,
        query: str,
        answer: str,
        labels: List[Dict],
        task_type: str,
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
                        question=query,
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
            cum_prec, cum_rec, cum_f1 = compute_metrics_class(
                cumulative_gt, cumulative_pred, positive_class=True
            )

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
        nine = compute_nine_metrics(all_gt, all_pred)

        return SampleResult(
            sample_index=sample_index,
            task_type=task_type,
            context=context,
            query=query,
            answer=answer,
            token_steps=token_steps,
            aggregate_precision=agg_prec,
            aggregate_recall=agg_rec,
            aggregate_f1=agg_f1,
            runtime_seconds=sample_runtime,
            precision_micro=nine["precision_micro"],
            recall_micro=nine["recall_micro"],
            f1_micro=nine["f1_micro"],
            precision_class_1=nine["precision_class_1"],
            recall_class_1=nine["recall_class_1"],
            f1_binary_class_1=nine["f1_binary_class_1"],
            precision_class_0=nine["precision_class_0"],
            recall_class_0=nine["recall_class_0"],
            f1_binary_class_0=nine["f1_binary_class_0"],
        )


# ============================================================================
# Dataset Loading — 150 unique (context, query) per task_type
# ============================================================================

def load_benchmark_samples() -> Tuple[List[Dict], List[Dict]]:
    """
    Load the test split and select 150 unique (context, query) combinations
    per task_type. Returns (benchmark_samples, original_rows) where
    original_rows contains every field of the HF dataset for each sample.
    """
    print(f"Loading HF dataset: {HF_DATASET_NAME} (split={HF_DATASET_SPLIT})...")
    hf_ds = load_dataset(HF_DATASET_NAME)
    test_split = hf_ds[HF_DATASET_SPLIT]

    # Filter out rows without valid hallucination labels
    filtered = [
        dict(row) for row in test_split
        if row.get("hallucination_labels") not in (None, "")
    ]

    # Per task_type, collect first-seen rows for each unique (context, query)
    # Preserves dataset order.
    by_task: Dict[str, Dict[Tuple[str, str], Dict]] = {}
    for row in filtered:
        tt = row.get("task_type", "unknown")
        key = (row["context"], row["query"])
        if tt not in by_task:
            by_task[tt] = {}
        if key not in by_task[tt] and len(by_task[tt]) < UNIQUE_PAIRS_PER_TASK:
            by_task[tt][key] = row

    benchmark_samples = []
    original_rows     = []
    for tt, mapping in by_task.items():
        print(f"  task_type={tt}: {len(mapping)} unique (context, query) pairs")
        for row in mapping.values():
            benchmark_samples.append({
                "task_type": tt,
                "context":   row["context"],
                "query":     row["query"],
                "answer":    row["output"],
                "labels":    parse_hallucination_labels(row["hallucination_labels"]),
            })
            original_rows.append(row)

    print(f"Total benchmark samples: {len(benchmark_samples)}")
    return benchmark_samples, original_rows


# ============================================================================
# Export: Per-model files (unchanged)
# ============================================================================

def export_per_step_csv(sample_results: List[SampleResult], filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index", "task_type", "step", "token",
            "token_char_start", "token_char_end",
            "ground_truth", "predicted", "confidence", "was_triggered",
            "cumulative_precision", "cumulative_recall", "cumulative_f1",
        ])
        for sr in sample_results:
            for ts in sr.token_steps:
                writer.writerow([
                    sr.sample_index, sr.task_type, ts.step, ts.token,
                    ts.token_char_start, ts.token_char_end,
                    ts.ground_truth, ts.predicted,
                    f"{ts.confidence:.6f}", ts.was_triggered,
                    f"{ts.cumulative_precision:.4f}",
                    f"{ts.cumulative_recall:.4f}",
                    f"{ts.cumulative_f1:.4f}",
                ])


def export_aggregate_csv(benchmark_results: List[BenchmarkResult], filepath: str):
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
            "task_type":           sr.task_type,
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
# Export: Sample-centric comparison JSON (new)
# ============================================================================

def export_per_sample_comparison(
    original_rows: List[Dict],
    benchmark_samples: List[Dict],
    model_results: Dict[str, List[SampleResult]],
    filepath: str,
):
    """
    Build a sample-centric JSON with both models' predictions side-by-side.

    Args:
        original_rows: list of original HF dataset rows (all fields)
        benchmark_samples: list of benchmark sample dicts (same order)
        model_results: dict model_name -> list of SampleResult (same order)
        filepath: output path
    """
    output = {
        "dataset":                HF_DATASET_NAME,
        "dataset_split":          HF_DATASET_SPLIT,
        "unique_pairs_per_task":  UNIQUE_PAIRS_PER_TASK,
        "tokenizer":              LLM_TOKENIZER_NAME,
        "models":                 list(model_results.keys()),
        "num_samples":            len(benchmark_samples),
        "samples":                [],
    }

    model_names = list(model_results.keys())

    for i, (orig_row, bench_sample) in enumerate(zip(original_rows, benchmark_samples)):
        # Pull the first model's tokens — token indices and ground truth are
        # identical across models because both use the same tokenizer.
        reference_result = model_results[model_names[0]][i]

        tokens_info = [
            {
                "index":        ts.step,
                "text":         ts.token,
                "char_start":   ts.token_char_start,
                "char_end":     ts.token_char_end,
                "ground_truth": ts.ground_truth,
            }
            for ts in reference_result.token_steps
        ]

        model_predictions = {}
        for model_name in model_names:
            sr = model_results[model_name][i]
            model_predictions[model_name] = {
                "predictions":     [ts.predicted   for ts in sr.token_steps],
                "confidences":     [ts.confidence  for ts in sr.token_steps],
                "runtime_seconds": sr.runtime_seconds,
                "metrics": {
                    "precision_micro":   sr.precision_micro,
                    "recall_micro":      sr.recall_micro,
                    "f1_micro":          sr.f1_micro,
                    "precision_class_1": sr.precision_class_1,
                    "recall_class_1":    sr.recall_class_1,
                    "f1_binary_class_1": sr.f1_binary_class_1,
                    "precision_class_0": sr.precision_class_0,
                    "recall_class_0":    sr.recall_class_0,
                    "f1_binary_class_0": sr.f1_binary_class_0,
                },
            }

        sample_entry = {
            "sample_index":      i,
            "task_type":         bench_sample["task_type"],
            "original_data":     orig_row,
            "parsed_labels":     [
                {"start": l["start"], "end": l["end"],
                 "label": l.get("label", "hallucination")}
                for l in bench_sample["labels"]
            ],
            "tokens":            tokens_info,
            "model_predictions": model_predictions,
        }
        output["samples"].append(sample_entry)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================================
# Benchmark Runner (per model)
# ============================================================================

def run_benchmark_for_model(
    samples: List[Dict],
    model_name: str,
    model_path: str,
    llm_tokenizer: LLMTokenizerWrapper,
    confidence_threshold: float,
    output_prefix: str,
    extra_artifact_paths: Optional[List[str]] = None,
) -> BenchmarkResult:
    """
    Run the benchmark for a single model.

    Args:
        extra_artifact_paths: optional list of additional files to upload as
            part of the W&B artifact for this run. Used by the second model
            to attach the comparison JSON.
    """
    print(f"\n{'#' * 70}")
    print(f"# Evaluating: {model_name}")
    print(f"# Path:       {model_path}")
    print(f"{'#' * 70}\n")

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=model_name,
        config={
            "model_name":            model_name,
            "model_path":            model_path,
            "trigger_strategy":      TriggerStrategy.EVERY_TOKEN.value,
            "tokenizer":             LLM_TOKENIZER_NAME,
            "confidence_threshold":  confidence_threshold,
            "dataset":               HF_DATASET_NAME,
            "dataset_split":         HF_DATASET_SPLIT,
            "unique_pairs_per_task": UNIQUE_PAIRS_PER_TASK,
            "num_samples":           len(samples),
            "seed":                  SEED,
        },
        reinit=True,
    )

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

    for i, sample in enumerate(samples):
        print(f"  [{model_name}] Evaluating sample {i + 1}/{len(samples)} "
              f"(task={sample['task_type']})...")

        result = engine.evaluate_sample(
            context=sample["context"],
            query=sample["query"],
            answer=sample["answer"],
            labels=sample["labels"],
            task_type=sample["task_type"],
            sample_index=i,
        )
        all_sample_results.append(result)

        for ts in result.token_steps:
            all_gt.append(ts.ground_truth)
            all_pred.append(ts.predicted)

        run.log({
            "sample_index":     i,
            "task_type":        sample["task_type"],
            "sample_precision": result.aggregate_precision,
            "sample_recall":    result.aggregate_recall,
            "sample_f1":        result.aggregate_f1,
            "sample_runtime_s": result.runtime_seconds,
        })

        print(
            f"    P={result.aggregate_precision:.4f} "
            f"R={result.aggregate_recall:.4f} "
            f"F1={result.aggregate_f1:.4f} "
            f"Runtime={result.runtime_seconds:.2f}s"
        )

    total_runtime = time.time() - total_start

    prec_micro, rec_micro, f1_micro = compute_metrics_micro(all_gt, all_pred)
    prec_c1, rec_c1, f1_c1 = compute_metrics_class(all_gt, all_pred, positive_class=True)
    prec_c0, rec_c0, f1_c0 = compute_metrics_class(all_gt, all_pred, positive_class=False)

    benchmark_result = BenchmarkResult(
        model_name=model_name,
        model_path=model_path,
        trigger_strategy=TriggerStrategy.EVERY_TOKEN.value,
        tokenizer_name=LLM_TOKENIZER_NAME,
        num_samples=len(samples),
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

    per_step_path = f"{output_prefix}_{model_name}_per_step.csv"
    json_path     = f"{output_prefix}_{model_name}_full_results.json"
    export_per_step_csv(all_sample_results, per_step_path)
    export_results_json(benchmark_result, json_path)

    artifact = wandb.Artifact(f"{model_name}_results", type="benchmark_results")
    artifact.add_file(per_step_path)
    artifact.add_file(json_path)
    if extra_artifact_paths:
        for p in extra_artifact_paths:
            artifact.add_file(p)
    run.log_artifact(artifact)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {model_name}")
    print(f"{'=' * 60}")
    print(f"Samples:              {len(samples)}")
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
# Main
# ============================================================================

def main(confidence_threshold: float, output_prefix: str):
    benchmark_samples, original_rows = load_benchmark_samples()

    print(f"Loading Llama tokenizer: {LLM_TOKENIZER_NAME}...")
    llm_tokenizer = LLMTokenizerWrapper(LLM_TOKENIZER_NAME)

    all_results: List[BenchmarkResult] = []
    model_results: Dict[str, List[SampleResult]] = {}

    for idx, model_cfg in enumerate(MODELS_TO_EVALUATE):
        is_last = (idx == len(MODELS_TO_EVALUATE) - 1)

        # For the last model we build the comparison JSON first, then pass it
        # as an extra artifact file to that model's run.
        extra_artifacts = None
        comparison_path = None
        if is_last:
            comparison_path = f"{output_prefix}_per_sample_comparison.json"
            # Don't write it yet — we need this model's results first.
            # We'll write after the run and upload separately via artifact.

        result = run_benchmark_for_model(
            samples=benchmark_samples,
            model_name=model_cfg["name"],
            model_path=model_cfg["model_path"],
            llm_tokenizer=llm_tokenizer,
            confidence_threshold=confidence_threshold,
            output_prefix=output_prefix,
            extra_artifact_paths=None,  # artifact for comparison handled below
        )
        all_results.append(result)
        model_results[model_cfg["name"]] = result.sample_results

    # Combined aggregate CSV
    aggregate_path = f"{output_prefix}_aggregate.csv"
    export_aggregate_csv(all_results, aggregate_path)
    print(f"\nCombined aggregate CSV written to: {aggregate_path}")

    # Combined per-sample comparison JSON
    comparison_path = f"{output_prefix}_per_sample_comparison.json"
    export_per_sample_comparison(
        original_rows=original_rows,
        benchmark_samples=benchmark_samples,
        model_results=model_results,
        filepath=comparison_path,
    )
    print(f"Per-sample comparison JSON written to: {comparison_path}")

    # Attach comparison JSON + aggregate CSV as a dedicated artifact to the
    # second (last) model's run. We reopen the last run briefly.
    last_model_name = MODELS_TO_EVALUATE[-1]["name"]
    print(f"\nAttaching comparison artifact to W&B run: {last_model_name}")

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=last_model_name,
        resume="allow",
        reinit=True,
    )
    artifact = wandb.Artifact("per_sample_comparison", type="comparison_results")
    artifact.add_file(comparison_path)
    artifact.add_file(aggregate_path)
    run.log_artifact(artifact)
    run.finish()

    # Final print
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
        "--confidence-threshold", type=float, default=0.90,
        help="Confidence threshold for hallucination prediction."
    )
    parser.add_argument(
        "--output-prefix", type=str, default="rq3_benchmark",
        help="Prefix for output files."
    )

    args = parser.parse_args()

    main(
        confidence_threshold=args.confidence_threshold,
        output_prefix=args.output_prefix,
    )