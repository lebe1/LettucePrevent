"""
Confidence threshold sweep for hallucination detection models — STANDALONE.

Runs a W&B grid sweep over:
- confidence_threshold: [0.6, 0.7, 0.8, 0.9]
- model_idx: [0, 1]  (TinyLettuce and LettuceDetect-base)

Dataset setup:
- wandb/RAGTruth-processed (test split)
- 15 unique (context, query) samples per task_type

After all sweep runs, prints a ranked summary showing the best threshold
per model based on F1 class 1, recall class 1, and runtime.

"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from datasets import load_dataset
import warnings

os.environ.setdefault("WEAVE_DISABLED", "true")

warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic serializer warnings.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Expected `list\[str\]` but got `tuple`.*",
)


# ============================================================================
# Configuration (mirrored from hallucination_benchmark.py)
# ============================================================================

SEED = 42
WANDB_ENTITY  = "lebeccard-technical-university-wien"
LLM_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"

HF_DATASET_NAME  = "wandb/RAGTruth-processed"
HF_DATASET_SPLIT = "test"

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
# Sweep-specific configuration
# ============================================================================

SWEEP_THRESHOLDS         = [0.6, 0.7, 0.8, 0.9]
UNIQUE_PAIRS_PER_TASK    = 15
DEFAULT_SWEEP_NAME       = "confidence-threshold-comparison-rq3"
DEFAULT_SWEEP_PROJECT    = "hdm-benchmark-rq3-threshold-sweep"
DEFAULT_OUTPUT_PREFIX    = "rq3_confident_treshold_sweep"


# ============================================================================
# Data structures (copied from hallucination_benchmark.py)
# ============================================================================

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

    precision_micro: float = 0.0
    recall_micro: float = 0.0
    f1_micro: float = 0.0
    precision_class_1: float = 0.0
    recall_class_1: float = 0.0
    f1_binary_class_1: float = 0.0
    precision_class_0: float = 0.0
    recall_class_0: float = 0.0
    f1_binary_class_0: float = 0.0


# ============================================================================
# Tokenizer wrapper
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
        tokens: List[TokenInfo] = []

        for i, _ in enumerate(encoded):
            prefix      = self.tokenizer.decode(encoded[: i + 1], skip_special_tokens=True)
            prev_prefix = self.tokenizer.decode(encoded[:i],      skip_special_tokens=True) if i > 0 else ""

            char_start = len(prev_prefix)
            char_end   = len(prefix)
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
# Detector wrapper
# ============================================================================

class HallucinationDetectorBase(ABC):
    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def predict(
        self,
        context: List[str],
        question: str,
        answer: str,
    ) -> List[Dict[str, Any]]: ...


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
# Offset mapper (handles tokenizer detokenization vs. original whitespace)
# ============================================================================

class OffsetMapper:
    def __init__(self, original: str, detokenized: str):
        self.original    = original
        self.detokenized = detokenized
        self._build_mapping()

    def _build_mapping(self):
        self.orig_to_detok: Dict[int, int] = {}
        self.detok_to_orig: Dict[int, int] = {}
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
        orig_end   = self.detok_to_orig.get(end - 1)

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
        return 1.0 - (len(self.detok_to_orig) / len(self.detokenized))


# ============================================================================
# Label parsing
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


def is_token_hallucinated(token: TokenInfo, labels: List[GroundTruthLabel]) -> bool:
    for lbl in labels:
        if token.char_start < lbl.end and token.char_end > lbl.start:
            return True
    return False


def map_predictions_to_tokens(
    predictions: List[Dict[str, Any]],
    tokens: List[TokenInfo],
    confidence_threshold: float = 0.90,
) -> Dict[int, Tuple[bool, float]]:
    result: Dict[int, Tuple[bool, float]] = {}
    for token in tokens:
        max_conf = 0.0
        is_hall  = False
        for pred in predictions:
            if token.char_start < pred["end"] and token.char_end > pred["start"]:
                if pred["confidence"] > max_conf:
                    max_conf = pred["confidence"]
                    is_hall  = pred["confidence"] >= confidence_threshold
        result[token.index] = (is_hall, max_conf)
    return result


# ============================================================================
# Metrics — pooled (used for headline RQ3 numbers)
# ============================================================================

def compute_metrics_class(
    ground_truths: List[bool],
    predictions: List[bool],
    positive_class: bool = True,
) -> Tuple[float, float, float]:
    tp = sum(1 for g, p in zip(ground_truths, predictions) if g == positive_class and p == positive_class)
    fp = sum(1 for g, p in zip(ground_truths, predictions) if g != positive_class and p == positive_class)
    fn = sum(1 for g, p in zip(ground_truths, predictions) if g == positive_class and p != positive_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
# NaN-aware per-sample helpers (for honest W&B charts)
# ============================================================================

def _per_sample_class_metric(
    gt: List[bool],
    pred: List[bool],
    positive_class: bool,
) -> Tuple[float, float, float]:
    """
    Per-sample (precision, recall, f1) for one class. Returns NaN when the
    metric is mathematically undefined (zero denominator), so W&B charts
    skip those points instead of plotting them as zeros.
    """
    tp = sum(1 for g, p in zip(gt, pred) if g == positive_class and p == positive_class)
    fp = sum(1 for g, p in zip(gt, pred) if g != positive_class and p == positive_class)
    fn = sum(1 for g, p in zip(gt, pred) if g == positive_class and p != positive_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _nanmean(values: List[float]) -> float:
    """Mean ignoring NaNs. Returns NaN if all values are NaN."""
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


# ============================================================================
# Trigger logic (every-token)
# ============================================================================

def should_trigger(step: int) -> bool:
    return True


# ============================================================================
# Evaluation engine (copied from hallucination_benchmark.py)
# ============================================================================

class EvaluationEngine:
    def __init__(
        self,
        detector: HallucinationDetectorBase,
        llm_tokenizer: LLMTokenizerWrapper,
        confidence_threshold: float = 0.90,
    ):
        self.detector             = detector
        self.llm_tokenizer        = llm_tokenizer
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

        tokens           = self.llm_tokenizer.tokenize_with_offsets(answer)
        detokenized_full = self.llm_tokenizer.get_detokenized_text(answer)
        offset_mapper    = OffsetMapper(answer, detokenized_full)
        divergence       = offset_mapper.check_divergence()
        if divergence > 0.05:
            print(f"  [WARNING] Sample {sample_index}: detokenization divergence = {divergence:.2%}")

        token_steps: List[TokenStepResult] = []
        last_prediction: Dict[int, Tuple[bool, float]] = {}
        cumulative_gt:   List[bool] = []
        cumulative_pred: List[bool] = []
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

                tokens_so_far   = tokens[: step + 1]
                pred_map        = map_predictions_to_tokens(
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
        all_gt   = [ts.ground_truth for ts in token_steps]
        all_pred = [ts.predicted     for ts in token_steps]
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
# Sweep machinery
# ============================================================================

_SAMPLES_CACHE:    Optional[List[Dict]]            = None
_TOKENIZER_CACHE:  Optional[LLMTokenizerWrapper]   = None
_ALL_RUN_RESULTS:  List[Dict]                      = []


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_benchmark_samples_for_sweep() -> List[Dict]:
    global _SAMPLES_CACHE
    if _SAMPLES_CACHE is not None:
        return _SAMPLES_CACHE

    print(
        f"Loading HF dataset: {HF_DATASET_NAME} "
        f"(split={HF_DATASET_SPLIT}, unique_pairs_per_task={UNIQUE_PAIRS_PER_TASK})..."
    )
    hf_ds      = load_dataset(HF_DATASET_NAME)
    test_split = hf_ds[HF_DATASET_SPLIT]

    filtered = [
        dict(row) for row in test_split
        if row.get("hallucination_labels") not in (None, "")
    ]

    by_task: Dict[str, Dict[tuple, Dict]] = {}
    for row in filtered:
        tt  = row.get("task_type", "unknown")
        key = (row["context"], row["query"])
        if tt not in by_task:
            by_task[tt] = {}
        if key not in by_task[tt] and len(by_task[tt]) < UNIQUE_PAIRS_PER_TASK:
            by_task[tt][key] = row

    benchmark_samples: List[Dict] = []
    for tt in sorted(by_task.keys()):
        mapping = by_task[tt]
        print(f"  task_type={tt}: {len(mapping)} unique (context, query) pairs")
        for row in mapping.values():
            benchmark_samples.append({
                "task_type": tt,
                "context":   row["context"],
                "query":     row["query"],
                "answer":    row["output"],
                "labels":    parse_hallucination_labels(row["hallucination_labels"]),
            })

    print(f"Total benchmark samples: {len(benchmark_samples)}")
    _SAMPLES_CACHE = benchmark_samples
    return benchmark_samples


def build_sample_manifest_hash(samples: List[Dict]) -> str:
    payload = [
        {"task_type": s["task_type"], "context": s["context"],
         "query": s["query"], "answer": s["answer"]}
        for s in samples
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_tokenizer() -> LLMTokenizerWrapper:
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        print(f"Loading tokenizer: {LLM_TOKENIZER_NAME}")
        _TOKENIZER_CACHE = LLMTokenizerWrapper(LLM_TOKENIZER_NAME)
    return _TOKENIZER_CACHE


def build_sweep_config(output_prefix: str, seed: int) -> Dict:
    return {
        "name":   DEFAULT_SWEEP_NAME,
        "method": "grid",
        "metric": {"name": "f1_binary_class_1", "goal": "maximize"},
        "parameters": {
            "confidence_threshold": {"values": SWEEP_THRESHOLDS},
            "model_idx":             {"values": list(range(len(MODELS_TO_EVALUATE)))},
            "output_prefix":         {"value":  output_prefix},
            "seed":                  {"value":  seed},
        },
    }


# ============================================================================
# Single sweep run
# ============================================================================

def evaluate_single_run():
    """Single W&B agent run for one (model_idx, confidence_threshold) pair."""
    run = wandb.init()
    cfg = wandb.config

    model_idx            = int(cfg.model_idx)
    confidence_threshold = float(cfg.confidence_threshold)
    output_prefix        = str(cfg.output_prefix)
    seed                 = int(cfg.seed)
    model_cfg            = MODELS_TO_EVALUATE[model_idx]

    thr_label = f"{confidence_threshold:.1f}".replace(".", "_")
    run_name  = f"{model_cfg['name']}_thr_{thr_label}"
    run.name  = run_name

    set_seed(seed)

    print("\n" + "#" * 70)
    print(f"# Run:       {run_name}")
    print(f"# Model:     {model_cfg['name']}")
    print(f"# Threshold: {confidence_threshold:.2f}")
    print("#" * 70 + "\n")

    samples              = load_benchmark_samples_for_sweep()
    sample_manifest_hash = build_sample_manifest_hash(samples)
    llm_tokenizer        = get_tokenizer()
    detector             = LettuceDetectWrapper(
        name=model_cfg["name"],
        model_path=model_cfg["model_path"],
    )
    engine = EvaluationEngine(
        detector=detector,
        llm_tokenizer=llm_tokenizer,
        confidence_threshold=confidence_threshold,
    )

    all_sample_results: List[SampleResult] = []
    all_gt:   List[bool] = []
    all_pred: List[bool] = []

    # Per-sample arrays for macro-style averaging at the end
    per_sample_c1    = {"precision": [], "recall": [], "f1": []}
    per_sample_c0    = {"precision": [], "recall": [], "f1": []}
    per_sample_micro = {"precision": [], "recall": [], "f1": []}

    total_start = time.time()

    for i, sample in enumerate(samples):
        print(
            f"  [{model_cfg['name']} @ thr={confidence_threshold}] "
            f"sample {i + 1}/{len(samples)} (task={sample['task_type']})"
        )
        result = engine.evaluate_sample(
            context=sample["context"],
            query=sample["query"],
            answer=sample["answer"],
            labels=sample["labels"],
            task_type=sample["task_type"],
            sample_index=i,
        )
        all_sample_results.append(result)

        sample_gt   = [ts.ground_truth for ts in result.token_steps]
        sample_pred = [ts.predicted     for ts in result.token_steps]
        all_gt.extend(sample_gt)
        all_pred.extend(sample_pred)

        # NaN-aware per-sample metrics (replace sklearn-style 0-on-undefined)
        prec_c1_s, rec_c1_s, f1_c1_s = _per_sample_class_metric(sample_gt, sample_pred, positive_class=True)
        prec_c0_s, rec_c0_s, f1_c0_s = _per_sample_class_metric(sample_gt, sample_pred, positive_class=False)
        prec_micro_s, rec_micro_s, f1_micro_s = result.precision_micro, result.recall_micro, result.f1_micro

        per_sample_c1["precision"].append(prec_c1_s)
        per_sample_c1["recall"].append(rec_c1_s)
        per_sample_c1["f1"].append(f1_c1_s)
        per_sample_c0["precision"].append(prec_c0_s)
        per_sample_c0["recall"].append(rec_c0_s)
        per_sample_c0["f1"].append(f1_c0_s)
        per_sample_micro["precision"].append(prec_micro_s)
        per_sample_micro["recall"].append(rec_micro_s)
        per_sample_micro["f1"].append(f1_micro_s)

        # Diagnostic counts so you can see WHY a metric is NaN
        n_pos_gt   = sum(1 for g in sample_gt   if g)
        n_pos_pred = sum(1 for p in sample_pred if p)

        run.log({
            "sample_index":             i,
            "task_type":                sample["task_type"],
            "sample_runtime_s":         result.runtime_seconds,
            "sample_n_tokens":          len(sample_gt),
            "sample_n_pos_gt":          n_pos_gt,
            "sample_n_pos_pred":        n_pos_pred,

            "sample_precision_micro":   prec_micro_s,
            "sample_recall_micro":      rec_micro_s,
            "sample_f1_micro":          f1_micro_s,

            "sample_precision_class_1": prec_c1_s,
            "sample_recall_class_1":    rec_c1_s,
            "sample_f1_class_1":        f1_c1_s,

            "sample_precision_class_0": prec_c0_s,
            "sample_recall_class_0":    rec_c0_s,
            "sample_f1_class_0":        f1_c0_s,
        })

    total_runtime = time.time() - total_start

    # ----- Pooled (headline RQ3 numbers) -----
    prec_micro, rec_micro, f1_micro = compute_metrics_micro(all_gt, all_pred)
    prec_c1,    rec_c1,    f1_c1    = compute_metrics_class(all_gt, all_pred, positive_class=True)
    prec_c0,    rec_c0,    f1_c0    = compute_metrics_class(all_gt, all_pred, positive_class=False)

    # ----- Macro (per-sample averaged, NaN-aware) -----
    avg_prec_c1 = _nanmean(per_sample_c1["precision"])
    avg_rec_c1  = _nanmean(per_sample_c1["recall"])
    avg_f1_c1   = _nanmean(per_sample_c1["f1"])
    avg_prec_c0 = _nanmean(per_sample_c0["precision"])
    avg_rec_c0  = _nanmean(per_sample_c0["recall"])
    avg_f1_c0   = _nanmean(per_sample_c0["f1"])
    avg_prec_mi = _nanmean(per_sample_micro["precision"])
    avg_rec_mi  = _nanmean(per_sample_micro["recall"])
    avg_f1_mi   = _nanmean(per_sample_micro["f1"])

    # ----- Diagnostics -----
    total_pos_gt   = sum(1 for g in all_gt   if g)
    total_pos_pred = sum(1 for p in all_pred if p)
    total_tokens   = len(all_gt)

    # ----- W&B summary -----
    run.summary["model_name"]            = model_cfg["name"]
    run.summary["model_path"]            = model_cfg["model_path"]
    run.summary["confidence_threshold"]  = confidence_threshold
    run.summary["num_samples"]           = len(samples)
    run.summary["unique_pairs_per_task"] = UNIQUE_PAIRS_PER_TASK
    run.summary["sample_manifest_hash"]  = sample_manifest_hash
    run.summary["seed"]                  = seed
    run.summary["tokenizer_name"]        = LLM_TOKENIZER_NAME

    # Pooled
    run.summary["precision_micro"]       = prec_micro
    run.summary["recall_micro"]          = rec_micro
    run.summary["f1_micro"]              = f1_micro
    run.summary["precision_class_1"]     = prec_c1
    run.summary["recall_class_1"]        = rec_c1
    run.summary["f1_binary_class_1"]     = f1_c1
    run.summary["precision_class_0"]     = prec_c0
    run.summary["recall_class_0"]        = rec_c0
    run.summary["f1_binary_class_0"]     = f1_c0

    # Macro
    run.summary["macro_precision_micro"]   = avg_prec_mi
    run.summary["macro_recall_micro"]      = avg_rec_mi
    run.summary["macro_f1_micro"]          = avg_f1_mi
    run.summary["macro_precision_class_1"] = avg_prec_c1
    run.summary["macro_recall_class_1"]    = avg_rec_c1
    run.summary["macro_f1_class_1"]        = avg_f1_c1
    run.summary["macro_precision_class_0"] = avg_prec_c0
    run.summary["macro_recall_class_0"]    = avg_rec_c0
    run.summary["macro_f1_class_0"]        = avg_f1_c0

    # Diagnostics
    run.summary["total_tokens"]   = total_tokens
    run.summary["total_pos_gt"]   = total_pos_gt
    run.summary["total_pos_pred"] = total_pos_pred
    run.summary["frac_pos_gt"]    = total_pos_gt   / total_tokens if total_tokens else 0.0
    run.summary["frac_pos_pred"]  = total_pos_pred / total_tokens if total_tokens else 0.0

    run.summary["total_runtime_seconds"] = total_runtime

    # ----- Local JSON dump -----
    safe_model  = model_cfg["name"].replace("/", "_")
    result_path = f"{output_prefix}_{safe_model}_thr_{thr_label}.json"
    result_dict = {
        "model_name":            model_cfg["name"],
        "model_path":            model_cfg["model_path"],
        "confidence_threshold":  confidence_threshold,
        "num_samples":           len(samples),
        "unique_pairs_per_task": UNIQUE_PAIRS_PER_TASK,
        "sample_manifest_hash":  sample_manifest_hash,
        "seed":                  seed,

        # Pooled
        "precision_micro":       prec_micro,
        "recall_micro":          rec_micro,
        "f1_micro":              f1_micro,
        "precision_class_1":     prec_c1,
        "recall_class_1":        rec_c1,
        "f1_binary_class_1":     f1_c1,
        "precision_class_0":     prec_c0,
        "recall_class_0":        rec_c0,
        "f1_binary_class_0":     f1_c0,

        # Macro
        "macro_precision_micro":   avg_prec_mi,
        "macro_recall_micro":      avg_rec_mi,
        "macro_f1_micro":          avg_f1_mi,
        "macro_precision_class_1": avg_prec_c1,
        "macro_recall_class_1":    avg_rec_c1,
        "macro_f1_class_1":        avg_f1_c1,
        "macro_precision_class_0": avg_prec_c0,
        "macro_recall_class_0":    avg_rec_c0,
        "macro_f1_class_0":        avg_f1_c0,

        # Diagnostics
        "total_tokens":          total_tokens,
        "total_pos_gt":          total_pos_gt,
        "total_pos_pred":        total_pos_pred,
        "total_runtime_seconds": total_runtime,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    artifact = wandb.Artifact(f"{safe_model}_thr_{thr_label}", type="benchmark_results")
    artifact.add_file(result_path)
    run.log_artifact(artifact)

    _ALL_RUN_RESULTS.append(result_dict)

    # ----- Console summary -----
    print(f"\n{'=' * 60}")
    print(f"RUN COMPLETE: {run_name}")
    print(f"{'=' * 60}")
    print(f"Model:              {model_cfg['name']}")
    print(f"Threshold:          {confidence_threshold}")
    print(f"Runtime:            {total_runtime:.2f}s")
    print(f"--- Diagnostics ---")
    print(f"  Total tokens:     {total_tokens}")
    if total_tokens:
        print(f"  Class-1 GT:       {total_pos_gt} ({100 * total_pos_gt / total_tokens:.2f}%)")
        print(f"  Class-1 pred:     {total_pos_pred} ({100 * total_pos_pred / total_tokens:.2f}%)")
    print(f"--- Global (micro) ---")
    print(f"  Precision:        {prec_micro:.4f}")
    print(f"  Recall:           {rec_micro:.4f}")
    print(f"  F1:               {f1_micro:.4f}")
    print(f"--- Class 1 (hallucinated) — POOLED ---")
    print(f"  Precision:        {prec_c1:.4f}")
    print(f"  Recall:           {rec_c1:.4f}")
    print(f"  F1:               {f1_c1:.4f}")
    print(f"--- Class 1 (hallucinated) — MACRO (avg over samples) ---")
    print(f"  Precision:        {avg_prec_c1:.4f}")
    print(f"  Recall:           {avg_rec_c1:.4f}")
    print(f"  F1:               {avg_f1_c1:.4f}")
    print(f"--- Class 0 (supported) — POOLED ---")
    print(f"  Precision:        {prec_c0:.4f}")
    print(f"  Recall:           {rec_c0:.4f}")
    print(f"  F1:               {f1_c0:.4f}")
    print(f"{'=' * 60}\n")

    run.finish()


# ============================================================================
# Final summary across all sweep runs
# ============================================================================

def print_final_summary():
    if not _ALL_RUN_RESULTS:
        print("No results collected — skipping summary.")
        return

    print("\n" + "=" * 100)
    print("CONFIDENCE THRESHOLD SWEEP — FINAL SUMMARY")
    print("=" * 100)

    models = sorted(set(r["model_name"] for r in _ALL_RUN_RESULTS))

    header = (
        f"{'Model':<45} {'Thr':>5} {'F1_c1':>8} {'Rec_c1':>8} "
        f"{'Prec_c1':>8} {'F1_c0':>8} {'F1_mic':>8} {'Runtime':>10}"
    )
    print(header)
    print("-" * 100)

    best_per_model: Dict[str, Dict] = {}

    for model in models:
        model_runs = sorted(
            [r for r in _ALL_RUN_RESULTS if r["model_name"] == model],
            key=lambda r: r["confidence_threshold"],
        )
        for r in model_runs:
            print(
                f"{r['model_name']:<45} "
                f"{r['confidence_threshold']:>5.1f} "
                f"{r['f1_binary_class_1']:>8.4f} "
                f"{r['recall_class_1']:>8.4f} "
                f"{r['precision_class_1']:>8.4f} "
                f"{r['f1_binary_class_0']:>8.4f} "
                f"{r['f1_micro']:>8.4f} "
                f"{r['total_runtime_seconds']:>9.2f}s"
            )

        best                 = max(model_runs, key=lambda r: r["f1_binary_class_1"])
        best_per_model[model] = best
        print()

    print("=" * 100)
    print("BEST THRESHOLD PER MODEL (by F1 class 1, pooled)")
    print("=" * 100)
    for model, best in best_per_model.items():
        print(f"\n  Model:     {model}")
        print(f"  Threshold: {best['confidence_threshold']}")
        print(f"  F1 c1:     {best['f1_binary_class_1']:.4f}")
        print(f"  Recall c1: {best['recall_class_1']:.4f}")
        print(f"  Prec c1:   {best['precision_class_1']:.4f}")
        print(f"  F1 c0:     {best['f1_binary_class_0']:.4f}")
        print(f"  F1 micro:  {best['f1_micro']:.4f}")
        print(f"  Runtime:   {best['total_runtime_seconds']:.2f}s")

    print("\n" + "=" * 100)

    summary = {
        "all_runs": _ALL_RUN_RESULTS,
        "best_per_model": {
            model: {
                "confidence_threshold":  best["confidence_threshold"],
                "f1_binary_class_1":     best["f1_binary_class_1"],
                "recall_class_1":        best["recall_class_1"],
                "precision_class_1":     best["precision_class_1"],
                "f1_binary_class_0":     best["f1_binary_class_0"],
                "f1_micro":              best["f1_micro"],
                "total_runtime_seconds": best["total_runtime_seconds"],
            }
            for model, best in best_per_model.items()
        },
    }
    summary_path = f"{DEFAULT_OUTPUT_PREFIX}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(
        description="W&B sweep: confidence threshold comparison for both models."
    )
    parser.add_argument("--entity",        type=str, default=WANDB_ENTITY,
                        help="W&B entity.")
    parser.add_argument("--project",       type=str, default=DEFAULT_SWEEP_PROJECT,
                        help="W&B project name for the sweep.")
    parser.add_argument("--sweep-id",      type=str, default=None,
                        help="Existing W&B sweep id. If omitted, a new sweep is created.")
    parser.add_argument("--count",         type=int,
                        default=len(SWEEP_THRESHOLDS) * len(MODELS_TO_EVALUATE),
                        help="How many agent runs to execute.")
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX,
                        help="Prefix for per-run local result files.")
    parser.add_argument("--seed",          type=int, default=SEED,
                        help="Global random seed.")
    parser.add_argument("--create-only",   action="store_true",
                        help="Create sweep and print id, but do not start the agent.")
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep_id: {sweep_id}")
    else:
        sweep_config = build_sweep_config(output_prefix=args.output_prefix, seed=args.seed)
        sweep_id     = wandb.sweep(sweep=sweep_config, entity=args.entity, project=args.project)
        print(f"Created sweep_id: {sweep_id}")

    if args.create_only:
        return

    print(f"Starting W&B agent for sweep_id={sweep_id} with count={args.count}")
    wandb.agent(
        sweep_id=sweep_id,
        function=evaluate_single_run,
        entity=args.entity,
        project=args.project,
        count=args.count,
    )

    print_final_summary()


if __name__ == "__main__":
    main()