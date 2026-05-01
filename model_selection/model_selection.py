"""
Hallucination Detection Benchmark — RQ3 Final Comparison

Runs three models in a single invocation on the same 450-sample evaluation set,
each at its own best confidence threshold (manually set below from the
threshold sweep results):

  - tinylettuce-ettin-68m-en              (encoder)
  - lettucedetect-base-modernbert-en-v1   (encoder)
  - lettuceprevent-ettin-decoder-68m-en   (decoder)

Simulates token-by-token incremental generation with a Llama-3.1-8B tokenizer.
Every detector receives prefixes that end exactly at a Llama token boundary,
mirroring how a real Llama-3.1-8B generator would emit text.

Dataset: wandb/RAGTruth-processed (test split)
Sample selection: first 150 unique (context, query) pairs per task type, in
dataset order. For each pair, the first-seen LLM answer in the dataset is
used. Total: 450 samples.

Outputs (per model):
  - {prefix}_{model}_per_step.csv         token-level details
  - {prefix}_{model}_full_results.json    full nested results

Outputs (combined):
  - {prefix}_aggregate.csv                model-level comparison
  - {prefix}_per_sample_comparison.json   sample-centric side-by-side

Logs to W&B (one run per model). The combined comparison JSON is logged as
an artifact to the last run.
"""

import argparse
import csv
import json
import random
import time
import warnings
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset

os.environ.setdefault("WEAVE_DISABLED", "true")
warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", message=r".*Expected `list\[str\]` but got `tuple`.*")


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

MAX_LENGTH = 4096

# ----------------------------------------------------------------------------
# Per-model best confidence thresholds (manually set from threshold sweep).
# ----------------------------------------------------------------------------
MODELS_TO_EVALUATE = [
    {
        "name":                 "tinylettuce-ettin-68m-en",
        "model_path":           "KRLabsOrg/tinylettuce-ettin-68m-en",
        "kind":                 "lettucedetect",
        "confidence_threshold": 0.5,   
    },
    {
        "name":                 "lettucedetect-base-modernbert-en-v1",
        "model_path":           "KRLabsOrg/lettucedect-base-modernbert-en-v1",
        "kind":                 "lettucedetect",
        "confidence_threshold": 0.6,    
    },
    {
        "name":                 "lettuceprevent-ettin-decoder-68m-en",
        "model_path":           "lebe1/lettuceprevent-ettin-decoder-68m-en",
        "kind":                 "lettuceprevent_decoder",
        "confidence_threshold": 0.8,    
    },
]

# Add near the other top-level config (after UNIQUE_PAIRS_PER_TASK)
LLMS_ROUND_ROBIN = [
    "gpt-4-0613",
    "gpt-3.5-turbo-0613",
    "mistral-7B-instruct",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
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
    confidence_threshold: float
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
# Decoder model definition
# ============================================================================

class EttinTokenClassifier(nn.Module):
    def __init__(self, config, num_labels: int = 2):
        super().__init__()
        from transformers import AutoModel
        self.num_labels = num_labels
        self.backbone   = AutoModel.from_config(config)
        self.dropout    = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.last_hidden_state))


# ============================================================================
# Llama-driven Ettin tokenization
# ============================================================================

def tokenize_text_via_llama_chunks(
    text: str,
    llama_tokenizer,
    ettin_tokenizer,
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    llama_enc     = llama_tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True,
    )
    llama_ids     = llama_enc["input_ids"]
    llama_offsets = llama_enc["offset_mapping"]

    ettin_ids: List[int]                = []
    char_offsets: List[Tuple[int, int]] = []
    llama_token_indices: List[int]      = []

    for li, (tok_id, (cs, ce)) in enumerate(zip(llama_ids, llama_offsets)):
        if cs == ce:
            continue
        chunk = llama_tokenizer.decode([tok_id])
        if not chunk:
            continue
        sub = ettin_tokenizer(chunk, add_special_tokens=False)["input_ids"]
        if not sub:
            continue
        ettin_ids.extend(sub)
        char_offsets.extend([(cs, ce)] * len(sub))
        llama_token_indices.extend([li] * len(sub))
    return ettin_ids, char_offsets, llama_token_indices


# ============================================================================
# Hallucination Detectors
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

    def reset_sample_cache(self) -> None:
        """Override to clear per-sample caches between samples."""
        pass


class LettuceDetectWrapper(HallucinationDetectorBase):
    """Wraps the LettuceDetect library (used for both TinyLettuce and base)."""

    def __init__(self, name: str, model_path: str):
        from lettucedetect.models.inference import HallucinationDetector
        self.name       = name
        self.model_path = model_path
        self.detector   = HallucinationDetector(
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


class LettucePreventDecoderWrapper(HallucinationDetectorBase):
    """
    Decoder model wrapper. Caches ctx/qry Llama tokenization per sample,
    cleared via reset_sample_cache between samples.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        llama_tokenizer_name: str,
        device: Optional[str] = None,
        max_length: int = MAX_LENGTH,
        torch_dtype: torch.dtype = torch.float32,
    ):
        from transformers import AutoTokenizer, AutoConfig
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors

        self.name       = name
        self.model_path = model_path
        self.max_length = max_length
        self.device     = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[{self.name}] Loading Ettin tokenizer from {model_path}")
        self.ettin_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.cls_id = self.ettin_tokenizer.cls_token_id
        self.sep_id = self.ettin_tokenizer.sep_token_id

        print(f"[{self.name}] Loading Llama tokenizer for boundary alignment")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer_name)

        print(f"[{self.name}] Loading model weights")
        config       = AutoConfig.from_pretrained(model_path)
        self.model   = EttinTokenClassifier(config, num_labels=2)
        weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        state_dict   = load_safetensors(weights_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[{self.name}] WARNING: missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[{self.name}] WARNING: unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        self.model.to(self.device).to(torch_dtype)
        self.model.eval()
        print(f"[{self.name}] Ready on {self.device} (raw softmax)")

        self._cached_ctx_ids: Optional[List[int]] = None
        self._cached_qry_ids: Optional[List[int]] = None
        self._cached_key:     Optional[Tuple[str, str]] = None

    def get_name(self) -> str:
        return self.name

    def reset_sample_cache(self) -> None:
        self._cached_ctx_ids = None
        self._cached_qry_ids = None
        self._cached_key     = None

    def _get_ctx_qry_ids(self, ctx_text: str, question: str) -> Tuple[List[int], List[int]]:
        key = (ctx_text, question)
        if self._cached_key == key and self._cached_ctx_ids is not None:
            return self._cached_ctx_ids, self._cached_qry_ids
        ctx_ids, _, _ = tokenize_text_via_llama_chunks(
            ctx_text, self.llama_tokenizer, self.ettin_tokenizer,
        )
        qry_ids, _, _ = tokenize_text_via_llama_chunks(
            question, self.llama_tokenizer, self.ettin_tokenizer,
        )
        self._cached_ctx_ids = ctx_ids
        self._cached_qry_ids = qry_ids
        self._cached_key     = key
        return ctx_ids, qry_ids

    @torch.no_grad()
    def predict(self, context, question, answer) -> List[Dict[str, Any]]:
        ctx_text = "\n".join(context) if isinstance(context, list) else context

        ctx_ids, qry_ids = self._get_ctx_qry_ids(ctx_text, question)
        ans_ids, ans_offs, ans_lidx = tokenize_text_via_llama_chunks(
            answer, self.llama_tokenizer, self.ettin_tokenizer,
        )

        if not ans_ids:
            return []

        budget = self.max_length - 2
        if len(ans_ids) >= budget:
            ans_ids   = ans_ids[:budget]
            ans_offs  = ans_offs[:budget]
            ans_lidx  = ans_lidx[:budget]
            input_ids = [self.cls_id] + ans_ids + [self.sep_id]
            answer_start = 1
        else:
            ctx_qry_budget = self.max_length - len(ans_ids) - 4
            ctx_local = ctx_ids
            qry_local = qry_ids
            if len(ctx_local) + len(qry_local) > ctx_qry_budget:
                excess = (len(ctx_local) + len(qry_local)) - ctx_qry_budget
                if excess <= len(ctx_local):
                    ctx_local = ctx_local[excess:]
                else:
                    leftover  = excess - len(ctx_local)
                    ctx_local = []
                    qry_local = qry_local[leftover:]
            prefix_ids = (
                [self.cls_id] + ctx_local + [self.sep_id]
                + qry_local + [self.sep_id]
            )
            input_ids    = prefix_ids + ans_ids + [self.sep_id]
            answer_start = len(prefix_ids)

        attention_mask = [1] * len(input_ids)
        input_ids_t      = torch.tensor([input_ids],      dtype=torch.long, device=self.device)
        attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=self.device)

        logits = self.model(input_ids=input_ids_t, attention_mask=attention_mask_t)
        probs_class_1 = torch.softmax(logits, dim=-1)[0, :, 1].float().cpu().numpy()
        ans_probs = probs_class_1[answer_start : answer_start + len(ans_ids)]

        groups: Dict[int, Dict[str, Any]] = {}
        for prob, (cs, ce), lidx in zip(ans_probs, ans_offs, ans_lidx):
            existing = groups.get(lidx)
            if existing is None or prob > existing["confidence"]:
                groups[lidx] = {
                    "start":      int(cs),
                    "end":        int(ce),
                    "confidence": float(prob),
                }

        return sorted(groups.values(), key=lambda d: d["start"])


def build_detector(model_cfg: Dict) -> HallucinationDetectorBase:
    kind = model_cfg["kind"]
    if kind == "lettucedetect":
        return LettuceDetectWrapper(
            name=model_cfg["name"],
            model_path=model_cfg["model_path"],
        )
    if kind == "lettuceprevent_decoder":
        return LettucePreventDecoderWrapper(
            name=model_cfg["name"],
            model_path=model_cfg["model_path"],
            llama_tokenizer_name=LLM_TOKENIZER_NAME,
        )
    raise ValueError(f"Unknown detector kind: {kind}")


# ============================================================================
# Tokenizer
# ============================================================================

class LLMTokenizerWrapper:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def get_name(self) -> str:
        return f"LLMTokenizer({self.model_name})"

    def tokenize_with_offsets(self, text: str) -> List[TokenInfo]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = []
        for i, _ in enumerate(encoded):
            prefix      = self.tokenizer.decode(encoded[: i + 1], skip_special_tokens=True)
            prev_prefix = self.tokenizer.decode(encoded[:i],      skip_special_tokens=True) if i > 0 else ""
            char_start  = len(prev_prefix)
            char_end    = len(prefix)
            tokens.append(TokenInfo(
                index=i, text=prefix[char_start:],
                char_start=char_start, char_end=char_end,
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
        self.original    = original
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
                i += 1; j += 1
            elif self.original[i] in (' ', '\n', '\t'):
                i += 1
            elif self.detokenized[j] in (' ', '\n', '\t'):
                j += 1
            else:
                i += 1; j += 1

    def detok_range_to_orig(self, start: int, end: int) -> Tuple[int, int]:
        orig_start = self.detok_to_orig.get(start)
        orig_end   = self.detok_to_orig.get(end - 1)
        if orig_start is None:
            for s in range(start, end):
                if s in self.detok_to_orig:
                    orig_start = self.detok_to_orig[s]; break
            if orig_start is None:
                orig_start = start
        if orig_end is None:
            for e in range(end - 1, start - 1, -1):
                if e in self.detok_to_orig:
                    orig_end = self.detok_to_orig[e]; break
            if orig_end is None:
                orig_end = end - 1
        return orig_start, orig_end + 1

    def check_divergence(self) -> float:
        if len(self.detokenized) == 0:
            return 0.0
        return 1.0 - (len(self.detok_to_orig) / len(self.detokenized))


# ============================================================================
# Label Parsing & Helpers
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
    for label in labels:
        if token.char_start < label.end and token.char_end > label.start:
            return True
    return False


def map_predictions_to_tokens(
    predictions: List[Dict[str, Any]],
    tokens: List[TokenInfo],
    confidence_threshold: float,
) -> Dict[int, Tuple[bool, float]]:
    result = {}
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
# Metrics
# ============================================================================

def compute_metrics_class(gts, preds, positive_class=True):
    tp = sum(1 for g, p in zip(gts, preds) if g == positive_class and p == positive_class)
    fp = sum(1 for g, p in zip(gts, preds) if g != positive_class and p == positive_class)
    fn = sum(1 for g, p in zip(gts, preds) if g == positive_class and p != positive_class)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_metrics_micro(gts, preds):
    if not gts:
        return 0.0, 0.0, 0.0
    correct = sum(1 for g, p in zip(gts, preds) if g == p)
    acc = correct / len(gts)
    return acc, acc, acc


def compute_nine_metrics(gts, preds) -> Dict[str, float]:
    p_mic, r_mic, f_mic = compute_metrics_micro(gts, preds)
    p_c1, r_c1, f_c1    = compute_metrics_class(gts, preds, True)
    p_c0, r_c0, f_c0    = compute_metrics_class(gts, preds, False)
    return {
        "precision_micro":   p_mic, "recall_micro":   r_mic, "f1_micro":          f_mic,
        "precision_class_1": p_c1, "recall_class_1": r_c1, "f1_binary_class_1": f_c1,
        "precision_class_0": p_c0, "recall_class_0": r_c0, "f1_binary_class_0": f_c0,
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
        confidence_threshold: float,
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
        # Reset detector's per-sample cache (decoder ctx/qry tokens).
        self.detector.reset_sample_cache()

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
            print(f"  [WARNING] Sample {sample_index}: detokenization divergence "
                  f"= {divergence:.2%}")

        token_steps:     List[TokenStepResult]            = []
        last_prediction: Dict[int, Tuple[bool, float]]    = {}
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
                        context=[context], question=query, answer=prefix_text,
                    )
                except Exception as e:
                    print(f"  [ERROR] Detection failed at step {step}: {e}")
                    predictions = []
                last_prediction = map_predictions_to_tokens(
                    predictions, tokens[: step + 1], self.confidence_threshold,
                )

            if token.index in last_prediction:
                is_pred_hall, pred_conf = last_prediction[token.index]
            else:
                is_pred_hall, pred_conf = False, 0.0

            orig_start, orig_end = offset_mapper.detok_range_to_orig(
                token.char_start, token.char_end
            )
            orig_token = TokenInfo(
                index=token.index, text=token.text,
                char_start=orig_start, char_end=orig_end,
            )
            is_gt_hall = is_token_hallucinated(orig_token, gt_labels)

            cumulative_gt.append(is_gt_hall)
            cumulative_pred.append(is_pred_hall)
            cum_prec, cum_rec, cum_f1 = compute_metrics_class(
                cumulative_gt, cumulative_pred, positive_class=True
            )

            token_steps.append(TokenStepResult(
                step=step, token=token.text,
                token_char_start=token.char_start, token_char_end=token.char_end,
                ground_truth=is_gt_hall, predicted=is_pred_hall,
                confidence=pred_conf, was_triggered=triggered,
                cumulative_precision=cum_prec, cumulative_recall=cum_rec, cumulative_f1=cum_f1,
            ))

        sample_runtime = time.time() - sample_start_time
        all_gt   = [ts.ground_truth for ts in token_steps]
        all_pred = [ts.predicted     for ts in token_steps]
        agg_prec, agg_rec, agg_f1 = compute_metrics_class(all_gt, all_pred, positive_class=True)
        nine = compute_nine_metrics(all_gt, all_pred)

        return SampleResult(
            sample_index=sample_index, task_type=task_type,
            context=context, query=query, answer=answer,
            token_steps=token_steps,
            aggregate_precision=agg_prec, aggregate_recall=agg_rec, aggregate_f1=agg_f1,
            runtime_seconds=sample_runtime,
            **{k: nine[k] for k in nine},
        )


# ============================================================================
# Dataset Loading — 150 unique (context, query) per task_type
# ============================================================================

def load_benchmark_samples() -> Tuple[List[Dict], List[Dict]]:
    """
    Select the first UNIQUE_PAIRS_PER_TASK unique (context, query) pairs per
    task type, in dataset order. For each pair, assign one LLM answer in
    round-robin order across LLMS_ROUND_ROBIN. This matches the threshold
    sweep's selection policy, so the sweep's 50-per-task subset is a strict
    subset of this 150-per-task evaluation set.

    Returns (benchmark_samples, original_rows) where original_rows contains
    the full HF dataset row for each chosen sample.
    """
    print(f"Loading HF dataset: {HF_DATASET_NAME} (split={HF_DATASET_SPLIT})...")
    hf_ds      = load_dataset(HF_DATASET_NAME)
    test_split = hf_ds[HF_DATASET_SPLIT]

    # Walk dataset in order. For each task, collect first UNIQUE_PAIRS_PER_TASK
    # unique prompts and, for each prompt, maintain a model -> row map.
    rows_by_prompt:        Dict[Tuple[str, str, str], Dict[str, Dict]] = {}
    prompt_order_per_task: Dict[str, List[Tuple[str, str]]]            = {}
    seen_per_task:         Dict[str, set]                              = {}

    for row in test_split:
        if row.get("hallucination_labels") in (None, ""):
            continue
        tt  = row.get("task_type", "unknown")
        ctx = row["context"]
        qry = row["query"]
        full_key = (tt, ctx, qry)

        rows_by_prompt.setdefault(full_key, {})
        rows_by_prompt[full_key][row.get("model", "unknown")] = dict(row)

        seen_per_task.setdefault(tt, set())
        prompt_order_per_task.setdefault(tt, [])
        if (ctx, qry) not in seen_per_task[tt] and len(prompt_order_per_task[tt]) < UNIQUE_PAIRS_PER_TASK:
            seen_per_task[tt].add((ctx, qry))
            prompt_order_per_task[tt].append((ctx, qry))

    # Build sample list with round-robin LLM assignment
    benchmark_samples: List[Dict] = []
    original_rows:     List[Dict] = []
    for tt in sorted(prompt_order_per_task.keys()):
        prompts = prompt_order_per_task[tt]
        print(f"  task_type={tt}: {len(prompts)} prompts selected")
        skipped = 0
        for i, (ctx, qry) in enumerate(prompts):
            full_key      = (tt, ctx, qry)
            preferred_llm = LLMS_ROUND_ROBIN[i % len(LLMS_ROUND_ROBIN)]
            row_map       = rows_by_prompt[full_key]
            if preferred_llm in row_map:
                row = row_map[preferred_llm]
            else:
                fallback = next(iter(row_map.values()))
                print(f"    [WARN] prompt {i} task={tt}: preferred LLM "
                      f"'{preferred_llm}' not found, using '{fallback.get('model')}'")
                row = fallback
                skipped += 1
            benchmark_samples.append({
                "task_type": tt,
                "context":   row["context"],
                "query":     row["query"],
                "answer":    row["output"],
                "model":     row.get("model", "unknown"),
                "labels":    parse_hallucination_labels(row["hallucination_labels"]),
            })
            original_rows.append(row)
        if skipped > 0:
            print(f"    [WARN] {skipped} prompts used fallback LLM in task {tt}")

    print(f"\nTotal benchmark samples: {len(benchmark_samples)}")

    # Distribution sanity check
    by_model: Dict[str, int] = {}
    for s in benchmark_samples:
        by_model[s["model"]] = by_model.get(s["model"], 0) + 1
    print("LLM distribution:")
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n}")

    return benchmark_samples, original_rows

# ============================================================================
# Export functions
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
            "model_name", "model_path", "confidence_threshold",
            "trigger_strategy", "tokenizer", "num_samples",
            "precision_micro", "recall_micro", "f1_micro",
            "precision_class_1", "recall_class_1", "f1_binary_class_1",
            "precision_class_0", "recall_class_0", "f1_binary_class_0",
            "total_runtime_seconds",
        ])
        for br in benchmark_results:
            writer.writerow([
                br.model_name, br.model_path, f"{br.confidence_threshold:.4f}",
                br.trigger_strategy, br.tokenizer_name, br.num_samples,
                f"{br.precision_micro:.4f}", f"{br.recall_micro:.4f}", f"{br.f1_micro:.4f}",
                f"{br.precision_class_1:.4f}", f"{br.recall_class_1:.4f}", f"{br.f1_binary_class_1:.4f}",
                f"{br.precision_class_0:.4f}", f"{br.recall_class_0:.4f}", f"{br.f1_binary_class_0:.4f}",
                f"{br.total_runtime_seconds:.2f}",
            ])


def export_results_json(benchmark_result: BenchmarkResult, filepath: str):
    output = {
        "model_name":            benchmark_result.model_name,
        "model_path":            benchmark_result.model_path,
        "confidence_threshold":  benchmark_result.confidence_threshold,
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
        output["samples"].append({
            "sample_index":        sr.sample_index,
            "task_type":           sr.task_type,
            "aggregate_precision": sr.aggregate_precision,
            "aggregate_recall":    sr.aggregate_recall,
            "aggregate_f1":        sr.aggregate_f1,
            "runtime_seconds":     sr.runtime_seconds,
            "token_steps":         [asdict(ts) for ts in sr.token_steps],
        })
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def export_per_sample_comparison(
    original_rows: List[Dict],
    benchmark_samples: List[Dict],
    model_results: Dict[str, List[SampleResult]],
    model_thresholds: Dict[str, float],
    filepath: str,
):
    output = {
        "dataset":               HF_DATASET_NAME,
        "dataset_split":         HF_DATASET_SPLIT,
        "unique_pairs_per_task": UNIQUE_PAIRS_PER_TASK,
        "tokenizer":             LLM_TOKENIZER_NAME,
        "models":                list(model_results.keys()),
        "model_thresholds":      model_thresholds,
        "num_samples":           len(benchmark_samples),
        "samples":               [],
    }

    model_names = list(model_results.keys())
    for i, (orig_row, bench_sample) in enumerate(zip(original_rows, benchmark_samples)):
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
                "confidence_threshold": model_thresholds[model_name],
                "predictions":          [ts.predicted   for ts in sr.token_steps],
                "confidences":          [ts.confidence  for ts in sr.token_steps],
                "runtime_seconds":      sr.runtime_seconds,
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

        output["samples"].append({
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
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ============================================================================
# Benchmark Runner (per model)
# ============================================================================

def run_benchmark_for_model(
    samples: List[Dict],
    model_cfg: Dict,
    llm_tokenizer: LLMTokenizerWrapper,
    output_prefix: str,
) -> BenchmarkResult:
    model_name           = model_cfg["name"]
    model_path           = model_cfg["model_path"]
    confidence_threshold = float(model_cfg["confidence_threshold"])

    print(f"\n{'#' * 70}")
    print(f"# Evaluating: {model_name}")
    print(f"# Path:       {model_path}")
    print(f"# Threshold:  {confidence_threshold}")
    print(f"{'#' * 70}\n")

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=model_name,
        config={
            "model_name":            model_name,
            "model_path":            model_path,
            "model_kind":            model_cfg["kind"],
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

    detector = build_detector(model_cfg)
    engine   = EvaluationEngine(
        detector=detector, llm_tokenizer=llm_tokenizer,
        confidence_threshold=confidence_threshold,
    )

    all_sample_results: List[SampleResult] = []
    all_gt:   List[bool] = []
    all_pred: List[bool] = []
    total_start = time.time()

    for i, sample in enumerate(samples):
        print(f"  [{model_name}] Evaluating sample {i + 1}/{len(samples)} "
              f"(task={sample['task_type']})...")
        result = engine.evaluate_sample(
            context=sample["context"], query=sample["query"],
            answer=sample["answer"], labels=sample["labels"],
            task_type=sample["task_type"], sample_index=i,
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
    prec_c1,    rec_c1,    f1_c1    = compute_metrics_class(all_gt, all_pred, positive_class=True)
    prec_c0,    rec_c0,    f1_c0    = compute_metrics_class(all_gt, all_pred, positive_class=False)

    benchmark_result = BenchmarkResult(
        model_name=model_name, model_path=model_path,
        confidence_threshold=confidence_threshold,
        trigger_strategy=TriggerStrategy.EVERY_TOKEN.value,
        tokenizer_name=LLM_TOKENIZER_NAME, num_samples=len(samples),
        precision_micro=prec_micro, recall_micro=rec_micro, f1_micro=f1_micro,
        precision_class_1=prec_c1, recall_class_1=rec_c1, f1_binary_class_1=f1_c1,
        precision_class_0=prec_c0, recall_class_0=rec_c0, f1_binary_class_0=f1_c0,
        total_runtime_seconds=total_runtime,
        sample_results=all_sample_results,
    )

    run.summary["confidence_threshold"]  = confidence_threshold
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
    run.log_artifact(artifact)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {model_name} (thr={confidence_threshold})")
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

    # Free GPU memory before loading the next detector.
    del detector, engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return benchmark_result


# ============================================================================
# Main
# ============================================================================

def main(output_prefix: str):
    benchmark_samples, original_rows = load_benchmark_samples()

    print(f"Loading Llama tokenizer: {LLM_TOKENIZER_NAME}...")
    llm_tokenizer = LLMTokenizerWrapper(LLM_TOKENIZER_NAME)

    all_results: List[BenchmarkResult] = []
    model_results:    Dict[str, List[SampleResult]] = {}
    model_thresholds: Dict[str, float]              = {}

    for model_cfg in MODELS_TO_EVALUATE:
        result = run_benchmark_for_model(
            samples=benchmark_samples, model_cfg=model_cfg,
            llm_tokenizer=llm_tokenizer, output_prefix=output_prefix,
        )
        all_results.append(result)
        model_results[model_cfg["name"]]    = result.sample_results
        model_thresholds[model_cfg["name"]] = float(model_cfg["confidence_threshold"])

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
        model_thresholds=model_thresholds,
        filepath=comparison_path,
    )
    print(f"Per-sample comparison JSON written to: {comparison_path}")

    # Attach combined artifacts to the last model's run.
    last_model_name = MODELS_TO_EVALUATE[-1]["name"]
    print(f"\nAttaching combined comparison artifact to W&B run: {last_model_name}")

    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, name=last_model_name,
        resume="allow", reinit=True,
    )
    artifact = wandb.Artifact("per_sample_comparison", type="comparison_results")
    artifact.add_file(comparison_path)
    artifact.add_file(aggregate_path)
    run.log_artifact(artifact)
    run.finish()

    # Final print
    print(f"\n{'=' * 80}")
    print(f"FINAL COMPARISON (each model at its own best threshold)")
    print(f"{'=' * 80}")
    print(f"{'Model':<42} {'Thr':>5} {'F1_c1':>8} {'F1_c0':>8} {'F1_mic':>8} {'Runtime':>10}")
    print("-" * 80)
    for br in all_results:
        print(f"{br.model_name:<42} "
              f"{br.confidence_threshold:>5.2f} "
              f"{br.f1_binary_class_1:>8.4f} "
              f"{br.f1_binary_class_0:>8.4f} "
              f"{br.f1_micro:>8.4f} "
              f"{br.total_runtime_seconds:>9.2f}s")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ3 Final Benchmark: TinyLettuce vs LettuceDetect-base vs LettucePrevent decoder"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="rq3_final_benchmark",
        help="Prefix for output files."
    )
    args = parser.parse_args()
    main(output_prefix=args.output_prefix)