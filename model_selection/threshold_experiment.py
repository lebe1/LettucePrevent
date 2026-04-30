"""
Unified threshold sweep for RQ3: fair comparison of HDMs for real-time
hallucination prevention.

Three models on equal footing:
  - tinylettuce-ettin-68m-en              (encoder, char-span output)
  - lettucedetect-base-modernbert-en-v1   (encoder, char-span output)
  - lettuceprevent-ettin-decoder-68m-en   (decoder, Llama-aligned output)

Thresholds: [0.5, 0.6, 0.7, 0.8, 0.9]
Sample selection: 50 unique (context, query) prompts per task type, in
dataset order, with one LLM answer assigned per prompt in round-robin order.
Total: 50 x 3 = 150 samples per (model, threshold) combination.

Timing scope (RQ3-fair):
  - Wraps ONLY detector.predict(...)         -> forward pass + alignment
  - Excludes Llama tokenization of prefix    -> shared infrastructure
  - Excludes ground-truth + metric computation
  - Reports both: total per-sample runtime AND mean per-step latency

Decoder optimization:
  - Context/query Llama-tokenization cached per sample (legitimate, since
    they are constant across all token steps).
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset

from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

os.environ.setdefault("WEAVE_DISABLED", "true")
warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", message=r".*Expected `list\[str\]` but got `tuple`.*")


# ============================================================================
# Configuration
# ============================================================================

SEED               = 42
WANDB_ENTITY       = "lebeccard-technical-university-wien"
LLM_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"
HF_DATASET_NAME    = "wandb/RAGTruth-processed"
HF_DATASET_SPLIT   = "test"

MODELS_TO_EVALUATE = [
    {
        "name":       "tinylettuce-ettin-68m-en",
        "model_path": "KRLabsOrg/tinylettuce-ettin-68m-en",
        "kind":       "lettucedetect",
    },
    {
        "name":       "lettucedetect-base-modernbert-en-v1",
        "model_path": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
        "kind":       "lettucedetect",
    },
    {
        "name":       "lettuceprevent-ettin-decoder-68m-en",
        "model_path": "lebe1/lettuceprevent-ettin-decoder-68m-en",
        "kind":       "lettuceprevent_decoder",
    },
]

MAX_LENGTH                 = 4096
SWEEP_PROMPTS_PER_TASK     = 50
SWEEP_THRESHOLDS           = [0.5, 0.6, 0.7, 0.8, 0.9]
LLMS_ROUND_ROBIN           = [
    "gpt-4-0613",
    "gpt-3.5-turbo-0613",
    "mistral-7B-instruct",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
]

DEFAULT_SWEEP_NAME    = "rq3-unified-threshold-sweep"
DEFAULT_SWEEP_PROJECT = "hdm-rq3-unified"
DEFAULT_OUTPUT_PREFIX = "rq3_unified_sweep"


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TokenInfo:
    index: int
    text: str
    char_start: int
    char_end: int


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
    predict_latency_s: float


@dataclass
class SampleResult:
    sample_index: int
    task_type: str
    model: str
    context: str
    query: str
    answer: str
    token_steps: List[TokenStepResult] = field(default_factory=list)
    total_predict_runtime_s: float = 0.0
    mean_step_latency_s: float = 0.0
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
# Llama tokenizer wrapper (shared, used by engine for token boundaries + GT)
# ============================================================================

class LLMTokenizerWrapper:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def tokenize_with_offsets(self, text: str) -> List[TokenInfo]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        tokens: List[TokenInfo] = []
        for i, _ in enumerate(encoded):
            prefix      = self.tokenizer.decode(encoded[: i + 1], skip_special_tokens=True)
            prev_prefix = self.tokenizer.decode(encoded[:i],      skip_special_tokens=True) if i > 0 else ""
            tokens.append(TokenInfo(
                index=i, text=prefix[len(prev_prefix):],
                char_start=len(prev_prefix), char_end=len(prefix),
            ))
        return tokens

    def get_detokenized_text(self, text: str) -> str:
        return self.tokenizer.decode(
            self.tokenizer.encode(text, add_special_tokens=False),
            skip_special_tokens=True,
        )


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
# Detector base
# ============================================================================

class HallucinationDetectorBase(ABC):
    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def predict(
        self,
        context: str,
        question: str,
        answer_prefix: str,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of {start, end, confidence} dicts in CHARACTER offsets
        of `answer_prefix`.
        """
        ...

    def reset_sample_cache(self) -> None:
        """Override to clear per-sample caches between samples."""
        pass


# ============================================================================
# LettuceDetect / TinyLettuce wrapper
# ============================================================================

class LettuceDetectWrapper(HallucinationDetectorBase):
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

    def predict(self, context, question, answer_prefix):
        return self.detector.predict(
            context=[context] if isinstance(context, str) else context,
            question=question,
            answer=answer_prefix,
            output_format="spans",
        )


# ============================================================================
# LettucePrevent decoder wrapper (with ctx/qry caching)
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


class LettucePreventDecoderWrapper(HallucinationDetectorBase):
    """
    Decoder model wrapper. Caches ctx/qry Llama tokenization per sample
    (cleared via reset_sample_cache).
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

        # Per-sample cache for ctx/qry tokenization
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
    def predict(self, context, question, answer_prefix) -> List[Dict[str, Any]]:
        ctx_text = "\n".join(context) if isinstance(context, list) else context

        ctx_ids, qry_ids = self._get_ctx_qry_ids(ctx_text, question)
        ans_ids, ans_offs, ans_lidx = tokenize_text_via_llama_chunks(
            answer_prefix, self.llama_tokenizer, self.ettin_tokenizer,
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


# ============================================================================
# Detector factory
# ============================================================================

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
# Offset mapping, label parsing, metrics (shared)
# ============================================================================

class OffsetMapper:
    def __init__(self, original: str, detokenized: str):
        self.original    = original
        self.detokenized = detokenized
        self._build_mapping()

    def _build_mapping(self):
        self.detok_to_orig: Dict[int, int] = {}
        i, j = 0, 0
        while i < len(self.original) and j < len(self.detokenized):
            if self.original[i] == self.detokenized[j]:
                self.detok_to_orig[j] = i
                i += 1; j += 1
            elif self.original[i] in (' ', '\n', '\t'):
                i += 1
            elif self.detokenized[j] in (' ', '\n', '\t'):
                j += 1
            else:
                i += 1; j += 1

    def detok_range_to_orig(self, start, end):
        s = self.detok_to_orig.get(start)
        e = self.detok_to_orig.get(end - 1)
        if s is None:
            for x in range(start, end):
                if x in self.detok_to_orig:
                    s = self.detok_to_orig[x]; break
            if s is None:
                s = start
        if e is None:
            for x in range(end - 1, start - 1, -1):
                if x in self.detok_to_orig:
                    e = self.detok_to_orig[x]; break
            if e is None:
                e = end - 1
        return s, e + 1

    def check_divergence(self):
        return 1.0 - (len(self.detok_to_orig) / max(len(self.detokenized), 1))


def parse_hallucination_labels(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    return [l for l in raw if isinstance(l, dict) and "start" in l and "end" in l]


def is_token_hallucinated(token, labels):
    return any(token.char_start < l.end and token.char_end > l.start for l in labels)


def map_predictions_to_tokens(predictions, tokens, threshold):
    result = {}
    for token in tokens:
        max_conf = 0.0
        is_hall  = False
        for pred in predictions:
            if token.char_start < pred["end"] and token.char_end > pred["start"]:
                if pred["confidence"] > max_conf:
                    max_conf = pred["confidence"]
                    is_hall  = pred["confidence"] >= threshold
        result[token.index] = (is_hall, max_conf)
    return result


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
    acc = sum(1 for g, p in zip(gts, preds) if g == p) / len(gts)
    return acc, acc, acc


def compute_nine_metrics(gts, preds):
    p_mic, r_mic, f_mic = compute_metrics_micro(gts, preds)
    p_c1, r_c1, f_c1    = compute_metrics_class(gts, preds, True)
    p_c0, r_c0, f_c0    = compute_metrics_class(gts, preds, False)
    return {
        "precision_micro":   p_mic, "recall_micro":   r_mic, "f1_micro":          f_mic,
        "precision_class_1": p_c1, "recall_class_1": r_c1, "f1_binary_class_1": f_c1,
        "precision_class_0": p_c0, "recall_class_0": r_c0, "f1_binary_class_0": f_c0,
    }


def _per_sample_class_metric(gt, pred, positive_class):
    tp = sum(1 for g, p in zip(gt, pred) if g == positive_class and p == positive_class)
    fp = sum(1 for g, p in zip(gt, pred) if g != positive_class and p == positive_class)
    fn = sum(1 for g, p in zip(gt, pred) if g == positive_class and p != positive_class)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def _nanmean(vs):
    clean = [v for v in vs if not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


# ============================================================================
# Evaluation engine (shared, identical for all detectors)
# ============================================================================

class EvaluationEngine:
    """
    The engine is detector-agnostic. It:
      - Llama-tokenizes the answer once per sample (shared infrastructure).
      - At each step, slices the prefix text and calls detector.predict(...).
      - Times ONLY detector.predict(...). All other work (Llama tokenization,
        ground-truth checks, metrics) is excluded from the runtime metric.
    """

    def __init__(self, detector, llm_tokenizer, confidence_threshold):
        self.detector             = detector
        self.llm_tokenizer        = llm_tokenizer
        self.confidence_threshold = confidence_threshold

    def evaluate_sample(self, context, query, answer, labels, task_type, model_name, sample_index):
        # Reset detector's per-sample cache (e.g., decoder ctx/qry cache).
        self.detector.reset_sample_cache()

        gt_labels = [
            GroundTruthLabel(l["start"], l["end"], l.get("label", "hallucination"))
            for l in labels
        ]
        tokens        = self.llm_tokenizer.tokenize_with_offsets(answer)
        detok_full    = self.llm_tokenizer.get_detokenized_text(answer)
        offset_mapper = OffsetMapper(answer, detok_full)
        if offset_mapper.check_divergence() > 0.05:
            print(f"  [WARN] Sample {sample_index}: detok divergence "
                  f"= {offset_mapper.check_divergence():.2%}")

        token_steps:     List[TokenStepResult]       = []
        last_prediction: Dict[int, Tuple[bool, float]] = {}
        encoded_full = self.llm_tokenizer.tokenizer.encode(answer, add_special_tokens=False)

        total_predict_runtime = 0.0
        for step, token in enumerate(tokens):
            # Llama prefix construction is SHARED INFRASTRUCTURE -> not timed.
            prefix_text = self.llm_tokenizer.tokenizer.decode(
                encoded_full[: step + 1], skip_special_tokens=True,
            )

            # === TIMED REGION: detector.predict only =========================
            t0 = time.perf_counter()
            try:
                preds = self.detector.predict(
                    context=context, question=query, answer_prefix=prefix_text,
                )
            except Exception as e:
                print(f"  [ERROR] step {step}: {e}")
                preds = []
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_latency = time.perf_counter() - t0
            # =================================================================

            total_predict_runtime += step_latency

            last_prediction = map_predictions_to_tokens(
                preds, tokens[: step + 1], self.confidence_threshold,
            )
            is_pred, pred_conf = last_prediction.get(token.index, (False, 0.0))

            os_, oe_   = offset_mapper.detok_range_to_orig(token.char_start, token.char_end)
            orig_token = TokenInfo(token.index, token.text, os_, oe_)
            is_gt      = is_token_hallucinated(orig_token, gt_labels)

            token_steps.append(TokenStepResult(
                step=step, token=token.text,
                token_char_start=token.char_start, token_char_end=token.char_end,
                ground_truth=is_gt, predicted=is_pred,
                confidence=pred_conf, predict_latency_s=step_latency,
            ))

        all_gt   = [ts.ground_truth for ts in token_steps]
        all_pred = [ts.predicted     for ts in token_steps]
        nine     = compute_nine_metrics(all_gt, all_pred)

        n_steps = max(len(token_steps), 1)
        return SampleResult(
            sample_index=sample_index, task_type=task_type, model=model_name,
            context=context, query=query, answer=answer,
            token_steps=token_steps,
            total_predict_runtime_s=total_predict_runtime,
            mean_step_latency_s=total_predict_runtime / n_steps,
            **{k: nine[k] for k in nine},
        )


# ============================================================================
# Sample selection (round-robin LLM, identical to both source scripts)
# ============================================================================

def load_sweep_samples(n_prompts_per_task: int) -> List[Dict]:
    print(f"Loading {HF_DATASET_NAME} (split={HF_DATASET_SPLIT})...")
    hf_ds      = load_dataset(HF_DATASET_NAME)
    test_split = hf_ds[HF_DATASET_SPLIT]

    rows_by_prompt:        Dict[Tuple[str, str, str], Dict[str, Dict]] = {}
    prompt_order_per_task: Dict[str, List[Tuple[str, str]]]            = {}
    seen_per_task:         Dict[str, set]                              = {}

    for row in test_split:
        if row.get("hallucination_labels") in (None, ""):
            continue
        tt  = row.get("task_type", "unknown")
        ctx = row["context"]; qry = row["query"]
        full_key = (tt, ctx, qry)
        rows_by_prompt.setdefault(full_key, {})
        rows_by_prompt[full_key][row.get("model", "unknown")] = dict(row)
        seen_per_task.setdefault(tt, set())
        prompt_order_per_task.setdefault(tt, [])
        if (ctx, qry) not in seen_per_task[tt]:
            seen_per_task[tt].add((ctx, qry))
            prompt_order_per_task[tt].append((ctx, qry))

    samples: List[Dict] = []
    for tt in sorted(prompt_order_per_task.keys()):
        prompts = prompt_order_per_task[tt][:n_prompts_per_task]
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
            samples.append({
                "task_type": tt,
                "context":   row["context"],
                "query":     row["query"],
                "answer":    row["output"],
                "model":     row.get("model", "unknown"),
                "labels":    parse_hallucination_labels(row["hallucination_labels"]),
            })
        if skipped > 0:
            print(f"    [WARN] {skipped} prompts used fallback LLM")

    print(f"\nTotal sweep samples: {len(samples)}")
    by_model: Dict[str, int] = {}
    for s in samples:
        by_model[s["model"]] = by_model.get(s["model"], 0) + 1
    print("LLM distribution:")
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n}")
    return samples


# ============================================================================
# Sweep machinery
# ============================================================================

_SAMPLES_CACHE:    Optional[List[Dict]]              = None
_TOKENIZER_CACHE:  Optional[LLMTokenizerWrapper]     = None
_DETECTOR_CACHE:   Optional[HallucinationDetectorBase] = None
_DETECTOR_NAME:    Optional[str]                     = None
_ALL_RUN_RESULTS:  List[Dict]                        = []


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_samples(prompts_per_task: int) -> List[Dict]:
    global _SAMPLES_CACHE
    if _SAMPLES_CACHE is not None:
        return _SAMPLES_CACHE
    _SAMPLES_CACHE = load_sweep_samples(prompts_per_task)
    return _SAMPLES_CACHE


def build_sample_manifest_hash(samples):
    payload = [
        {"task_type": s["task_type"], "model": s["model"],
         "context": s["context"], "query": s["query"], "answer": s["answer"]}
        for s in samples
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_tokenizer():
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = LLMTokenizerWrapper(LLM_TOKENIZER_NAME)
    return _TOKENIZER_CACHE


def get_detector(model_cfg):
    global _DETECTOR_CACHE, _DETECTOR_NAME
    if _DETECTOR_CACHE is None or _DETECTOR_NAME != model_cfg["name"]:
        # Free GPU memory from any previously-loaded detector
        if _DETECTOR_CACHE is not None:
            del _DETECTOR_CACHE
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _DETECTOR_CACHE = build_detector(model_cfg)
        _DETECTOR_NAME  = model_cfg["name"]
    return _DETECTOR_CACHE


def build_sweep_config(output_prefix, seed, prompts_per_task):
    return {
        "name":   DEFAULT_SWEEP_NAME,
        "method": "grid",
        "metric": {"name": "f1_binary_class_1", "goal": "maximize"},
        "parameters": {
            "confidence_threshold": {"values": SWEEP_THRESHOLDS},
            "model_idx":             {"values": list(range(len(MODELS_TO_EVALUATE)))},
            "output_prefix":         {"value":  output_prefix},
            "seed":                  {"value":  seed},
            "prompts_per_task":      {"value":  prompts_per_task},
        },
    }


# ============================================================================
# Single sweep run
# ============================================================================

def evaluate_single_run():
    run = wandb.init()
    cfg = wandb.config

    model_idx        = int(cfg.model_idx)
    threshold        = float(cfg.confidence_threshold)
    output_prefix    = str(cfg.output_prefix)
    seed             = int(cfg.seed)
    prompts_per_task = int(cfg.prompts_per_task)
    model_cfg        = MODELS_TO_EVALUATE[model_idx]

    thr_label = f"{threshold:.1f}".replace(".", "_")
    run.name  = f"{model_cfg['name']}_thr_{thr_label}"

    set_seed(seed)

    print("\n" + "#" * 70)
    print(f"# Run:        {run.name}")
    print(f"# Model:      {model_cfg['name']} ({model_cfg['kind']})")
    print(f"# Threshold:  {threshold:.2f}")
    print("#" * 70 + "\n")

    samples       = load_samples(prompts_per_task)
    manifest_hash = build_sample_manifest_hash(samples)
    llm_tokenizer = get_tokenizer()
    detector      = get_detector(model_cfg)
    engine        = EvaluationEngine(detector, llm_tokenizer, threshold)

    all_gt:   List[bool] = []
    all_pred: List[bool] = []
    per_sample_c1    = {"precision": [], "recall": [], "f1": []}
    per_sample_c0    = {"precision": [], "recall": [], "f1": []}
    per_sample_micro = {"precision": [], "recall": [], "f1": []}
    per_sample_runtime: List[float] = []
    per_sample_mean_lat: List[float] = []
    all_step_latencies: List[float]  = []

    wallclock_start = time.time()

    for i, sample in enumerate(samples):
        print(f"  sample {i+1}/{len(samples)} (task={sample['task_type']}, model={sample['model']})")
        result = engine.evaluate_sample(
            context=sample["context"], query=sample["query"],
            answer=sample["answer"], labels=sample["labels"],
            task_type=sample["task_type"], model_name=sample["model"],
            sample_index=i,
        )

        sgt   = [ts.ground_truth for ts in result.token_steps]
        spred = [ts.predicted     for ts in result.token_steps]
        all_gt.extend(sgt); all_pred.extend(spred)

        pc1, rc1, fc1 = _per_sample_class_metric(sgt, spred, True)
        pc0, rc0, fc0 = _per_sample_class_metric(sgt, spred, False)
        per_sample_c1["precision"].append(pc1); per_sample_c1["recall"].append(rc1); per_sample_c1["f1"].append(fc1)
        per_sample_c0["precision"].append(pc0); per_sample_c0["recall"].append(rc0); per_sample_c0["f1"].append(fc0)
        per_sample_micro["precision"].append(result.precision_micro)
        per_sample_micro["recall"].append(result.recall_micro)
        per_sample_micro["f1"].append(result.f1_micro)
        per_sample_runtime.append(result.total_predict_runtime_s)
        per_sample_mean_lat.append(result.mean_step_latency_s)
        all_step_latencies.extend(ts.predict_latency_s for ts in result.token_steps)

        run.log({
            "sample_index":              i,
            "task_type":                 sample["task_type"],
            "model":                     sample["model"],
            "sample_predict_runtime_s":  result.total_predict_runtime_s,
            "sample_mean_step_lat_s":    result.mean_step_latency_s,
            "sample_n_tokens":           len(sgt),
            "sample_n_pos_gt":           sum(1 for g in sgt if g),
            "sample_n_pos_pred":         sum(1 for p in spred if p),
            "sample_precision_micro":    result.precision_micro,
            "sample_recall_micro":       result.recall_micro,
            "sample_f1_micro":           result.f1_micro,
            "sample_precision_class_1":  pc1, "sample_recall_class_1": rc1, "sample_f1_class_1": fc1,
            "sample_precision_class_0":  pc0, "sample_recall_class_0": rc0, "sample_f1_class_0": fc0,
        })

    wallclock_runtime = time.time() - wallclock_start

    # Pooled metrics
    p_mic, r_mic, f_mic = compute_metrics_micro(all_gt, all_pred)
    p_c1, r_c1, f_c1    = compute_metrics_class(all_gt, all_pred, True)
    p_c0, r_c0, f_c0    = compute_metrics_class(all_gt, all_pred, False)

    # Macro metrics
    macro = {
        "macro_precision_class_1": _nanmean(per_sample_c1["precision"]),
        "macro_recall_class_1":    _nanmean(per_sample_c1["recall"]),
        "macro_f1_class_1":        _nanmean(per_sample_c1["f1"]),
        "macro_precision_class_0": _nanmean(per_sample_c0["precision"]),
        "macro_recall_class_0":    _nanmean(per_sample_c0["recall"]),
        "macro_f1_class_0":        _nanmean(per_sample_c0["f1"]),
        "macro_precision_micro":   _nanmean(per_sample_micro["precision"]),
        "macro_recall_micro":      _nanmean(per_sample_micro["recall"]),
        "macro_f1_micro":          _nanmean(per_sample_micro["f1"]),
    }

    # Runtime stats
    total_predict_runtime = sum(per_sample_runtime)
    mean_sample_runtime   = total_predict_runtime / max(len(per_sample_runtime), 1)
    mean_step_latency     = (sum(all_step_latencies) / len(all_step_latencies)
                             if all_step_latencies else 0.0)
    median_step_latency   = (float(np.median(all_step_latencies))
                             if all_step_latencies else 0.0)
    p95_step_latency      = (float(np.percentile(all_step_latencies, 95))
                             if all_step_latencies else 0.0)

    # Token-level diagnostics
    total_pos_gt   = sum(1 for g in all_gt   if g)
    total_pos_pred = sum(1 for p in all_pred if p)
    total_tokens   = len(all_gt)

    summary_payload = {
        "model_name": model_cfg["name"], "model_path": model_cfg["model_path"],
        "model_kind": model_cfg["kind"],
        "confidence_threshold": threshold,
        "num_samples": len(samples),
        "prompts_per_task": prompts_per_task,
        "sample_manifest_hash": manifest_hash,
        "seed": seed, "tokenizer_name": LLM_TOKENIZER_NAME,

        "precision_micro":   p_mic, "recall_micro":   r_mic, "f1_micro":          f_mic,
        "precision_class_1": p_c1, "recall_class_1": r_c1, "f1_binary_class_1": f_c1,
        "precision_class_0": p_c0, "recall_class_0": r_c0, "f1_binary_class_0": f_c0,

        "total_tokens":   total_tokens,
        "total_pos_gt":   total_pos_gt,
        "total_pos_pred": total_pos_pred,
        "frac_pos_gt":    total_pos_gt   / total_tokens if total_tokens else 0.0,
        "frac_pos_pred":  total_pos_pred / total_tokens if total_tokens else 0.0,

        "total_predict_runtime_s": total_predict_runtime,
        "mean_sample_runtime_s":   mean_sample_runtime,
        "mean_step_latency_s":     mean_step_latency,
        "median_step_latency_s":   median_step_latency,
        "p95_step_latency_s":      p95_step_latency,
        "wallclock_runtime_s":     wallclock_runtime,
        **macro,
    }
    for k, v in summary_payload.items():
        run.summary[k] = v

    safe_model = model_cfg["name"].replace("/", "_")
    out_path   = f"{output_prefix}_{safe_model}_thr_{thr_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    artifact = wandb.Artifact(f"{safe_model}_thr_{thr_label}", type="benchmark_results")
    artifact.add_file(out_path)
    run.log_artifact(artifact)

    _ALL_RUN_RESULTS.append(summary_payload)

    print(f"\n{'=' * 60}")
    print(f"RUN COMPLETE: {run.name}")
    print(f"  Threshold:           {threshold} ")
    print(f"  Total predict time:  {total_predict_runtime:.2f}s")
    print(f"  Wallclock:           {wallclock_runtime:.2f}s")
    print(f"  Mean step latency:   {mean_step_latency*1000:.2f}ms")
    print(f"  Median step lat.:    {median_step_latency*1000:.2f}ms")
    print(f"  P95 step latency:    {p95_step_latency*1000:.2f}ms")
    print(f"  Total tokens:        {total_tokens}")
    if total_tokens:
        print(f"  Class-1 GT:          {total_pos_gt} ({100*total_pos_gt/total_tokens:.2f}%)")
        print(f"  Class-1 pred:        {total_pos_pred} ({100*total_pos_pred/total_tokens:.2f}%)")
    print(f"  Pooled F1 c1:        {f_c1:.4f}  Recall: {r_c1:.4f}  Prec: {p_c1:.4f}")
    print(f"  Macro  F1 c1:        {macro['macro_f1_class_1']:.4f}")
    print(f"{'=' * 60}\n")
    run.finish()


# ============================================================================
# Final summary
# ============================================================================

def print_final_summary():
    if not _ALL_RUN_RESULTS:
        print("No results - skipping summary.")
        return

    print("\n" + "=" * 110)
    print("RQ3 UNIFIED THRESHOLD SWEEP - FINAL SUMMARY")
    print("=" * 110)

    models = sorted(set(r["model_name"] for r in _ALL_RUN_RESULTS))

    header = (
        f"{'Model':<42} {'Thr':>5} {'F1_c1':>8} {'Rec_c1':>8} {'Prec_c1':>8} "
        f"{'F1_mic':>8} {'PredTime':>10} {'StepLat(ms)':>12}"
    )
    print(header); print("-" * 110)

    best_per_model: Dict[str, Dict] = {}
    for model in models:
        runs = sorted(
            [r for r in _ALL_RUN_RESULTS if r["model_name"] == model],
            key=lambda r: r["confidence_threshold"],
        )
        for r in runs:
            print(
                f"{r['model_name']:<42} "
                f"{r['confidence_threshold']:>5.1f} "
                f"{r['f1_binary_class_1']:>8.4f} "
                f"{r['recall_class_1']:>8.4f} "
                f"{r['precision_class_1']:>8.4f} "
                f"{r['f1_micro']:>8.4f} "
                f"{r['total_predict_runtime_s']:>9.2f}s "
                f"{r['mean_step_latency_s']*1000:>11.2f}"
            )
        best_per_model[model] = max(runs, key=lambda r: r["f1_binary_class_1"])
        print()

    print("=" * 110)
    print("BEST THRESHOLD PER MODEL (by F1 class 1, pooled)")
    print("=" * 110)
    for model, best in best_per_model.items():
        print(f"\n  Model:               {model}")
        print(f"  Threshold:           {best['confidence_threshold']}")
        print(f"  F1 c1:               {best['f1_binary_class_1']:.4f}")
        print(f"  Recall c1:           {best['recall_class_1']:.4f}")
        print(f"  Prec c1:             {best['precision_class_1']:.4f}")
        print(f"  F1 micro:            {best['f1_micro']:.4f}")
        print(f"  Total predict time:  {best['total_predict_runtime_s']:.2f}s")
        print(f"  Mean step latency:   {best['mean_step_latency_s']*1000:.2f}ms")
        print(f"  P95 step latency:    {best['p95_step_latency_s']*1000:.2f}ms")
    print("\n" + "=" * 110)

    summary_path = f"{DEFAULT_OUTPUT_PREFIX}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "all_runs": _ALL_RUN_RESULTS,
            "best_per_model": {
                m: {
                    "confidence_threshold":    b["confidence_threshold"],
                    "f1_binary_class_1":       b["f1_binary_class_1"],
                    "recall_class_1":          b["recall_class_1"],
                    "precision_class_1":       b["precision_class_1"],
                    "f1_micro":                b["f1_micro"],
                    "total_predict_runtime_s": b["total_predict_runtime_s"],
                    "mean_step_latency_s":     b["mean_step_latency_s"],
                    "p95_step_latency_s":      b["p95_step_latency_s"],
                }
                for m, b in best_per_model.items()
            },
        }, f, indent=2)
    print(f"Summary saved: {summary_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--entity",            type=str, default=WANDB_ENTITY)
    parser.add_argument("--project",           type=str, default=DEFAULT_SWEEP_PROJECT)
    parser.add_argument("--sweep-id",          type=str, default=None)
    parser.add_argument("--count",             type=int,
                        default=len(SWEEP_THRESHOLDS) * len(MODELS_TO_EVALUATE))
    parser.add_argument("--output-prefix",     type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--seed",              type=int, default=SEED)
    parser.add_argument("--prompts-per-task",  type=int, default=SWEEP_PROMPTS_PER_TASK)
    parser.add_argument("--create-only",       action="store_true")
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep_id: {sweep_id}")
    else:
        cfg = build_sweep_config(
            output_prefix=args.output_prefix, seed=args.seed,
            prompts_per_task=args.prompts_per_task,
        )
        sweep_id = wandb.sweep(sweep=cfg, entity=args.entity, project=args.project)
        print(f"Created sweep_id: {sweep_id}")

    if args.create_only:
        return

    print(f"Starting agent for sweep_id={sweep_id}, count={args.count}")
    wandb.agent(
        sweep_id=sweep_id, function=evaluate_single_run,
        entity=args.entity, project=args.project, count=args.count,
    )
    print_final_summary()


if __name__ == "__main__":
    main()