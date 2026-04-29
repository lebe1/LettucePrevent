"""
Threshold sweep for the lettuceprevent decoder model

Sample selection:
- First 50 unique (context, query) prompts per task type, in dataset order.
- Each prompt is assigned ONE LLM answer in round-robin order
- Total: 50 × 3 = 150 sample-answer pairs.
"""

import argparse
import csv
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
        "name":       "lettuceprevent-ettin-decoder-68m-en",
        "model_path": "lebe1/lettuceprevent-ettin-decoder-68m-en",
    },
]

MAX_LENGTH                = 4096
SWEEP_PROMPTS_PER_TASK    = 50
LLMS_ROUND_ROBIN          = [
    "gpt-4-0613",
    "gpt-3.5-turbo-0613",
    "mistral-7B-instruct",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
]

SWEEP_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DEFAULT_SWEEP_NAME    = "confidence-threshold-lettuceprevent-rq3"
DEFAULT_SWEEP_PROJECT = "hdm-rq3-threshold-sweep-lettuceprevent"
DEFAULT_OUTPUT_PREFIX = "rq3_lettuceprevent_sweep"


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
    was_triggered: bool


@dataclass
class SampleResult:
    sample_index: int
    task_type: str
    model: str
    context: str
    query: str
    answer: str
    token_steps: List[TokenStepResult] = field(default_factory=list)
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
# Model definition
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
# Llama tokenizer wrapper
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
# Detector wrapper
# ============================================================================

class HallucinationDetectorBase(ABC):
    @abstractmethod
    def get_name(self) -> str: ...
    @abstractmethod
    def predict(self, context, question, answer) -> List[Dict[str, Any]]: ...


class LettucePreventDecoderWrapper(HallucinationDetectorBase):
    """
    Inference wrapper for the lettuceprevent decoder model.
    Uses raw softmax (no temperature scaling).
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
        self.name        = name
        self.model_path  = model_path
        self.max_length  = max_length

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[{self.name}] Loading Ettin tokenizer from {model_path}")
        self.ettin_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.cls_id = self.ettin_tokenizer.cls_token_id
        self.sep_id = self.ettin_tokenizer.sep_token_id

        print(f"[{self.name}] Loading Llama tokenizer for boundary alignment")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer_name)

        print(f"[{self.name}] Loading model weights from {model_path}")
        config = AutoConfig.from_pretrained(model_path)
        self.model = EttinTokenClassifier(config, num_labels=2)

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors
        weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        state_dict   = load_safetensors(weights_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[{self.name}] WARNING: missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[{self.name}] WARNING: unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")

        self.model.to(self.device).to(torch_dtype)
        self.model.eval()
        print(f"[{self.name}] Ready on {self.device} (Raw softmax)")

    def get_name(self) -> str:
        return self.name

    @torch.no_grad()
    def predict(self, context, question, answer) -> List[Dict[str, Any]]:
        ctx_text = "\n".join(context) if isinstance(context, list) else context

        ctx_ids, _, _      = tokenize_text_via_llama_chunks(ctx_text, self.llama_tokenizer, self.ettin_tokenizer)
        qry_ids, _, _      = tokenize_text_via_llama_chunks(question, self.llama_tokenizer, self.ettin_tokenizer)
        ans_ids, ans_offs, ans_lidx = tokenize_text_via_llama_chunks(answer, self.llama_tokenizer, self.ettin_tokenizer)

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
            if len(ctx_ids) + len(qry_ids) > ctx_qry_budget:
                excess = (len(ctx_ids) + len(qry_ids)) - ctx_qry_budget
                if excess <= len(ctx_ids):
                    ctx_ids = ctx_ids[excess:]
                else:
                    ctx_ids = []
                    qry_ids = qry_ids[(excess - len(ctx_ids)):]
            prefix_ids = (
                [self.cls_id] + ctx_ids + [self.sep_id]
                + qry_ids + [self.sep_id]
            )
            input_ids = prefix_ids + ans_ids + [self.sep_id]
            answer_start = len(prefix_ids)

        attention_mask = [1] * len(input_ids)

        input_ids_t      = torch.tensor([input_ids],      dtype=torch.long, device=self.device)
        attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        logits = self.model(input_ids=input_ids_t, attention_mask=attention_mask_t)

        # ---- RAW SOFTMAX (no temperature scaling) ----------------------
        probs_class_1 = torch.softmax(logits, dim=-1)[0, :, 1].float().cpu().numpy()
        # ----------------------------------------------------------------

        ans_probs = probs_class_1[answer_start : answer_start + len(ans_ids)]

        # Aggregate Ettin sub-token probs to Llama-token level (max).
        groups: Dict[int, Dict[str, Any]] = {}
        for prob, (cs, ce), lidx in zip(ans_probs, ans_offs, ans_lidx):
            existing = groups.get(lidx)
            if existing is None or prob > existing["confidence"]:
                groups[lidx] = {
                    "start": int(cs),
                    "end":   int(ce),
                    "confidence": float(prob),
                }

        return sorted(groups.values(), key=lambda d: d["start"])


# ============================================================================
# Offset mapper, label parsing, metrics
# ============================================================================

class OffsetMapper:
    def __init__(self, original: str, detokenized: str):
        self.original = original
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
        "precision_micro": p_mic, "recall_micro": r_mic, "f1_micro": f_mic,
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
# Evaluation engine
# ============================================================================

class EvaluationEngine:
    def __init__(self, detector, llm_tokenizer, confidence_threshold):
        self.detector             = detector
        self.llm_tokenizer        = llm_tokenizer
        self.confidence_threshold = confidence_threshold

    def evaluate_sample(self, context, query, answer, labels, task_type, model_name, sample_index):
        gt_labels = [
            GroundTruthLabel(l["start"], l["end"], l.get("label", "hallucination"))
            for l in labels
        ]
        tokens         = self.llm_tokenizer.tokenize_with_offsets(answer)
        detok_full     = self.llm_tokenizer.get_detokenized_text(answer)
        offset_mapper  = OffsetMapper(answer, detok_full)
        if offset_mapper.check_divergence() > 0.05:
            print(f"  [WARN] Sample {sample_index}: detok divergence "
                  f"= {offset_mapper.check_divergence():.2%}")

        token_steps:     List[TokenStepResult]            = []
        last_prediction: Dict[int, Tuple[bool, float]]    = {}
        sample_start = time.time()
        encoded_full = self.llm_tokenizer.tokenizer.encode(answer, add_special_tokens=False)

        for step, token in enumerate(tokens):
            prefix_text = self.llm_tokenizer.tokenizer.decode(
                encoded_full[: step + 1], skip_special_tokens=True
            )
            try:
                preds = self.detector.predict(
                    context=[context], question=query, answer=prefix_text,
                )
            except Exception as e:
                print(f"  [ERROR] step {step}: {e}")
                preds = []
            last_prediction = map_predictions_to_tokens(
                preds, tokens[: step + 1], self.confidence_threshold,
            )

            is_pred, pred_conf = last_prediction.get(token.index, (False, 0.0))

            os_, oe_ = offset_mapper.detok_range_to_orig(token.char_start, token.char_end)
            orig_token = TokenInfo(token.index, token.text, os_, oe_)
            is_gt = is_token_hallucinated(orig_token, gt_labels)

            token_steps.append(TokenStepResult(
                step=step, token=token.text,
                token_char_start=token.char_start, token_char_end=token.char_end,
                ground_truth=is_gt, predicted=is_pred,
                confidence=pred_conf, was_triggered=True,
            ))

        runtime  = time.time() - sample_start
        all_gt   = [ts.ground_truth for ts in token_steps]
        all_pred = [ts.predicted     for ts in token_steps]
        nine = compute_nine_metrics(all_gt, all_pred)

        return SampleResult(
            sample_index=sample_index, task_type=task_type, model=model_name,
            context=context, query=query, answer=answer,
            token_steps=token_steps, runtime_seconds=runtime,
            **{k: nine[k] for k in nine},
        )


# ============================================================================
# Sample selection: first N prompts per task, round-robin LLM
# ============================================================================

def load_sweep_samples(n_prompts_per_task: int) -> List[Dict]:
    """
    Select the first n_prompts_per_task unique (context, query) prompts per
    task type, in dataset order. Each prompt gets one LLM answer assigned
    in round-robin order across LLMS_ROUND_ROBIN.
    """
    print(f"Loading {HF_DATASET_NAME} (split={HF_DATASET_SPLIT})...")
    hf_ds      = load_dataset(HF_DATASET_NAME)
    test_split = hf_ds[HF_DATASET_SPLIT]

    # For each (task, ctx, qry), collect a model -> row mapping
    rows_by_prompt: Dict[Tuple[str, str, str], Dict[str, Dict]] = {}
    prompt_order_per_task: Dict[str, List[Tuple[str, str]]]      = {}
    seen_per_task: Dict[str, set]                                = {}

    for row in test_split:
        if row.get("hallucination_labels") in (None, ""):
            continue
        tt = row.get("task_type", "unknown")
        ctx = row["context"]; qry = row["query"]
        full_key = (tt, ctx, qry)
        rows_by_prompt.setdefault(full_key, {})
        rows_by_prompt[full_key][row.get("model", "unknown")] = dict(row)
        seen_per_task.setdefault(tt, set())
        prompt_order_per_task.setdefault(tt, [])
        if (ctx, qry) not in seen_per_task[tt]:
            seen_per_task[tt].add((ctx, qry))
            prompt_order_per_task[tt].append((ctx, qry))

    # Pick first n_prompts_per_task per task, assign LLM round-robin
    samples: List[Dict] = []
    for tt in sorted(prompt_order_per_task.keys()):
        prompts = prompt_order_per_task[tt][:n_prompts_per_task]
        print(f"  task_type={tt}: {len(prompts)} prompts selected")
        skipped = 0
        for i, (ctx, qry) in enumerate(prompts):
            full_key = (tt, ctx, qry)
            preferred_llm = LLMS_ROUND_ROBIN[i % len(LLMS_ROUND_ROBIN)]
            row_map = rows_by_prompt[full_key]
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

    by_model = {}
    for s in samples:
        by_model[s["model"]] = by_model.get(s["model"], 0) + 1
    print("LLM distribution:")
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n}")

    return samples


# ============================================================================
# Sweep machinery
# ============================================================================

_SAMPLES_CACHE:    Optional[List[Dict]]                          = None
_TOKENIZER_CACHE:  Optional[LLMTokenizerWrapper]                 = None
_DETECTOR_CACHE:   Optional[LettucePreventDecoderWrapper]        = None
_ALL_RUN_RESULTS:  List[Dict]                                    = []


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
    global _DETECTOR_CACHE
    if _DETECTOR_CACHE is None or _DETECTOR_CACHE.name != model_cfg["name"]:
        _DETECTOR_CACHE = LettucePreventDecoderWrapper(
            name=model_cfg["name"],
            model_path=model_cfg["model_path"],
            llama_tokenizer_name=LLM_TOKENIZER_NAME,
        )
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
    run.name  = f"{model_cfg['name']}_uncal_thr_{thr_label}"

    set_seed(seed)

    print("\n" + "#" * 70)
    print(f"# Run:        {run.name}")
    print(f"# Model:      {model_cfg['name']}")
    print(f"# Threshold:  {threshold:.2f}")
    print(f"# Calibration: NONE (raw softmax)")
    print("#" * 70 + "\n")

    samples         = load_samples(prompts_per_task)
    manifest_hash   = build_sample_manifest_hash(samples)
    llm_tokenizer   = get_tokenizer()
    detector        = get_detector(model_cfg)
    engine          = EvaluationEngine(detector, llm_tokenizer, threshold)

    all_gt:   List[bool] = []
    all_pred: List[bool] = []
    per_sample_c1    = {"precision": [], "recall": [], "f1": []}
    per_sample_c0    = {"precision": [], "recall": [], "f1": []}
    per_sample_micro = {"precision": [], "recall": [], "f1": []}

    total_start = time.time()

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

        run.log({
            "sample_index": i, "task_type": sample["task_type"], "model": sample["model"],
            "sample_runtime_s": result.runtime_seconds,
            "sample_n_tokens": len(sgt),
            "sample_n_pos_gt": sum(1 for g in sgt if g),
            "sample_n_pos_pred": sum(1 for p in spred if p),
            "sample_precision_micro":   result.precision_micro,
            "sample_recall_micro":      result.recall_micro,
            "sample_f1_micro":          result.f1_micro,
            "sample_precision_class_1": pc1, "sample_recall_class_1": rc1, "sample_f1_class_1": fc1,
            "sample_precision_class_0": pc0, "sample_recall_class_0": rc0, "sample_f1_class_0": fc0,
        })

    total_runtime = time.time() - total_start

    p_mic, r_mic, f_mic = compute_metrics_micro(all_gt, all_pred)
    p_c1, r_c1, f_c1    = compute_metrics_class(all_gt, all_pred, True)
    p_c0, r_c0, f_c0    = compute_metrics_class(all_gt, all_pred, False)

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

    total_pos_gt   = sum(1 for g in all_gt   if g)
    total_pos_pred = sum(1 for p in all_pred if p)
    total_tokens   = len(all_gt)

    summary_payload = {
        "model_name": model_cfg["name"], "model_path": model_cfg["model_path"],
        "confidence_threshold": threshold,
        "num_samples": len(samples),
        "prompts_per_task": prompts_per_task,
        "sample_manifest_hash": manifest_hash,
        "seed": seed, "tokenizer_name": LLM_TOKENIZER_NAME,
        "precision_micro": p_mic, "recall_micro": r_mic, "f1_micro": f_mic,
        "precision_class_1": p_c1, "recall_class_1": r_c1, "f1_binary_class_1": f_c1,
        "precision_class_0": p_c0, "recall_class_0": r_c0, "f1_binary_class_0": f_c0,
        "total_tokens": total_tokens, "total_pos_gt": total_pos_gt, "total_pos_pred": total_pos_pred,
        "frac_pos_gt":   total_pos_gt   / total_tokens if total_tokens else 0.0,
        "frac_pos_pred": total_pos_pred / total_tokens if total_tokens else 0.0,
        "total_runtime_seconds": total_runtime,
        **macro,
    }
    for k, v in summary_payload.items():
        run.summary[k] = v

    safe_model = model_cfg["name"].replace("/", "_")
    out_path   = f"{output_prefix}_{safe_model}_thr_{thr_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    artifact = wandb.Artifact(f"{safe_model}_uncal_thr_{thr_label}", type="benchmark_results")
    artifact.add_file(out_path)
    run.log_artifact(artifact)

    _ALL_RUN_RESULTS.append(summary_payload)

    print(f"\n{'=' * 60}")
    print(f"RUN COMPLETE: {run.name}")
    print(f"  Threshold:        {threshold} ")
    print(f"  Runtime:          {total_runtime:.2f}s")
    print(f"  Total tokens:     {total_tokens}")
    if total_tokens:
        print(f"  Class-1 GT:       {total_pos_gt} ({100*total_pos_gt/total_tokens:.2f}%)")
        print(f"  Class-1 pred:     {total_pos_pred} ({100*total_pos_pred/total_tokens:.2f}%)")
    print(f"  Pooled F1 c1:     {f_c1:.4f}  Recall: {r_c1:.4f}  Prec: {p_c1:.4f}")
    print(f"  Macro  F1 c1:     {macro['macro_f1_class_1']:.4f}")
    print(f"{'=' * 60}\n")
    run.finish()


def print_final_summary():
    if not _ALL_RUN_RESULTS:
        print("No results — skipping summary."); return
    print("\n" + "=" * 80)
    print("LETTUCEPREVENT — THRESHOLD SWEEP SUMMARY")
    print("=" * 80)
    runs = sorted(_ALL_RUN_RESULTS, key=lambda r: r["confidence_threshold"])
    print(f"{'Thr':>5} {'F1_c1':>8} {'Rec_c1':>8} {'Prec_c1':>8} {'F1_mic':>8} {'Runtime':>10}")
    print("-" * 60)
    for r in runs:
        print(f"{r['confidence_threshold']:>5.2f} "
              f"{r['f1_binary_class_1']:>8.4f} {r['recall_class_1']:>8.4f} "
              f"{r['precision_class_1']:>8.4f} {r['f1_micro']:>8.4f} "
              f"{r['total_runtime_seconds']:>9.2f}s")
    best = max(runs, key=lambda r: r["f1_binary_class_1"])
    print(f"\nBest: thr={best['confidence_threshold']} "
          f"F1_c1={best['f1_binary_class_1']:.4f}")
    summary_path = f"{DEFAULT_OUTPUT_PREFIX}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"all_runs": runs, "best": best}, f, indent=2)
    print(f"Summary saved: {summary_path}")


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