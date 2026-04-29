"""
Temperature scaling for the lettuceprevent decoder model.

Calibration set: first 20 unique (context, query) prompts per task type
(stratified, deterministic), with all 6 LLM-generated answers per prompt
= 360 answers total.

The selected calibration prompts are saved to JSON so downstream scripts
(threshold sweep, RQ1/RQ2 prevention experiments) can exclude them and
avoid data leakage.

Output:
- {prefix}.json — temperature, calibration prompts, before/after metrics
"""

import argparse
import json
import math
import os
import random
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy.optimize import minimize_scalar
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    AutoTokenizer
)
from transformers.modeling_outputs import TokenClassifierOutput
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors


os.environ.setdefault("WEAVE_DISABLED", "true")
warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*")


# ============================================================================
# Configuration
# ============================================================================

SEED               = 42
LLM_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"
HF_DATASET_NAME    = "wandb/RAGTruth-processed"
HF_DATASET_SPLIT   = "test"

MODEL_PATH = "lebe1/lettuceprevent-ettin-decoder-68m-en"
MODEL_NAME = "lettuceprevent-ettin-decoder-68m-en"

CALIBRATION_PROMPTS_PER_TASK = 20      # First 20 prompts per task
MAX_LENGTH                   = 4096
DEFAULT_OUTPUT_PREFIX        = "lettuceprevent_calibration"

# All 6 LLMs in RAGTruth — calibration uses every answer.
LLMS = [
    "gpt-4-0613",
    "gpt-3.5-turbo-0613",
    "mistral-7B-instruct",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
]


# ============================================================================
# Model class — must match training-time definition exactly
# ============================================================================

class EttinTokenClassifier(PreTrainedModel):
    def __init__(self, config, num_labels: int = 2):
        super().__init__(config)
        self.num_labels = num_labels
        self.backbone   = AutoModel.from_config(config)
        self.dropout    = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs         = self.backbone(input_ids=input_ids,
                                        attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits          = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            weight = self.class_weights if hasattr(self, "class_weights") else None
            loss   = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_class_weights(self, weights: torch.Tensor):
        self.register_buffer("class_weights", weights)




# ============================================================================
# Llama-driven Ettin tokenization (mirrors training preprocessing)
# ============================================================================

def tokenize_text_via_llama_chunks(
    text: str,
    llama_tokenizer,
    ettin_tokenizer,
    char_mask: List[int] = None,
):
    llama_enc = llama_tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True,
    )
    llama_ids     = llama_enc["input_ids"]
    llama_offsets = llama_enc["offset_mapping"]

    ettin_ids: List[int] = []
    labels:    List[int] = [] if char_mask is not None else None

    for tok_id, (cs, ce) in zip(llama_ids, llama_offsets):
        if cs == ce:
            continue
        chunk = llama_tokenizer.decode([tok_id])
        if not chunk:
            continue
        sub_ids = ettin_tokenizer(chunk, add_special_tokens=False)["input_ids"]
        if not sub_ids:
            continue
        ettin_ids.extend(sub_ids)
        if char_mask is not None:
            is_hall = int(any(
                char_mask[i] == 1
                for i in range(cs, min(ce, len(char_mask)))
            ))
            labels.extend([is_hall] * len(sub_ids))
    return ettin_ids, labels


def build_char_mask(answer: str, hallucination_labels: List[Dict]) -> List[int]:
    mask = [0] * len(answer)
    for span in hallucination_labels:
        for i in range(span["start"], min(span["end"], len(answer))):
            mask[i] = 1
    return mask


def parse_hallucination_labels(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    parsed = [l for l in raw if isinstance(l, dict) and "start" in l and "end" in l]
    if len(parsed) != len(raw):
        print(f"  WARNING: dropped {len(raw) - len(parsed)} malformed labels")
    return parsed

# ============================================================================
# Calibration set loader — first N prompts per task, all 6 LLMs each
# ============================================================================

def load_calibration_set(
    n_prompts_per_task: int,
) -> Tuple[List[Dict], List[Tuple[str, str, str]]]:
    """
    Load the first n_prompts_per_task unique (context, query) prompts per
    task type, in dataset order. For each selected prompt, use all 6 LLM
    answers.

    Returns:
        samples: list of {context, query, answer, labels, task_type, model}
        cal_prompts: list of (task_type, context, query) tuples — the
                     unique prompts used for calibration. Saved to JSON
                     so downstream scripts can exclude them.
    """
    print(f"Loading {HF_DATASET_NAME} (split={HF_DATASET_SPLIT})...")
    hf_ds      = load_dataset(HF_DATASET_NAME)
    test_split = hf_ds[HF_DATASET_SPLIT]

    # Walk the dataset in order. For each task, collect the first
    # n_prompts_per_task unique (context, query) prompts. For each prompt,
    # collect every row that matches it (one per LLM).
    by_task_prompts:  Dict[str, List[Tuple[str, str]]]            = {}
    rows_by_prompt:   Dict[Tuple[str, str, str], List[Dict]]       = {}
    seen_per_task:    Dict[str, set]                              = {}

    for row in test_split:
        if row.get("hallucination_labels") in (None, ""):
            continue
        tt = row.get("task_type", "unknown")
        ctx = row["context"]
        qry = row["query"]
        key  = (ctx, qry)
        full_key = (tt, ctx, qry)

        seen_per_task.setdefault(tt, set())
        by_task_prompts.setdefault(tt, [])
        rows_by_prompt.setdefault(full_key, [])

        if key in seen_per_task[tt]:
            # Already-known prompt for this task — collect another LLM answer
            rows_by_prompt[full_key].append(dict(row))
        elif len(by_task_prompts[tt]) < n_prompts_per_task:
            # New prompt within budget for this task — add it
            seen_per_task[tt].add(key)
            by_task_prompts[tt].append(key)
            rows_by_prompt[full_key].append(dict(row))
        # else: new prompt for this task but budget exhausted — skip

    # Build sample list, ordered by task → prompt → LLM
    samples:     List[Dict]                       = []
    cal_prompts: List[Tuple[str, str, str]]        = []
    for tt in sorted(by_task_prompts.keys()):
        prompts = by_task_prompts[tt]
        print(f"  task_type={tt}: {len(prompts)} unique prompts selected")
        for ctx, qry in prompts:
            full_key = (tt, ctx, qry)
            cal_prompts.append(full_key)
            llms_seen = []
            for row in rows_by_prompt[full_key]:
                samples.append({
                    "task_type": tt,
                    "context":   row["context"],
                    "query":     row["query"],
                    "answer":    row["output"],
                    "model":     row.get("model", "unknown"),
                    "labels":    parse_hallucination_labels(row["hallucination_labels"]),
                })
                llms_seen.append(row.get("model", "unknown"))
            # Sanity check — warn if a prompt has fewer than 6 LLM answers
            if len(llms_seen) != len(LLMS):
                print(f"    WARNING: prompt has {len(llms_seen)} LLM answers, "
                      f"expected {len(LLMS)}: {llms_seen}")

    print(f"\nTotal calibration samples (prompt × LLM): {len(samples)}")
    return samples, cal_prompts


# ============================================================================
# Logit collection
# ============================================================================

def collect_logits_and_labels(
    samples: List[Dict],
    model,
    ettin_tokenizer,
    llama_tokenizer,
    device: torch.device,
    max_length: int = MAX_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    cls_id = ettin_tokenizer.cls_token_id
    sep_id = ettin_tokenizer.sep_token_id

    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i, s in enumerate(samples):
            char_mask = build_char_mask(s["answer"], s["labels"])

            ctx_ids, _      = tokenize_text_via_llama_chunks(
                s["context"], llama_tokenizer, ettin_tokenizer, char_mask=None,
            )
            qry_ids, _      = tokenize_text_via_llama_chunks(
                s["query"],   llama_tokenizer, ettin_tokenizer, char_mask=None,
            )
            ans_ids, ans_labels = tokenize_text_via_llama_chunks(
                s["answer"],  llama_tokenizer, ettin_tokenizer,
                char_mask=char_mask,
            )

            if not ans_ids:
                continue

            budget = max_length - len(ans_ids) - 4
            if len(ctx_ids) + len(qry_ids) > budget:
                excess = (len(ctx_ids) + len(qry_ids)) - budget
                if excess <= len(ctx_ids):
                    ctx_ids = ctx_ids[excess:]
                else:
                    remaining = excess - len(ctx_ids)
                    ctx_ids = []
                    qry_ids = qry_ids[remaining:]

            input_ids = (
                [cls_id] + ctx_ids + [sep_id]
                + qry_ids + [sep_id]
                + ans_ids + [sep_id]
            )
            input_ids = input_ids[:max_length]
            attention_mask = [1] * len(input_ids)
            answer_start = 1 + len(ctx_ids) + 1 + len(qry_ids) + 1
            answer_end   = min(answer_start + len(ans_ids), len(input_ids))
            ans_labels   = ans_labels[:answer_end - answer_start]

            if answer_end <= answer_start:
                continue  

            input_ids_t      = torch.tensor([input_ids],      dtype=torch.long, device=device)
            attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
            logits  = outputs.logits

            ans_logits = logits[0, answer_start:answer_end, :].cpu().float().numpy()
            assert len(ans_logits) == len(ans_labels)

            all_logits.append(ans_logits)
            all_labels.append(np.array(ans_labels, dtype=np.int64))

            if (i + 1) % 25 == 0:
                print(f"  processed {i + 1}/{len(samples)} samples")

    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    print(f"\nCollected {len(logits_arr)} (logits, label) pairs")
    print(f"  Class 1: {labels_arr.sum()} ({100 * labels_arr.mean():.2f}%)")
    print(f"  Class 0: {(labels_arr == 0).sum()}")
    return logits_arr, labels_arr


# ============================================================================
# Temperature fitting & calibration metrics
# ============================================================================

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()

    def nll(log_T: float) -> float:
        T = math.exp(log_T)
        scaled = logits_t / T
        log_probs = torch.log_softmax(scaled, dim=-1)
        nll_per_token = -log_probs.gather(1, labels_t.unsqueeze(1)).squeeze(1)
        return float(nll_per_token.mean().item())

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return math.exp(result.x)


def compute_nll(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    logits_t = torch.from_numpy(logits).float() / T
    labels_t = torch.from_numpy(labels).long()
    log_probs = torch.log_softmax(logits_t, dim=-1)
    return float(-log_probs.gather(1, labels_t.unsqueeze(1)).squeeze(1).mean().item())


def compute_ece(
    logits: np.ndarray,
    labels: np.ndarray,
    T: float = 1.0,
    n_bins: int = 15,
) -> float:
    probs = torch.softmax(torch.from_numpy(logits).float() / T, dim=-1).numpy()
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies  = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc  = accuracies[in_bin].mean()
        ece += (in_bin.sum() / len(confidences)) * abs(bin_conf - bin_acc)
    return float(ece)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling on the lettuceprevent decoder."
    )
    parser.add_argument("--prompts-per-task", type=int,
                        default=CALIBRATION_PROMPTS_PER_TASK)
    parser.add_argument("--seed",          type=int, default=SEED)
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--device",        type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    samples, cal_prompts = load_calibration_set(args.prompts_per_task)

    print(f"Loading Ettin tokenizer from {MODEL_PATH}")
    ettin_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"Loading Llama tokenizer ({LLM_TOKENIZER_NAME})")
    llama_tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_NAME)

    # In main(), right after loading the tokenizers:
    assert ettin_tokenizer.cls_token_id is not None, \
        "Ettin tokenizer has no CLS token — check special-token config"
    assert ettin_tokenizer.sep_token_id is not None, \
        "Ettin tokenizer has no SEP token — check special-token config"

    print(f"Loading model weights from {MODEL_PATH}")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    model  = EttinTokenClassifier(config, num_labels=2)

    weights_path = hf_hub_download(repo_id=MODEL_PATH, filename="model.safetensors")
    state_dict   = load_safetensors(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"WARNING: unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")

    model = model.to(device)
    model.eval()

    t0 = time.time()
    logits, labels = collect_logits_and_labels(
        samples, model, ettin_tokenizer, llama_tokenizer, device,
    )
    print(f"Logit collection took {time.time() - t0:.1f}s")

    print("\nFitting temperature...")
    nll_before = compute_nll(logits, labels, T=1.0)
    ece_before = compute_ece(logits, labels, T=1.0)
    T_star    = fit_temperature(logits, labels)
    if abs(math.log(T_star)) > 2.9:
        print(f"WARNING: T_star={T_star:.4f} near optimization boundary — "
            f"calibration may be unreliable")
    nll_after  = compute_nll(logits, labels, T=T_star)
    ece_after  = compute_ece(logits, labels, T=T_star)

    print(f"\n{'=' * 60}")
    print(f"CALIBRATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Optimal temperature T*:  {T_star:.4f}")
    print(f"  NLL  before / after:     {nll_before:.4f} / {nll_after:.4f}")
    print(f"  ECE  before / after:     {ece_before:.4f} / {ece_after:.4f}")
    print(f"{'=' * 60}\n")

    output = {
        "model_path":                   MODEL_PATH,
        "model_name":                   MODEL_NAME,
        "tokenizer_name":               LLM_TOKENIZER_NAME,
        "calibration_prompts_per_task": args.prompts_per_task,
        "n_calibration_samples":        len(samples),
        "n_calibration_tokens":         int(len(logits)),
        "n_class_1_tokens":             int(labels.sum()),
        "frac_class_1":                 float(labels.mean()),
        "seed":                         args.seed,
        "temperature":                  T_star,
        "nll_before":                   nll_before,
        "nll_after":                    nll_after,
        "ece_before":                   ece_before,
        "ece_after":                    ece_after,
        # Save the prompts so downstream scripts can exclude them
        "calibration_prompts":          [
            {"task_type": tt, "context": ctx, "query": qry}
            for (tt, ctx, qry) in cal_prompts
        ],
    }

    output_path = f"{args.output_prefix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Calibration result saved to: {output_path}")
    print(f"\nUse T = {T_star:.4f} in the threshold sweep script.")


if __name__ == "__main__":
    main()