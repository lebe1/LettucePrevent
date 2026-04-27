"""
Confidence threshold sweep for hallucination detection models.

Runs a W&B grid sweep over:
- confidence_threshold: [0.6, 0.7, 0.8, 0.9]
- model_idx: [0, 1]  (TinyLettuce and LettuceDetect-base)

Dataset setup:
- wandb/RAGTruth-processed (test split)
- 50 unique (context, query) samples per task_type

After all sweep runs, prints a ranked summary showing the best threshold
per model based on F1 class 1, recall class 1, and runtime.
"""

import argparse
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from datasets import load_dataset

os.environ.setdefault("WEAVE_DISABLED", "true")

import hallucination_benchmark as hb


SWEEP_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
UNIQUE_PAIRS_PER_TASK = 50
DEFAULT_SWEEP_NAME = "confidence-threshold-comparison-rq3"
DEFAULT_SWEEP_PROJECT = "hdm-benchmark-rq3-threshold-sweep"
DEFAULT_OUTPUT_PREFIX = "rq3_confident_treshold_sweep"


_SAMPLES_CACHE: Optional[List[Dict]] = None
_TOKENIZER_CACHE: Optional[hb.LLMTokenizerWrapper] = None

# Collects results across all sweep runs for final summary
_ALL_RUN_RESULTS: List[Dict] = []


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_benchmark_samples_for_sweep() -> List[Dict]:
    global _SAMPLES_CACHE
    if _SAMPLES_CACHE is not None:
        return _SAMPLES_CACHE

    print(
        f"Loading HF dataset: {hb.HF_DATASET_NAME} "
        f"(split={hb.HF_DATASET_SPLIT}, unique_pairs_per_task={UNIQUE_PAIRS_PER_TASK})..."
    )
    hf_ds = load_dataset(hb.HF_DATASET_NAME)
    test_split = hf_ds[hb.HF_DATASET_SPLIT]

    filtered = [
        dict(row) for row in test_split
        if row.get("hallucination_labels") not in (None, "")
    ]

    by_task: Dict[str, Dict[tuple, Dict]] = {}
    for row in filtered:
        tt = row.get("task_type", "unknown")
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
            benchmark_samples.append(
                {
                    "task_type": tt,
                    "context": row["context"],
                    "query": row["query"],
                    "answer": row["output"],
                    "labels": hb.parse_hallucination_labels(row["hallucination_labels"]),
                }
            )

    print(f"Total benchmark samples: {len(benchmark_samples)}")
    _SAMPLES_CACHE = benchmark_samples
    return benchmark_samples


def build_sample_manifest_hash(samples: List[Dict]) -> str:
    payload = [
        {
            "task_type": s["task_type"],
            "context": s["context"],
            "query": s["query"],
            "answer": s["answer"],
        }
        for s in samples
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_tokenizer() -> hb.LLMTokenizerWrapper:
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        print(f"Loading tokenizer: {hb.LLM_TOKENIZER_NAME}")
        _TOKENIZER_CACHE = hb.LLMTokenizerWrapper(hb.LLM_TOKENIZER_NAME)
    return _TOKENIZER_CACHE


def build_sweep_config(output_prefix: str, seed: int) -> Dict:
    return {
        "name": DEFAULT_SWEEP_NAME,
        "method": "grid",
        "metric": {"name": "f1_binary_class_1", "goal": "maximize"},
        "parameters": {
            "confidence_threshold": {"values": SWEEP_THRESHOLDS},
            "model_idx": {"values": list(range(len(hb.MODELS_TO_EVALUATE)))},
            "output_prefix": {"value": output_prefix},
            "seed": {"value": seed},
        },
    }


def evaluate_single_run():
    """Single W&B agent run for one (model_idx, confidence_threshold) pair."""
    # Peek at config before init to build a meaningful run name
    # W&B sweep agent pre-populates wandb.config before init
    run = wandb.init()
    cfg = wandb.config

    model_idx = int(cfg.model_idx)
    confidence_threshold = float(cfg.confidence_threshold)
    output_prefix = str(cfg.output_prefix)
    seed = int(cfg.seed)
    model_cfg = hb.MODELS_TO_EVALUATE[model_idx]

    # Set a descriptive run name so it's visible in the W&B dashboard
    thr_label = f"{confidence_threshold:.1f}".replace(".", "_")
    run_name = f"{model_cfg['name']}_thr_{thr_label}"
    run.name = run_name
    run.save()

    set_seed(seed)

    print("\n" + "#" * 70)
    print(f"# Run:       {run_name}")
    print(f"# Model:     {model_cfg['name']}")
    print(f"# Threshold: {confidence_threshold:.2f}")
    print("#" * 70 + "\n")

    samples = load_benchmark_samples_for_sweep()
    sample_manifest_hash = build_sample_manifest_hash(samples)
    llm_tokenizer = get_tokenizer()
    detector = hb.LettuceDetectWrapper(
        name=model_cfg["name"],
        model_path=model_cfg["model_path"],
    )
    engine = hb.EvaluationEngine(
        detector=detector,
        llm_tokenizer=llm_tokenizer,
        confidence_threshold=confidence_threshold,
    )

    all_sample_results = []
    all_gt = []
    all_pred = []
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
        all_gt.extend(ts.ground_truth for ts in result.token_steps)
        all_pred.extend(ts.predicted for ts in result.token_steps)

        # Log all 9 per-sample metrics
        run.log({
            "sample_index":              i,
            "task_type":                 sample["task_type"],
            "sample_runtime_s":          result.runtime_seconds,
            "sample_precision_micro":    result.precision_micro,
            "sample_recall_micro":       result.recall_micro,
            "sample_f1_micro":           result.f1_micro,
            "sample_precision_class_1":  result.precision_class_1,
            "sample_recall_class_1":     result.recall_class_1,
            "sample_f1_class_1":         result.f1_binary_class_1,
            "sample_precision_class_0":  result.precision_class_0,
            "sample_recall_class_0":     result.recall_class_0,
            "sample_f1_class_0":         result.f1_binary_class_0,
        })

    total_runtime = time.time() - total_start

    # Global aggregate metrics
    prec_micro, rec_micro, f1_micro = hb.compute_metrics_micro(all_gt, all_pred)
    prec_c1, rec_c1, f1_c1 = hb.compute_metrics_class(
        all_gt, all_pred, positive_class=True
    )
    prec_c0, rec_c0, f1_c0 = hb.compute_metrics_class(
        all_gt, all_pred, positive_class=False
    )

    # W&B summary — all 9 metrics + metadata
    run.summary["model_name"]            = model_cfg["name"]
    run.summary["model_path"]            = model_cfg["model_path"]
    run.summary["confidence_threshold"]  = confidence_threshold
    run.summary["num_samples"]           = len(samples)
    run.summary["unique_pairs_per_task"] = UNIQUE_PAIRS_PER_TASK
    run.summary["sample_manifest_hash"]  = sample_manifest_hash
    run.summary["seed"]                  = seed
    run.summary["tokenizer_name"]        = hb.LLM_TOKENIZER_NAME

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

    # Save JSON locally
    safe_model = model_cfg["name"].replace("/", "_")
    result_path = f"{output_prefix}_{safe_model}_thr_{thr_label}.json"
    result_dict = {
        "model_name":            model_cfg["name"],
        "model_path":            model_cfg["model_path"],
        "confidence_threshold":  confidence_threshold,
        "num_samples":           len(samples),
        "unique_pairs_per_task": UNIQUE_PAIRS_PER_TASK,
        "sample_manifest_hash":  sample_manifest_hash,
        "seed":                  seed,
        "precision_micro":       prec_micro,
        "recall_micro":          rec_micro,
        "f1_micro":              f1_micro,
        "precision_class_1":     prec_c1,
        "recall_class_1":        rec_c1,
        "f1_binary_class_1":     f1_c1,
        "precision_class_0":     prec_c0,
        "recall_class_0":        rec_c0,
        "f1_binary_class_0":     f1_c0,
        "total_runtime_seconds": total_runtime,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    artifact = wandb.Artifact(f"{safe_model}_thr_{thr_label}", type="benchmark_results")
    artifact.add_file(result_path)
    run.log_artifact(artifact)

    # Store for final summary
    _ALL_RUN_RESULTS.append(result_dict)

    # Per-run console summary
    print(f"\n{'=' * 60}")
    print(f"RUN COMPLETE: {run_name}")
    print(f"{'=' * 60}")
    print(f"Model:              {model_cfg['name']}")
    print(f"Threshold:          {confidence_threshold}")
    print(f"Runtime:            {total_runtime:.2f}s")
    print(f"--- Global (micro) ---")
    print(f"  Precision:        {prec_micro:.4f}")
    print(f"  Recall:           {rec_micro:.4f}")
    print(f"  F1:               {f1_micro:.4f}")
    print(f"--- Class 1 (hallucinated) ---")
    print(f"  Precision:        {prec_c1:.4f}")
    print(f"  Recall:           {rec_c1:.4f}")
    print(f"  F1:               {f1_c1:.4f}")
    print(f"--- Class 0 (supported) ---")
    print(f"  Precision:        {prec_c0:.4f}")
    print(f"  Recall:           {rec_c0:.4f}")
    print(f"  F1:               {f1_c0:.4f}")
    print(f"{'=' * 60}\n")

    run.finish()


def print_final_summary():
    """Print a ranked comparison table after all sweep runs complete."""
    if not _ALL_RUN_RESULTS:
        print("No results collected — skipping summary.")
        return

    print("\n")
    print("=" * 100)
    print("CONFIDENCE THRESHOLD SWEEP — FINAL SUMMARY")
    print("=" * 100)

    # Group by model
    models = sorted(set(r["model_name"] for r in _ALL_RUN_RESULTS))

    header = (
        f"{'Model':<45} {'Thr':>5} {'F1_c1':>8} {'Rec_c1':>8} "
        f"{'Prec_c1':>8} {'F1_c0':>8} {'F1_mic':>8} {'Runtime':>10}"
    )
    print(header)
    print("-" * 100)

    best_per_model = {}

    for model in models:
        model_runs = sorted(
            [r for r in _ALL_RUN_RESULTS if r["model_name"] == model],
            key=lambda r: r["confidence_threshold"],
        )
        for r in model_runs:
            marker = ""
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

        # Find best threshold for this model by F1 class 1
        best = max(model_runs, key=lambda r: r["f1_binary_class_1"])
        best_per_model[model] = best
        print()

    print("=" * 100)
    print("BEST THRESHOLD PER MODEL (by F1 class 1)")
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

    # Also save summary to JSON
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


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(
        description="W&B sweep: confidence threshold comparison for both models."
    )
    parser.add_argument(
        "--entity", type=str, default=hb.WANDB_ENTITY,
        help="W&B entity.",
    )
    parser.add_argument(
        "--project", type=str, default=DEFAULT_SWEEP_PROJECT,
        help="W&B project name for the sweep.",
    )
    parser.add_argument(
        "--sweep-id", type=str, default=None,
        help="Existing W&B sweep id. If omitted, a new sweep is created.",
    )
    parser.add_argument(
        "--count", type=int,
        default=len(SWEEP_THRESHOLDS) * len(hb.MODELS_TO_EVALUATE),
        help="How many agent runs to execute.",
    )
    parser.add_argument(
        "--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for per-run local result files.",
    )
    parser.add_argument(
        "--seed", type=int, default=hb.SEED,
        help="Global random seed.",
    )
    parser.add_argument(
        "--create-only", action="store_true",
        help="Create sweep and print id, but do not start the agent.",
    )
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep_id: {sweep_id}")
    else:
        sweep_config = build_sweep_config(
            output_prefix=args.output_prefix,
            seed=args.seed,
        )
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            entity=args.entity,
            project=args.project,
        )
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

    # After all sweep runs complete, print the final comparison
    print_final_summary()


if __name__ == "__main__":
    main()