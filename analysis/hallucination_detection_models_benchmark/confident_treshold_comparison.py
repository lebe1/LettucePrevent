"""
Confidence threshold sweep for hallucination detection models.

Runs a W&B grid sweep over:
- confidence_threshold: [0.6, 0.7, 0.8, 0.9]
- model_idx: [0, 1]  (TinyLettuce and LettuceDetect-base)

Dataset setup:
- wandb/RAGTruth-processed (test split)
- 15 unique (context, query) samples per task_type
"""

import argparse
import hashlib
import json
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from datasets import load_dataset

import hallucination_benchmark as hb


SWEEP_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
UNIQUE_PAIRS_PER_TASK = 15
DEFAULT_SWEEP_NAME = "confidence-threshold-comparison-rq3"
DEFAULT_SWEEP_PROJECT = "hdm-benchmark-rq3-threshold-sweep"
DEFAULT_OUTPUT_PREFIX = "rq3_confident_treshold_sweep"


_SAMPLES_CACHE: Optional[List[Dict]] = None
_TOKENIZER_CACHE: Optional[hb.LLMTokenizerWrapper] = None


def set_seed(seed: int):
    """Seed all local RNGs used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_benchmark_samples_for_sweep() -> List[Dict]:
    """
    Build benchmark samples deterministically:
    - filter rows with labels
    - keep first UNIQUE_PAIRS_PER_TASK unique (context, query) per task
    - emit samples in sorted task_type order for stable global ordering
    """
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
    # Sort task types to keep output ordering deterministic across runs/processes.
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
    """
    Create a compact fingerprint of the exact sample list used in a run.
    If two runs have the same hash, they evaluated the same samples in the
    same order.
    """
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
    """Load and cache tokenizer once per process."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        print(f"Loading tokenizer: {hb.LLM_TOKENIZER_NAME}")
        _TOKENIZER_CACHE = hb.LLMTokenizerWrapper(hb.LLM_TOKENIZER_NAME)
    return _TOKENIZER_CACHE


def build_sweep_config(output_prefix: str, seed: int) -> Dict:
    """Return W&B grid-sweep config over model and confidence threshold."""
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
    with wandb.init() as run:
        cfg = wandb.config

        model_idx = int(cfg.model_idx)
        confidence_threshold = float(cfg.confidence_threshold)
        output_prefix = str(cfg.output_prefix)
        seed = int(cfg.seed)
        model_cfg = hb.MODELS_TO_EVALUATE[model_idx]
        set_seed(seed)

        print("\n" + "#" * 70)
        print(
            f"# Sweep run: model={model_cfg['name']} "
            f"threshold={confidence_threshold:.2f}"
        )
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
                f"  [{model_cfg['name']}] sample {i + 1}/{len(samples)} "
                f"(task={sample['task_type']})"
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

            run.log(
                {
                    "sample_index": i,
                    "task_type": sample["task_type"],
                    "sample_precision": result.aggregate_precision,
                    "sample_recall": result.aggregate_recall,
                    "sample_f1": result.aggregate_f1,
                    "sample_runtime_s": result.runtime_seconds,
                }
            )

        total_runtime = time.time() - total_start

        prec_micro, rec_micro, f1_micro = hb.compute_metrics_micro(all_gt, all_pred)
        prec_c1, rec_c1, f1_c1 = hb.compute_metrics_class(
            all_gt, all_pred, positive_class=True
        )
        prec_c0, rec_c0, f1_c0 = hb.compute_metrics_class(
            all_gt, all_pred, positive_class=False
        )

        run.summary["model_name"] = model_cfg["name"]
        run.summary["model_path"] = model_cfg["model_path"]
        run.summary["num_samples"] = len(samples)
        run.summary["unique_pairs_per_task"] = UNIQUE_PAIRS_PER_TASK
        run.summary["sample_manifest_hash"] = sample_manifest_hash
        run.summary["seed"] = seed
        run.summary["tokenizer_name"] = hb.LLM_TOKENIZER_NAME
        run.summary["precision_micro"] = prec_micro
        run.summary["recall_micro"] = rec_micro
        run.summary["f1_micro"] = f1_micro
        run.summary["precision_class_1"] = prec_c1
        run.summary["recall_class_1"] = rec_c1
        run.summary["f1_binary_class_1"] = f1_c1
        run.summary["precision_class_0"] = prec_c0
        run.summary["recall_class_0"] = rec_c0
        run.summary["f1_binary_class_0"] = f1_c0
        run.summary["total_runtime_seconds"] = total_runtime

        safe_model = model_cfg["name"].replace("/", "_")
        thr_label = str(confidence_threshold).replace(".", "_")
        result_path = f"{output_prefix}_{safe_model}_thr_{thr_label}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model_cfg["name"],
                    "model_path": model_cfg["model_path"],
                    "confidence_threshold": confidence_threshold,
                    "num_samples": len(samples),
                    "unique_pairs_per_task": UNIQUE_PAIRS_PER_TASK,
                    "sample_manifest_hash": sample_manifest_hash,
                    "seed": seed,
                    "precision_micro": prec_micro,
                    "recall_micro": rec_micro,
                    "f1_micro": f1_micro,
                    "precision_class_1": prec_c1,
                    "recall_class_1": rec_c1,
                    "f1_binary_class_1": f1_c1,
                    "precision_class_0": prec_c0,
                    "recall_class_0": rec_c0,
                    "f1_binary_class_0": f1_c0,
                    "total_runtime_seconds": total_runtime,
                },
                f,
                indent=2,
            )
        artifact = wandb.Artifact("threshold_sweep_result", type="benchmark_results")
        artifact.add_file(result_path)
        run.log_artifact(artifact)

        print(f"\nCompleted run: {model_cfg['name']} @ threshold={confidence_threshold:.2f}")
        print(f"F1 class 1: {f1_c1:.4f} | F1 class 0: {f1_c0:.4f} | Runtime: {total_runtime:.2f}s")


def main():
    # ---------------------------------------------------------------------
    # CLI for two modes:
    # 1) Create + run a new sweep (default)
    # 2) Attach to existing sweep via --sweep-id
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="W&B sweep: confidence threshold comparison for both models."
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=hb.WANDB_ENTITY,
        help="W&B entity.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_SWEEP_PROJECT,
        help="W&B project name for the sweep.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing W&B sweep id. If omitted, a new sweep is created.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=len(SWEEP_THRESHOLDS) * len(hb.MODELS_TO_EVALUATE),
        help="How many agent runs to execute.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for per-run local result files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=hb.SEED,
        help="Global random seed for reproducible sweep runs.",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
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


if __name__ == "__main__":
    main()
