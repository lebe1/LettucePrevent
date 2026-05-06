"""Dataset loader for RQ1 experiments."""

import json
import math
from typing import Dict, List, Tuple

from datasets import load_dataset


HF_DATASET_NAME = "wandb/RAGTruth-processed"
HF_DATASET_SPLIT = "test"
UNIQUE_PAIRS_PER_TASK = 150


def load_local_summary_prompts(filepath: str) -> List[Dict]:
    """
    Load local RAGTruth summary prompts (used for 'number' and
    'baseline-run-numbers' detector types).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} local summary prompts from {filepath}")
    # Normalize to a common shape with prompt + task_type.
    normalized = []
    for item in data:
        normalized.append({
            "prompt": item["prompt"],
            "task_type": "Summary",
            "counts": item.get("counts"),
            "_source": "local-summary",
        })
    return normalized


def load_hf_ragtruth_prompts(
    n_per_task: int | None = UNIQUE_PAIRS_PER_TASK,
) -> List[Dict]:
    """
    Load wandb/RAGTruth-processed test split, select the first n_per_task
    unique (context, query) pairs per task type in dataset order. All three
    task types (Summary, QA, Data2txt) are included.

    If n_per_task is None, no per-task cap is applied (full test split).

    No LLM-answer assignment is performed: RQ1 generates the answers itself.
    The 'prompt' field given to the generator is just the row's context, since
    that is what was used at training time for these task types in RAGTruth.
    """
    # Treat None as "no cap" so the >= comparison below stays simple.
    cap: float = math.inf if n_per_task is None else n_per_task

    print(f"Loading HF dataset: {HF_DATASET_NAME} (split={HF_DATASET_SPLIT})...")
    hf_ds = load_dataset(HF_DATASET_NAME)
    test_split = hf_ds[HF_DATASET_SPLIT]

    seen_per_task: Dict[str, set] = {}
    counts_per_task: Dict[str, int] = {}
    selected: List[Dict] = []

    for row in test_split:
        if row.get("hallucination_labels") in (None, ""):
            # We don't strictly need labels for generation, but skipping
            # malformed rows keeps the prompt set consistent with RQ3.
            pass
        tt = row.get("task_type", "unknown")
        ctx = row["context"]
        qry = row["query"]
        seen_per_task.setdefault(tt, set())

        # Per-task cap: first n_per_task unique (context, query) pairs only.
        already_have = counts_per_task.get(tt, 0)
        if already_have >= cap:
            continue
        if (ctx, qry) in seen_per_task[tt]:
            continue
        seen_per_task[tt].add((ctx, qry))

        selected.append({
            "prompt": ctx,
            "query": qry,
            "task_type": tt,
            "counts": None,
            "_source": "hf-ragtruth",
        })
        counts_per_task[tt] = already_have + 1

    print(f"Selected {len(selected)} HF prompts:")
    by_task: Dict[str, int] = {}
    for s in selected:
        by_task[s["task_type"]] = by_task.get(s["task_type"], 0) + 1
    for tt, n in sorted(by_task.items()):
        print(f"  task_type={tt}: {n}")

    return selected


def load_prompts_for_detector(
    detector_type: str,
    local_summary_filepath: str = "./data/ragtruth_unique_summary_prompts.json",
    n_per_task: int | None = UNIQUE_PAIRS_PER_TASK,
) -> Tuple[List[Dict], str]:
    """
    Load the appropriate prompt set for the given detector type.
    Returns (prompts, source_label).
    """
    det = detector_type.lower()
    if det in ("number", "baseline-run-numbers"):
        return load_local_summary_prompts(local_summary_filepath), "local-summary"
    if det in ("lettucedetect", "lettuceprevent", "baseline-run-facts"):
        return load_hf_ragtruth_prompts(n_per_task=n_per_task), "hf-ragtruth"
    raise ValueError(f"Unknown detector_type: {detector_type}")