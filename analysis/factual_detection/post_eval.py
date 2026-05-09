"""
Post-generation evaluation: re-run LettuceDetect-base over the generated
answers to measure remaining hallucinations and bucket them by confidence.
"""

import json
from typing import Dict, List, Tuple

from lettucedetect.models.inference import HallucinationDetector


POST_EVAL_MODEL_PATH = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

CONFIDENCE_BUCKETS_DEF = [
    ("95%-100%", 0.95, 1.001),
    ("90%-95%",  0.90, 0.95),
    ("85%-90%",  0.85, 0.90),
    ("80%-85%",  0.80, 0.85),
    ("75%-80%",  0.75, 0.80),
    ("70%-75%",  0.70, 0.75),
]


def _bucket_for(conf: float, floor: float) -> str:
    """Return bucket label for confidence value, or '' if below the floor."""
    if conf < floor:
        return ""
    for label, lo, hi in CONFIDENCE_BUCKETS_DEF:
        if lo < conf <= hi:
            return label
    return ""


def evaluate_generated_answers(
    items: List[Dict],
    confidence_floor: float = 0.70,
) -> Tuple[Dict, List[Dict]]:
    """
    Run LettuceDetect-base over each generated answer and bucket detected
    hallucinations by confidence per task.

    Args:
        items: list of generation result dicts. Each must have
               'prompt', 'answer', 'task_type'.
        confidence_floor: minimum confidence for a hallucination to be kept.

    Returns:
        (stats_dict, hallucinated_items)
        - stats_dict: aggregate + per-task counts and bucket breakdown
        - hallucinated_items: list of items with 'hallucinations_detected' added
    """
    print(f"Loading post-eval detector: {POST_EVAL_MODEL_PATH}")
    detector = HallucinationDetector(
        method="transformer",
        model_path=POST_EVAL_MODEL_PATH,
    )

    task_types = ["Summary", "QA", "Data2txt"]
    counts_per_task = {
        tt: {label: 0 for label, _, _ in CONFIDENCE_BUCKETS_DEF}
        for tt in task_types
    }
    items_per_task = {tt: 0 for tt in task_types}
    items_with_halluc_per_task = {tt: 0 for tt in task_types}
    hallucinated_items: List[Dict] = []

    for i, item in enumerate(items):
        if "_meta" in item:
            continue
        tt = item.get("task_type", "unknown")
        if tt not in counts_per_task:
            counts_per_task[tt] = {label: 0 for label, _, _ in CONFIDENCE_BUCKETS_DEF}
            items_per_task[tt] = 0
            items_with_halluc_per_task[tt] = 0

        items_per_task[tt] += 1

        prompt = item.get("prompt", "")
        answer = item.get("answer", "")
        if not answer:
            continue

        try:
            preds = detector.predict(
                context=[prompt], answer=answer, output_format="spans",
            )
        except Exception as e:
            print(f"  [post-eval] item {i} ({tt}): detector error: {e}")
            continue

        kept = []
        for h in preds:
            conf = h.get("confidence", 0.0)
            if conf < confidence_floor:
                continue
            bucket = _bucket_for(conf, confidence_floor)
            if not bucket:
                continue
            kept.append(h)
            counts_per_task[tt][bucket] += 1

        if kept:
            items_with_halluc_per_task[tt] += 1
            enriched = dict(item)
            enriched["hallucinations_detected"] = kept
            enriched["prompt_number"] = i
            hallucinated_items.append(enriched)

    # Aggregate across tasks.
    bucket_labels = [label for label, _, _ in CONFIDENCE_BUCKETS_DEF]
    counts_total = {label: 0 for label in bucket_labels}
    for tt, counts in counts_per_task.items():
        for label in bucket_labels:
            counts_total[label] += counts.get(label, 0)

    items_total = sum(items_per_task.values())
    items_with_halluc_total = sum(items_with_halluc_per_task.values())
    halluc_total = sum(counts_total.values())

    stats = {
        "post_eval_model_path": POST_EVAL_MODEL_PATH,
        "confidence_floor": confidence_floor,
        "total": {
            "items": items_total,
            "items_with_hallucinations": items_with_halluc_total,
            "hallucinations": halluc_total,
            "buckets": counts_total,
        },
        "per_task": {
            tt: {
                "items": items_per_task[tt],
                "items_with_hallucinations": items_with_halluc_per_task[tt],
                "hallucinations": sum(counts_per_task[tt].values()),
                "buckets": counts_per_task[tt],
            }
            for tt in counts_per_task
        },
    }

    return stats, hallucinated_items


def write_stats_txt(stats: Dict, filepath: str) -> None:
    """Mirror the original script's TXT format, extended for per-task breakdown."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Post-eval model: {stats['post_eval_model_path']}\n")
        f.write(f"Confidence floor: {stats['confidence_floor']}\n\n")

        for tt, info in stats["per_task"].items():
            f.write(f"--- {tt} ---\n")
            f.write(f"  Items: {info['items']}\n")
            f.write(f"  Items with hallucinations: {info['items_with_hallucinations']}\n")
            f.write(f"  Hallucinations: {info['hallucinations']}\n")
            for label, n in info["buckets"].items():
                f.write(f"    {label}: {n}\n")
            f.write("\n")

        t = stats["total"]
        f.write("--- TOTAL ---\n")
        f.write(f"  Items: {t['items']}\n")
        f.write(f"  Items with hallucinations: {t['items_with_hallucinations']}\n")
        f.write(f"  Hallucinations: {t['hallucinations']}\n")
        for label, n in t["buckets"].items():
            f.write(f"    {label}: {n}\n")


def write_hallucinations_json(
    hallucinated_items: List[Dict],
    filepath: str,
    confidence_floor: float,
) -> None:
    """
    Emit hallucinations grouped by confidence bucket and broken down per task.
    Compatible with downstream tooling expecting bucket-keyed structures.
    """
    by_task: Dict[str, Dict[str, List]] = {}
    for item in hallucinated_items:
        tt = item.get("task_type", "unknown")
        by_task.setdefault(tt, {label: [] for label, _, _ in CONFIDENCE_BUCKETS_DEF})
        for h in item["hallucinations_detected"]:
            bucket = _bucket_for(h.get("confidence", 0.0), confidence_floor)
            if not bucket:
                continue
            by_task[tt][bucket].append({
                **h,
                "prompt_number": item.get("prompt_number"),
                "task_type": tt,
            })

    output = []
    for tt, buckets in by_task.items():
        for bucket_label, halls in buckets.items():
            if halls:
                output.append({
                    f"{tt} :: Confidence interval {bucket_label}": {
                        "Hallucinations": halls
                    }
                })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)