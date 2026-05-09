"""
RQ1 + RQ2 evaluation for the hallucination-prevention experiment.

Sample: 450 paired prompts (150 Summary + 150 Data2Txt + 150 QA), pooled.

RQ1: Does logits-based hallucination prevention reduce hallucinations per text?
     -> Paired test on LettuceDetect span counts (>=70% confidence, all pooled).

RQ2: What is the runtime trade-off?
     -> Paired test on duration_seconds, reported (a) on all data and
        (b) excluding upper-tail runtime-diff outliers whose
        logits_modifications == 0 ("unrelated factors", e.g. system interruptions).

Bonferroni correction (family-wise across the full study):
    alpha = 0.05 / 6 ~= 0.00833
    (2 metrics x 3 models = 6 tests)

The script is run once per model. The corrected alpha is fixed regardless,
because the correction applies to the family of 6 tests, not to a single run.

Inputs
------
gen_baseline.json    : full generation file for baseline run
gen_prevention.json  : full generation file for prevention run
spans_baseline.json  : LettuceDetect spans (>=70%) for baseline outputs
spans_prevention.json: LettuceDetect spans (>=70%) for prevention outputs

Usage
-----
python extract_factual_hallucinations.py \
    gen_baseline.json gen_prevention.json \
    spans_baseline.json spans_prevention.json \
    --model-name mistral -o results_mistral.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


# Family-wise Bonferroni: 2 parameters (hallucinations, runtime) x 3 models = 6 tests.
N_BONFERRONI_TESTS = 6
ALPHA_FAMILY = 0.05
ALPHA_BONFERRONI = ALPHA_FAMILY / N_BONFERRONI_TESTS  # ~0.008333


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_generation_file(path: str) -> Dict[int, dict]:
    """
    Load a generation JSON. Returns {prompt_number -> metadata dict}.

    The generation files in this pipeline do NOT carry an explicit
    prompt_number field. We assign prompt_number positionally:
    the i-th non-meta entry (0-indexed) is prompt_number = i.

    This matches the upstream convention used to build the spans files
    (where prompt_number is the global test-set index over all 450 prompts).

    Skips non-dict entries and trailing _meta blocks.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: Dict[int, dict] = {}
    positional = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        if "_meta" in item:
            continue
        # Honour explicit prompt_number if present, else assign positionally
        pn = item.get("prompt_number", positional)
        records[pn] = {
            "task_type": item.get("task_type"),
            "duration_seconds": item.get("duration_seconds"),
            "logits_modifications": item.get("logits_modifications", 0),
        }
        positional += 1
    return records


def load_span_counts(path: str) -> Counter:
    """
    Load LettuceDetect spans JSON, return {prompt_number -> total span count}.
    Pools across all confidence buckets (>=70% threshold already applied upstream).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counts: Counter = Counter()
    for bucket_dict in data:
        if not isinstance(bucket_dict, dict):
            continue
        for _bucket_label, payload in bucket_dict.items():
            if not isinstance(payload, dict):
                continue
            for span in payload.get("Hallucinations", []):
                pn = span.get("prompt_number")
                if pn is None:
                    continue
                counts[pn] += 1
    return counts


def build_paired_arrays(
    base_records: Dict[int, dict],
    prev_records: Dict[int, dict],
    base_counts: Counter,
    prev_counts: Counter,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           List[int], List[str]]:
    """
    Pair on prompt_number. Returns hallucination counts, durations,
    prevention's logits_modifications, plus the prompt_number list and
    matching task-type list.
    """
    common = sorted(set(base_records) & set(prev_records))
    hall_base = np.array([base_counts.get(pn, 0) for pn in common], dtype=float)
    hall_prev = np.array([prev_counts.get(pn, 0) for pn in common], dtype=float)
    rt_base = np.array([base_records[pn]["duration_seconds"] for pn in common],
                       dtype=float)
    rt_prev = np.array([prev_records[pn]["duration_seconds"] for pn in common],
                       dtype=float)
    mods_prev = np.array([prev_records[pn]["logits_modifications"] for pn in common],
                         dtype=float)
    task_types = [base_records[pn]["task_type"] for pn in common]
    return hall_base, hall_prev, rt_base, rt_prev, mods_prev, common, task_types


# ----------------------------------------------------------------------------
# Statistical tests
# ----------------------------------------------------------------------------

def shapiro_on_differences(x: np.ndarray, y: np.ndarray) -> dict:
    diffs = y - x
    if len(diffs) < 3:
        return {"n": len(diffs), "error": "n < 3"}
    if np.all(diffs == diffs[0]):
        return {"n": len(diffs), "error": "zero variance in differences"}
    stat, p = stats.shapiro(diffs)
    return {
        "n": int(len(diffs)),
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal_at_alpha_05": bool(p > 0.05),
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)),
    }


def paired_t(x: np.ndarray, y: np.ndarray) -> dict:
    result = stats.ttest_rel(y, x)
    diffs = y - x
    sd = np.std(diffs, ddof=1)
    cohens_d = float(np.mean(diffs) / sd) if sd > 0 else 0.0
    return {
        "test": "paired_t",
        "n": int(len(diffs)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "df": int(len(diffs) - 1),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "std_diff": float(sd),
        "cohens_d": cohens_d,
        "significant_bonferroni": bool(result.pvalue < ALPHA_BONFERRONI),
    }


def wilcoxon(x: np.ndarray, y: np.ndarray) -> dict:
    diffs = y - x
    non_zero = diffs[diffs != 0]
    if len(non_zero) == 0:
        return {
            "test": "wilcoxon",
            "n": int(len(diffs)),
            "n_non_zero": 0,
            "error": "all differences are zero",
        }

    result = stats.wilcoxon(y, x, zero_method="wilcox", correction=False,
                            alternative="two-sided")
    ranks = stats.rankdata(np.abs(non_zero))
    w_plus = float(np.sum(ranks[non_zero > 0]))
    w_minus = float(np.sum(ranks[non_zero < 0]))
    total = w_plus + w_minus
    r_effect = (w_plus - w_minus) / total if total > 0 else 0.0

    return {
        "test": "wilcoxon",
        "n": int(len(diffs)),
        "n_non_zero": int(len(non_zero)),
        "n_zero_diff": int(len(diffs) - len(non_zero)),
        "statistic": float(result.statistic),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_value": float(result.pvalue),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "rank_biserial_r": float(r_effect),
        "significant_bonferroni": bool(result.pvalue < ALPHA_BONFERRONI),
    }


def choose_and_run(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    normality = shapiro_on_differences(x, y)
    if "error" in normality:
        test_result = wilcoxon(x, y)
        normality["fallback"] = "Wilcoxon (normality not assessable)"
    elif normality["is_normal_at_alpha_05"]:
        test_result = paired_t(x, y)
    else:
        test_result = wilcoxon(x, y)
    return {"label": label, "normality": normality, "test": test_result}


# ----------------------------------------------------------------------------
# Runtime outlier rule (per proposal)
# ----------------------------------------------------------------------------

def identify_unexplained_upper_outliers(
    rt_diffs: np.ndarray, mods_prev: np.ndarray, iqr_multiplier: float = 1.5,
) -> np.ndarray:
    """
    Mark paired observations as outliers iff:
      - (prev - base) runtime-diff is in the upper tail (> Q3 + 1.5*IQR), AND
      - prevention applied 0 logits modifications (i.e. no algorithmic reason
        for the prevention run to be slower; the slowdown is from "unrelated
        factors" such as system interruptions).
    """
    q1, q3 = np.percentile(rt_diffs, [25, 75])
    upper_bound = q3 + iqr_multiplier * (q3 - q1)
    return (rt_diffs > upper_bound) & (mods_prev == 0)


# ----------------------------------------------------------------------------
# Descriptives
# ----------------------------------------------------------------------------

def describe(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "total": float(np.sum(arr)),
        "n_with_value_gt_0": int(np.sum(arr > 0)),
        "pct_with_value_gt_0": float(np.mean(arr > 0) * 100),
    }


def contingency_2x2(hall_base: np.ndarray, hall_prev: np.ndarray) -> dict:
    a = int(np.sum((hall_base > 0) & (hall_prev > 0)))
    b = int(np.sum((hall_base > 0) & (hall_prev == 0)))
    c = int(np.sum((hall_base == 0) & (hall_prev > 0)))
    d = int(np.sum((hall_base == 0) & (hall_prev == 0)))
    return {
        "both_have_hallucinations": a,
        "only_baseline_has": b,
        "only_prevention_has": c,
        "neither_has": d,
        "row_baseline_has_total": a + b,
        "row_baseline_no_total": c + d,
        "col_prevention_has_total": a + c,
        "col_prevention_no_total": b + d,
        "grand_total": a + b + c + d,
    }


def task_breakdown(values_base: np.ndarray, values_prev: np.ndarray,
                   task_types: List[str]) -> dict:
    """Per-task descriptives (exploratory; primary analysis is pooled)."""
    out: dict = {}
    arr_tasks = np.array(task_types)
    for t in sorted(set(task_types)):
        mask = arr_tasks == t
        out[t] = {
            "n": int(mask.sum()),
            "baseline": describe(values_base[mask]),
            "prevention": describe(values_prev[mask]),
            "mean_diff_prev_minus_base": float(
                np.mean(values_prev[mask] - values_base[mask])
            ),
        }
    return out


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def print_header(gen1: str, gen2: str, n: int, model_name: str = None) -> None:
    print("=" * 78)
    title = "RQ1 + RQ2: hallucination prevention - baseline vs prevention"
    if model_name:
        title += f"  [{model_name}]"
    print(title)
    print("=" * 78)
    print(f"Baseline generations  : {gen1}")
    print(f"Prevention generations: {gen2}")
    print(f"Paired prompts (n)    : {n}")
    print(f"Bonferroni alpha      : {ALPHA_BONFERRONI:.6f} "
          f"(= {ALPHA_FAMILY} / {N_BONFERRONI_TESTS} tests; "
          f"2 metrics x 3 models)")
    print("=" * 78)


def print_descriptives(label: str, base_stats: dict, prev_stats: dict) -> None:
    print(f"\n--- Descriptive statistics: {label} ---")
    print(f"{'metric':<22} {'baseline':>14} {'prevention':>14}")
    for key in ("n", "mean", "median", "std", "min", "q1", "q3", "max", "total",
                "n_with_value_gt_0", "pct_with_value_gt_0"):
        bv = base_stats.get(key, "")
        pv = prev_stats.get(key, "")
        if isinstance(bv, float):
            print(f"{key:<22} {bv:>14.4f} {pv:>14.4f}")
        else:
            print(f"{key:<22} {bv:>14} {pv:>14}")


def print_contingency(label: str, c: dict) -> None:
    print(f"\n--- Contingency table (2x2): {label} ---")
    print(f"{'':30} {'prev: HAS':>12} {'prev: NONE':>12} {'total':>10}")
    print(f"{'baseline: HAS hallucination':<30} "
          f"{c['both_have_hallucinations']:>12} "
          f"{c['only_baseline_has']:>12} "
          f"{c['row_baseline_has_total']:>10}")
    print(f"{'baseline: NO hallucination':<30} "
          f"{c['only_prevention_has']:>12} "
          f"{c['neither_has']:>12} "
          f"{c['row_baseline_no_total']:>10}")
    print(f"{'total':<30} "
          f"{c['col_prevention_has_total']:>12} "
          f"{c['col_prevention_no_total']:>12} "
          f"{c['grand_total']:>10}")


def print_test(result: dict) -> None:
    norm = result["normality"]
    test = result["test"]
    print(f"\n--- Hypothesis test: {result['label']} ---")

    if "error" in norm:
        print(f"  Shapiro-Wilk: {norm['error']}  "
              f"(fallback: {norm.get('fallback', '-')})")
    else:
        print(f"  Shapiro-Wilk on differences: W = {norm['statistic']:.4f}, "
              f"p = {_fmt_p(norm['p_value'])}  "
              f"-> {'normal' if norm['is_normal_at_alpha_05'] else 'non-normal'}")
        print(f"  mean diff = {norm['mean_diff']:.4f}, "
              f"sd diff = {norm['std_diff']:.4f}")

    print(f"  Test used: {test.get('test', 'n/a')}")
    if "error" in test:
        print(f"  {test['error']}")
        return

    if test["test"] == "paired_t":
        print(f"  t = {test['statistic']:.4f}, df = {test['df']}, "
              f"p = {_fmt_p(test['p_value'])}")
        print(f"  mean diff (prev - base) = {test['mean_diff']:.4f}")
        print(f"  Cohen's d               = {test['cohens_d']:.4f}")
    else:
        print(f"  W = {test['statistic']:.4f}, p = {_fmt_p(test['p_value'])}")
        print(f"  n non-zero diffs        = {test['n_non_zero']}")
        print(f"  W+ = {test['w_plus']:.1f}, W- = {test['w_minus']:.1f}")
        print(f"  median diff (prev-base) = {test['median_diff']:.4f}")
        print(f"  rank-biserial r         = {test['rank_biserial_r']:.4f}")

    marker = "YES" if test["significant_bonferroni"] else "no"
    print(f"  Significant at alpha={ALPHA_BONFERRONI:.6f}: {marker}")


def print_task_breakdown(label: str, breakdown: dict) -> None:
    print(f"\n--- Per-task-type descriptives: {label} (exploratory, not tested) ---")
    print(f"{'task':<10} {'n':>5} {'base mean':>12} {'prev mean':>12} "
          f"{'mean diff':>12} {'base total':>12} {'prev total':>12}")
    for task, d in breakdown.items():
        print(f"{task:<10} {d['n']:>5} "
              f"{d['baseline']['mean']:>12.4f} "
              f"{d['prevention']['mean']:>12.4f} "
              f"{d['mean_diff_prev_minus_base']:>12.4f} "
              f"{d['baseline']['total']:>12.2f} "
              f"{d['prevention']['total']:>12.2f}")


def print_outlier_block(n_excluded: int, excluded_idx: List[int],
                        prompts: List[int]) -> None:
    print(f"\n--- Runtime outlier handling ---")
    print(f"  Rule: upper-tail runtime-diff (> Q3 + 1.5*IQR) "
          f"AND logits_modifications == 0")
    print(f"  Excluded count: {n_excluded}")
    if 0 < n_excluded <= 30:
        print(f"  Excluded prompt_numbers: {[prompts[i] for i in excluded_idx]}")


# ----------------------------------------------------------------------------
# RQ1 analysis (hallucinations)
# ----------------------------------------------------------------------------

def analyze_rq1(hall_base: np.ndarray, hall_prev: np.ndarray,
                task_types: List[str], n: int) -> dict:
    print("\n" + "#" * 78)
    print("#  RQ1: HALLUCINATIONS PER TEXT (LettuceDetect spans, >=70%, pooled)")
    print("#" * 78)

    desc_base = describe(hall_base)
    desc_prev = describe(hall_prev)
    print_descriptives("hallucinations per text (pooled)", desc_base, desc_prev)

    cont = contingency_2x2(hall_base, hall_prev)
    print_contingency("hallucination presence per text (pooled)", cont)

    primary = choose_and_run(
        hall_base, hall_prev,
        f"hallucinations per text (pooled, n={n})",
    )
    print_test(primary)

    breakdown = task_breakdown(hall_base, hall_prev, task_types)
    print_task_breakdown("hallucinations", breakdown)

    return {
        "metric": "lettucedetect_span_count_per_text",
        "confidence_threshold": ">=70% (all spans in input file)",
        "descriptives_baseline": desc_base,
        "descriptives_prevention": desc_prev,
        "contingency_2x2": cont,
        "analysis_pooled": primary,
        "exploratory_per_task_descriptives": breakdown,
    }


# ----------------------------------------------------------------------------
# RQ2 analysis (runtime)
# ----------------------------------------------------------------------------

def analyze_rq2(rt_base: np.ndarray, rt_prev: np.ndarray, mods_prev: np.ndarray,
                task_types: List[str], prompts: List[int]) -> dict:
    print("\n" + "#" * 78)
    print("#  RQ2: RUNTIME PER TEXT (duration_seconds)")
    print("#" * 78)

    # All data
    desc_base_all = describe(rt_base)
    desc_prev_all = describe(rt_prev)
    print_descriptives("runtime (seconds) - ALL data",
                       desc_base_all, desc_prev_all)
    result_all = choose_and_run(rt_base, rt_prev, "runtime - ALL data")
    print_test(result_all)
    breakdown_all = task_breakdown(rt_base, rt_prev, task_types)
    print_task_breakdown("runtime - ALL data", breakdown_all)

    # Outlier removal per proposal
    rt_diffs = rt_prev - rt_base
    exclude_mask = identify_unexplained_upper_outliers(rt_diffs, mods_prev)
    excluded_idx = np.where(exclude_mask)[0].tolist()
    keep_mask = ~exclude_mask
    print_outlier_block(int(exclude_mask.sum()), excluded_idx, prompts)

    rt_base_clean = rt_base[keep_mask]
    rt_prev_clean = rt_prev[keep_mask]
    task_types_clean = [task_types[i] for i, k in enumerate(keep_mask) if k]

    desc_base_clean = describe(rt_base_clean)
    desc_prev_clean = describe(rt_prev_clean)
    print_descriptives("runtime (seconds) - outliers excluded",
                       desc_base_clean, desc_prev_clean)
    result_clean = choose_and_run(rt_base_clean, rt_prev_clean,
                                  "runtime - outliers excluded")
    print_test(result_clean)
    breakdown_clean = task_breakdown(rt_base_clean, rt_prev_clean,
                                     task_types_clean)
    print_task_breakdown("runtime - outliers excluded", breakdown_clean)

    return {
        "all_data": {
            "descriptives_baseline": desc_base_all,
            "descriptives_prevention": desc_prev_all,
            "analysis_pooled": result_all,
            "exploratory_per_task_descriptives": breakdown_all,
        },
        "outliers_excluded": {
            "rule": ("upper-tail runtime-diff (> Q3 + 1.5*IQR) "
                     "AND logits_modifications == 0"),
            "n_excluded": int(exclude_mask.sum()),
            "excluded_prompt_numbers": [prompts[i] for i in excluded_idx],
            "descriptives_baseline": desc_base_clean,
            "descriptives_prevention": desc_prev_clean,
            "analysis_pooled": result_clean,
            "exploratory_per_task_descriptives": breakdown_clean,
        },
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ1 + RQ2 paired tests with family-wise Bonferroni "
                    "(alpha = 0.05/6).",
    )
    parser.add_argument("gen_baseline", help="Baseline generation JSON")
    parser.add_argument("gen_prevention", help="Prevention generation JSON")
    parser.add_argument("spans_baseline", help="Baseline LettuceDetect spans JSON")
    parser.add_argument("spans_prevention", help="Prevention LettuceDetect spans JSON")
    parser.add_argument("-o", "--output", help="Optional JSON output file")
    parser.add_argument("--model-name", default=None,
                        help="Optional label for the LLM (e.g. mistral, qwen, llama)")
    args = parser.parse_args()

    print("Loading generations and spans...")
    base_records = load_generation_file(args.gen_baseline)
    prev_records = load_generation_file(args.gen_prevention)
    base_counts = load_span_counts(args.spans_baseline)
    prev_counts = load_span_counts(args.spans_prevention)

    print(f"  baseline   prompts: {len(base_records)} | "
          f"prompts with spans: {len(base_counts)} | "
          f"total spans: {sum(base_counts.values())}")
    print(f"  prevention prompts: {len(prev_records)} | "
          f"prompts with spans: {len(prev_counts)} | "
          f"total spans: {sum(prev_counts.values())}")

    hall_base, hall_prev, rt_base, rt_prev, mods_prev, prompts, task_types = (
        build_paired_arrays(base_records, prev_records, base_counts, prev_counts)
    )
    if len(prompts) == 0:
        print("\nNo common prompts. Aborting.")
        return

    print_header(Path(args.gen_baseline).name, Path(args.gen_prevention).name,
                 len(prompts), args.model_name)

    print("\nTask-type composition of paired sample:")
    for t, c in sorted(Counter(task_types).items()):
        print(f"  {t:<12} {c}")

    rq1 = analyze_rq1(hall_base, hall_prev, task_types, len(prompts))
    rq2 = analyze_rq2(rt_base, rt_prev, mods_prev, task_types, prompts)

    if args.output:
        payload = {
            "model_name": args.model_name,
            "n_paired_prompts": len(prompts),
            "task_type_composition": dict(Counter(task_types)),
            "bonferroni": {
                "family_alpha": ALPHA_FAMILY,
                "n_tests": N_BONFERRONI_TESTS,
                "rationale": "2 parameters (hallucinations, runtime) x 3 models",
                "corrected_alpha": ALPHA_BONFERRONI,
            },
            "extraction_summary": {
                Path(args.gen_baseline).name: {
                    "n_prompts": len(base_records),
                    "n_prompts_with_spans": len(base_counts),
                    "total_spans": int(sum(base_counts.values())),
                },
                Path(args.gen_prevention).name: {
                    "n_prompts": len(prev_records),
                    "n_prompts_with_spans": len(prev_counts),
                    "total_spans": int(sum(prev_counts.values())),
                },
            },
            "rq1_hallucinations": rq1,
            "rq2_runtime": rq2,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results written to: {args.output}")


if __name__ == "__main__":
    main()