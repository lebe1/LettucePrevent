"""
End-to-end hallucination evaluation: extraction + statistical comparison.

  1. For each run JSON, identify Summary items.
  2. Extract numbers in TWO modes per run:
       - symmetric: digits AND number-words from both prompt and answer
       - digit-only: digit sequences from both prompt and answer
     A number from the answer is "hallucinated" if it does not appear in the
     prompt's number set. Token-counted: "2023" said three times in an answer
     when not in the prompt counts as 3 hallucinations.
  3. Pair the two runs on prompt_number. Items with zero hallucinations are
     INCLUDED so the paired sample is complete (n = 942 expected).
  4. Shapiro-Wilk on paired differences -> paired-t (normal) or Wilcoxon (non-normal).
  5. Bonferroni correction: alpha = 0.025 per test.
  6. Runtime reported (a) all data, (b) excluding upper-tail outliers whose
     logits_modifications == 0 (the "unrelated factor" case).

Output: prints both extraction modes side-by-side. Optional JSON dump and
optional legacy-format hallucination JSONs.

Usage:
    python evaluate_hallucinations.py baseline.json prevention.json \
        --model-name mistral -o results_mistral.json

    python evaluate_hallucinations.py baseline.json prevention.json \
        --model-name qwen -o results_qwen.json --write-hallucination-jsons
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from word2number import w2n


ALPHA_BONFERRONI = 0.05 / 6  # 0.05 / 2 tests * 3 models


# ----------------------------------------------------------------------------
# Number extraction
# ----------------------------------------------------------------------------

_NUMBER_WORD_PATTERN = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|and|[-])+\b",
    re.IGNORECASE,
)


def extract_cardinal_digits(text: str) -> List[str]:
    """Return all standalone digit sequences in `text`, in document order."""
    return re.findall(r"\b\d+\b", text)


def extract_number_words(text: str) -> List[str]:
    """Return number-word phrases in `text` converted to digit strings."""
    out: List[str] = []
    for match in _NUMBER_WORD_PATTERN.finditer(text):
        phrase = match.group().replace("-", " ").lower()
        try:
            out.append(str(w2n.word_to_num(phrase)))
        except ValueError:
            continue
    return out


def hallucinations_for_item(item: dict, mode: str) -> List[str]:
    """
    Numbers in the answer that are not in the prompt's number set.

    mode = "symmetric" : digits + number-words extracted from BOTH sides
    mode = "digits"    : digits extracted from BOTH sides (legacy mode)

    Token-counted: duplicates in the answer count separately.
    """
    prompt_text = item.get("prompt", "")
    answer_text = item.get("answer", "")

    if mode == "symmetric":
        prompt_numbers = set(extract_cardinal_digits(prompt_text)
                             + extract_number_words(prompt_text))
        answer_numbers = (extract_cardinal_digits(answer_text)
                          + extract_number_words(answer_text))
    elif mode == "digits":
        prompt_numbers = set(extract_cardinal_digits(prompt_text))
        answer_numbers = extract_cardinal_digits(answer_text)
    else:
        raise ValueError(f"Unknown extraction mode: {mode}")

    return [n for n in answer_numbers if n not in prompt_numbers]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_run(path: str) -> Tuple[Dict[int, dict], List[dict], List[dict]]:
    """
    Load a run JSON and compute hallucinations per Summary item under both
    extraction modes.

    Returns
    -------
    records : {prompt_number -> dict with hallucination counts (both modes),
                                  duration_seconds, logits_modifications}
    annotated_symmetric : items with at least one symmetric hallucination
    annotated_digits    : items with at least one digit-only hallucination
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: Dict[int, dict] = {}
    annotated_sym: List[dict] = []
    annotated_dig: List[dict] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict) or item.get("task_type") != "Summary":
            continue

        prompt_number = item.get("prompt_number", i)
        hall_sym = hallucinations_for_item(item, "symmetric")
        hall_dig = hallucinations_for_item(item, "digits")

        records[prompt_number] = {
            "hallucination_count_symmetric": len(hall_sym),
            "hallucination_count_digits": len(hall_dig),
            "duration_seconds": item.get("duration_seconds"),
            "logits_modifications": item.get("logits_modifications", 0),
        }

        if hall_sym:
            enriched = dict(item)
            enriched["hallucinated_numbers"] = hall_sym
            enriched["prompt_number"] = prompt_number
            annotated_sym.append(enriched)
        if hall_dig:
            enriched = dict(item)
            enriched["hallucinated_numbers"] = hall_dig
            enriched["prompt_number"] = prompt_number
            annotated_dig.append(enriched)

    return records, annotated_sym, annotated_dig


def build_paired_arrays(
    baseline: Dict[int, dict],
    prevention: Dict[int, dict],
    hall_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    common = sorted(set(baseline) & set(prevention))
    hall_base = np.array([baseline[p][hall_key] for p in common], dtype=float)
    hall_prev = np.array([prevention[p][hall_key] for p in common], dtype=float)
    rt_base = np.array([baseline[p]["duration_seconds"] for p in common], dtype=float)
    rt_prev = np.array([prevention[p]["duration_seconds"] for p in common], dtype=float)
    mods_prev = np.array([prevention[p]["logits_modifications"] for p in common], dtype=float)
    return hall_base, hall_prev, rt_base, rt_prev, mods_prev, common


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

    result = stats.wilcoxon(y, x, zero_method="wilcox", correction=False, alternative="two-sided")
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
# Runtime outliers (per proposal)
# ----------------------------------------------------------------------------

def identify_unexplained_upper_outliers(
    rt_diffs: np.ndarray, mods_prev: np.ndarray, iqr_multiplier: float = 1.5,
) -> np.ndarray:
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


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def print_header(file1: str, file2: str, n: int, model_name: str = None) -> None:
    print("=" * 78)
    title = "STATISTICAL COMPARISON: baseline vs hallucination-prevention"
    if model_name:
        title += f"  [{model_name}]"
    print(title)
    print("=" * 78)
    print(f"Baseline   (file 1): {file1}")
    print(f"Prevention (file 2): {file2}")
    print(f"Matched prompts    : {n}")
    print(f"Bonferroni alpha   : {ALPHA_BONFERRONI} (= 0.05 / 2 tests)")
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
        print(f"  Shapiro-Wilk: {norm['error']}  (fallback: {norm.get('fallback', '-')})")
    else:
        print(f"  Shapiro-Wilk on differences: W = {norm['statistic']:.4f}, "
              f"p = {_fmt_p(norm['p_value'])}  "
              f"-> {'normal' if norm['is_normal_at_alpha_05'] else 'non-normal'}")
        print(f"  mean diff = {norm['mean_diff']:.4f}, sd diff = {norm['std_diff']:.4f}")

    print(f"  Test used: {test.get('test', 'n/a')}")
    if "error" in test:
        print(f"  {test['error']}")
        return

    if test["test"] == "paired_t":
        print(f"  t = {test['statistic']:.4f}, df = {test['df']}, p = {_fmt_p(test['p_value'])}")
        print(f"  mean diff (prev - base) = {test['mean_diff']:.4f}")
        print(f"  Cohen's d               = {test['cohens_d']:.4f}")
    else:
        print(f"  W = {test['statistic']:.4f}, p = {_fmt_p(test['p_value'])}")
        print(f"  n non-zero diffs        = {test['n_non_zero']}")
        print(f"  W+ = {test['w_plus']:.1f}, W- = {test['w_minus']:.1f}")
        print(f"  median diff (prev-base) = {test['median_diff']:.4f}")
        print(f"  rank-biserial r         = {test['rank_biserial_r']:.4f}")

    marker = "YES" if test["significant_bonferroni"] else "no"
    print(f"  Significant at alpha={ALPHA_BONFERRONI}: {marker}")


def print_outlier_block(n_excluded: int, excluded_idx: List[int], prompts: List[int]) -> None:
    print(f"\n--- Runtime outlier handling ---")
    print(f"  Rule: upper-tail runtime-diff outlier (> Q3 + 1.5*IQR) AND logits_modifications == 0")
    print(f"  Excluded count: {n_excluded}")
    if 0 < n_excluded <= 20:
        print(f"  Excluded prompt_numbers: {[prompts[i] for i in excluded_idx]}")


def print_extraction_summary(name: str, records: Dict[int, dict],
                             annotated_sym: List[dict],
                             annotated_dig: List[dict]) -> None:
    sym = [r["hallucination_count_symmetric"] for r in records.values()]
    dig = [r["hallucination_count_digits"] for r in records.values()]
    print(f"\n[{name}] Summary items: {len(records)}")
    print(f"  Symmetric (digits + words): "
          f"{len(annotated_sym)} texts with hallucinations, total {sum(sym)}")
    print(f"  Digit-only               : "
          f"{len(annotated_dig)} texts with hallucinations, total {sum(dig)}")


# ----------------------------------------------------------------------------
# Per-mode analysis
# ----------------------------------------------------------------------------

def analyze_mode(
    baseline: Dict[int, dict],
    prevention: Dict[int, dict],
    mode: str,
) -> dict:
    """Run the full hallucination + runtime analysis for a given extraction mode."""
    hall_key = f"hallucination_count_{mode}"
    hall_base, hall_prev, rt_base, rt_prev, mods_prev, prompts = build_paired_arrays(
        baseline, prevention, hall_key,
    )

    print("\n" + "#" * 78)
    print(f"#  EXTRACTION MODE: {mode.upper()}")
    print("#" * 78)

    desc_base = describe(hall_base)
    desc_prev = describe(hall_prev)
    print_descriptives(f"hallucinations per text [{mode}]", desc_base, desc_prev)

    cont = contingency_2x2(hall_base, hall_prev)
    print_contingency(f"hallucination presence per text [{mode}]", cont)

    hall_result = choose_and_run(hall_base, hall_prev,
                                 f"hallucinations per text [{mode}]")
    print_test(hall_result)

    rt_desc_base_all = describe(rt_base)
    rt_desc_prev_all = describe(rt_prev)
    print_descriptives("runtime (seconds) - ALL data", rt_desc_base_all, rt_desc_prev_all)
    rt_result_all = choose_and_run(rt_base, rt_prev, "runtime - ALL data")
    print_test(rt_result_all)

    rt_diffs = rt_prev - rt_base
    exclude_mask = identify_unexplained_upper_outliers(rt_diffs, mods_prev)
    excluded_idx = np.where(exclude_mask)[0].tolist()
    keep_mask = ~exclude_mask
    print_outlier_block(int(exclude_mask.sum()), excluded_idx, prompts)

    rt_base_clean = rt_base[keep_mask]
    rt_prev_clean = rt_prev[keep_mask]
    rt_desc_base_clean = describe(rt_base_clean)
    rt_desc_prev_clean = describe(rt_prev_clean)
    print_descriptives("runtime (seconds) - outliers excluded",
                       rt_desc_base_clean, rt_desc_prev_clean)
    rt_result_clean = choose_and_run(rt_base_clean, rt_prev_clean,
                                     "runtime - outliers excluded")
    print_test(rt_result_clean)

    return {
        "mode": mode,
        "hallucinations_per_text": {
            "descriptives_baseline": desc_base,
            "descriptives_prevention": desc_prev,
            "contingency_2x2": cont,
            "analysis": hall_result,
        },
        "runtime_all_data": {
            "descriptives_baseline": rt_desc_base_all,
            "descriptives_prevention": rt_desc_prev_all,
            "analysis": rt_result_all,
        },
        "runtime_outliers_excluded": {
            "n_excluded": int(exclude_mask.sum()),
            "excluded_prompt_numbers": [prompts[i] for i in excluded_idx],
            "rule": "upper-tail runtime-diff outlier (> Q3 + 1.5*IQR) AND logits_modifications == 0",
            "descriptives_baseline": rt_desc_base_clean,
            "descriptives_prevention": rt_desc_prev_clean,
            "analysis": rt_result_clean,
        },
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hallucinations (symmetric + digit-only) and run paired tests.",
    )
    parser.add_argument("file1", help="Baseline run JSON")
    parser.add_argument("file2", help="Prevention run JSON")
    parser.add_argument("-o", "--output", help="Optional JSON file with full results")
    parser.add_argument("--model-name", default=None,
                        help="Optional label for the LLM under evaluation (e.g. mistral, qwen, llama)")
    parser.add_argument(
        "--write-hallucination-jsons",
        action="store_true",
        help="Also write per-run hallucinations_<file>_<mode>.json (legacy extraction output)",
    )
    args = parser.parse_args()

    print("Loading and extracting hallucinations (both modes)...")
    baseline, base_sym, base_dig = load_run(args.file1)
    prevention, prev_sym, prev_dig = load_run(args.file2)

    print_extraction_summary(Path(args.file1).name, baseline, base_sym, base_dig)
    print_extraction_summary(Path(args.file2).name, prevention, prev_sym, prev_dig)

    if args.write_hallucination_jsons:
        for path, sym, dig in [(args.file1, base_sym, base_dig),
                               (args.file2, prev_sym, prev_dig)]:
            stem = Path(path).stem
            for label, items in (("symmetric", sym), ("digits", dig)):
                out_path = Path(path).parent / f"hallucinations_{stem}_{label}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(items, f, indent=2, ensure_ascii=False)
                print(f"  -> wrote {out_path}")

    common = sorted(set(baseline) & set(prevention))
    if not common:
        print("\nNo common prompts between the two files. Aborting.")
        return

    print_header(Path(args.file1).name, Path(args.file2).name, len(common),
                 args.model_name)

    sym_results = analyze_mode(baseline, prevention, "symmetric")
    dig_results = analyze_mode(baseline, prevention, "digits")

    if args.output:
        payload = {
            "model_name": args.model_name,
            "n_matched_prompts": len(common),
            "alpha_bonferroni": ALPHA_BONFERRONI,
            "extraction_summary": {
                Path(args.file1).name: {
                    "summary_items": len(baseline),
                    "items_with_hallucinations_symmetric": len(base_sym),
                    "items_with_hallucinations_digits": len(base_dig),
                    "total_hallucinations_symmetric": int(sum(
                        r["hallucination_count_symmetric"] for r in baseline.values())),
                    "total_hallucinations_digits": int(sum(
                        r["hallucination_count_digits"] for r in baseline.values())),
                },
                Path(args.file2).name: {
                    "summary_items": len(prevention),
                    "items_with_hallucinations_symmetric": len(prev_sym),
                    "items_with_hallucinations_digits": len(prev_dig),
                    "total_hallucinations_symmetric": int(sum(
                        r["hallucination_count_symmetric"] for r in prevention.values())),
                    "total_hallucinations_digits": int(sum(
                        r["hallucination_count_digits"] for r in prevention.values())),
                },
            },
            "symmetric_extraction": sym_results,
            "digit_only_extraction": dig_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results written to: {args.output}")


if __name__ == "__main__":
    main()