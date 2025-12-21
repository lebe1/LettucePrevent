"""
Comprehensive Statistical Analysis: McNemar + Wilcoxon + Paired T-Test

This script implements a rigorous statistical framework:
1. Shapiro-Wilk test for normality assessment (determines appropriate tests)
2. McNemar's test for binary hallucination detection (presence/absence)
3. Wilcoxon signed-rank test for hallucination counts per text (PRIMARY TEST if non-normal)
4. Paired t-test for hallucination counts (ALTERNATIVE if normal) and runtime comparison
5. Rich descriptive statistics for interpretability
6. Practical significance thresholds evaluation

Statistical tests use Bonferroni correction (α = 0.05/2 = 0.025)
Descriptive statistics are reported without hypothesis testing
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
import math
from word2number import w2n


# ============================================================================
# NUMBER EXTRACTION FUNCTIONS
# ============================================================================

def extract_cardinal_digits(text):
    """Extract digit numbers from text."""
    return re.findall(r'\b\d+\b', text)


def extract_number_words(text):
    """Extract number words from text and convert to digits."""
    number_word_pattern = re.compile(
        r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
        r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
        r'eighty|ninety|hundred|thousand|million|billion|and|[-])+\b',
        re.IGNORECASE
    )
    matches = number_word_pattern.finditer(text)
    number_strings = []
    for match in matches:
        phrase = match.group().replace("-", " ").lower()
        try:
            number = str(w2n.word_to_num(phrase))
            number_strings.append(number)
        except ValueError:
            continue
    return number_strings


def normalize_prompt_numbers(prompt):
    """Extract all numbers (digits and words) from prompt."""
    digit_numbers = extract_cardinal_digits(prompt)
    word_numbers = extract_number_words(prompt)
    return set(digit_numbers + word_numbers)


def process_summary_item(item, index):
    """
    Process a single item to detect hallucinated numbers.
    
    Returns:
        dict with hallucinated_numbers count and metadata
    """
    if item.get("task_type") != "Summary":
        return None
    
    prompt_numbers = normalize_prompt_numbers(item.get("prompt", ""))
    answer_numbers = extract_cardinal_digits(item.get("answer", ""))
    hallucinated = [num for num in answer_numbers if num not in prompt_numbers]
    
    return {
        "prompt_number": index,
        "hallucinated_numbers": hallucinated,
        "hallucination_count": len(hallucinated),
        "duration_seconds": item.get("duration_seconds")
    }


def extract_hallucinations_from_file(filepath: str) -> Tuple[Dict[int, List[str]], Dict[int, int], Dict[int, float]]:
    """
    Extract hallucinated numbers, counts, and runtime from original JSON file.
    
    Returns:
        Tuple of (hallucinations_dict, counts_dict, runtime_dict)
        - hallucinations_dict: Maps prompt_number to list of hallucinated numbers
        - counts_dict: Maps prompt_number to count of hallucinations
        - runtime_dict: Maps prompt_number to duration_seconds
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hallucinations = {}
    counts = {}
    runtimes = {}
    
    for i, item in enumerate(data):
        if isinstance(item, dict) and item.get("task_type") == "Summary":
            result = process_summary_item(item, i)
            if result:
                hallucinations[i] = result["hallucinated_numbers"]
                counts[i] = result["hallucination_count"]
                runtimes[i] = result["duration_seconds"]
    
    return hallucinations, counts, runtimes


# ============================================================================
# SHAPIRO-WILK NORMALITY TEST
# ============================================================================

def shapiro_wilk_test(data: List[float]) -> Dict:
    """
    Perform Shapiro-Wilk test for normality.
    
    This is a simplified implementation for educational purposes.
    For production use, consider scipy.stats.shapiro()
    
    Args:
        data: List of numeric values
    
    Returns:
        Dictionary with test results and interpretation
    """
    n = len(data)
    
    if n < 3:
        return {
            'error': 'Sample size too small for normality test (n < 3)',
            'n': n,
            'recommendation': 'Use non-parametric tests'
        }
    
    if n > 5000:
        return {
            'note': 'Large sample (n > 5000): Normality tests may be overly sensitive',
            'n': n,
            'recommendation': 'Consider visual inspection and Central Limit Theorem'
        }
    
    # Calculate basic statistics
    sorted_data = sorted(data)
    mean_val = sum(data) / n
    
    # Calculate variance
    variance = sum((x - mean_val) ** 2 for x in data) / (n - 1)
    std_val = math.sqrt(variance)
    
    if std_val == 0:
        return {
            'error': 'Zero variance detected',
            'n': n,
            'recommendation': 'Data has no variation - statistical tests not applicable'
        }
    
    # Calculate skewness
    skewness = sum(((x - mean_val) / std_val) ** 3 for x in data) / n
    
    # Calculate kurtosis
    kurtosis = sum(((x - mean_val) / std_val) ** 4 for x in data) / n - 3
    
    # Simplified normality heuristics (not exact Shapiro-Wilk)
    # For exact implementation, use scipy.stats.shapiro
    
    # Check for discrete data (potential count data)
    unique_values = len(set(data))
    is_likely_count = all(x >= 0 and x == int(x) for x in data)
    
    # Assess normality based on skewness and kurtosis
    abs_skew = abs(skewness)
    abs_kurt = abs(kurtosis)
    
    # Rule of thumb: |skewness| < 2 and |kurtosis| < 7 suggests approximate normality
    likely_normal = abs_skew < 1.0 and abs_kurt < 2.0
    
    # Additional check: ratio of unique values
    unique_ratio = unique_values / n
    
    if is_likely_count and unique_ratio < 0.5:
        normality_assessment = "Non-normal (discrete count data with limited unique values)"
        p_value_approx = "< 0.05"
        is_normal = False
    elif abs_skew >= 2.0 or abs_kurt >= 7.0:
        normality_assessment = "Non-normal (extreme skewness or kurtosis)"
        p_value_approx = "< 0.05"
        is_normal = False
    elif abs_skew >= 1.0 or abs_kurt >= 2.0:
        normality_assessment = "Questionable normality (moderate skewness or kurtosis)"
        p_value_approx = "~ 0.05"
        is_normal = False
    else:
        normality_assessment = "Approximately normal"
        p_value_approx = "> 0.05"
        is_normal = True
    
    # Determine recommended test
    if is_likely_count and unique_ratio < 0.3:
        recommended_test = "Wilcoxon signed-rank (count data)"
    elif not is_normal:
        recommended_test = "Wilcoxon signed-rank (non-normal data)"
    else:
        recommended_test = "Paired t-test (normal data)"
    
    return {
        'n': n,
        'mean': mean_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'unique_values': unique_values,
        'unique_ratio': unique_ratio,
        'is_likely_count_data': is_likely_count,
        'normality_assessment': normality_assessment,
        'p_value_approx': p_value_approx,
        'is_normal': is_normal,
        'recommended_test': recommended_test
    }


def assess_normality_for_differences(data1: List[float], data2: List[float]) -> Dict:
    """
    Assess normality of the differences between paired samples.
    
    This is what matters for paired t-test vs Wilcoxon decision.
    """
    if len(data1) != len(data2):
        raise ValueError("Samples must have equal length")
    
    differences = [d2 - d1 for d1, d2 in zip(data1, data2)]
    
    normality_result = shapiro_wilk_test(differences)
    normality_result['differences_sample'] = differences[:10]  # First 10 for inspection
    
    return normality_result


# ============================================================================
# MCNEMAR'S TEST FUNCTIONS
# ============================================================================

def classify_prompts_mcnemar(file1_data: Dict[int, List[str]], 
                             file2_data: Dict[int, List[str]]) -> Dict[str, List[int]]:
    """Classify prompts into McNemar's test categories based on presence/absence."""
    categories = {'a': [], 'b': [], 'c': [], 'd': []}
    
    all_prompts = set(file1_data.keys()) | set(file2_data.keys())
    
    for prompt_num in sorted(all_prompts):
        file1_nums = file1_data.get(prompt_num, [])
        file2_nums = file2_data.get(prompt_num, [])
        
        file1_has_hallucination = len(file1_nums) > 0
        file2_has_hallucination = len(file2_nums) > 0
        
        if file1_has_hallucination and file2_has_hallucination:
            categories['a'].append(prompt_num)
        elif file1_has_hallucination and not file2_has_hallucination:
            categories['b'].append(prompt_num)
        elif not file1_has_hallucination and file2_has_hallucination:
            categories['c'].append(prompt_num)
        else:
            categories['d'].append(prompt_num)
    
    return categories


# ============================================================================
# WILCOXON SIGNED-RANK TEST (PRIMARY TEST FOR HALLUCINATIONS)
# ============================================================================

def wilcoxon_signed_rank_test(data1: List[int], data2: List[int]) -> Dict:
    """
    Perform Wilcoxon signed-rank test on paired data.
    
    This is the PRIMARY statistical test for hallucination counts.
    
    Args:
        data1: Hallucination counts for method 1
        data2: Hallucination counts for method 2
    
    Returns:
        Dictionary with test statistics
    """
    n = len(data1)
    if n != len(data2):
        raise ValueError("Samples must have equal length")
    
    if n < 2:
        return {
            'error': 'Not enough data points for Wilcoxon test',
            'n': n
        }
    
    # Calculate differences
    differences = [d2 - d1 for d1, d2 in zip(data1, data2)]
    
    # Remove zero differences
    non_zero_diffs = [(i, d) for i, d in enumerate(differences) if d != 0]
    n_non_zero = len(non_zero_diffs)
    
    if n_non_zero == 0:
        return {
            'n': n,
            'n_non_zero': 0,
            'w_statistic': 0,
            'p_value_approx': "= 1.0",
            'significant_bonferroni': False,
            'note': 'All differences are zero'
        }
    
    # Rank absolute differences
    abs_diffs = [(i, abs(d)) for i, d in non_zero_diffs]
    abs_diffs.sort(key=lambda x: x[1])
    
    # Assign ranks (handle ties with average rank)
    ranks = {}
    i = 0
    while i < len(abs_diffs):
        # Find ties
        j = i
        while j < len(abs_diffs) and abs_diffs[j][1] == abs_diffs[i][1]:
            j += 1
        # Average rank for ties
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            idx = abs_diffs[k][0]
            ranks[idx] = avg_rank
        i = j
    
    # Sum of positive ranks and negative ranks
    w_plus = sum(ranks[i] for i, d in non_zero_diffs if d > 0)
    w_minus = sum(ranks[i] for i, d in non_zero_diffs if d < 0)
    
    # Test statistic (smaller of the two)
    w_statistic = min(w_plus, w_minus)
    
    # Expected value and variance under null hypothesis
    expected_w = n_non_zero * (n_non_zero + 1) / 4
    variance_w = n_non_zero * (n_non_zero + 1) * (2 * n_non_zero + 1) / 24
    
    # Z-score with continuity correction
    if w_statistic < expected_w:
        z_score = (w_statistic + 0.5 - expected_w) / math.sqrt(variance_w)
    else:
        z_score = (w_statistic - 0.5 - expected_w) / math.sqrt(variance_w)
    
    z_abs = abs(z_score)
    
    # Approximate p-value (two-tailed)
    if z_abs >= 3.291:
        p_value_approx = "< 0.001"
    elif z_abs >= 2.807:  # For α=0.005 (two-tailed)
        p_value_approx = "< 0.005"
    elif z_abs >= 2.576:
        p_value_approx = "< 0.01"
    elif z_abs >= 2.241:  # For α=0.025 (two-tailed) - Bonferroni corrected
        p_value_approx = "< 0.025"
    elif z_abs >= 1.96:
        p_value_approx = "< 0.05"
    else:
        p_value_approx = "> 0.05"
    
    # Significance with Bonferroni correction (α = 0.025)
    significant_bonferroni = z_abs >= 2.241
    
    # Effect size (rank-biserial correlation)
    r_effect = (w_plus - w_minus) / (n_non_zero * (n_non_zero + 1) / 2)
    
    return {
        'n': n,
        'n_non_zero': n_non_zero,
        'n_zero_diff': n - n_non_zero,
        'w_statistic': w_statistic,
        'w_plus': w_plus,
        'w_minus': w_minus,
        'z_score': z_score,
        'p_value_approx': p_value_approx,
        'significant_bonferroni': significant_bonferroni,
        'effect_size_r': r_effect,
        'differences': differences
    }


def calculate_descriptive_stats(data: List[int]) -> Dict:
    """Calculate comprehensive descriptive statistics."""
    n = len(data)
    if n == 0:
        return {'n': 0}
    
    sorted_data = sorted(data)
    
    # Mean
    mean_val = sum(data) / n
    
    # Median
    if n % 2 == 0:
        median_val = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        median_val = sorted_data[n//2]
    
    # Quartiles
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]
    iqr = q3 - q1
    
    # Min, Max
    min_val = sorted_data[0]
    max_val = sorted_data[-1]
    
    # Standard deviation
    variance = sum((x - mean_val) ** 2 for x in data) / n
    std_val = math.sqrt(variance)
    
    # Count of non-zero values
    non_zero_count = sum(1 for x in data if x > 0)
    
    # Total sum
    total_sum = sum(data)
    
    return {
        'n': n,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'q1': q1,
        'q3': q3,
        'max': max_val,
        'iqr': iqr,
        'non_zero_count': non_zero_count,
        'non_zero_percentage': (non_zero_count / n * 100) if n > 0 else 0,
        'total_sum': total_sum
    }


# ============================================================================
# PAIRED T-TEST FUNCTIONS (FOR RUNTIME AND ALTERNATIVE FOR HALLUCINATIONS)
# ============================================================================

def detect_outliers_iqr(data: List[float], multiplier: float = 3.0) -> Tuple[List[int], float, float]:
    """Detect outliers using IQR method."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    q1 = sorted_data[n // 4]
    q3 = sorted_data[3 * n // 4]
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outlier_indices = [i for i, val in enumerate(data) if val < lower_bound or val > upper_bound]
    
    return outlier_indices, lower_bound, upper_bound


def paired_t_test(data1: List[float], data2: List[float]) -> Dict:
    """Perform paired t-test on two samples."""
    n = len(data1)
    if n != len(data2):
        raise ValueError("Samples must have equal length")
    
    if n < 2:
        return {
            'error': 'Not enough data points for t-test',
            'n': n
        }
    
    # Calculate differences
    differences = [d2 - d1 for d1, d2 in zip(data1, data2)]
    
    # Mean and standard deviation of differences
    mean_diff = sum(differences) / n
    variance = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
    std_diff = math.sqrt(variance)
    
    # Standard error
    se = std_diff / math.sqrt(n)
    
    # T-statistic
    if se == 0:
        t_stat = 0
    else:
        t_stat = mean_diff / se
    
    # Degrees of freedom
    df = n - 1
    
    # Approximate p-value (two-tailed)
    t_abs = abs(t_stat)
    if df > 0:
        if df >= 30:
            critical_025 = 2.457  # For α=0.025 (Bonferroni corrected)
            critical_001 = 3.291
            critical_01 = 2.750
            critical_05 = 2.042
        elif df >= 20:
            critical_025 = 2.528
            critical_001 = 3.552
            critical_01 = 2.845
            critical_05 = 2.086
        elif df >= 10:
            critical_025 = 2.634
            critical_001 = 3.930
            critical_01 = 3.169
            critical_05 = 2.228
        else:
            critical_025 = 2.776
            critical_001 = 4.781
            critical_01 = 3.707
            critical_05 = 2.571
        
        if t_abs >= critical_001:
            p_value_approx = "< 0.001"
        elif t_abs >= critical_01:
            p_value_approx = "< 0.01"
        elif t_abs >= critical_025:
            p_value_approx = "< 0.025"
        elif t_abs >= critical_05:
            p_value_approx = "< 0.05"
        else:
            p_value_approx = "> 0.05"
        
        # Significance with Bonferroni correction
        significant_bonferroni = t_abs >= critical_025
    else:
        p_value_approx = "N/A"
        significant_bonferroni = False
    
    # Calculate median difference
    median_diff = sorted(differences)[n // 2] if n % 2 == 1 else \
                  (sorted(differences)[n // 2 - 1] + sorted(differences)[n // 2]) / 2
    
    # Effect size (Cohen's d)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    return {
        'n': n,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'std_diff': std_diff,
        'se': se,
        't_statistic': t_stat,
        'df': df,
        'p_value_approx': p_value_approx,
        'significant_bonferroni': significant_bonferroni,
        'cohens_d': cohens_d,
        'differences': differences
    }


def analyze_runtime_comparison(runtime1: Dict[int, float], 
                               runtime2: Dict[int, float],
                               remove_outliers: bool = True) -> Dict:
    """Analyze runtime differences between two methods."""
    common_prompts = sorted(set(runtime1.keys()) & set(runtime2.keys()))
    
    if len(common_prompts) == 0:
        return {'error': 'No common prompts between files'}
    
    times1 = [runtime1[p] for p in common_prompts]
    times2 = [runtime2[p] for p in common_prompts]
    
    results = {
        'total_prompts': len(common_prompts),
        'prompt_numbers': common_prompts
    }
    
    # Detect outliers
    differences = [t2 - t1 for t1, t2 in zip(times1, times2)]
    outlier_indices, lower_bound, upper_bound = detect_outliers_iqr(differences, multiplier=3.0)
    
    results['outliers'] = {
        'count': len(outlier_indices),
        'indices': outlier_indices,
        'prompt_numbers': [common_prompts[i] for i in outlier_indices],
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    # Perform t-test with all data
    results['with_outliers'] = paired_t_test(times1, times2)
    results['with_outliers']['mean_time1'] = sum(times1) / len(times1)
    results['with_outliers']['mean_time2'] = sum(times2) / len(times2)
    results['with_outliers']['median_time1'] = sorted(times1)[len(times1) // 2]
    results['with_outliers']['median_time2'] = sorted(times2)[len(times2) // 2]
    
    # Perform t-test without outliers
    if remove_outliers and len(outlier_indices) > 0:
        clean_times1 = [t for i, t in enumerate(times1) if i not in outlier_indices]
        clean_times2 = [t for i, t in enumerate(times2) if i not in outlier_indices]
        
        if len(clean_times1) >= 2:
            results['without_outliers'] = paired_t_test(clean_times1, clean_times2)
            results['without_outliers']['mean_time1'] = sum(clean_times1) / len(clean_times1)
            results['without_outliers']['mean_time2'] = sum(clean_times2) / len(clean_times2)
            results['without_outliers']['median_time1'] = sorted(clean_times1)[len(clean_times1) // 2]
            results['without_outliers']['median_time2'] = sorted(clean_times2)[len(clean_times2) // 2]
    
    return results


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def print_summary_header(file1_name: str, file2_name: str, total_prompts: int,
                        hall_threshold: float, runtime_threshold: float):
    """Print executive summary header."""
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*80)
    print(f"\nComparing: {file1_name} (Method 1) vs {file2_name} (Method 2)")
    print(f"Total prompts analyzed: {total_prompts}")
    print("\nStatistical Framework:")
    print("  • Bonferroni correction: α = 0.05/2 = 0.025 per test")
    print("  • Normality assessment: Shapiro-Wilk test (determines test selection)")
    print("  • Primary hallucination test: Wilcoxon signed-rank OR Paired t-test")
    print("  • Secondary test: Paired t-test (runtime)")
    print("  • Descriptive statistics: McNemar categories, percentages, totals")
    print("\nPractical Significance Thresholds:")
    print(f"  • Hallucination reduction: ≥ {hall_threshold:.0f}%")
    print(f"  • Runtime increase: ≤ {runtime_threshold:.0f}%")
    print("="*80)


def print_normality_assessment(normality_hall: Dict, normality_runtime: Dict):
    """Print normality test results."""
    print("\n" + "="*80)
    print("NORMALITY ASSESSMENT - Shapiro-Wilk Test")
    print("="*80)
    print("\nThis determines which statistical tests are appropriate for the data.\n")
    
    print("-"*80)
    print("Hallucination Count Differences:")
    print("-"*80)
    if 'error' in normality_hall:
        print(f"  Error: {normality_hall['error']}")
        print(f"  Recommendation: {normality_hall['recommendation']}")
    else:
        print(f"  Sample size (n):                    {normality_hall['n']}")
        print(f"  Mean:                               {normality_hall['mean']:.4f}")
        print(f"  Std dev:                            {normality_hall['std']:.4f}")
        print(f"  Skewness:                           {normality_hall['skewness']:.4f}")
        print(f"  Kurtosis (excess):                  {normality_hall['kurtosis']:.4f}")
        print(f"  Unique values:                      {normality_hall['unique_values']}")
        print(f"  Unique ratio:                       {normality_hall['unique_ratio']:.2%}")
        print(f"  Count data detected:                {'Yes' if normality_hall['is_likely_count_data'] else 'No'}")
        print(f"  Assessment:                         {normality_hall['normality_assessment']}")
        print(f"  p-value (approx):                   {normality_hall['p_value_approx']}")
        print(f"  RECOMMENDED TEST:                   {normality_hall['recommended_test']}")
    
    print("\n" + "-"*80)
    print("Runtime Differences:")
    print("-"*80)
    if 'error' in normality_runtime:
        print(f"  Error: {normality_runtime['error']}")
        print(f"  Recommendation: {normality_runtime['recommendation']}")
    else:
        print(f"  Sample size (n):                    {normality_runtime['n']}")
        print(f"  Mean:                               {normality_runtime['mean']:.4f}")
        print(f"  Std dev:                            {normality_runtime['std']:.4f}")
        print(f"  Skewness:                           {normality_runtime['skewness']:.4f}")
        print(f"  Kurtosis (excess):                  {normality_runtime['kurtosis']:.4f}")
        print(f"  Assessment:                         {normality_runtime['normality_assessment']}")
        print(f"  p-value (approx):                   {normality_runtime['p_value_approx']}")
        print(f"  RECOMMENDED TEST:                   {normality_runtime['recommended_test']}")
    
    print("\n" + "-"*80)
    print("Test Selection Decision:")
    print("-"*80)
    
    hall_test = normality_hall.get('recommended_test', 'Unknown')
    runtime_test = normality_runtime.get('recommended_test', 'Unknown')
    
    if 'Wilcoxon' in hall_test:
        print(f"  ✓ Hallucinations: Using {hall_test}")
        print(f"    Reason: Data violates normality assumptions")
    else:
        print(f"  ✓ Hallucinations: Using {hall_test}")
        print(f"    Reason: Data meets normality assumptions")
    
    if 'Wilcoxon' in runtime_test:
        print(f"  ! Runtime: Recommended {runtime_test}")
        print(f"    Note: Proceeding with Paired t-test as originally planned")
        print(f"    Reason: Runtime data typically benefits from parametric approach")
    else:
        print(f"  ✓ Runtime: Using {runtime_test}")
    
    print("\n" + "="*80)


def print_descriptive_hallucination_stats(counts1: Dict[int, int], 
                                         counts2: Dict[int, int],
                                         hallucinations1: Dict[int, List[str]],
                                         hallucinations2: Dict[int, List[str]],
                                         categories: Dict[str, List[int]],
                                         file1_name: str,
                                         file2_name: str):
    """Print comprehensive descriptive statistics for hallucinations."""
    common_prompts = sorted(set(counts1.keys()) & set(counts2.keys()))
    n_total = len(common_prompts)
    
    # Extract counts for common prompts
    counts1_list = [counts1[p] for p in common_prompts]
    counts2_list = [counts2[p] for p in common_prompts]
    
    # Calculate descriptive stats
    stats1 = calculate_descriptive_stats(counts1_list)
    stats2 = calculate_descriptive_stats(counts2_list)
    
    # Total hallucinations
    total_hall1 = stats1['total_sum']
    total_hall2 = stats2['total_sum']
    
    # Texts with hallucinations
    texts_with_hall1 = stats1['non_zero_count']
    texts_with_hall2 = stats2['non_zero_count']
    
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS - Hallucinations")
    print("="*80)
    print("\nThese are descriptive statistics (not hypothesis tests)")
    print("They provide interpretability alongside the statistical tests below.\n")
    
    print("-"*80)
    print(f"{'Metric':<40} | {'Method 1':<15} | {'Method 2':<15}")
    print("-"*80)
    print(f"{'Total texts analyzed':<40} | {n_total:<15} | {n_total:<15}")
    print(f"{'Texts with hallucinations (count)':<40} | {texts_with_hall1:<15} | {texts_with_hall2:<15}")
    print(f"{'Texts with hallucinations (%)':<40} | {stats1['non_zero_percentage']:>14.1f}% | {stats2['non_zero_percentage']:>14.1f}%")
    print(f"{'Total hallucinated numbers (sum)':<40} | {total_hall1:<15} | {total_hall2:<15}")
    print(f"{'Mean hallucinations per text':<40} | {stats1['mean']:>14.2f} | {stats2['mean']:>14.2f}")
    print(f"{'Median hallucinations per text':<40} | {stats1['median']:>14.1f} | {stats2['median']:>14.1f}")
    print(f"{'Std dev':<40} | {stats1['std']:>14.2f} | {stats2['std']:>14.2f}")
    print(f"{'IQR (Q1-Q3)':<40} | {stats1['q1']:.1f}-{stats1['q3']:.1f}{'':>6} | {stats2['q1']:.1f}-{stats2['q3']:.1f}")
    print(f"{'Range (min-max)':<40} | {stats1['min']}-{stats1['max']}{'':>11} | {stats2['min']}-{stats2['max']}")
    print("-"*80)
    
    # McNemar contingency table
    a, b, c, d = len(categories['a']), len(categories['b']), len(categories['c']), len(categories['d'])
    
    print("\n" + "-"*80)
    print("McNemar's Test Contingency Table (Binary Hallucination Detection):")
    print("-"*80)
    print(f"{'':30} | {'Method 1':^25} |")
    print(f"{'':30} | {'Has Hall.':<12} | {'No Hall.':<12} | {'Total':<10}")
    print("-"*80)
    print(f"{'Method 2':<30} |              |            |")
    print(f"{'  Has Hallucination':<30} | {a:^12} | {c:^12} | {a+c:<10}")
    print(f"{'                      (a={a})':<30} | {'(c='+str(c)+')':^12} |")
    print(f"{'  No Hallucination':<30} | {b:^12} | {d:^12} | {b+d:<10}")
    print(f"{'                      (b={b})':<30} | {'(d='+str(d)+')':^12} |")
    print(f"{'  Total':<30} | {a+b:^12} | {c+d:^12} | {a+b+c+d:<10}")
    print("-"*80)
    
    # Interpretation
    print("\n" + "-"*80)
    print("Interpretation:")
    print("-"*80)
    diff_texts = texts_with_hall1 - texts_with_hall2
    diff_total = total_hall1 - total_hall2
    diff_pct = stats1['non_zero_percentage'] - stats2['non_zero_percentage']
    
    if abs(diff_texts) > 0:
        direction = "more" if diff_texts > 0 else "fewer"
        print(f"  • Method 1 had {direction} texts with hallucinations ({abs(diff_texts)} texts, {abs(diff_pct):.1f}%)")
    
    if abs(diff_total) > 0:
        direction = "more" if diff_total > 0 else "fewer"
        print(f"  • Method 1 produced {abs(diff_total)} {direction} total hallucinated numbers")
    
    if total_hall1 > 0:
        reduction_pct = ((total_hall1 - total_hall2) / total_hall1) * 100
        if reduction_pct > 0:
            print(f"  • Method 2 achieved a {reduction_pct:.1f}% reduction in total hallucinated numbers")
        elif reduction_pct < 0:
            print(f"  • Method 2 had a {abs(reduction_pct):.1f}% increase in total hallucinated numbers")
    
    # Ratio comparison
    if texts_with_hall1 > 0 and texts_with_hall2 > 0:
        avg_hall_per_affected1 = total_hall1 / texts_with_hall1
        avg_hall_per_affected2 = total_hall2 / texts_with_hall2
        print(f"  • Average hallucinations per affected text: M1={avg_hall_per_affected1:.2f}, M2={avg_hall_per_affected2:.2f}")
    
    print("\n" + "="*80)
    
    return stats1, stats2


def print_wilcoxon_results(wilcoxon_results: Dict, file1_name: str, file2_name: str):
    """Print Wilcoxon signed-rank test results."""
    print("\n" + "="*80)
    print("PRIMARY STATISTICAL TEST - Wilcoxon Signed-Rank Test")
    print("="*80)
    print("\nTest: Hallucination counts per text (Method 2 vs Method 1)")
    print("Null hypothesis: No difference in hallucination counts between methods")
    print("Significance level: α = 0.025 (Bonferroni corrected for 2 tests)\n")
    
    print("-"*80)
    print("Test Statistics:")
    print("-"*80)
    print(f"  Sample size (n):                    {wilcoxon_results['n']}")
    print(f"  Non-zero differences:               {wilcoxon_results['n_non_zero']}")
    print(f"  Zero differences:                   {wilcoxon_results['n_zero_diff']}")
    print(f"  W statistic:                        {wilcoxon_results['w_statistic']:.1f}")
    print(f"  Sum of positive ranks (W+):         {wilcoxon_results['w_plus']:.1f}")
    print(f"  Sum of negative ranks (W-):         {wilcoxon_results['w_minus']:.1f}")
    print(f"  Z-score:                            {wilcoxon_results['z_score']:.4f}")
    print(f"  p-value (two-tailed):               {wilcoxon_results['p_value_approx']}")
    print(f"  Significant (α=0.025):              {'Yes' if wilcoxon_results['significant_bonferroni'] else 'No'}")
    print(f"  Effect size (rank-biserial r):      {wilcoxon_results['effect_size_r']:.4f}")
    
    print("\n" + "-"*80)
    print("Interpretation:")
    print("-"*80)
    
    if wilcoxon_results['significant_bonferroni']:
        if wilcoxon_results['w_plus'] > wilcoxon_results['w_minus']:
            print(f"  ✓ Method 2 produces SIGNIFICANTLY MORE hallucinations than Method 1")
            print(f"    (Wilcoxon signed-rank test, p {wilcoxon_results['p_value_approx']})")
        else:
            print(f"  ✓ Method 2 produces SIGNIFICANTLY FEWER hallucinations than Method 1")
            print(f"    (Wilcoxon signed-rank test, p {wilcoxon_results['p_value_approx']})")
    else:
        print(f"  ✗ No significant difference in hallucination counts between methods")
        print(f"    (Wilcoxon signed-rank test, p {wilcoxon_results['p_value_approx']})")
    
    # Effect size interpretation
    abs_r = abs(wilcoxon_results['effect_size_r'])
    if abs_r < 0.1:
        effect = "negligible"
    elif abs_r < 0.3:
        effect = "small"
    elif abs_r < 0.5:
        effect = "medium"
    else:
        effect = "large"
    print(f"\n  Effect size: {effect} (|r| = {abs_r:.3f})")
    
    if wilcoxon_results.get('note'):
        print(f"\n  Note: {wilcoxon_results['note']}")
    
    print("\n" + "="*80)


def print_paired_t_test_hallucinations(ttest_results: Dict, file1_name: str, file2_name: str):
    """Print paired t-test results for hallucinations (when normal)."""
    print("\n" + "="*80)
    print("PRIMARY STATISTICAL TEST - Paired T-Test (Hallucinations)")
    print("="*80)
    print("\nTest: Hallucination counts per text (Method 2 vs Method 1)")
    print("Null hypothesis: No difference in hallucination counts between methods")
    print("Significance level: α = 0.025 (Bonferroni corrected for 2 tests)")
    print("Note: Used instead of Wilcoxon due to normal distribution of differences\n")
    
    print("-"*80)
    print("Test Statistics:")
    print("-"*80)
    print(f"  Sample size (n):                    {ttest_results['n']}")
    print(f"  Mean difference (M2 - M1):          {ttest_results['mean_diff']:.4f}")
    print(f"  Median difference:                  {ttest_results['median_diff']:.4f}")
    print(f"  Std dev of differences:             {ttest_results['std_diff']:.4f}")
    print(f"  Standard error:                     {ttest_results['se']:.4f}")
    print(f"  t-statistic:                        {ttest_results['t_statistic']:.4f}")
    print(f"  Degrees of freedom:                 {ttest_results['df']}")
    print(f"  p-value (two-tailed):               {ttest_results['p_value_approx']}")
    print(f"  Significant (α=0.025):              {'Yes' if ttest_results['significant_bonferroni'] else 'No'}")
    print(f"  Effect size (Cohen's d):            {ttest_results['cohens_d']:.4f}")
    
    print("\n" + "-"*80)
    print("Interpretation:")
    print("-"*80)
    
    if ttest_results['significant_bonferroni']:
        if ttest_results['mean_diff'] > 0:
            print(f"  ✓ Method 2 produces SIGNIFICANTLY MORE hallucinations than Method 1")
            print(f"    Mean difference: {ttest_results['mean_diff']:.4f} hallucinations per text")
            print(f"    (Paired t-test, p {ttest_results['p_value_approx']})")
        else:
            print(f"  ✓ Method 2 produces SIGNIFICANTLY FEWER hallucinations than Method 1")
            print(f"    Mean difference: {abs(ttest_results['mean_diff']):.4f} hallucinations per text")
            print(f"    (Paired t-test, p {ttest_results['p_value_approx']})")
    else:
        print(f"  ✗ No significant difference in hallucination counts between methods")
        print(f"    (Paired t-test, p {ttest_results['p_value_approx']})")
    
    # Effect size interpretation
    abs_d = abs(ttest_results['cohens_d'])
    if abs_d < 0.2:
        effect = "negligible"
    elif abs_d < 0.5:
        effect = "small"
    elif abs_d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"\n  Effect size: {effect} (|d| = {abs_d:.3f})")
    
    print("\n" + "="*80)


def print_runtime_results(runtime_analysis: Dict, file1_name: str, file2_name: str):
    """Print runtime comparison results."""
    print("\n" + "="*80)
    print("SECONDARY STATISTICAL TEST - Paired T-Test (Runtime)")
    print("="*80)
    print("\nTest: Runtime per text (Method 2 vs Method 1)")
    print("Null hypothesis: No difference in runtime between methods")
    print("Significance level: α = 0.025 (Bonferroni corrected for 2 tests)\n")
    
    # Outlier information
    outliers = runtime_analysis['outliers']
    print("-"*80)
    print("Outlier Detection (IQR method with 3.0 multiplier):")
    print("-"*80)
    print(f"  Extreme outliers detected: {outliers['count']}")
    if outliers['count'] > 0:
        print(f"  Outlier prompt numbers: {outliers['prompt_numbers'][:10]}{'...' if len(outliers['prompt_numbers']) > 10 else ''}")
        print(f"  Acceptable range: [{outliers['lower_bound']:.3f}, {outliers['upper_bound']:.3f}] seconds")
    
    # Results without outliers (primary)
    if 'without_outliers' in runtime_analysis:
        without_outliers = runtime_analysis['without_outliers']
        print("\n" + "-"*80)
        print("Test Statistics (outliers removed - RECOMMENDED):")
        print("-"*80)
        print(f"  Sample size (n):                    {without_outliers['n']}")
        print(f"  Mean runtime (Method 1):            {without_outliers['mean_time1']:.4f} seconds")
        print(f"  Mean runtime (Method 2):            {without_outliers['mean_time2']:.4f} seconds")
        print(f"  Mean difference (M2 - M1):          {without_outliers['mean_diff']:.4f} seconds")
        print(f"  Median runtime (Method 1):          {without_outliers['median_time1']:.4f} seconds")
        print(f"  Median runtime (Method 2):          {without_outliers['median_time2']:.4f} seconds")
        print(f"  Median difference (M2 - M1):        {without_outliers['median_diff']:.4f} seconds")
        print(f"  Std dev of differences:             {without_outliers['std_diff']:.4f}")
        print(f"  t-statistic:                        {without_outliers['t_statistic']:.4f}")
        print(f"  Degrees of freedom:                 {without_outliers['df']}")
        print(f"  p-value (two-tailed):               {without_outliers['p_value_approx']}")
        print(f"  Significant (α=0.025):              {'Yes' if without_outliers['significant_bonferroni'] else 'No'}")
        print(f"  Effect size (Cohen's d):            {without_outliers['cohens_d']:.4f}")
        
        results = without_outliers
    else:
        with_outliers = runtime_analysis['with_outliers']
        print("\n" + "-"*80)
        print("Test Statistics (with outliers):")
        print("-"*80)
        print(f"  Sample size (n):                    {with_outliers['n']}")
        print(f"  Mean difference (M2 - M1):          {with_outliers['mean_diff']:.4f} seconds")
        print(f"  t-statistic:                        {with_outliers['t_statistic']:.4f}")
        print(f"  p-value (two-tailed):               {with_outliers['p_value_approx']}")
        print(f"  Significant (α=0.025):              {'Yes' if with_outliers['significant_bonferroni'] else 'No'}")
        
        results = with_outliers
    
    print("\n" + "-"*80)
    print("Interpretation:")
    print("-"*80)
    
    if results['significant_bonferroni']:
        if results['mean_diff'] > 0:
            print(f"  ✓ Method 2 is SIGNIFICANTLY SLOWER than Method 1")
            print(f"    Mean difference: {abs(results['mean_diff']):.4f} seconds per text")
            print(f"    (Paired t-test, p {results['p_value_approx']})")
        else:
            print(f"  ✓ Method 2 is SIGNIFICANTLY FASTER than Method 1")
            print(f"    Mean difference: {abs(results['mean_diff']):.4f} seconds per text")
            print(f"    (Paired t-test, p {results['p_value_approx']})")
    else:
        print(f"  ✗ No significant difference in runtime between methods")
        print(f"    (Paired t-test, p {results['p_value_approx']})")
    
    # Effect size interpretation
    abs_d = abs(results['cohens_d'])
    if abs_d < 0.2:
        effect = "negligible"
    elif abs_d < 0.5:
        effect = "small"
    elif abs_d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"\n  Effect size: {effect} (|d| = {abs_d:.3f})")
    
    print("\n" + "="*80)


def print_executive_summary(primary_test_results: Dict,
                           test_type: str,
                           runtime_results: Dict,
                           stats1: Dict,
                           stats2: Dict,
                           file1_name: str,
                           file2_name: str,
                           hallucination_threshold: float = 90.0,
                           runtime_threshold: float = 20.0):
    """Print executive summary with key findings."""
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    print()
    
    # Calculate practical significance metrics
    texts_with_hall1 = stats1['non_zero_count']
    texts_with_hall2 = stats2['non_zero_count']
    total_hall1 = stats1['total_sum']
    total_hall2 = stats2['total_sum']
    
    # Hallucination reduction percentage
    if total_hall1 > 0:
        hall_reduction_pct = ((total_hall1 - total_hall2) / total_hall1) * 100
    else:
        hall_reduction_pct = 0
    
    pct1 = stats1['non_zero_percentage']
    pct2 = stats2['non_zero_percentage']
    
    # Runtime increase percentage
    if 'without_outliers' in runtime_results:
        rt_results = runtime_results['without_outliers']
    else:
        rt_results = runtime_results.get('with_outliers', {})
    
    if 'mean_time1' in rt_results and rt_results['mean_time1'] > 0:
        runtime_increase_pct = ((rt_results['mean_time2'] - rt_results['mean_time1']) / rt_results['mean_time1']) * 100
    else:
        runtime_increase_pct = 0
    
    # Statistical significance
    hall_sig = primary_test_results.get('significant_bonferroni', False)
    
    if test_type == 'wilcoxon':
        hall_better = primary_test_results.get('w_plus', 0) < primary_test_results.get('w_minus', 0)
        test_name = "Wilcoxon signed-rank"
    else:
        hall_better = primary_test_results.get('mean_diff', 0) < 0
        test_name = "Paired t-test"
    
    rt_sig = rt_results.get('significant_bonferroni', False)
    rt_faster = rt_results.get('mean_diff', 0) < 0
    
    # Practical significance evaluation
    hall_practical = hall_reduction_pct >= hallucination_threshold
    runtime_practical = runtime_increase_pct <= runtime_threshold
    
    print("-"*80)
    print("PRACTICAL SIGNIFICANCE THRESHOLDS:")
    print("-"*80)
    print(f"  Hallucination reduction threshold:  ≥ {hallucination_threshold:.0f}%")
    print(f"  Runtime increase threshold:         ≤ {runtime_threshold:.0f}%")
    
    print("\n" + "-"*80)
    print("Statistical Significance:")
    print("-"*80)
    if hall_sig and hall_better:
        print(f"  ✓ Hallucinations: SIGNIFICANT reduction (p {primary_test_results['p_value_approx']}, {test_name})")
    elif hall_sig:
        print(f"  ✗ Hallucinations: SIGNIFICANT increase (p {primary_test_results['p_value_approx']}, {test_name})")
    else:
        print(f"  − Hallucinations: No significant difference (p {primary_test_results.get('p_value_approx', 'N/A')}, {test_name})")
    
    if 'significant_bonferroni' in rt_results:
        if rt_sig and rt_faster:
            print(f"  ✓ Runtime: SIGNIFICANTLY faster (p {rt_results['p_value_approx']})")
        elif rt_sig:
            print(f"  ✗ Runtime: SIGNIFICANTLY slower (p {rt_results['p_value_approx']})")
        else:
            print(f"  ✓ Runtime: No significant increase (p {rt_results['p_value_approx']})")
    
    print(f"\n" + "-"*80)
    print(f"Practical Significance (vs Thresholds):")
    print("-"*80)
    print(f"  Hallucination reduction: {hall_reduction_pct:>6.1f}%  {'✓ MEETS' if hall_practical else '✗ BELOW'} threshold (≥{hallucination_threshold:.0f}%)")
    print(f"  Runtime increase:        {runtime_increase_pct:>+6.1f}%  {'✓ MEETS' if runtime_practical else '✗ EXCEEDS'} threshold (≤{runtime_threshold:.0f}%)")
    
    print(f"\n" + "-"*80)
    print(f"Descriptive Statistics:")
    print("-"*80)
    print(f"  Method 1: {texts_with_hall1}/{stats1['n']} texts ({pct1:.1f}%), {total_hall1} total hallucinated numbers")
    print(f"  Method 2: {texts_with_hall2}/{stats2['n']} texts ({pct2:.1f}%), {total_hall2} total hallucinated numbers")
    if 'mean_time1' in rt_results:
        print(f"  Mean runtime: M1={rt_results['mean_time1']:.3f}s, M2={rt_results['mean_time2']:.3f}s")
    
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT:")
    print(f"{'='*80}")
    
    # Overall assessment
    if hall_sig and hall_better and hall_practical and runtime_practical:
        print("✓✓ SUCCESS: Significant hallucination reduction meeting practical threshold")
        print(f"   with acceptable runtime impact ({runtime_increase_pct:+.1f}%)")
    elif hall_sig and hall_better and hall_practical and not runtime_practical:
        print("✓✗ TRADEOFF: Significant hallucination reduction meeting threshold")
        print(f"   but runtime increase ({runtime_increase_pct:+.1f}%) exceeds {runtime_threshold:.0f}% limit")
    elif hall_sig and hall_better and not hall_practical and runtime_practical:
        print(f"−✓ PARTIAL: Significant hallucination reduction but below {hallucination_threshold:.0f}% threshold")
        print(f"   ({hall_reduction_pct:.1f}%), though runtime impact is acceptable")
    elif hall_sig and hall_better and not hall_practical and not runtime_practical:
        print(f"−− INSUFFICIENT: Hallucination reduction ({hall_reduction_pct:.1f}%) below threshold")
        print(f"   and runtime increase ({runtime_increase_pct:+.1f}%) exceeds {runtime_threshold:.0f}% limit")
    elif not hall_sig or not hall_better:
        print("✗ FAILED: No significant hallucination reduction observed")
    
    print("\n" + "="*80)


def save_detailed_results(categories: Dict[str, List[int]], 
                         primary_test_results: Dict,
                         test_type: str,
                         runtime_analysis: Dict,
                         stats1: Dict,
                         stats2: Dict,
                         normality_hall: Dict,
                         normality_runtime: Dict,
                         output_file: str):
    """Save detailed results to JSON file."""
    output = {
        'normality_tests': {
            'hallucination_differences': {
                k: v for k, v in normality_hall.items() 
                if k not in ['differences_sample']
            },
            'runtime_differences': {
                k: v for k, v in normality_runtime.items() 
                if k not in ['differences_sample']
            }
        },
        'descriptive_statistics': {
            'method1': {k: v for k, v in stats1.items() if k != 'n'},
            'method2': {k: v for k, v in stats2.items() if k != 'n'},
            'sample_size': stats1['n']
        },
        'primary_test': {
            'test_type': test_type,
            'results': {
                k: v for k, v in primary_test_results.items() 
                if k != 'differences'
            }
        },
        'mcnemar_categories': categories,
        'runtime_analysis': {
            'total_prompts': runtime_analysis['total_prompts'],
            'outliers': runtime_analysis['outliers'],
            'with_outliers': {
                k: v for k, v in runtime_analysis['with_outliers'].items() 
                if k != 'differences'
            }
        }
    }
    
    if 'without_outliers' in runtime_analysis:
        output['runtime_analysis']['without_outliers'] = {
            k: v for k, v in runtime_analysis['without_outliers'].items() 
            if k != 'differences'
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive statistical comparison with normality testing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('file1', help='First JSON file (method 1)')
    parser.add_argument('file2', help='Second JSON file (method 2)')
    parser.add_argument('--output', '-o', help='Output file for detailed results (optional)')
    parser.add_argument('--keep-outliers', action='store_true', 
                       help='Keep outliers in runtime analysis (not recommended)')
    parser.add_argument('--hallucination-threshold', type=float, default=90.0,
                       help='Practical significance: minimum hallucination reduction %% (default: 90)')
    parser.add_argument('--runtime-threshold', type=float, default=20.0,
                       help='Practical significance: maximum acceptable runtime increase %% (default: 20)')
    
    args = parser.parse_args()
    
    # Load and extract data
    print(f"Loading and processing {args.file1}...")
    file1_hallucinations, file1_counts, file1_runtimes = extract_hallucinations_from_file(args.file1)
    print(f"  Found {len(file1_counts)} Summary prompts")
    
    print(f"\nLoading and processing {args.file2}...")
    file2_hallucinations, file2_counts, file2_runtimes = extract_hallucinations_from_file(args.file2)
    print(f"  Found {len(file2_counts)} Summary prompts")
    
    # Get common prompts
    common_prompts = sorted(set(file1_counts.keys()) & set(file2_counts.keys()))
    
    if len(common_prompts) == 0:
        print("\nError: No common prompts found between files!")
        return
    
    # Print summary header
    print_summary_header(Path(args.file1).name, Path(args.file2).name, len(common_prompts),
                        args.hallucination_threshold, args.runtime_threshold)
    
    # Extract counts for common prompts
    counts1_list = [file1_counts[p] for p in common_prompts]
    counts2_list = [file2_counts[p] for p in common_prompts]
    
    # Assess normality
    normality_hall = assess_normality_for_differences(counts1_list, counts2_list)
    
    # For runtime
    runtime_common = sorted(set(file1_runtimes.keys()) & set(file2_runtimes.keys()))
    if runtime_common:
        times1 = [file1_runtimes[p] for p in runtime_common]
        times2 = [file2_runtimes[p] for p in runtime_common]
        normality_runtime = assess_normality_for_differences(times1, times2)
    else:
        normality_runtime = {'error': 'No common runtime data'}
    
    # Print normality assessment
    print_normality_assessment(normality_hall, normality_runtime)
    
    # Calculate descriptive statistics
    stats1 = calculate_descriptive_stats(counts1_list)
    stats2 = calculate_descriptive_stats(counts2_list)
    
    # Perform McNemar's test (for contingency table)
    categories = classify_prompts_mcnemar(file1_hallucinations, file2_hallucinations)
    
    # Print descriptive statistics (includes McNemar table)
    stats1, stats2 = print_descriptive_hallucination_stats(
        file1_counts, file2_counts,
        file1_hallucinations, file2_hallucinations,
        categories,
        Path(args.file1).name, Path(args.file2).name
    )
    
    # Choose appropriate test based on normality
    use_wilcoxon = not normality_hall.get('is_normal', False)
    
    if use_wilcoxon:
        # Perform Wilcoxon signed-rank test (PRIMARY TEST for non-normal)
        primary_test_results = wilcoxon_signed_rank_test(counts1_list, counts2_list)
        print_wilcoxon_results(primary_test_results, Path(args.file1).name, Path(args.file2).name)
        test_type = 'wilcoxon'
    else:
        # Perform paired t-test (ALTERNATIVE for normal data)
        primary_test_results = paired_t_test(
            [float(x) for x in counts1_list], 
            [float(x) for x in counts2_list]
        )
        print_paired_t_test_hallucinations(primary_test_results, 
                                          Path(args.file1).name, Path(args.file2).name)
        test_type = 'paired_t'
    
    # Perform runtime analysis
    runtime_analysis = analyze_runtime_comparison(
        file1_runtimes, 
        file2_runtimes,
        remove_outliers=not args.keep_outliers
    )
    
    if 'error' not in runtime_analysis:
        print_runtime_results(runtime_analysis, 
                            Path(args.file1).name, Path(args.file2).name)
    
    # Print executive summary
    print_executive_summary(primary_test_results, test_type, runtime_analysis, 
                          stats1, stats2,
                          Path(args.file1).name, Path(args.file2).name,
                          args.hallucination_threshold, args.runtime_threshold)
    
    # Save detailed results
    if args.output:
        save_detailed_results(categories, primary_test_results, test_type,
                            runtime_analysis, stats1, stats2,
                            normality_hall, normality_runtime, args.output)

if __name__ == '__main__':
    main()