"""
Generate figures for the "Detection of hallucinated numbers" section.

Usage:
    python plot_number_detection_figures.py \
        --raw-mistral baseline_mistral.json prevention_mistral.json \
        --raw-llama   baseline_llama.json   prevention_llama.json \
        --raw-qwen    baseline_qwen.json    prevention_qwen.json \
        --out graphics/number-detection/

"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from word2number import w2n


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

# Tableau muted palette: "Tableau 10 - Color Blind"-ish picks
COLOR_BASELINE   = "#5778A4"   # muted blue
COLOR_PREVENTION = "#E49444"   # muted orange
COLOR_MISTRAL    = "#5778A4"
COLOR_LLAMA      = "#85B6B2"
COLOR_QWEN       = "#E49444"
EDGECOLOR        = "#333333"

MODEL_DISPLAY = {
    "mistral": "Mistral-7B-Instruct-v0.2",
    "llama":   "Llama-2-7b-chat-hf",
    "qwen":    "Qwen2.5-14B-Instruct",
}
MODEL_COLORS = {
    "mistral": COLOR_MISTRAL,
    "llama":   COLOR_LLAMA,
    "qwen":    COLOR_QWEN,
}


def configure_style() -> None:
    """Apply thesis-friendly matplotlib rcParams."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.frameon":     False,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


# ----------------------------------------------------------------------------
# Number extraction (re-implemented here to avoid cross-script imports)
# ----------------------------------------------------------------------------

_NUMBER_WORD_PATTERN = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|and|[-])+\b",
    re.IGNORECASE,
)


def _extract_digits(text: str) -> List[str]:
    return re.findall(r"\b\d+\b", text)


def _extract_words(text: str) -> List[str]:
    out = []
    for m in _NUMBER_WORD_PATTERN.finditer(text):
        try:
            out.append(str(w2n.word_to_num(m.group().replace("-", " ").lower())))
        except ValueError:
            continue
    return out


def _hallucinations(item: dict) -> List[str]:
    p = set(_extract_digits(item.get("prompt", "")) + _extract_words(item.get("prompt", "")))
    a = _extract_digits(item.get("answer", "")) + _extract_words(item.get("answer", ""))
    return [n for n in a if n not in p]


def _load_paired_per_text(baseline_path: str, prevention_path: str) -> Dict[str, np.ndarray]:
    """Load per-text hallucination counts and runtimes for two paired runs."""
    with open(baseline_path) as f:
        base_data = json.load(f)
    with open(prevention_path) as f:
        prev_data = json.load(f)

    def index(items):
        out = {}
        for i, item in enumerate(items):
            if not isinstance(item, dict) or item.get("task_type") != "Summary":
                continue
            pn = item.get("prompt_number", i)
            out[pn] = item
        return out

    base_idx = index(base_data)
    prev_idx = index(prev_data)
    common = sorted(set(base_idx) & set(prev_idx))

    return {
        "hall_base": np.array([len(_hallucinations(base_idx[p])) for p in common]),
        "hall_prev": np.array([len(_hallucinations(prev_idx[p])) for p in common]),
        "rt_base":   np.array([base_idx[p].get("duration_seconds") for p in common], dtype=float),
        "rt_prev":   np.array([prev_idx[p].get("duration_seconds") for p in common], dtype=float),
    }


# ----------------------------------------------------------------------------
# Runtime difference histograms
# ----------------------------------------------------------------------------

def _runtime_diff_hist(ax, diffs: np.ndarray, color: str, label: str,
                       bins=None, alpha=0.7) -> None:
    if bins is None:
        # Common bins across panels: -2 to 6 seconds in 0.25 s steps
        bins = np.arange(-2.0, 6.01, 0.25)
    ax.hist(diffs, bins=bins, color=color, edgecolor=EDGECOLOR, linewidth=0.4,
            alpha=alpha, label=label)
    ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Runtime difference (prevention $-$ baseline) [s]")
    ax.set_ylabel("Number of texts")


def plot_runtime_difference_per_model(paired_by_model: Dict[str, dict],
                                      out_dir: Path) -> None:
    """Side-by-side panels, one per model."""
    keys = ["mistral", "llama", "qwen"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), sharey=True)
    bins = np.arange(-2.0, 6.01, 0.25)
    for ax, key in zip(axes, keys):
        diffs = paired_by_model[key]["rt_prev"] - paired_by_model[key]["rt_base"]
        _runtime_diff_hist(ax, diffs, MODEL_COLORS[key], MODEL_DISPLAY[key],
                           bins=bins, alpha=0.85)
        ax.set_title(MODEL_DISPLAY[key])
    fig.tight_layout()
    out_path = out_dir / "runtime_difference_panels.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


def plot_runtime_difference_overlay(paired_by_model: Dict[str, dict],
                                    out_dir: Path) -> None:
    """All three models overlaid on a single axis."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = np.arange(-2.0, 6.01, 0.25)
    for key in ["mistral", "llama", "qwen"]:
        diffs = paired_by_model[key]["rt_prev"] - paired_by_model[key]["rt_base"]
        ax.hist(diffs, bins=bins, color=MODEL_COLORS[key],
                edgecolor=EDGECOLOR, linewidth=0.3, alpha=0.55,
                label=MODEL_DISPLAY[key])
    ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Runtime difference (prevention $-$ baseline) [s]")
    ax.set_ylabel("Number of texts")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = out_dir / "runtime_difference_overlay.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render hallucination-count and runtime-difference histograms.",
    )
    parser.add_argument("--raw-mistral", nargs=2, required=True,
                        metavar=("BASELINE_JSON", "PREVENTION_JSON"),
                        help="Two raw run JSONs for Mistral (baseline + prevention)")
    parser.add_argument("--raw-llama", nargs=2, required=True,
                        metavar=("BASELINE_JSON", "PREVENTION_JSON"),
                        help="Two raw run JSONs for Llama")
    parser.add_argument("--raw-qwen", nargs=2, required=True,
                        metavar=("BASELINE_JSON", "PREVENTION_JSON"),
                        help="Two raw run JSONs for Qwen")
    parser.add_argument("--out", default="graphics/number-detection/",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    configure_style()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading paired per-text data ...")
    paired = {
        "mistral": _load_paired_per_text(*args.raw_mistral),
        "llama":   _load_paired_per_text(*args.raw_llama),
        "qwen":    _load_paired_per_text(*args.raw_qwen),
    }
    for key, p in paired.items():
        print(f"  {key}: n = {len(p['hall_base'])}  "
              f"baseline total hall = {int(p['hall_base'].sum())}  "
              f"prevention total hall = {int(p['hall_prev'].sum())}")

    print("\nRendering runtime-difference figure (panels)...")
    plot_runtime_difference_per_model(paired, out_dir)

    print("\nRendering runtime-difference figure (overlay)...")
    plot_runtime_difference_overlay(paired, out_dir)

    print(f"\nAll figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()