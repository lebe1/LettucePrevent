# LettucePrevent

<div align="center">
  <img src="./huggingface_posts/visualizations/LettucePrevent.png" alt="Alt Text" style="width:40%; height:auto;">
</div>


Real-time prevention of factual hallucinations in Retrieval-Augmented Generation (RAG) by hooking a token-level hallucination detection model (HDM) into the generator's logits during decoding.

At each decoding step, the candidate top-k tokens are scored by an HDM (e.g. [LettucePrevent](https://huggingface.co/lebe1/lettuceprevent-ettin-decoder-68m-en) or [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)). Tokens predicted to introduce a hallucination are penalized before the next token is sampled, so unsupported content is suppressed *before* it is generated.

This repository contains the experimental code for the master thesis [*"Real-time Prevention of Factual Hallucinations in Retrieval-Augmented Generation"*](https://repositum.tuwien.at/handle/20.500.12708/229242)



## Mechanism visually explained

<img src="./huggingface_posts/visualizations/NumberLogitsProcessor.gif" alt="Alt Text" style="width:70%; height:auto;">

Scroll to the bottom to see each slide of this GIF.

## Repository layout

```
LettucePrevent/
├── main.py               # Experiment entry point: single runs + RQ1/RQ2 W&B sweeps
├── detectors/            # HDM wrappers (LettucePrevent, LettuceDetect, number detector), detector factory, dataset loading
├── logits_processors/    # HallucinationLogitsProcessor — hooks the detector into the generator's decoding loop (the core mechanism)
├── preprocess/           # Extracts the unique RAGTruth summary prompts consumed by the number detector (see Dataset preparation)
├── analysis/             # Post-hoc evaluation of run outputs + raw thesis result data
├── model_selection/      # Benchmark of candidate HDMs (threshold sweep + head-to-head comparison)
├── model_training/       # Training of the LettucePrevent HDM on RAGTruth (W&B hyperparameter sweep)
├── slurm_scripts/        # sbatch submit scripts for --rq1 / --rq2 on an HPC cluster (Slurm)
├── playground/           # Exploratory one-off scripts (beam search, tokenizer experiments) — not part of the pipeline
├── huggingface_posts/    # Hugging Face blog post draft + the visualizations embedded in this README
├── data/                 # Inputs (RAGTruth prompts) and run outputs (generations, hallucinations, stats)
└── requirements.txt
```

Three subprojects ship their own README with full instructions and results:

- [`analysis/README.md`](analysis/README.md) — evaluation pipelines for the number and factual experiments.
- [`model_selection/README.md`](model_selection/README.md) — how the three candidate HDMs were compared and why `lettuceprevent` was selected.
- [`model_training/README.md`](model_training/README.md) — how the LettucePrevent HDM was trained, including how to download the model archives from the GitHub release.

## Requirements

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU. Tested with Python 3.10+ and PyTorch with CUDA.
The default generator model is `meta-llama/Llama-3.3-70B-Instruct`, which needs A100-class hardware — pass `--generator-model` (e.g. `mistralai/Mistral-7B-Instruct-v0.2`) to run on a smaller GPU.

### 2. Hugging Face login

Some models (e.g. `meta-llama/Llama-3.1-8B`) are gated and the dataset is pulled via 🤗 `datasets`, so a Hugging Face account and CLI login are required:

1. Create an account at <https://huggingface.co/join> and request access to the gated Llama models on their model pages.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Log in once on the machine running the experiments (the `hf` CLI is already installed via `requirements.txt`):
```bash
   hf auth login   # paste your token when prompted
```

After this, `wandb/RAGTruth-processed` and all model weights are downloaded automatically — no manual dataset download needed.

### 3. Weights & Biases

Single runs can skip W&B entirely with `--no-wandb`. The sweep modes (`--rq1` / `--rq2`) always log to W&B, so for those log in once (`wandb` is already installed via `requirements.txt`):

```bash
wandb login   # paste your API key from https://wandb.ai/authorize
```

Then update `WANDB_ENTITY` (and optionally `WANDB_PROJECT_RQ1` / `WANDB_PROJECT_RQ2`) at the top of `main.py` to point to your own workspace, or pass `--entity` on the command line.

### 4. RAGTruth summary prompts — only for the number detector

The `number` and `baseline-run-numbers` detector types read local RAGTruth summary prompts from `./data/ragtruth_unique_summary_prompts.json`. To reconstruct, how this file was created, we explain it in the [Dataset preparation](#dataset-preparation) steps. All other detector types pull their dataset automatically from Hugging Face — no preparation needed.

### 5. Verify the setup

Each of the following commands runs a minimal end-to-end experiment (one prompt per task type, `--skip-threshold 1.0` = detector consulted at every step) and writes its outputs to `./data/`:

```bash
# LettucePrevent HDM — the detector used in both RQ1 and RQ2
python main.py --detector-type lettuceprevent --n-per-task 1 --skip-threshold 1.0 --no-wandb

# LettuceDetect HDM
python main.py --detector-type lettucedetect --n-per-task 1 --skip-threshold 1.0 --no-wandb

# Baseline (no logits processor) — the RQ1 control arm
python main.py --detector-type baseline-run-facts --n-per-task 1 --skip-threshold 1.0 --no-wandb

# Number detector — requires step 4 (local RAGTruth summary prompts)
python main.py --detector-type number --n-per-task 1 --skip-threshold 1.0 --no-wandb

# Minimal RQ2 sweep: creates a W&B sweep and runs a single cell — requires step 3
python main.py --rq2 --count 1 --output-prefix smoke
```

If these finish without errors, the full experiments described under [Usage](#usage) will run as well.

## Dataset preparation

In the very beginning the experiments were executed with the numbers detector only, where the experiments use the [RAGTruth](https://github.com/ParticleMedia/RAGTruth) summary prompts.
To reconstruct the creation of `./data/ragtruth_unique_summary_prompts.json`,, which is the file consumed by `main.py` (`LOCAL_SUMMARY_FILE`), the two steps are necessary.

1. Download `ragtruth_data.json` from the official repository:
   <https://github.com/ParticleMedia/RAGTruth/tree/main/dataset>
   and place it under `./data/ragtruth_data.json`.

2. Extract the unique summary prompts by running the preprocessing script from inside the `preprocess/` directory:

```bash
   cd preprocess
   python extract_unique_summary_prompts.py
```


## Usage

### Single run

```bash
python main.py \
  --generator-model meta-llama/Llama-3.3-70B-Instruct \
  --detector-type lettuceprevent \
  --skip-threshold 0.99 \
  --n-per-task 20
```

Disable W&B with `--no-wandb`. Outputs (generations, post-eval hallucinations, stats) are written to `./data/`.

Currently there are three experiments available:

1. DETECTOR_TYPE `'number'`
   - `NumberLogitsProcessor()` tries to reject all numbers that are not mentioned in the input text
2. DETECTOR_TYPE `'lettucedetect'` / `'lettuceprevent'` / `'tinylettuce'`
   - Token-level HDM rejects candidate tokens with a high hallucination score during decoding
3. DETECTOR_TYPE `'baseline-run-numbers'` / `'baseline-run-facts'`
   - Unmodified generation, used as baseline to compare experiment runs

### Configuration Parameters

All parameters are set at the top of `main.py` under the *Top-of-file configuration* section.

| Parameter | Default | Description |
| --- | --- | --- |
| `GENERATOR_MODELS` | `["meta-llama/Llama-3.3-70B-Instruct"]` | List of generator LLMs to sweep over |
| `DETECTOR_TYPES_SWEEPED` | `["lettuceprevent", "baseline-run-facts"]` | Detectors compared in the RQ1 sweep |
| `SKIP_THRESHOLDS` | `[0.8, 0.9, 0.99, 1.0]` | RQ2 grid of skip thresholds (runtime/accuracy trade-off) |
| `DETECTOR_TYPE_RQ2` | `"lettuceprevent"` | Detector fixed during the RQ2 skip-threshold sweep |
| `N_PER_TASK` | `20` | Prompts per task type during RQ2 (RQ1 uses the full set) |
| `DETECTORS_BEST_CONFIDENCE_THRESHOLDS` | `{tinylettuce: 0.5, lettucedetect: 0.6, lettuceprevent: 0.8}` | Hallucination-confidence threshold per detector |
| `MODELS_BEST_SKIP_THRESHOLDS` | `{Mistral-7B: 0.8, Llama-2-7B: 1.0, Qwen2.5-14B: 0.99}` | Best skip threshold per generator (used in RQ1) |
| `LETTUCEDETECT_MODEL_PATH` | `KRLabsOrg/lettucedect-base-modernbert-en-v1` | HF path of the LettuceDetect HDM |
| `LETTUCEPREVENT_MODEL_PATH` | `lebe1/lettuceprevent-ettin-decoder-68m-en` | HF path of the LettucePrevent HDM |
| `NUM_BEAMS` | `1` | `1` = greedy decoding (≈ 3–4× faster than 4-beam) |
| `GENERATION_CONFIG_KWARGS.max_new_tokens` | `300` | Max tokens generated per prompt |
| `GENERATION_CONFIG_KWARGS.min_length` | `150` | Min generation length |
| `LAST_K_TOKENS_TO_CONSIDER` | `10` | Recent-token context window (ignored if `USE_ALL_TOKENS=True`) |
| `TOP_K_LOGITS` | `10` | Number of top candidate tokens scored per generation step |
| `PENALTY_VALUE` | `0` | Score assigned to penalised tokens (`float('-inf')` to hard-block) |
| `USE_ALL_TOKENS` | `True` | Use the full generated prefix as context instead of only the last `LAST_K_TOKENS_TO_CONSIDER` |
| `SEED` | `42` | Random seed for deterministic runs |
| `SYSTEM_PROMPT` | *(see main.py)* | System message prepended to every generator prompt |
| `WANDB_ENTITY` / `WANDB_PROJECT_RQ1` / `WANDB_PROJECT_RQ2` | *(see main.py)* | W&B targets — update to your own workspace |

## Experiment Results on Number Detector

All experiments of the number detector have been executed on a NVIDIA A40 on the 942 unique summary prompts of the RAGTruth dataset.

### Total Hallucinations
Total number of hallucinations excluding written numbers (twenty-five), which were in the scope of the prevention mechanism.

| Model                    | Plain run | NumberDetector run |
| ------------------------ | --------- | ------------------ |
| Qwen2.5 14B Instruct     | 68        | 23                 |
| Mistral 7B Instruct v0.2 | 157       | 22                 |
| Llama 7 2B               | 71        | 25                 |


### Average runtime per generated answer
Total runtime divided by 942

| Model                | Plain run [s] | NumberDetector run [s] |
| -------------------- | ------------- | ---------------------- |
| Qwen2.5 14B Instruct | 10.03         | 10.55                  |
| Mistral 7B Instruct  | 8.09          | 8.82                   |
| Llama 7 2B           | 10.70         | 11.31                  |

## Experiment Results on Factual Detector

Hallucinated spans per text on the full evaluation set of 450 prompts, with each model paired to its tuned operating point (`MODELS_BEST_SKIP_THRESHOLDS` × `lettuceprevent` detector). All experiments have been executed on the NVIDIA A100s.


| Model                    | Plain run | LettucePrevent run | Relative change |
| ------------------------ | --------- | ------------------ | --------------- |
| Qwen2.5 14B Instruct     | 1 808     | 1 741              | −3.71 %         |
| Mistral 7B Instruct v0.2 | 2 232     | 2 269              | +1.66 %         |
| Llama 7 2B               | 2 837     | 2 082              | −26.61 %        |

The strongest reduction is observed on Llama-2-7B, while the other two models show essentially no net change at its tuned operating point.

For more experiment results, see [`analysis/README.md`](analysis/README.md) (evaluation details behind the tables above), [`model_selection/README.md`](model_selection/README.md) (HDM benchmark results) and [`model_training/README.md`](model_training/README.md) (training sweep results).

### W&B sweeps (RQ1 / RQ2)

Originally, the thesis is structured around two research questions, each backed by a W&B grid sweep. 
First, we wanted to evaluate to what extent we can reduce hallucinations with this approach and second what is the best trade-off in regards of runtime.
Both sweeps log to the entity/project defined at the top of `main.py` (`WANDB_ENTITY`, `WANDB_PROJECT_RQ1`, `WANDB_PROJECT_RQ2`).

**RQ1 — Hallucination reduction vs. baseline (full test set).**
Grid: `GENERATOR_MODELS × DETECTOR_TYPES_SWEEPED`. The skip threshold is *not* swept; it is resolved per generator model from `MODELS_BEST_SKIP_THRESHOLDS` (fallback `1.0` = no skipping). `n_per_task=None`, i.e. all available prompts.

```bash
python main.py --rq1
```

**RQ2 — Runtime / accuracy trade-off across skip thresholds.**
Grid: `GENERATOR_MODELS × SKIP_THRESHOLDS`. Detector is fixed to `DETECTOR_TYPE_RQ2` (`lettuceprevent`) so that the skip threshold is the only varying dimension besides the model. `n_per_task=N_PER_TASK` (small, for fast turnaround).

```bash
python main.py --rq2
```

**Sweep mechanics.**
- `--rq1` / `--rq2` create a new sweep on W&B and immediately launch an agent against it. The agent count defaults to the full grid size (`|models| × |detectors|` for RQ1, `|models| × |thresholds|` for RQ2).
- To attach additional agents to an existing sweep (e.g. across multiple GPUs or nodes), pass `--sweep-id <id>`. This skips sweep creation and just spawns an agent.
- Cap the number of runs per agent with `--count N`.
- Each sweep run resolves its config inside `sweep_fn_rq1` / `sweep_fn_rq2`, which delegates to `run_one_cell` with `use_wandb=False` (the outer agent run is already open). The cell logs per-prompt metrics, a results summary, and uploads `generations.json`, `hallucinations.json`, and `stats.txt` as a W&B artifact.


## License

MIT — see [LICENSE](LICENSE).

## Slide deck

<img src="./huggingface_posts/visualizations/1.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./huggingface_posts/visualizations/2.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./huggingface_posts/visualizations/3.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./huggingface_posts/visualizations/4.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./huggingface_posts/visualizations/5.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./huggingface_posts/visualizations/6.jpg" alt="Alt Text" style="width:100%; height:auto;">

Note: This slide deck was created to illustrate the whole process in a simplified way. Therefore, it states probability instead of logits score (where the magic actually happens in the transformers library) and it also uses bad probability sample figures being bigger than one when summed up.