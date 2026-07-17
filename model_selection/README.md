# `Model selection`

This experiment evaluates the best hallucination detection model (HDM) for real-time prevention by sweeping confidence thresholds, then running a final head-to-head comparison at each model's best threshold. 

Models compared:

- `KRLabsOrg/tinylettuce-ettin-68m-en` (encoder)
- `KRLabsOrg/lettucedect-base-modernbert-en-v1` (encoder)
- `lebe1/lettuceprevent-ettin-decoder-68m-en` (decoder)

All three models are evaluated on the same prefixes, produced by simulating token-by-token generation with the `meta-llama/Llama-3.1-8B` tokenizer on the `wandb/RAGTruth-processed` test split. Samples are unique (context, query) prompts per task type in dataset order, with one LLM answer assigned per prompt in round-robin fashion over the six RAGTruth generator LLMs — the same selection policy in both steps, so the sweep's subset is a strict subset of the final evaluation set.

## Layout

```
model_selection/
├── threshold_experiment.py   # step 1: per-model threshold sweep (W&B grid sweep)
├── model_selection.py        # step 2: final 3-way comparison
├── results/
│   ├── rq3-unified-sweep-results/       # step 1 outputs: per-run JSONs + sweep summary
│   └── rq3-model-selection-results/     # step 2 outputs: CSVs + JSONs
├── slurm/                    # sbatch submit scripts + job logs (.txt)
└── README.md
```

## Pipeline

### Step 1 — `threshold_experiment.py`

W&B grid sweep over `MODELS_TO_EVALUATE × SWEEP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]` on **50 unique (context, query) prompts per task type** (150 samples per cell). Only `detector.predict(...)` is timed, so the runtime metric is fair across encoders and the decoder.

```bash
python threshold_experiment.py --entity <wandb-entity> --project <wandb-project>
```

Further flags: `--sweep-id` (attach to an existing sweep), `--count`, `--seed`, `--prompts-per-task`, `--output-prefix`, `--create-only`.

Outputs one JSON per (model, threshold), a `*_summary.json` with the best threshold per model, and a final printed summary table.

### Step 2 — `model_selection.py`

Final benchmark on the **first 150 unique (context, query) pairs per task type (450 samples total)**, each model evaluated at its own best threshold from step 1. Set those thresholds manually in the `MODELS_TO_EVALUATE` list at the top of the script (`confidence_threshold` field) before running:

```bash
python model_selection.py
```

Outputs (in the working dir, then moved to `results/`):

- `{prefix}_{model}_per_step.csv` — token-level details
- `{prefix}_{model}_full_results.json` — full nested results per model
- `{prefix}_aggregate.csv` — model-level comparison
- `{prefix}_per_sample_comparison.json` — sample-centric side-by-side

### SLURM

`slurm/` contains sbatch scripts for both steps (`run_general_threshold_sweep.sh` for step 1, `run_model_selection.sh` for step 2) plus the corresponding job logs. Paths (repo, venv, results dir) are hardcoded/env-overridable at the top of each script — adjust them to your cluster before submitting.

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt   # requirements.txt lives at the repo root
```

Requires a CUDA-capable GPU. Tested with Python 3.10+ and PyTorch with CUDA.

**Hugging Face.** Some models (e.g. `meta-llama/Llama-3.1-8B`) are gated and the dataset is pulled via 🤗 `datasets`, so a Hugging Face account and CLI login are required:

1. Create an account at <https://huggingface.co/join> and request access to the gated Llama models on their model pages.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Log in once on the machine running the experiments:
```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login   # paste your token when prompted
```

After this, `wandb/RAGTruth-processed` and all model weights are downloaded automatically — no manual dataset download needed.

**Weights & Biases.** Both scripts log to W&B. Log in once:

```bash
pip install wandb
wandb login   # paste your API key from https://wandb.ai/authorize
```

Then point the scripts to your own W&B workspace:

- `threshold_experiment.py`: pass `--entity` / `--project` (or edit `WANDB_ENTITY`, `DEFAULT_SWEEP_PROJECT`)
- `model_selection.py`: edit `WANDB_ENTITY`, `WANDB_PROJECT` at the top of the script

## Experiment Results

All token-level metrics report performance on detecting the **hallucinated class (c1 — class 1)**, i.e. the positive class in the binary token classifier. *Class 0* would correspond to faithful, evidence-supported tokens. 
Micro-averaged F1 (`F1mic`) is reported across both classes and is dominated by the much more frequent class 0, so it stays close to ~0.95 even when c1 performance varies widely. 
The c1 metrics are therefore the more informative ones for hallucination detection.

### Step 1 — Threshold sweep

For each model, the threshold maximising **F1 on the hallucinated class (F1c1)** is selected as its operating point. 
The corresponding row per model is highlighted in bold. Latency is the mean per-step `detector.predict(...)` time in milliseconds.


| Model | τ | F1c1 | Recc1 | Precc1 | F1mic | Lat. (ms) |
|---|---|---|---|---|---|---|
| tinylettuce-ettin-68m-en | **0.5** | **0.3010** | **0.2168** | 0.4921 | 0.9592 | **24.96** |
| | 0.6 | 0.2909 | 0.2050 | 0.5012 | 0.9595 | 25.04 |
| | 0.7 | 0.2814 | 0.1921 | 0.5257 | 0.9602 | 25.12 |
| | 0.8 | 0.2500 | 0.1624 | 0.5430 | 0.9605 | 25.63 |
| | 0.9 | 0.2152 | 0.1307 | 0.6083 | 0.9614 | 25.99 |
| lettucedetect-base-modernbert-en-v1 | 0.5 | 0.3597 | 0.3307 | 0.3943 | 0.9523 | 27.57 |
| | **0.6** | **0.3598** | 0.3168 | 0.4161 | 0.9543 | 27.53 |
| | 0.7 | 0.3547 | 0.2990 | 0.4358 | 0.9559 | 27.41 |
| | 0.8 | 0.3497 | 0.2792 | 0.4677 | 0.9579 | 27.75 |
| | 0.9 | 0.3445 | 0.2505 | 0.5512 | 0.9614 | 28.33 |
| lettuceprevent-ettin-decoder-68m-en | 0.5 | 0.3439 | 0.5673 | 0.2468 | 0.9123 | 20.93 |
| | 0.6 | 0.3694 | 0.5218 | 0.2859 | 0.9278 | 20.94 |
| | 0.7 | 0.3867 | 0.4663 | 0.3303 | 0.9400 | 21.32 |
| | **0.8** | **0.3974** | 0.3980 | 0.3968 | 0.9511 | 21.05 |
| | 0.9 | 0.2730 | 0.1911 | 0.4777 | 0.9587 | 20.98 |


The two encoder models behave as expected: raising τ trades recall on c1 for precision on c1 while leaving F1c1 fairly flat. 
The decoder model `lettuceprevent` shows a F1c1 peak at τ = 0.8 while also producing the lowest per-step latency.

### Step 2 — Final head-to-head (450 samples)

Each model is rerun on the full 450-sample evaluation set at its tuned τ from step 1. `Runtime` is the total wall-clock time spent inside `detector.predict(...)` across all samples.

| Model | τ | F1c1 | Recc1 | Precc1 | F1mic | Runtime (s) |
|---|---|---|---|---|---|---|
| tinylettuce | 0.5 | 0.2856 | 0.2243 | 0.3930 | 0.9587 | 1411.75 |
| lettucedetect | 0.6 | 0.3556 | 0.3427 | 0.3695 | 0.9543 | 1989.76 |
| lettuceprevent | 0.8 | **0.3886** | **0.4410** | 0.3473 | 0.9489 | **1247.89** |

The ranking from the threshold sweep holds on the larger evaluation set: `lettuceprevent` wins on both quality (highest F1c1 and recall on hallucinated tokens) and speed (lowest total runtime). 
This makes it the model selected for downstream real-time prevention. 
`lettucedetect-base` is the strongest encoder on c1 quality but ~60 % slower than `lettuceprevent`.