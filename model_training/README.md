# model_training

Contains everything needed to reproduce the hyperparameter sweep and inspect the results for the token-classification model trained on RAGTruth.

## Training scripts

Two scripts are present:

- **`train_tokenized_decoder_model.py`** — the script used in the thesis. It runs a WandB hyperparameter sweep (12 runs) over learning rate, batch size, weight decay, and warmup ratio. During training it uses `hf_ds["test"]` as the eval set for early stopping and best-model selection, which means the held-out test split is seen during model selection.

- **`train_tokenized_decoder_model_improved.py`** — an improved version that was **not used in the thesis**. It correctly carves out a separate validation split from the training data for early stopping, keeping the test set truly held-out. Use this script for future experiments.

## Subfolders

| Folder | Contents |
|---|---|
| `results/model_archives/` | Compressed checkpoints for all 12 sweep runs (~13 GB total), plus `SHA256SUMS.txt` for integrity verification |
| `results/wandb_export/` | Per-run metric history CSVs and a combined sweep overview, exported from WandB |
| `converting_CSV_results/` | Scripts and pre-converted CSVs that reshape the raw WandB exports into a format suitable for plotting |
| `analyze_tokenizer_differences_for_training/` | One-off analysis comparing the Ettin and Llama-3.1 tokenizers as used in the training pipeline |

## Downloading a model archive

The model archives are attached as release assets on GitHub (~13 GB combined). To download and unpack a single run:

```bash
wget https://github.com/yourname/yourrepo/releases/download/thesis-sweep-results-v1/sweep_run_jumping-sweep-1_2026-04-16-11-53.tar.gz
tar -xzf sweep_run_jumping-sweep-1_2026-04-16-11-53.tar.gz
```

Verify integrity:

```bash
sha256sum -c SHA256SUMS.txt
```
