# Hallucination Detection Benchmark — Left-Encoder Evaluation

Evaluates hallucination detection models on their ability to detect hallucinations in incrementally generated answers, simulating token-by-token LLM generation.

## Requirements

```bash
pip install lettucedetect spacy transformers torch --break-system-packages
python -m spacy download en_core_web_sm
```

## Usage

### Base mode (spaCy tokenizer splitting sentences by every word)

```bash
python hallucination_benchmark.py \
    --dataset ragtruth_data.json \
    --max-samples 3 \
    --trigger every_token \
    --output-prefix results/base_run
```

### With LLM tokenizer (Mistral)

```bash
python hallucination_benchmark.py \
    --dataset ragtruth_data.json \
    --max-samples 5 \
    --trigger every_token \
    --llm-tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
    --output-prefix results/mistral_run
```

### Other LLM tokenizers

```bash
# Qwen
--llm-tokenizer Qwen/Qwen2.5-14B-Instruct

# Llama
--llm-tokenizer meta-llama/Llama-2-7b-chat-hf
```

### Custom confidence threshold

```bash
python hallucination_benchmark.py \
    --dataset ragtruth_data.json \
    --confidence-threshold 0.85 \
    --output-prefix results/threshold85
```

## Output Files

Each run produces three files:

| File | Description |
|------|-------------|
| `{prefix}_per_step.csv` | Token-by-token results with cumulative metrics |
| `{prefix}_aggregate.csv` | Single-row summary (P, R, F1, runtime) |
| `{prefix}_full_results.json` | Complete results including all token steps |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    run_benchmark()                      │
│  - loads dataset                                        │
│  - initializes tokenizer + detector                     │
│  - iterates samples                                     │
│  - exports results                                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 EvaluationEngine                        │
│  - tokenizes answer                                     │
│  - iterates tokens with trigger strategy                │
│  - builds prefix at each triggered step                 │
│  - calls detector.predict() on prefix                   │
│  - maps predictions to tokens                           │
│  - carries forward predictions for skipped steps        │
│  - computes cumulative + aggregate metrics              │
└──────────┬─────────────────┬────────────────────────────┘
           │                 │
           ▼                 ▼
┌──────────────────┐ ┌──────────────────────────────┐
│ HallucinationDe- │ │ TokenizerBase                │
│ tectorBase       │ │  ├─ SpacyTokenizer (default) │
│  └─ LettuceWrap  │ │  └─ LLMTokenizerWrapper      │
└──────────────────┘ │     (+ OffsetMapper)         │
                     └──────────────────────────────┘
```

## Metrics

**Token-level evaluation:**
- A token is **ground-truth hallucinated** if its character range overlaps any label span
- A token is **predicted hallucinated** if the detector's confidence ≥ threshold (default 0.90)

**Per-step:** cumulative precision, recall, F1 up to each token

**Aggregate:** global precision, recall, F1 across all tokens in all samples
