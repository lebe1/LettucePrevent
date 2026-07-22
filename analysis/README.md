# analysis

Post-hoc evaluation of the experiment runs produced by `main.py`, plus the raw result data used in the thesis. Nothing here runs during generation - these scripts consume the `*_generations.json` / `*_hallucinations.json` / `*_stats.txt` outputs after the fact.

## `number_detection/`

Evaluation of the `number` detector vs. its baseline on the 942 unique RAGTruth summary prompts.

- `extract_number_hallucinations.py` — extracts digit sequences and written number words (via `word2number`) from prompts and answers.
- `evaluate_number_hallucinations.py` — end-to-end evaluation: a number in the answer counts as hallucinated if it does not appear in the prompt; runs are paired on `prompt_number` and compared statistically (symmetric and digit-only extraction modes).
- `create_runtime_histograms.py` — generates the runtime figures for the number-detection chapter.
- `results_{llama,mistral,qwen}.json` — extracted hallucination results per generator model.

## `factual_detection/`

Evaluation of the factual (`lettuceprevent`) experiments.

- `post_eval.py` — re-runs `KRLabsOrg/lettucedect-base-modernbert-en-v1` over the generated answers to count remaining hallucinated spans, bucketed by detection confidence.
- `extract_factual_hallucinations.py` — RQ1 + RQ2 statistics on the 450 paired prompts (150 Summary + 150 Data2Txt + 150 QA): paired tests on hallucinated span counts (≥ 70 % confidence) and on runtime, with Bonferroni correction across the study.
- `factual_results_{llama,mistral,qwen}.json` — extracted results per generator model.
