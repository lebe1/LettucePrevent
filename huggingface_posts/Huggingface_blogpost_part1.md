# Real-Time Prevention of Factual Hallucinations in RAG via Custom Generation Loops

<div align="center">
  <img src="https://huggingface.co/lebe1/lettuceprevent-ettin-decoder-68m-en/resolve/main/assets/LettucePrevent.png" alt="LettucePrevent" style="width:40%; height:auto;">
</div>

**TL;DR:** 
- **LettucePrevent** integrates a token-level detector directly into the generation loop with a custom `LogitsProcessor`, outperforming state-of-the-art hallucination detection models (HDMs)  
- Our regex-verifiable number detector achieves a **>60% relative reduction in numeric hallucinations** across evaluated models with minimal latency overhead.
- Try it out yourself with the [Hugging Face's `custom_generate` module](https://huggingface.co/lebe1/lettuceprevent-generate)
---

Retrieval-Augmented Generation (RAG) systems frequently generate numerical or factual claims unsupported by the retrieved context. Current mitigation strategies rely on post-hoc detection, requiring generate-detect-regenerate cycles that increase computational cost. LettucePrevent introduces a preventive mechanism by integrating a token-level hallucination detector directly into the inference loop via a custom `LogitsProcessor`, penalizing unsupported tokens in the logit space prior to sampling.

🤗 [Custom Generate Module](https://huggingface.co/lebe1/lettuceprevent-generate) 🤗 [LettucePrevent 68M HDM](https://huggingface.co/lebe1/lettuceprevent-ettin-decoder-68m-en) 💻 [GitHub Repository](https://github.com/lebe1/LettucePrevent) 📜 [Thesis Publication](https://repositum.tuwien.at/handle/20.500.12708/229242)

## The Mechanism: Modifying the Autoregressive Loop

Standard autoregressive decoding samples subsequent tokens directly from the model's logit distribution. LettucePrevent modifies this process via a custom `LogitsProcessor`:

1. **Candidate Extraction:** At each decoding step, the top-$k$ candidate tokens are identified.
2. **Hallucination Scoring:** A Hallucination Detection Model (HDM) evaluates these candidates against the grounding context.
3. **Logit Penalization:** Candidates predicted to induce a hallucination are assigned a significant negative penalty (e.g., $-\infty$) within the logit distribution.
4. **Modified Sampling:** The model samples from the adjusted distribution, intrinsically suppressing unsupported content.

A `skip_threshold` parameter bypasses HDM evaluation when the top-candidate probability exceeds the threshold, preserving baseline throughput for high-confidence steps.

## The Detector Toolkit

The `lettuceprevent-generate` module provides three distinct detector architectures:

| Detector | Architecture | Mechanism | Overhead |
| :--- | :--- | :--- | :--- |
| **`number`** | Deterministic (Regex) | Rejects numeric strings absent from the context. | Negligible |
| **`lettuceprevent`** | 68M Causal Decoder | Neural token classifier identifying unsupported factual claims. | Low (CUDA) |
| **`lettucedetect`** | Encoder (ModernBERT) | Designed for post-hoc evaluation of complete sequences. | High |

### Architectural Rationale

Existing HDMs typically utilize bidirectional encoders, which exhibit a streaming prefix mismatch when deployed within a decoding loop as they are optimized for complete sequences. The 68M Ettin Decoder (`lebe1/lettuceprevent-ettin-decoder-68m-en`) utilizes a causal architecture. Processing input autoregressively ensures predictions at position *i* perfectly align with the probability distribution required for incomplete prefixes during streaming inference.

## Integration via `custom_generate`

### Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

context = "Revenue was 2400 million in 2021 and 3100 million in 2022."
question = "What is the percentage increase in revenue from 2021 to 2022?"

text = tok.apply_chat_template(
    [{"role": "user", "content": f"{context}\n{question}"}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tok(text, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    custom_generate="lebe1/lettuceprevent-generate",
    trust_remote_code=True,
    tokenizer=tok,
    input_text=context,
    detector_type="lettuceprevent",
    skip_threshold=0.9,
    max_new_tokens=300,
)

print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Parameter Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `tokenizer` | **required** | Generator tokenizer (used to decode candidates). |
| `input_text` | **required** | Grounding context validated against by the detector. |
| `detector_type` | `"lettuceprevent"` | `"number"`, `"lettuceprevent"`, or a `baseline-*` switch. |
| `skip_threshold` | `1.0` | Skip the HDM check when the top-token probability exceeds this value. `1.0` = always check. |
| `penalty_value` | `0.0` | Score assigned to hallucinated tokens (`float('-inf')` to hard-block). |
| `confidence_threshold` | `0.9` | Hallucination probability threshold for `lettuceprevent`. |
| `top_k_logits` | `10` | Candidate tokens scored per decoding step. |
| `last_k_tokens_to_consider` | `10` | Context window used by the detector. |
| `use_all_tokens` | `True` | If `False`, only the last `last_k_tokens_to_consider` tokens are considered. |
| `model_path` | detector default | Override the hallucination detector checkpoint. |
| `query` | `""` | Optional query string for certain detectors. |
| `debug_print` | `False` | Print per-step debugging information. |

## Experimental Results

Evaluated on the **RAGTruth** evaluation set (**942 numeric prompts**; **450 factual prompts**).

### 1. Number Detection

| Host Model | Baseline Hallucinations | Number Detector | Relative Reduction | Runtime Overhead |
|------------|------------------------:|----------------:|-------------------:|-----------------:|
| Qwen2.5 14B | 68 | 23 | **66.2%** | +0.53 s |
| Mistral 7B v0.2 | 157 | 22 | **86.0%** | +0.73 s |
| Llama 3 2B | 71 | 25 | **64.8%** | +0.62 s |

The deterministic detector achieves a relative reduction exceeding **60%** across all evaluated architectures. Logit-level interception effectively filters unsupported numerical values prior to token selection, adding only marginal latency (**+0.5–0.7 s**) to the inference process.

### 2. Factual Detection (LettucePrevent)

| Host Model | Baseline Hallucinated Spans | LettucePrevent | Relative Reduction |
|------------|----------------------------:|---------------:|-------------------:|
| Qwen2.5 14B | 1,808 | 1,741 | **−3.71%** |
| Mistral 7B v0.2 | 2,232 | 2,269 | **+1.66%** |
| Llama-2 7B | 2,837 | 2,082 | **−26.61%** |

The 68M decoder achieves substantial reductions on smaller parameter models (Llama-2 7B). Larger models operate closer to their factual ceiling at this evaluation scale, exhibiting minimal variance.

## Limitations

- **Recall vs. Precision:** HDMs are calibrated for high recall. False negatives permit hallucinations; false positives force suboptimal token selection.
- **Computational Overhead:** Invoking a 68M model at each decoding step increases latency, necessitating `skip_threshold` utilization in production.
- **Tokenizer Alignment:** The neural detector was trained using Llama-style tokenization boundaries. Divergent tokenization schemes may degrade alignment.
- **Language Constraints:** Optimized exclusively for English RAGTruth distributions.



## Citation

```bibtex
@mastersthesis{Beccard:2026,
  title  = {Real-time Prevention of Factual Hallucinations in Retrieval-Augmented Generation},
  author = {Leon Beccard},
  school = {Technische Universität Wien},
  year   = {2026},
  url    = {https://repositum.tuwien.at/handle/20.500.12708/229242}
}
```