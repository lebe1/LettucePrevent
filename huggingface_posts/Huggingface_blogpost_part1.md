
# LettucePrevent: Real-Time Hallucination Prevention on Number Hallucinations (Part 1)

Me and @Adam present LettucePrevent, a novel approach to hallucination prevention that stops errors **before they happen**. By integrating directly into the text generation process of the transformers library through custom LogitsProcessors, we achieve real-time hallucination prevention with minimal performance overhead.

<img src="../visualizations/NumberLogitsProcessor.gif" alt="Alt Text" style="width:70%; height:auto;">

Scroll to the bottom to see each slide of this GIF.

## Results sneak peak
In our first implementation we implemented a LogitsProcessor for a summary task preventing any number not to be created if not mentioned in the input with remarkable results.

### Total Hallucinations

| Model | Plain Generation | With NumberLogitsProcessor | Reduction |
|-------|------------------|---------------------------|-----------|
| **Qwen2.5 14B Instruct** | 56        | 14   | **75.00%** | 
| **Mistral 7B Instruct v0.2** | 141 | 8 | **94.33%** |
| **Llama 2 7B** | 47 | 4 | **91.49%** |

### Average runtime per generated answer

| Model | Plain (s) | With Processor (s) | Overhead |
|-------|-----------|-------------------|----------|
| **Qwen2.5 14B Instruct** | 10.03 | 10.55   | +0.52s | 
| **Mistral 7B Instruct v0.2** | 8.09 | 8.82 | +0.73s |
| **Llama 2 7B** | 10.70 | 11.31 | +0.61s |

For a more comprehensive overview, scroll down to the chapter [Results](#Results) below.

## TL;DR

- **Prevention over Detection**: Stop hallucinations during generation rather than filtering afterward
- **NumberLogitsProcessor**: Prevents numerical hallucinations by masking invalid tokens in real-time
- **Seamless Integration**: Works with any Hugging Face Transformers model via LogitsProcessor API
- **Proven Results**: Reduced hallucinated numbers by 93% (Mistral 7B) and 90% (Llama 2 7B) on RAGTruth dataset
- **Low Overhead**: Only 9% average latency increase while dramatically improving accuracy
- **Extensible Architecture**: Base framework supports any type of hallucination detector
- All code is open-source and production-ready
- **Future Look-out:** Implementing a detector using [TinyLettuce](https://huggingface.co/blog/adaamko/tinylettuce) to prevent factual hallucinations
- **Community Questions:** 
	1. Any other suitable low-latency models known for factual hallucination prevention?
	2. Any suggestions or improvements? Please shoot them in the comments

## Quick Links

- **GitHub**: [github.com/lebe1/LettucePrevent](https://github.com/lebe1/LettucePrevent)
- **TinyLettuce:** https://huggingface.co/blog/adaamko/tinylettuce


## Motivation

While post-hoc detection systems excel at identifying hallucinations after they occur, production RAG systems need something more: **Prevention**

### Our Novel Approach: Prevention at Generation Time

**What if we could stop hallucinations before they happen?**

LettucePrevent introduces a new paradigm inside the transformers library: **real-time intervention during token generation**. By integrating directly into the model's decoding process via custom LogitsProcessors, we can:

1. Examine each candidate token before it's selected
2. Check if that token would create a hallucination
3. Mask invalid tokens with penalty values
4. Force the model to select valid alternatives

This approach provides **hard guarantees** that certain types of hallucinations cannot occur, while maintaining natural language generation quality.

### How they differ from current LogitsProcessors

The [SequenceBiasLogitsProcessor](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1202) is the most suitable LogitsProcessor and accepts a list of tokens to be positively or negatively biased.   
In our first and very simple case of not allowing unmentioned numbers for a summary tasks, the first attempt was to positively bias all numbers found in the input text. This lead not to any significant improvement though. Either the bias was too high and the language generation quality suffered extremely or the bias was too low to reduce hallucinated numbers.
Experiment results can also be found in the `/data` directory.

## The Architecture

### Core Components

#### 1. HallucinationLogitsProcessor

The heart of LettucePrevent is a generalized `LogitsProcessor` that integrates with Hugging Face's generation pipeline. Here is a strongly simplified version:

```python
class HallucinationLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # For each batch
        for batch_idx in range(batch_size):
            # Get current sequence
            current_text = self.tokenizer.decode(input_ids[batch_idx])
            
            # Get top-k candidate tokens
            _, top_k_indices = torch.topk(scores[batch_idx], k=self.top_k_logits)
            
            # Check each candidate
            for token_id in top_k_indices:
                if self.hallucination_detector.check_hallucination(current_text, token_id):
                    # Mask this token - force model to choose alternative
                    scores[batch_idx][token_id] = self.penalty_value
                else:
                    break  # First valid token found
        
        return scores
```

**Key Design Decisions:**

- **Top-K Checking**: Only examines top candidate tokens (default: 10) for efficiency
- **Greedy Breaking**: Stops at first valid token to minimize overhead
- **Configurable Context**: Can consider last K tokens or full sequence
- **Hard Penalties**: Sets hallucinating tokens to `-inf` to guarantee rejection

#### 2. BaseHallucinationDetector Interface

Extensible base class for implementing specific hallucination detectors:

```python
class BaseHallucinationDetector:
    def __init__(self, tokenizer, input_text: str):
        self.tokenizer = tokenizer
        self.input_text = input_text
    
    @abstractmethod
    def check_hallucination(self, 
                          current_sequence: str, 
                          next_token_id: int, 
                          k_tokens: int) -> bool:
        """Returns True if next_token would create a hallucination."""
        pass
```

This abstraction allows implementing detectors for different hallucination types while reusing the core LogitsProcessor logic.

### Implementation: NumberHallucinationDetector

Our first concrete implementation prevents numerical hallucinations. Again, all shown code snippets below are simplified and shortened only to demonstrate the idea:

**Step 1: Extract Ground Truth Numbers**

```python
def _extract_all_numbers_from_input(self):
    # Find numeric formats: 2400, 2,400, 2400.5
    numeric = self._extract_numeric_numbers(self.input_text)
    
    # Find written numbers: "twenty-four hundred"
    written = self._extract_written_numbers(self.input_text)
    
    return numeric.union(written)
```

**Step 2: Real-Time Validation**

```python
def check_hallucination(self, current_sequence: str, 
                       next_token_id: int, k_tokens: int = 4):
    next_token = self.tokenizer.decode([next_token_id]).strip()
    
    # Only check digit/punctuation tokens
    if not self._is_digit_space_punctuation_token(next_token_id):
        return False
    
    # Build potential number from recent tokens + next token
    tokens = self.tokenizer.encode(current_sequence, add_special_tokens=False)
    recent_tokens = tokens[-k_tokens:] if len(tokens) >= k_tokens else tokens
    recent_str = self.tokenizer.decode(recent_tokens).strip()
    potential_sequence = recent_str + next_token_str
    
    # Extract number candidate
    number_match = re.search(r'[\d,.]+$', potential_sequence)
    if not number_match:
        return False
    
    # Check against allowed numbers
    for allowed_number in self.allowed_numbers:
        if number_candidate == allowed_number:
            return False  # Valid!
        if allowed_number.startswith(number_candidate):
            return False  # Valid prefix - could complete to allowed number
    
    # Not in allowed set and not a valid prefix
    return True  # HALLUCINATION!
```

**Key Features:**

- **Format-Aware**: Handles commas, decimals, written forms
- **Prefix Matching**: Allows numbers being built token-by-token
- **Context Window**: Examines last K tokens to detect multi-token numbers
- **Conservative**: Only blocks clearly invalid numbers

## Results

We evaluated LettucePrevent on 942 unique summary prompts from the RAGTruth dataset containing numerical information. All experiments were executed on an NVIDIA A40.

### Hallucination Reduction

**Text-Level Hallucinations**

*Number of texts containing one or more hallucinated numbers:*

| Model | Plain Generation | With NumberLogitsProcessor | Reduction |
|-------|------------------|---------------------------|-----------|
| **Qwen2.5 14B Instruct** | 46        | 12   | **73.91%** | 
| **Mistral 7B Instruct v0.2** | 116 | 7 | **93.97%** |
| **Llama 2 7B** | 41 | 4 | **90.24%** |


**Total Hallucinations**

*Total number of hallucinated instances across all texts:*

| Model | Plain Generation | With NumberLogitsProcessor | Reduction |
|-------|------------------|---------------------------|-----------|
| **Qwen2.5 14B Instruct** | 56        | 14   | **75.00%** | 
| **Mistral 7B Instruct v0.2** | 141 | 8 | **94.33%** |
| **Llama 2 7B** | 47 | 4 | **91.49%** |

**Dramatic results**: Both models reduced number hallucinations by over 90% at both the text level and total hallucination count. This demonstrates the effectiveness of generation-time prevention - not only do fewer texts contain errors, but the total number of errors is drastically reduced.

### Performance Overhead

*Total runtime across 942 prompts (NVIDIA A40):*

| Model | Plain Runtime (s) | With Processor (s) | Overhead |
|-------|-------------------|-------------------|----------|
| **Qwen2.5 14B Instruct** | 9450.35       | 9947.45   | +5.26% | 
| **Mistral 7B Instruct v0.2** | 7630.61 | 8322.47 | +9.06% |
| **Llama 2 7B** | 10093.12 | 10672.77 | +5.74% |

*Average runtime per answer:*

| Model | Plain (s) | With Processor (s) | Overhead |
|-------|-----------|-------------------|----------|
| **Qwen2.5 14B Instruct** | 10.03 | 10.55   | +0.52s | 
| **Mistral 7B Instruct v0.2** | 8.09 | 8.82 | +0.73s |
| **Llama 2 7B** | 10.70 | 11.31 | +0.61s |

**Minimal overhead**: The additional processing adds less than 1 second per generation on average - a small price for 90%+ hallucination reduction.

## Quickstart

### Installation

Clone the [repository](https://github.com/lebe1/LettucePrevent)
Install all necessary packages via the `requirements.txt` from the repository:

```shell
pip install -r requirements.txt
```

### Execution

``` bash
python main.py
```

## Conclusion

LettucePrevent demonstrates that prevention is the next step after detection for many production use cases. By intervening during generation rather than filtering afterward, we achieve:

- **up to 90% reduction** in number hallucinations
- **<10% latency overhead**
- **Seamless integration** with existing models

The extensible architecture makes it straightforward to add new detectors for different hallucination types, while the LogitsProcessor API ensures compatibility with the entire Hugging Face ecosystem.
Another suitable detector approach could also be to detect valid URLs during generation.
There are several ideas also to ultimately shrink down the generation of hallucinated numbers to 0.
This would provide a next important step for production-ready systems to ensure zero hallucinations.
We are also looking forward to implement a strategy preventing factual hallucinations.

## Slide deck

<img src="./visualizations/1.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./visualizations/2.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./visualizations/3.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./visualizations/4.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./visualizations/5.jpg" alt="Alt Text" style="width:100%; height:auto;">

<img src="./visualizations/6.jpg" alt="Alt Text" style="width:100%; height:auto;">
