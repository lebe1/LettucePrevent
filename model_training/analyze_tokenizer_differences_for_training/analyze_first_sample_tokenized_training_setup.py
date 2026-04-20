import json
import random
from transformers import AutoTokenizer
from datasets import load_dataset

SEED = 42
random.seed(SEED)

MODEL_NAME = "jhu-clsp/ettin-decoder-68m"
LLAMA_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"
MAX_LENGTH = 4096

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_NAME)

CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id


def parse_hallucination_labels(raw_labels):
    if isinstance(raw_labels, str):
        try:
            raw_labels = json.loads(raw_labels)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw_labels, list):
        return []
    return [lbl for lbl in raw_labels
            if isinstance(lbl, dict) and "start" in lbl and "end" in lbl]


def build_char_mask(answer: str, labels: list) -> list:
    mask = [0] * len(answer)
    for span in labels:
        for i in range(span["start"], min(span["end"], len(answer))):
            mask[i] = 1
    return mask


def tokenize_text_via_llama_chunks(text: str, char_mask: list = None):
    """
    Returns:
        input_ids:    List[int]   — Ettin token IDs
        labels:       List[int]   — hallucination labels (or None)
        llama_chunks: List[dict]  — debug info per Llama token
    """
    llama_enc = llama_tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    llama_ids     = llama_enc["input_ids"]
    llama_offsets = llama_enc["offset_mapping"]

    all_input_ids = []
    all_labels    = [] if char_mask is not None else None
    llama_chunks  = []  # debug info

    for llama_tok_id, (cs, ce) in zip(llama_ids, llama_offsets):
        if cs == ce:
            continue

        chunk_text = llama_tokenizer.decode([llama_tok_id])
        if not chunk_text:
            continue

        ettin_enc = tokenizer(chunk_text, add_special_tokens=False)
        ettin_ids = ettin_enc["input_ids"]
        if not ettin_ids:
            continue

        all_input_ids.extend(ettin_ids)

        is_hallucinated = None
        if char_mask is not None:
            is_hallucinated = int(any(
                char_mask[i] == 1
                for i in range(cs, min(ce, len(char_mask)))
            ))
            all_labels.extend([is_hallucinated] * len(ettin_ids))

        llama_chunks.append({
            "llama_id":        llama_tok_id,
            "llama_token":     llama_tokenizer.convert_ids_to_tokens([llama_tok_id])[0],
            "chunk_text":      chunk_text,
            "char_span":       (cs, ce),
            "ettin_ids":       ettin_ids,
            "ettin_tokens":    tokenizer.convert_ids_to_tokens(ettin_ids),
            "is_hallucinated": is_hallucinated,
        })

    return all_input_ids, all_labels, llama_chunks


def preprocess(sample: dict) -> dict:
    context   = sample["context"]
    query     = sample["query"]
    answer    = sample["answer"]
    char_mask = build_char_mask(answer, sample["labels"])

    context_ids, _, _ = tokenize_text_via_llama_chunks(context, char_mask=None)
    query_ids,   _, _ = tokenize_text_via_llama_chunks(query,   char_mask=None)
    answer_ids, answer_labels, answer_chunks = tokenize_text_via_llama_chunks(
        answer, char_mask=char_mask
    )

    input_ids = (
        [CLS_ID]
        + context_ids
        + [SEP_ID]
        + query_ids
        + [SEP_ID]
        + answer_ids
        + [SEP_ID]
    )
    attention_mask = [1] * len(input_ids)

    labels = (
        [-100]
        + [-100] * len(context_ids)
        + [-100]
        + [-100] * len(query_ids)
        + [-100]
        + answer_labels
        + [-100]
    )

    input_ids      = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels         = labels[:MAX_LENGTH]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        # debug extras
        "_answer":        answer,
        "_hal_spans":     sample["labels"],
        "_answer_chunks": answer_chunks,
        "_context_len":   len(context_ids),
        "_query_len":     len(query_ids),
    }


# --- Load dataset ---
hf_ds = load_dataset("wandb/RAGTruth-processed")


def hf_row_to_sample(row):
    return {
        "context": row["context"],
        "query":   row["query"],
        "answer":  row["output"],
        "labels":  parse_hallucination_labels(row["hallucination_labels"]),
    }


train_raw = [hf_row_to_sample(row) for row in hf_ds["train"]
             if row.get("hallucination_labels") not in (None, "")]
random.shuffle(train_raw)

# --- Find first sample with hallucinated tokens ---
for idx, sample in enumerate(train_raw):
    preprocessed = preprocess(sample)
    if any(l == 1 for l in preprocessed["labels"]):
        break

p = preprocessed

print("=" * 100)
print("DEBUG: First training sample (LLAMA-CHUNKED ETTIN TOKENIZATION)")
print("=" * 100)
print(f"Answer (first 300 chars): {p['_answer'][:300]}...")
print(f"Hallucinated spans      : {p['_hal_spans']}")
print()

# === View 1: full Ettin sequence ===
print("-" * 100)
print("FULL ETTIN TOKEN SEQUENCE")
print("-" * 100)
print(f"{'idx':>5}  {'tok_id':>8}  {'token':>25}  {'label':>5}  {'meaning'}")
print("-" * 100)

n_masked = 0
n_supported = 0
n_hallucinated = 0

for i, (tid, lbl) in enumerate(zip(p["input_ids"], p["labels"])):
    tok_str = tokenizer.convert_ids_to_tokens([tid])[0]
    if lbl == -100:
        meaning = "masked"
        n_masked += 1
    elif lbl == 0:
        meaning = "supported"
        n_supported += 1
    else:
        meaning = "HALLUCINATED"
        n_hallucinated += 1
    print(f"  {i:>3}  {tid:>8}  {tok_str:>25}  {lbl:>5}  {meaning}")

print("-" * 100)
print(f"Total tokens              : {len(p['input_ids'])}")
print(f"  masked (ctx+query+spec) : {n_masked}")
print(f"  supported               : {n_supported}")
print(f"  hallucinated            : {n_hallucinated}")

# === View 2: Llama-chunk-by-chunk alignment (answer only) ===
print()
print("=" * 100)
print("LLAMA → ETTIN ALIGNMENT FOR ANSWER TOKENS")
print("=" * 100)
print(f"{'#':>4}  {'llama_token':>20}  {'chunk':>20}  "
      f"{'ettin_tokens':>40}  {'label'}")
print("-" * 100)

for i, chunk in enumerate(p["_answer_chunks"]):
    label_str = "HALLUCINATED" if chunk["is_hallucinated"] == 1 else "supported"
    ettin_repr = " ".join(repr(t) for t in chunk["ettin_tokens"])
    print(f"  {i:>3}  "
          f"{repr(chunk['llama_token']):>20}  "
          f"{repr(chunk['chunk_text']):>20}  "
          f"{ettin_repr:>40}  "
          f"{label_str}")

# === View 3: summary of split behaviour ===
print()
print("=" * 100)
print("SPLIT SUMMARY (Llama tokens that produced >1 Ettin sub-tokens)")
print("=" * 100)
splits = [c for c in p["_answer_chunks"] if len(c["ettin_ids"]) > 1]
print(f"Llama tokens that split into multiple Ettin tokens: {len(splits)} / {len(p['_answer_chunks'])}")
for c in splits[:20]:
    print(f"  {repr(c['llama_token']):>20} → {c['ettin_tokens']}")
if len(splits) > 20:
    print(f"  ... and {len(splits) - 20} more")

print()
print("=" * 100)
print("COMPARISON: direct Ettin tokenization of the full answer (no Llama chunking)")
print("=" * 100)
direct_ids = tokenizer(p["_answer"], add_special_tokens=False)["input_ids"]
direct_tokens = tokenizer.convert_ids_to_tokens(direct_ids)
print(f"Direct Ettin tokens      : {len(direct_ids)}")
print(f"Llama-chunked Ettin tokens: {sum(len(c['ettin_ids']) for c in p['_answer_chunks'])}")
print()
print(f"First 30 direct tokens        : {direct_tokens[:30]}")
print(f"First 30 Llama-chunked tokens : "
      f"{[t for c in p['_answer_chunks'][:15] for t in c['ettin_tokens']][:30]}")