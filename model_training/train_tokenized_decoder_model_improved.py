import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput
from safetensors.torch import load_file
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_dataset

# ============================================================
# 0. Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# 1. Config
# ============================================================
# Path or HF repo id of the best model saved by the sweep script.
# Either a local dir like "./sweep_run_xyz_2026-07-17-10:00"
# or the HF repo "lebe1/lettuceprevent-ettin-decoder-68m-en"
MODEL_PATH = "lebe1/lettuceprevent-ettin-decoder-68m-en"

BASE_MODEL_NAME      = "jhu-clsp/ettin-decoder-68m"
LLAMA_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"
MAX_LENGTH           = 4096
NUM_LABELS           = 2
EVAL_BATCH_SIZE      = 8
RESULTS_FILE         = "test_results.json"

# ============================================================
# 2. Tokenizers
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_NAME)

# ============================================================
# 3. Model definition (must match the training script)
# ============================================================
class EttinTokenClassifier(PreTrainedModel):
    def __init__(self, config, num_labels: int = 2):
        super().__init__(config)
        self.num_labels = num_labels
        self.backbone   = AutoModel.from_config(config)
        self.dropout    = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs         = self.backbone(input_ids=input_ids,
                                        attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits          = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_finetuned_model(model_path: str) -> EttinTokenClassifier:
    """Load the fine-tuned classifier from a local dir or HF repo."""
    config = AutoConfig.from_pretrained(model_path)
    model  = EttinTokenClassifier(config, num_labels=NUM_LABELS)

    if os.path.isdir(model_path):
        weights_path = os.path.join(model_path, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")

    state_dict = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # "class_weights" from training may show up as unexpected — that is fine,
    # it only affected the training loss, not predictions.
    print(f"Missing keys   : {missing}")
    print(f"Unexpected keys: {unexpected}")
    return model


# ============================================================
# 4. Preprocessing (identical to the training script)
# ============================================================
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
    llama_enc = llama_tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    llama_ids     = llama_enc["input_ids"]
    llama_offsets = llama_enc["offset_mapping"]

    all_input_ids = []
    all_labels    = [] if char_mask is not None else None

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

        if char_mask is not None:
            is_hallucinated = int(any(
                char_mask[i] == 1
                for i in range(cs, min(ce, len(char_mask)))
            ))
            all_labels.extend([is_hallucinated] * len(ettin_ids))

    return all_input_ids, all_labels


def preprocess(sample: dict) -> dict:
    context   = sample["context"]
    query     = sample["query"]
    answer    = sample["answer"]
    char_mask = build_char_mask(answer, sample["labels"])

    context_ids, _ = tokenize_text_via_llama_chunks(context, char_mask=None)
    query_ids,   _ = tokenize_text_via_llama_chunks(query,   char_mask=None)

    answer_ids, answer_labels = tokenize_text_via_llama_chunks(answer, char_mask=char_mask)

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

    assert len(input_ids) == len(attention_mask) == len(labels)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


# ============================================================
# 5. Dataset class
# ============================================================
class HallucinationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids":      torch.tensor(s["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(s["labels"],         dtype=torch.long),
        }


# ============================================================
# 6. Metrics (identical to the training script)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = np.concatenate(logits, axis=0)
    if isinstance(labels, (list, tuple)):
        labels = np.concatenate(labels, axis=0)
    predictions = np.argmax(logits, axis=-1)
    flat_preds  = predictions.flatten()
    flat_labels = labels.flatten()
    mask        = flat_labels != -100

    return {
        "f1_binary_class_1" : f1_score(flat_labels[mask], flat_preds[mask], average="binary", pos_label=1, zero_division="warn"),
        "precision_class_1" : precision_score(flat_labels[mask], flat_preds[mask], average="binary", pos_label=1, zero_division="warn"),
        "recall_class_1"    : recall_score(flat_labels[mask], flat_preds[mask], average="binary", pos_label=1, zero_division="warn"),
        "f1_binary_class_0" : f1_score(flat_labels[mask], flat_preds[mask], average="binary", pos_label=0, zero_division="warn"),
        "precision_class_0" : precision_score(flat_labels[mask], flat_preds[mask], average="binary", pos_label=0, zero_division="warn"),
        "recall_class_0"    : recall_score(flat_labels[mask], flat_preds[mask], average="binary", pos_label=0, zero_division="warn"),
        "f1_micro"          : f1_score(flat_labels[mask], flat_preds[mask], average="micro", zero_division="warn"),
        "precision_micro"   : precision_score(flat_labels[mask], flat_preds[mask], average="micro", zero_division="warn"),
        "recall_micro"      : recall_score(flat_labels[mask], flat_preds[mask], average="micro", zero_division="warn"),
    }


# ============================================================
# 7. Load & preprocess the TEST split
# ============================================================
if __name__ == "__main__":
    print("Loading test split ...")
    hf_ds = load_dataset("wandb/RAGTruth-processed")

    def hf_row_to_sample(row: dict) -> dict:
        return {
            "context": row["context"],
            "query":   row["query"],
            "answer":  row["output"],
            "labels":  parse_hallucination_labels(row["hallucination_labels"]),
        }

    test_raw = [hf_row_to_sample(row) for row in hf_ds["test"]
                if row.get("hallucination_labels") not in (None, "")]
    print(f"Test samples: {len(test_raw)}")

    test_preprocessed = [preprocess(s) for s in test_raw]
    test_dataset      = HallucinationDataset(test_preprocessed)

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ========================================================
    # 8. Load model & evaluate once
    # ========================================================
    print(f"Loading fine-tuned model from: {MODEL_PATH}")
    model  = load_finetuned_model(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).float()
    model.eval()

    eval_args = TrainingArguments(
        output_dir="./test_eval_tmp",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        eval_accumulation_steps=16,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    test_metrics = trainer.evaluate()

    print("\n" + "=" * 60)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"{k:30s}: {v}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")