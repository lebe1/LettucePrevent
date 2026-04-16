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
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import TokenClassifierOutput
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_dataset
from datetime import datetime
from huggingface_hub import HfApi
import wandb

# ============================================================
# 0. Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ============================================================
# 1. Fixed Config (not swept)
# ============================================================
MODEL_NAME      = "jhu-clsp/ettin-decoder-68m"
MAX_LENGTH      = 4096
NUM_SAMPLES     = "FULL"
EPOCHS          = 6
NUM_LABELS      = 2
FREEZE_BACKBONE = False
STRATEGY        = "weighted"

LLAMA_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"

WANDB_ENTITY  = "lebeccard-technical-university-wien"
WANDB_PROJECT = "train-tokenized-decoder-model"

HF_USERNAME   = "lebe1"
HF_MODEL_NAME = "lettuceprevent-ettin-decoder-68m-en-tokenized"
HF_REPO_ID    = f"{HF_USERNAME}/{HF_MODEL_NAME}"

# ============================================================
# 2. Sweep configuration
# ============================================================
sweep_config = {
    "method": "grid",
    "metric": {
        "name": "eval/f1_binary_class_1",
        "goal": "maximize",
    },
    "parameters": {
        "learning_rate": {"values": [1e-6, 5e-6, 1e-5]},
        "batch_size":    {"values": [4, 8]},
        "weight_decay":  {"values": [0.01, 0.05]},
        "warmup_ratio":  {"values": [0.1]},
    }
}


# ============================================================
# 3. Tokenizers
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id
print(f"CLS token : {tokenizer.convert_ids_to_tokens([CLS_ID])} (id={CLS_ID})")
print(f"SEP token : {tokenizer.convert_ids_to_tokens([SEP_ID])} (id={SEP_ID})")

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_NAME)
print(f"Llama tokenizer loaded: vocab_size={llama_tokenizer.vocab_size}")


# ============================================================
# 4. Token classifier (unchanged)
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
            weight = self.class_weights if hasattr(self, "class_weights") else None
            loss   = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_class_weights(self, weights: torch.Tensor):
        self.register_buffer("class_weights", weights)

    @classmethod
    def from_pretrained_model(cls, model_name, num_labels=2, freeze_backbone=True):
        config     = AutoConfig.from_pretrained(model_name)
        instance   = cls(config, num_labels=num_labels)
        pretrained = AutoModel.from_pretrained(model_name)
        instance.backbone.load_state_dict(pretrained.state_dict(), strict=False)
        del pretrained

        for param in instance.backbone.parameters():
            param.requires_grad = not freeze_backbone
        for param in instance.classifier.parameters():
            param.requires_grad = True

        nn.init.normal_(instance.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(instance.classifier.bias)
        return instance


# ============================================================
# 5. Preprocessing helpers
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
    """
    Tokenize text by first splitting it into Llama token chunks, decoding each
    chunk individually, then tokenizing each chunk with the Ettin tokenizer
    SEPARATELY. This forces Ettin's tokenization to respect Llama token
    boundaries — simulating the way the HDM receives text during real
    generation, where it sees accumulated text arriving at Llama-token
    boundaries.

    Returns:
        input_ids: List[int] — Ettin token IDs
        labels:    List[int] — hallucination labels per Ettin token
                              (or None if char_mask is None — used for context/query)
    """
    llama_enc = llama_tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    llama_ids      = llama_enc["input_ids"]
    llama_offsets  = llama_enc["offset_mapping"]

    all_input_ids = []
    all_labels    = [] if char_mask is not None else None

    for llama_tok_id, (cs, ce) in zip(llama_ids, llama_offsets):
        if cs == ce:  # safety: skip empty offsets
            continue

        # Decode this single Llama token back to text
        chunk_text = llama_tokenizer.decode([llama_tok_id])
        if not chunk_text:
            continue

        # Tokenize this chunk with Ettin IN ISOLATION
        ettin_enc = tokenizer(chunk_text, add_special_tokens=False)
        ettin_ids = ettin_enc["input_ids"]
        if not ettin_ids:
            continue

        all_input_ids.extend(ettin_ids)

        # If labels are needed, assign the Llama-token-level label to every
        # Ettin sub-token produced from this chunk
        if char_mask is not None:
            # A Llama token is hallucinated if ANY char in its span is hallucinated
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

    # Tokenize context and query via Llama-chunk-by-chunk Ettin tokenization
    # (labels are all -100 for these, so pass char_mask=None)
    context_ids, _ = tokenize_text_via_llama_chunks(context, char_mask=None)
    query_ids,   _ = tokenize_text_via_llama_chunks(query,   char_mask=None)

    # Tokenize answer the same way, with labels aligned to Llama token boundaries
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


def has_hallucination(preprocessed: dict) -> bool:
    return any(l == 1 for l in preprocessed["labels"])


# ============================================================
# 6. Load & preprocess dataset
# ============================================================
print("Loading and preprocessing dataset ...")
hf_ds = load_dataset("wandb/RAGTruth-processed")


def hf_row_to_sample(row: dict) -> dict:
    return {
        "context": row["context"],
        "query":   row["query"],
        "answer":  row["output"],
        "labels":  parse_hallucination_labels(row["hallucination_labels"]),
    }


train_raw = [hf_row_to_sample(row) for row in hf_ds["train"]
             if row.get("hallucination_labels") not in (None, "")]
eval_raw  = [hf_row_to_sample(row) for row in hf_ds["test"]
             if row.get("hallucination_labels") not in (None, "")]

random.shuffle(train_raw)
random.shuffle(eval_raw)

print(f"Train samples : {len(train_raw)}")
print(f"Eval  samples : {len(eval_raw)}")

train_preprocessed = [preprocess(s) for s in train_raw]
eval_preprocessed  = [preprocess(s) for s in eval_raw]

n0 = sum(l == 0 for p in train_preprocessed for l in p["labels"])
n1 = sum(l == 1 for p in train_preprocessed for l in p["labels"])
total_active  = n0 + n1
w0            = total_active / (2.0 * n0)
w1            = total_active / (2.0 * n1)
class_weights = torch.tensor([w0, w1], dtype=torch.float32)
print(f"[weighted] Token counts  — supported: {n0:,}  hallucinated: {n1:,}")
print(f"[weighted] Class weights — w0={w0:.4f}  w1={w1:.4f}")
print("Preprocessing done.")


# ============================================================
# 7. Dataset class (unchanged)
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


train_dataset = HallucinationDataset(train_preprocessed)
eval_dataset  = HallucinationDataset(eval_preprocessed)

collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)


# ============================================================
# 8. Metrics (unchanged)
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
# 9. Single sweep run
# ============================================================
def train_sweep():
    timestamp_str = datetime.now().strftime("%Y-%m-%d-%H:%M")

    run = wandb.init()
    cfg = run.config

    lr           = cfg.learning_rate
    batch_size   = cfg.batch_size
    weight_decay = cfg.weight_decay
    warmup_ratio = cfg.warmup_ratio

    print(f"\n{'='*60}")
    print(f"Starting sweep run: {run.name}")
    print(f"  learning_rate : {lr}")
    print(f"  batch_size    : {batch_size}")
    print(f"  weight_decay  : {weight_decay}")
    print(f"  warmup_ratio  : {warmup_ratio}")
    print(f"{'='*60}\n")

    scratch_dir = os.environ.get("SCRATCH_DIR", ".")
    output_dir  = os.path.join(scratch_dir, f"sweep_run_{run.name}_{timestamp_str}")

    model = EttinTokenClassifier.from_pretrained_model(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        freeze_backbone=FREEZE_BACKBONE,
    )
    model.set_class_weights(class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).float()
    model.backbone.config.use_cache = False
    model.backbone.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_binary_class_1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="wandb",
        seed=SEED,
        prediction_loss_only=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=1,
        eval_accumulation_steps=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    final_metrics = trainer.evaluate()
    print(f"\n=== Final evaluation for run {run.name} ===")
    print(final_metrics)

    run.log({f"final/{k}": v for k, v in final_metrics.items()})

    best_f1 = final_metrics.get("eval_f1_binary_class_1", 0.0)
    run.summary["best_f1_binary_class_1"] = best_f1
    run.summary["eval_loss"]              = final_metrics.get("eval_loss", None)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Run model saved to {output_dir}")

    wandb.finish()

    return best_f1, output_dir


# ============================================================
# 10. Run the sweep
# ============================================================
if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_config,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )
    print(f"Sweep created: {sweep_id}")
    print(f"View at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps/{sweep_id}")

    all_run_results = []

    def tracked_train():
        f1, output_dir = train_sweep()
        all_run_results.append({"f1": f1, "output_dir": output_dir})

    wandb.agent(sweep_id, function=tracked_train, count=None)

# ============================================================
# 11. Upload only the best model to HuggingFace
# ============================================================
if not all_run_results:
    print("Not all runs completed — skipping HuggingFace upload.")
else:
    best_run = max(all_run_results, key=lambda x: x["f1"])
    print(f"\nBest run — F1: {best_run['f1']:.4f}  |  dir: {best_run['output_dir']}")

    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
    api.upload_folder(
        folder_path=best_run["output_dir"],
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print(f"Best model uploaded to https://huggingface.co/{HF_REPO_ID}")