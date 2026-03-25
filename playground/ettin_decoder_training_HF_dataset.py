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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ============================================================
# 1. Config
# ============================================================
MODEL_NAME   = "jhu-clsp/ettin-decoder-68m"
MAX_LENGTH   = 4096
BATCH_SIZE   = 16
NUM_SAMPLES  = 2000
EPOCHS       = 6
LR           = 1e-5
WEIGHT_DECAY = 0.01
NUM_LABELS   = 2

FREEZE_BACKBONE = False

# "filter"   → keep only samples with at least one hallucinated token
# "weighted" → keep all samples, scale loss by inverse class frequency
STRATEGY = "weighted"

scope_tag  = "frozen" if FREEZE_BACKBONE else "full"
OUTPUT_DIR = f"./ettin_{scope_tag}_{STRATEGY}"
print(f"Model          : {MODEL_NAME}")
print(f"Freeze backbone: {FREEZE_BACKBONE}  ({scope_tag} fine-tune)")
print(f"Strategy       : {STRATEGY}")
print(f"Output dir     : {OUTPUT_DIR}")


# ============================================================
# 2. Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

CLS_ID = tokenizer.cls_token_id 
SEP_ID = tokenizer.sep_token_id 
print(f"CLS token : {tokenizer.convert_ids_to_tokens([CLS_ID])} (id={CLS_ID})")
print(f"SEP token : {tokenizer.convert_ids_to_tokens([SEP_ID])} (id={SEP_ID})")


# ============================================================
# 3. Token classifier  (unchanged)
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
            loss = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)(
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
# 4. Preprocessing helpers
# ============================================================
def parse_hallucination_labels(raw_labels):
    """
    raw_labels is either:
      - a list of dicts already (HuggingFace deserialised it)
      - a JSON string  '[]'  or  '[{"start":…}, …]'
    Returns a list of dicts with at least 'start' and 'end' keys.
    """
    if isinstance(raw_labels, str):
        try:
            raw_labels = json.loads(raw_labels)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw_labels, list):
        return []
    # Keep only entries that carry character offsets
    return [lbl for lbl in raw_labels
            if isinstance(lbl, dict) and "start" in lbl and "end" in lbl]


def build_char_mask(answer: str, labels: list) -> list:
    """Character-level binary mask: 0 = supported, 1 = hallucinated."""
    mask = [0] * len(answer)
    for span in labels:
        for i in range(span["start"], min(span["end"], len(answer))):
            mask[i] = 1
    return mask


def token_is_hallucinated(char_start, char_end, char_mask) -> int:
    return int(any(char_mask[i] == 1 for i in range(char_start, char_end)))


def preprocess(sample: dict) -> dict:
    """
    Tokenizes as: [CLS] context [SEP] query [SEP] answer [SEP]

    Labels:
      -100  → context / query / special tokens  (ignored in loss)
         0  → supported answer token
         1  → hallucinated answer token
    """
    context   = sample["context"]   # long source passage
    query     = sample["query"]     # short instruction / question
    answer    = sample["answer"]    # model-generated response
    char_mask = build_char_mask(answer, sample["labels"])

    # Tokenize all three parts separately for clean offset mappings
    context_enc = tokenizer(context, add_special_tokens=False,
                            return_offsets_mapping=True)
    query_enc   = tokenizer(query,   add_special_tokens=False,
                            return_offsets_mapping=True)
    answer_enc  = tokenizer(answer,  add_special_tokens=False,
                            return_offsets_mapping=True)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # [CLS] context [SEP] query [SEP] answer [SEP]
    input_ids = (
        [cls_id]
        + context_enc["input_ids"]
        + [sep_id]
        + query_enc["input_ids"]
        + [sep_id]
        + answer_enc["input_ids"]
        + [sep_id]
    )

    attention_mask = [1] * len(input_ids)

    # Labels: only answer tokens get 0/1; everything else is -100
    answer_labels = []
    for cs, ce in answer_enc["offset_mapping"]:
        if cs == 0 and ce == 0:           # padding / special offset
            answer_labels.append(-100)
        else:
            answer_labels.append(token_is_hallucinated(cs, ce, char_mask))

    labels = (
        [-100]                                       # [CLS]
        + [-100] * len(context_enc["input_ids"])     # context
        + [-100]                                     # [SEP]
        + [-100] * len(query_enc["input_ids"])       # query
        + [-100]                                     # [SEP]
        + answer_labels                              # answer tokens
        + [-100]                                     # final [SEP]
    )

    # Truncate to MAX_LENGTH
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
# 5. Load and convert HuggingFace dataset
# ============================================================
hf_ds = load_dataset("wandb/RAGTruth-processed")

def hf_row_to_sample(row: dict) -> dict:
    """
    Convert one HuggingFace row into the flat dict expected by preprocess().
    hallucination_labels is a JSON string in the HF dataset.
    """
    return {
        "context": row["context"],
        "query":   row["query"],
        "answer":  row["output"],
        "labels":  parse_hallucination_labels(row["hallucination_labels"]),
    }

# Use HuggingFace splits; filter rows that have no labels at all
train_raw = [hf_row_to_sample(row) for row in hf_ds["train"]
             if row.get("hallucination_labels") not in (None, "")]
eval_raw  = [hf_row_to_sample(row) for row in hf_ds["test"]
             if row.get("hallucination_labels") not in (None, "")]

random.shuffle(train_raw)
random.shuffle(eval_raw)

if NUM_SAMPLES == "FULL":
    print(f"Using FULL dataset")
    print(f"Train samples (before strategy) : {len(train_raw)}")
    print(f"Eval  samples                   : {len(eval_raw)}")
else:
    total   = len(train_raw) + len(eval_raw)
    frac    = NUM_SAMPLES / total
    n_train = min(int(len(train_raw) * frac), len(train_raw))
    n_eval  = min(NUM_SAMPLES - n_train, len(eval_raw))

    train_raw = train_raw[:n_train]
    eval_raw  = eval_raw[:n_eval]

    print(f"Total available                    : {total}")
    print(f"Train samples (before strategy)    : {len(train_raw)}")
    print(f"Eval  samples                      : {len(eval_raw)}")

train_preprocessed = [preprocess(s) for s in train_raw]
eval_preprocessed  = [preprocess(s) for s in eval_raw]


# ============================================================
# 6. Apply class-imbalance strategy  (unchanged logic)
# ============================================================
if STRATEGY == "filter":
    before = len(train_preprocessed)
    pairs  = [(r, p) for r, p in zip(train_raw, train_preprocessed)
              if has_hallucination(p)]
    train_raw          = [p[0] for p in pairs]
    train_preprocessed = [p[1] for p in pairs]
    print(f"[filter] Kept {len(train_preprocessed)} / {before} samples "
          f"(removed {before - len(train_preprocessed)} fully-supported)")

elif STRATEGY == "weighted":
    n0 = sum(l == 0 for p in train_preprocessed for l in p["labels"])
    n1 = sum(l == 1 for p in train_preprocessed for l in p["labels"])
    total_active = n0 + n1
    w0 = total_active / (2.0 * n0)
    w1 = total_active / (2.0 * n1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    print(f"[weighted] Token counts  — supported: {n0:,}  hallucinated: {n1:,}")
    print(f"[weighted] Class weights — w0={w0:.4f}  w1={w1:.4f}")

print("Preprocessing done.")


# ============================================================
# 7. Debug print — first sample with hallucinated labels
# ============================================================
def print_first_sample_tokens(raw_sample: dict, preprocessed: dict) -> None:
    input_ids = preprocessed["input_ids"]
    labels    = preprocessed["labels"]
    tokens    = tokenizer.convert_ids_to_tokens(input_ids)

    print("\n" + "=" * 72)
    print("DEBUG: First training sample containing hallucinated tokens")
    print("=" * 72)
    print(f"Context (first 200 chars) : {raw_sample['context'][:200]}...")
    print(f"Query                     : {raw_sample['query']}")
    print(f"Answer (first 300 chars)  : {raw_sample['answer'][:300]}"
          f"{'...' if len(raw_sample['answer']) > 300 else ''}")
    print(f"Hallucinated spans        : {raw_sample['labels']}\n")

    label_meaning = {-100: "masked", 0: "supported", 1: "HALLUCINATED"}
    print(f"{'idx':>5}  {'tok_id':>8}  {'token':>25}  {'label':>6}  meaning")
    print("-" * 72)
    for idx, (tok, tok_id, lbl) in enumerate(zip(tokens, input_ids, labels)):
        print(f"{idx:>5}  {tok_id:>8}  {str(tok):>25}  "
              f"{lbl:>6}  {label_meaning.get(lbl, lbl)}")

    n_masked    = sum(1 for l in labels if l == -100)
    n_supported = sum(1 for l in labels if l == 0)
    n_halluc    = sum(1 for l in labels if l == 1)
    print("-" * 72)
    print(f"Total tokens              : {len(input_ids)}")
    print(f"  masked (ctx+query+spec) : {n_masked}")
    print(f"  supported               : {n_supported}")
    print(f"  hallucinated            : {n_halluc}")
    print("=" * 72 + "\n")


# Find the first training sample that actually contains a hallucinated token
debug_pair = next(
    ((raw, pre) for raw, pre in zip(train_raw, train_preprocessed)
     if has_hallucination(pre)),
    None,
)

if debug_pair is None:
    print("WARNING: No training sample with hallucinated tokens found for debug print.")
else:
    print_first_sample_tokens(*debug_pair)


# ============================================================
# 8. Dataset  (unchanged)
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


# ============================================================
# 9. Collator
# ============================================================
collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)


# ============================================================
# 10. Metrics
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

    f1_bin = f1_score(
        flat_labels[mask], flat_preds[mask],
        average="binary", pos_label=1, zero_division="warn",
    )
    f1_micro = f1_score(
        flat_labels[mask], flat_preds[mask],
        average="micro", zero_division="warn",
    )
    precision = precision_score(
        flat_labels[mask], flat_preds[mask],
        average="binary", pos_label=1, zero_division="warn",
    )
    recall = recall_score(
        flat_labels[mask], flat_preds[mask],
        average="binary", pos_label=1, zero_division="warn",
    )
    return {
        "f1_binary"  : f1_bin,
        "f1_micro"   : f1_micro,
        "precision"  : precision,
        "recall"     : recall,
    }


# ============================================================
# 11. Model
# ============================================================
model = EttinTokenClassifier.from_pretrained_model(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    freeze_backbone=FREEZE_BACKBONE,
)

# Register class weights as a model buffer for strategy "weighted"
# so they are automatically moved to the correct device with the model.
if STRATEGY == "weighted":
    model.set_class_weights(class_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")
model = model.to(device).float()
model.backbone.config.use_cache = False # 
model.backbone.gradient_checkpointing_enable()

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params    = total_params - trainable_params
print(f"Trainable params : {trainable_params:,}")
print(f"Frozen params    : {frozen_params:,}")
print(f"Total params     : {total_params:,}")


# ============================================================
# 12. Pre-training sanity check
# ============================================================
model.eval()
batch = collator([train_dataset[0], train_dataset[1]])
batch = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    out = model(**batch)

print(f"Pre-training loss : {out.loss.item():.4f}")
print(f"Logits nan/inf    : {torch.isnan(out.logits).any().item()} / "
      f"{torch.isinf(out.logits).any().item()}")
print(f"Logits min/max    : {out.logits.min().item():.4f} / "
      f"{out.logits.max().item():.4f}\n")


# ============================================================
# 13. Training
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=1.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="precision",
    greater_is_better=True,
    logging_strategy="epoch",
    report_to="none",
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
)

trainer.train()

print("\n=== Final evaluation ===")
metrics = trainer.evaluate()
print(metrics)