import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .base_detector import BaseHallucinationDetector


# ============================================================
# Model architecture — must match train.py exactly
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


# ============================================================
# Detector
# ============================================================
class EttinDecoderDetector(BaseHallucinationDetector):

    def __init__(
        self,
        tokenizer,
        input_text: str,
        model_path: str,
        confidence_threshold: float = 0.9,
    ):
        super().__init__(tokenizer, input_text)
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector_tokenizer = AutoTokenizer.from_pretrained(model_path)
        config      = AutoConfig.from_pretrained(model_path)
        self.model  = EttinTokenClassifier(config, num_labels=2)

        # ── Resolve model weights regardless of local path or HF repo ID ──
        weights_filename = "model.safetensors"
        if os.path.isdir(model_path):
            # Local directory — build the path directly
            weights_path = os.path.join(model_path, weights_filename)
        else:
            # Treat as a HuggingFace repo ID and download/cache the file
            weights_path = hf_hub_download(
                repo_id=model_path,
                filename=weights_filename,
            )

        state_dict = load_file(weights_path, device=str(self.device))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()

        self.context = input_text
        self.query   = ""

        print(
            f"Initialized EttinDecoderDetector from '{model_path}' "
            f"with threshold: {confidence_threshold}"
        )

    def _build_input(self, answer: str) -> dict:
        det_tok = self.detector_tokenizer

        context_enc = det_tok(self.context, add_special_tokens=False)
        query_enc   = det_tok(self.query,   add_special_tokens=False)
        answer_enc  = det_tok(answer,       add_special_tokens=False)

        cls_id = det_tok.cls_token_id
        sep_id = det_tok.sep_token_id

        input_ids = (
            [cls_id]
            + context_enc["input_ids"]
            + [sep_id]
            + query_enc["input_ids"]
            + [sep_id]
            + answer_enc["input_ids"]
            + [sep_id]
        )
        n_prefix     = 1 + len(context_enc["input_ids"]) + 1 \
                         + len(query_enc["input_ids"])   + 1
        answer_mask  = (
            [False] * n_prefix
            + [True]  * len(answer_enc["input_ids"])
            + [False]
        )
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "answer_mask":    answer_mask,
        }

    def check_hallucination(
        self,
        current_sequence: str,
        next_token_id: int,
        k_tokens: int = 4,
    ) -> bool:
        next_token_raw = self.tokenizer.convert_ids_to_tokens([next_token_id])[0]
        next_token_str = next_token_raw.replace("▁", " ")

        if self.input_text in current_sequence:
            context_end    = current_sequence.find(self.input_text) + len(self.input_text)
            current_answer = current_sequence[context_end:]
        else:
            current_answer = current_sequence

        potential_answer = current_answer + next_token_str

        if len(potential_answer.strip()) < 3:
            return False

        try:
            enc = self._build_input(potential_answer)

            input_ids      = torch.tensor([enc["input_ids"]],
                                          dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([enc["attention_mask"]],
                                          dtype=torch.long).to(self.device)

            with torch.no_grad():
                out = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)

            probs = torch.softmax(out.logits[0], dim=-1)

            answer_positions = [
                i for i, is_ans in enumerate(enc["answer_mask"]) if is_ans
            ]
            if not answer_positions:
                return False

            last_answer_pos = answer_positions[-1]
            halluc_prob     = probs[last_answer_pos, 1].item()
            predicted_class = probs[last_answer_pos].argmax().item()

            if predicted_class == 1 and halluc_prob >= self.confidence_threshold:
                print(
                    f"EttinDecoder detected hallucination: '{next_token_str}' "
                    f"(confidence: {halluc_prob:.3f})"
                )
                return True

            return False

        except Exception as e:
            print(f"Error in EttinDecoder detection: {e}")
            return False