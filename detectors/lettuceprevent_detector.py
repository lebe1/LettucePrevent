import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from .base_detector import BaseHallucinationDetector


# ---------------------------------------------------------------------------
# Model architecture (matches train_tokenized_decoder_model.py)
# ---------------------------------------------------------------------------

class EttinTokenClassifier(PreTrainedModel):
    def __init__(self, config, num_labels: int = 2):
        super().__init__(config)
        self.num_labels = num_labels
        self.backbone = AutoModel.from_config(config)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

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


# ---------------------------------------------------------------------------
# LettucePrevent decoder detector
# ---------------------------------------------------------------------------

class LettucePreventDetector(BaseHallucinationDetector):
    """
    Decoder-style hallucination detector using a fine-tuned Ettin classifier.

    Inference pipeline:
      1. Generator emits token IDs from its own tokenizer (Mistral, Llama, Qwen).
      2. We decode generator IDs to text with the generator's tokenizer.
      3. Ettin tokenizer encodes context + query + answer text into Ettin IDs.
      4. The Ettin classifier scores each Ettin token; we read the class-1
         probability at the last answer position to flag the candidate.

    Only ONE tokenizer touches the detector model: Ettin's. The generator's
    tokenizer is used only to decode IDs back to text.

    Caches at __init__:
      - Ettin-tokenized context IDs
      - Ettin-tokenized query IDs (empty by default — set self.query if used)
      - the input_text length, for cheap answer-extraction
    """

    DEFAULT_MODEL_PATH = "lebe1/lettuceprevent-ettin-decoder-68m-en"

    def __init__(
        self,
        tokenizer,
        input_text: str,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.9,
        query: str = "",
    ):
        super().__init__(tokenizer, input_text)
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load detector tokenizer + model.
        self.detector_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.cls_id = self.detector_tokenizer.cls_token_id
        self.sep_id = self.detector_tokenizer.sep_token_id

        config = AutoConfig.from_pretrained(self.model_path)
        self.model = EttinTokenClassifier(config, num_labels=2)

        weights_filename = "model.safetensors"
        if os.path.isdir(self.model_path):
            weights_path = os.path.join(self.model_path, weights_filename)
        else:
            weights_path = hf_hub_download(
                repo_id=self.model_path, filename=weights_filename,
            )
        state_dict = load_file(weights_path, device=str(self.device))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()

        # ----- Per-prompt cache (context + query Ettin IDs) -----
        self.context = input_text
        self.query = query
        self._cached_context_ids = self.detector_tokenizer(
            self.context, add_special_tokens=False,
        )["input_ids"]
        self._cached_query_ids = self.detector_tokenizer(
            self.query, add_special_tokens=False,
        )["input_ids"]
        self._input_text_len = len(input_text)

        print(
            f"Initialized LettucePreventDetector (model_path='{self.model_path}', "
            f"threshold={confidence_threshold})"
        )
        print(
            f"  Cached context tokens: {len(self._cached_context_ids)}, "
            f"query tokens: {len(self._cached_query_ids)}"
        )

    # ---------- Helpers ----------

    def _build_input(self, answer_text: str) -> dict:
        """Build Ettin input IDs using cached context+query and freshly-tokenized answer."""
        answer_ids = self.detector_tokenizer(
            answer_text, add_special_tokens=False,
        )["input_ids"]

        input_ids = (
            [self.cls_id]
            + self._cached_context_ids
            + [self.sep_id]
            + self._cached_query_ids
            + [self.sep_id]
            + answer_ids
            + [self.sep_id]
        )

        n_prefix = (
            1
            + len(self._cached_context_ids)
            + 1
            + len(self._cached_query_ids)
            + 1
        )
        answer_mask = (
            [False] * n_prefix
            + [True] * len(answer_ids)
            + [False]
        )
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_mask": answer_mask,
        }

    # ---------- Main API ----------

    def check_hallucination(
        self,
        current_sequence: str,
        next_token_id: int,
        k_tokens: int = 4,
    ) -> bool:
        # Use decode-the-difference for tokenizer-family-agnostic next-token text.
        # Falling back to single-ID decode here because the base interface only
        # passes us current_sequence as text, not the ID list. Single-ID decode
        # is correct for SentencePiece (Mistral/Llama) since those tokenizers
        # encode leading-space markers in the token text itself, and works for
        # BPE (Qwen) for non-leading positions. For pathological first-token
        # cases the small inaccuracy is bounded.
        next_token_str = self.tokenizer.decode(
            [next_token_id], skip_special_tokens=True,
        )

        if self.input_text in current_sequence:
            current_answer = current_sequence[
                current_sequence.find(self.input_text) + self._input_text_len:
            ]
        else:
            current_answer = current_sequence

        potential_answer = current_answer + next_token_str

        if len(potential_answer.strip()) < 3:
            return False

        try:
            enc = self._build_input(potential_answer)

            input_ids_t = torch.tensor(
                [enc["input_ids"]], dtype=torch.long, device=self.device,
            )
            attention_mask_t = torch.tensor(
                [enc["attention_mask"]], dtype=torch.long, device=self.device,
            )

            with torch.no_grad():
                out = self.model(input_ids=input_ids_t, attention_mask=attention_mask_t)

            probs = torch.softmax(out.logits[0], dim=-1)
            answer_positions = [
                i for i, is_ans in enumerate(enc["answer_mask"]) if is_ans
            ]
            if not answer_positions:
                return False

            last_pos = answer_positions[-1]
            halluc_prob = probs[last_pos, 1].item()
            predicted_class = probs[last_pos].argmax().item()

            if predicted_class == 1 and halluc_prob >= self.confidence_threshold:
                print(
                    f"LettucePrevent detected hallucination: '{next_token_str}' "
                    f"(confidence: {halluc_prob:.3f})"
                )
                return True
            return False

        except Exception as e:
            print(f"Error in LettucePrevent detection: {e}")
            return False