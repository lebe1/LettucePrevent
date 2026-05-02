import torch
from transformers import LogitsProcessor

from detectors.base_detector import BaseHallucinationDetector


class HallucinationLogitsProcessor(LogitsProcessor):
    """
    Generalized logits processor that works with any BaseHallucinationDetector
    implementation.

    Skip-threshold semantics: when the generator is more than `skip_threshold`
    confident on its top candidate token (top-k softmax), the HDM check is
    bypassed. With skip_threshold = 1.0 the check is never bypassed, since
    a softmax over k=10 entries is always <= 1.0 by definition. Lower values
    let the processor skip on increasingly less-confident tokens.
    """

    def __init__(
        self,
        hallucination_detector: BaseHallucinationDetector,
        last_k_tokens_to_consider: int = 4,
        top_k_logits: int = 10,
        penalty_value: float = float("-inf"),
        use_all_tokens: bool = False,
        skip_threshold: float = 1.0,
    ):
        self.hallucination_detector    = hallucination_detector
        self.last_k_tokens_to_consider = last_k_tokens_to_consider
        self.penalty_value             = penalty_value
        self.top_k_logits              = top_k_logits
        self.use_all_tokens            = use_all_tokens
        self.skip_threshold            = skip_threshold

        # Counters
        self.modifications_count = 0   # times we applied the penalty
        self.skip_count          = 0   # times we skipped HDM due to confidence
        self.check_count         = 0   # total HDM check opportunities

        tokens_context = (
            "all tokens" if use_all_tokens
            else f"last {last_k_tokens_to_consider} tokens"
        )
        print(
            f"Initialized LogitsProcessor with "
            f"{type(hallucination_detector).__name__}"
        )
        print(f"Context window: {tokens_context}")
        print(f"Skip threshold: {skip_threshold}")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            current_ids = input_ids[batch_idx]
            current_text = self.hallucination_detector.tokenizer.decode(
                current_ids, skip_special_tokens=True,
            )

            if self.use_all_tokens:
                context_window = len(current_ids)
            else:
                context_window = min(
                    self.last_k_tokens_to_consider, len(current_ids),
                )

            top_k_scores, top_k_indices = torch.topk(
                scores[batch_idx],
                k=min(self.top_k_logits, scores.shape[-1]),
            )
            top_k_probs = torch.softmax(top_k_scores, dim=-1)
            top_k_tokens = [
                self.hallucination_detector.tokenizer.decode([idx.item()])
                for idx in top_k_indices
            ]

            print("Top K tokens:")
            for rank, (token_str, prob) in enumerate(
                zip(top_k_tokens, top_k_probs.tolist())
            ):
                print(f"  {rank}: {repr(token_str)} ({prob:.4f})")

            # Count this opportunity to check (whether or not we skip).
            self.check_count += 1

            # Skip HDM check when generator confidence STRICTLY exceeds the
            # threshold. With skip_threshold = 1.0, this branch is never taken
            # (top-k softmax is bounded above by 1.0 exactly).
            if top_k_probs[0].item() > self.skip_threshold:
                self.skip_count += 1
                print(
                    f"Skipping check: top token {repr(top_k_tokens[0])} "
                    f"has prob {top_k_probs[0].item():.4f} > {self.skip_threshold}"
                )
                continue

            # Otherwise: walk top-k candidates, penalize the first hallucinatory one.
            for i, token_id in enumerate(top_k_indices):
                token_id_item = token_id.item()

                if self.hallucination_detector.check_hallucination(
                    current_text, token_id_item, context_window,
                ):
                    scores[batch_idx][token_id_item] = self.penalty_value
                    self.modifications_count += 1
                    token_str = self.hallucination_detector.tokenizer.decode(
                        [token_id_item]
                    )
                    print(f"----------Modified logit for token: {token_str}----------")
                    continue

                # Analysis-only logging.
                chosen_token_str = self.hallucination_detector.tokenizer.decode(
                    [token_id_item]
                )
                chosen_prob = torch.softmax(
                    scores[batch_idx], dim=-1
                )[token_id_item].item()
                input_text = self.hallucination_detector.input_text
                if input_text in current_text:
                    generated_text = current_text[
                        current_text.find(input_text) + len(input_text):
                    ]
                else:
                    generated_text = current_text
                print(
                    f"Chosen token: {repr(chosen_token_str)} "
                    f"({chosen_prob:.4f}) | Generated so far: {repr(generated_text)}"
                )

                # Greedy break after first non-hallucinatory candidate.
                break

        return scores