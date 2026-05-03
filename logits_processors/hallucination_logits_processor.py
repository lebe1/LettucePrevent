import torch
from transformers import LogitsProcessor

from detectors.base_detector import BaseHallucinationDetector


class HallucinationLogitsProcessor(LogitsProcessor):
    """
    Generalized logits processor that works with any BaseHallucinationDetector.

    Optimizations:
      - debug_print: when False, ALL per-token prints are suppressed
        (counters still update). Significant runtime impact.
      - decoded-prefix cache: avoid O(N^2) re-decoding of the growing
        sequence on every step. With beam search, beams reorder per step,
        so the cache validates against the actual current_ids and falls
        back to a full decode on mismatch.
    """

    def __init__(
        self,
        hallucination_detector: BaseHallucinationDetector,
        last_k_tokens_to_consider: int = 4,
        top_k_logits: int = 10,
        penalty_value: float = float("-inf"),
        use_all_tokens: bool = False,
        skip_threshold: float = 1.0,
        debug_print: bool = False,
    ):
        self.hallucination_detector    = hallucination_detector
        self.last_k_tokens_to_consider = last_k_tokens_to_consider
        self.penalty_value             = penalty_value
        self.top_k_logits              = top_k_logits
        self.use_all_tokens            = use_all_tokens
        self.skip_threshold            = skip_threshold
        self.debug_print               = debug_print

        # Counters
        self.modifications_count = 0
        self.skip_count          = 0
        self.check_count         = 0

        # Decoded-prefix cache, per beam batch index. Each entry is a tuple
        # (last_id_int, decoded_text_so_far). On mismatch, we fall back to
        # full re-decode.
        self._decoded_cache: dict = {}

        if self.debug_print:
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

    def _get_current_text(self, batch_idx: int, current_ids: torch.LongTensor) -> str:
        """
        Return the decoded text for current_ids, using a per-beam cache to
        avoid O(N^2) work. Cache invalidates on:
          - first call for this batch_idx (cold start)
          - the trailing token id changed without growing (beam reorder)
          - the sequence shrank (shouldn't happen, defensive)
        """
        cur_len = len(current_ids)
        cached = self._decoded_cache.get(batch_idx)
        tokenizer = self.hallucination_detector.tokenizer

        # Hot path: pure single-token append from a known previous state.
        if cached is not None:
            prev_len, prev_text = cached
            if cur_len == prev_len + 1:
                # Decode-the-difference: get just the appended token's text.
                # Decode from prev_len-1 onwards so leading-space markers
                # resolve correctly (BPE quirk on first token of a chunk).
                if prev_len > 0:
                    boundary_text = tokenizer.decode(
                        current_ids[prev_len - 1: prev_len + 1].tolist(),
                        skip_special_tokens=True,
                    )
                    last_token_only = tokenizer.decode(
                        current_ids[prev_len - 1: prev_len].tolist(),
                        skip_special_tokens=True,
                    )
                    appended_text = boundary_text[len(last_token_only):]
                else:
                    appended_text = tokenizer.decode(
                        current_ids[: cur_len].tolist(),
                        skip_special_tokens=True,
                    )
                new_text = prev_text + appended_text
                self._decoded_cache[batch_idx] = (cur_len, new_text)
                return new_text

        # Cold path / cache invalidation: full decode.
        new_text = tokenizer.decode(current_ids, skip_special_tokens=True)
        self._decoded_cache[batch_idx] = (cur_len, new_text)
        return new_text

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            current_ids = input_ids[batch_idx]
            current_text = self._get_current_text(batch_idx, current_ids)

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

            if self.debug_print:
                top_k_tokens = [
                    self.hallucination_detector.tokenizer.decode([idx.item()])
                    for idx in top_k_indices
                ]
                print("Top K tokens:")
                for rank, (token_str, prob) in enumerate(
                    zip(top_k_tokens, top_k_probs.tolist())
                ):
                    print(f"  {rank}: {repr(token_str)} ({prob:.4f})")

            self.check_count += 1

            # Skip HDM check when generator confidence STRICTLY exceeds the
            # threshold. With skip_threshold = 1.0, this branch is never taken.
            if top_k_probs[0].item() > self.skip_threshold:
                self.skip_count += 1
                if self.debug_print:
                    top_token_str = self.hallucination_detector.tokenizer.decode(
                        [top_k_indices[0].item()]
                    )
                    print(
                        f"Skipping check: top token {repr(top_token_str)} "
                        f"has prob {top_k_probs[0].item():.4f} > "
                        f"{self.skip_threshold}"
                    )
                continue

            # Walk top-k candidates, penalize the first hallucinatory one.
            for token_id in top_k_indices:
                token_id_item = token_id.item()

                if self.hallucination_detector.check_hallucination(
                    current_text, token_id_item, context_window,
                ):
                    scores[batch_idx][token_id_item] = self.penalty_value
                    self.modifications_count += 1
                    if self.debug_print:
                        token_str = self.hallucination_detector.tokenizer.decode(
                            [token_id_item]
                        )
                        print(
                            f"----------Modified logit for token: "
                            f"{token_str}----------"
                        )
                    continue

                if self.debug_print:
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
                        f"({chosen_prob:.4f}) | Generated so far: "
                        f"{repr(generated_text)}"
                    )

                # Greedy break after first non-hallucinatory candidate.
                break

        return scores