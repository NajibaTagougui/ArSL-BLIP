"""
Template Constraint for Controlled Generation

Prevents mode collapse by forcing the model to generate outputs
in the form: "arabic sign language letter <name>"

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026
"""

import torch
from transformers import LogitsProcessor
from typing import List


TEMPLATE_PREFIX = "arabic sign language letter"


class TemplateConstraintLogitsProcessor(LogitsProcessor):
    """Force the decoder to start with the template prefix.

    This is the key contribution that raises accuracy from 22.1% to 92.19%.
    By constraining the output space to a fixed prefix, the model is guided
    to produce valid Arabic letter names instead of free-form captions.
    """

    def __init__(self, prefix_token_ids: List[int], eos_token_id: int):
        """
        Args:
            prefix_token_ids: Token IDs of the template prefix.
            eos_token_id: End-of-sequence token ID.
        """
        self.prefix_ids = prefix_token_ids
        self.prefix_len = len(prefix_token_ids)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """At each decoding step, enforce template prefix tokens."""
        current_len = input_ids.shape[1]

        # During prefix phase, force the next required token
        if current_len < self.prefix_len:
            required_token = self.prefix_ids[current_len]
            # Set all logits to -inf except the required token
            mask = torch.full_like(scores, float("-inf"))
            mask[:, required_token] = 0.0
            return scores + mask

        # After prefix, allow free generation
        return scores


def build_template_processor(processor, template: str = TEMPLATE_PREFIX):
    """Build the prefix token IDs from a processor tokenizer."""
    tokenizer = processor.tokenizer
    prefix_ids = tokenizer.encode(template, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    return TemplateConstraintLogitsProcessor(prefix_ids, eos_id)
