from __future__ import annotations

import random
from functools import partial
from typing import Any

from core.representations.base import SequenceAbstractionSet, RuleTemplate, RuleSpec
from core.symbols import distinct_symbol_subset, make_rng
from core.sampling import sample_unique_pseudowords

# --------------------------------------------------
# String-specific primitive implementations
# --------------------------------------------------

def sequence_to_text(seq: list[str]) -> str:
    """Render a token sequence as a space-separated string."""
    return " ".join(seq)


# P1: Permutation
def permutation_reverse(seq: list[str]) -> list[str]:
    return list(reversed(seq))


# P2: Reindexing
def reindexing_rotate(seq: list[str], k: int = 1) -> list[str]:
    if not seq:
        return []
    k = k % len(seq)
    return seq[-k:] + seq[:-k] if k else list(seq)


# P3: Iteration
def iteration_duplicate(seq: list[str]) -> list[str]:
    return seq + seq


# P4: Partitioning
def partitioning_by_offset(seq: list[str], offset: int = 0) -> list[str]:
    return seq[offset::2]


# P5: Extension
def extension_append_boundary(seq: list[str], side: str = "last") -> list[str]:
    if not seq:
        return []
    if side == "first":
        return seq + [seq[0]]
    if side == "last":
        return seq + [seq[-1]]
    raise ValueError(f"Unknown boundary side: {side}")


# P6: Reduction
def reduction_majority(seq: list[str]) -> list[str]:
    if not seq:
        return []
    counts: dict[str, int] = {}
    for token in seq:
        counts[token] = counts.get(token, 0) + 1
    best_token, best_count = max(counts.items(), key=lambda item: item[1])
    num_best = sum(1 for count in counts.values() if count == best_count)
    if num_best != 1:
        raise ValueError("reduction_majority requires a unique majority token")
    return [best_token]


def resolve_string_rule(spec: RuleSpec):
    if spec.family == "permutation":
        return permutation_reverse
    if spec.family == "reindexing":
        return partial(reindexing_rotate, **spec.params)
    if spec.family == "iteration":
        return iteration_duplicate
    if spec.family == "partitioning":
        return partial(partitioning_by_offset, **spec.params)
    if spec.family == "extension":
        return partial(extension_append_boundary, **spec.params)
    if spec.family == "reduction":
        return reduction_majority
    raise KeyError(f"Unknown string rule family: {spec.family}")


STRING_RULE_TEMPLATES: dict[str, RuleTemplate] = {
    "permutation": RuleTemplate(
        name="permutation_reverse",
        family="permutation",
        sample_params=lambda rng: {},
    ),
    "reindexing": RuleTemplate(
        name="reindexing_rotate",
        family="reindexing",
        sample_params=lambda rng: {"k": rng.choice([-1, 1])},
    ),
    "iteration": RuleTemplate(
        name="iteration_duplicate",
        family="iteration",
        sample_params=lambda rng: {},
    ),
    "partitioning": RuleTemplate(
        name="partitioning_by_offset",
        family="partitioning",
        sample_params=lambda rng: {"offset": rng.choice([0, 1])},
    ),
    "extension": RuleTemplate(
        name="extension_append_boundary",
        family="extension",
        sample_params=lambda rng: {"side": rng.choice(["first", "last"])},
    ),
    "reduction": RuleTemplate(
        name="reduction_majority",
        family="reduction",
        sample_params=lambda rng: {},
    ),
}


class StringAbstractionSet(SequenceAbstractionSet):
    """Concrete abstraction-set implementation for string pseudo-word sequences."""

    @property
    def rule_templates(self) -> dict[str, RuleTemplate]:
        return STRING_RULE_TEMPLATES

    def resolve_rule(self, spec: RuleSpec):
        return resolve_string_rule(spec)

    def build_rule(self, family: str, **params: Any):
        spec = RuleSpec(name=f"custom_{family}", family=family, params=params)
        return resolve_string_rule(spec)

    def sequence_to_text(self, seq: list[str]) -> str:
        return sequence_to_text(seq)

    def soft_score(self, gold: list[str], pred: str) -> float:
        """Calculate token-match similarity between gold sequence and predicted string."""
        if not pred or not isinstance(pred, str):
            return 0.0
        
        pred_tokens = pred.strip().split()
        if not pred_tokens and not gold:
            return 1.0
        if not pred_tokens or not gold:
            return 0.0
            
        # Basic matching: how many gold tokens are in the pred string (ignoring order for now)
        # or better: simple jaccard or token-level accuracy if lengths match
        matches = 0
        for g, p in zip(gold, pred_tokens):
            if g == p:
                matches += 1
        
        # Penalty for length mismatch
        length_factor = min(len(gold), len(pred_tokens)) / max(len(gold), len(pred_tokens))
        return (matches / len(gold)) * length_factor if gold else 0.0

    def sample_input(
        self,
        *,
        rng: random.Random,
        pool: list[str],
        size_constraints: int,
        repetition_mode: str = "medium",
        **kwargs: Any
    ) -> list[str]:
        length = size_constraints
        if length < 0:
            raise ValueError("length must be non-negative")
        if not pool and length > 0:
            raise ValueError("pool must be non-empty when length > 0")
        if length == 0:
            return []

        if repetition_mode == "low":
            if len(pool) >= length:
                return rng.sample(pool, length)
            return [rng.choice(pool) for _ in range(length)]
        if repetition_mode == "medium":
            return [rng.choice(pool) for _ in range(length)]
        if repetition_mode == "high":
            repeated = rng.choice(pool)
            seq = [rng.choice(pool) for _ in range(length)]
            seq[rng.randrange(length)] = repeated
            return seq

        raise ValueError(f"Unknown repetition_mode: {repetition_mode}")
