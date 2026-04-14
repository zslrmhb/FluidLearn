from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from .types import ContextSpec


class ContextBackend(ABC):
    """Representation-specific context interface.

    Each backend implements:
    - how to evaluate a structural context for one input
    - how to sample an input satisfying a target context value
    """

    @abstractmethod
    def eval_context(self, x: Any, context_family: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def sample_input_for_context(
        self,
        rng: random.Random,
        target_value: str,
        context_family: str,
        allowed_sizes: list[int],
        used_inputs: set,
        **kwargs,
    ) -> Any:
        raise NotImplementedError


def eval_size_parity(seq: list[str]) -> str:
    return "odd" if len(seq) % 2 == 1 else "even"


def eval_repetition_presence(seq: list[str]) -> str:
    return "repeated" if len(set(seq)) < len(seq) else "unique"


def eval_boundary_relation(seq: list[str]) -> str:
    if not seq:
        return "different"
    return "equal" if seq[0] == seq[-1] else "different"


def eval_majority_existence(seq: list[str]) -> str:
    if not seq:
        return "no_majority"

    counts: dict[str, int] = {}
    for token in seq:
        counts[token] = counts.get(token, 0) + 1

    best_count = max(counts.values())
    num_best = sum(1 for c in counts.values() if c == best_count)

    if num_best != 1:
        return "no_majority"

    return "has_majority" if best_count > len(seq) // 2 else "no_majority"


def eval_local_pattern_consistency(seq: list[str]) -> str:
    if len(seq) < 4:
        return "inconsistent"

    for i in range(2, len(seq)):
        if seq[i] != seq[i - 2]:
            return "inconsistent"

    return "consistent"


CONTEXT_REGISTRY: dict[str, ContextSpec] = {
    "size_parity": ContextSpec(
        family="size_parity",
        values=["odd", "even"],
    ),
    "repetition_presence": ContextSpec(
        family="repetition_presence",
        values=["repeated", "unique"],
    ),
    "boundary_relation": ContextSpec(
        family="boundary_relation",
        values=["equal", "different"],
    ),
    "majority_existence": ContextSpec(
        family="majority_existence",
        values=["has_majority", "no_majority"],
    ),
    "local_pattern_consistency": ContextSpec(
        family="local_pattern_consistency",
        values=["consistent", "inconsistent"],
    ),
}


CONTEXT_EVALUATORS = {
    "size_parity": eval_size_parity,
    "repetition_presence": eval_repetition_presence,
    "boundary_relation": eval_boundary_relation,
    "majority_existence": eval_majority_existence,
    "local_pattern_consistency": eval_local_pattern_consistency,
}


def eval_context(item: Any, context_family: str) -> str:
    """Evaluate one abstract structural context on a string sequence or grid."""
    if context_family not in CONTEXT_EVALUATORS:
        raise KeyError(f"Unknown context family: {context_family}")
    
    # Flatten grid to 1D for generic context logic
    if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
        flattened = [str(x) for row in item for x in row]
    else:
        flattened = [str(x) for x in item]
        
    return CONTEXT_EVALUATORS[context_family](flattened)


def sample_context_spec(rng: random.Random) -> ContextSpec:
    return rng.choice(list(CONTEXT_REGISTRY.values()))