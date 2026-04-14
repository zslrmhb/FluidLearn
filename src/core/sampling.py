import random
from typing import Any, Optional

from core.symbols import ONSETS, VOWELS, SYMBOL_PATTERNS, make_rng, distinct_symbol_subset
from .contexts import eval_context
from .representations.base import AbstractionSet

def _sample_from_slot(rng: random.Random, slot: str) -> str:
    if slot == "C":
        return rng.choice(ONSETS)
    if slot == "V":
        return rng.choice(VOWELS)
    raise ValueError(f"Unknown slot type: {slot}")

def sample_pseudoword(rng: random.Random, pattern: Optional[str] = None) -> str:
    if pattern is None:
        pattern = rng.choice(SYMBOL_PATTERNS)
    return "".join(_sample_from_slot(rng, ch) for ch in pattern)

def sample_unique_pseudowords(
    rng: random.Random,
    n: int,
    min_len: int = 3,
    max_tries: int = 10000,
) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    tries = 0
    while len(results) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Could not generate enough unique pseudo-words.")
        word = sample_pseudoword(rng)
        if len(word) < min_len or word in seen:
            continue
        seen.add(word)
        results.append(word)
    return results

def sample_symbol_pool(seed: int, n: int) -> list[str]:
    rng = make_rng(seed)
    pool = sample_unique_pseudowords(rng=rng, n=max(n, 30))
    return distinct_symbol_subset(rng=rng, pool=pool, n=n)

def _make_hashable(item: Any) -> Any:
    if isinstance(item, list):
        if len(item) > 0 and isinstance(item[0], list):
            return tuple(tuple(row) for row in item)
        return tuple(item)
    return item

def sample_unique_core_item(
    rng: random.Random,
    abstraction: AbstractionSet,
    pool: list[Any],
    size_constraints: Any,
    repetition_mode: str,
    used_inputs: set[Any],
) -> Any:
    for _ in range(200):
        length = rng.choice(size_constraints) if isinstance(size_constraints, list) else size_constraints
        
        item = abstraction.sample_input(
            rng=rng,
            pool=pool,
            size_constraints=length,
            repetition_mode=repetition_mode,
        )
        key = _make_hashable(item)
        if key not in used_inputs:
            used_inputs.add(key)
            return item
    raise RuntimeError("Could not sample enough unique core items.")


def sample_unique_majority_item(
    rng: random.Random,
    abstraction: AbstractionSet,
    pool: list[Any],
    size_constraints: Any,
    used_inputs: set[Any],
) -> Any:
    # Handle GridAbstractionSet detection via its class name or checking the type
    # Since sampling.py is in core/, we can import from core.representations
    # but to avoid some circularity at top level, keeping it here.
    from .representations import GridAbstractionSet
    is_grid_abstraction = isinstance(abstraction, GridAbstractionSet)
    
    length = size_constraints
    # size can be a tuple (R,C) or int (total cells) for grids
    is_tuple_size = isinstance(length, tuple)
    
    if is_grid_abstraction:
        if is_tuple_size:
            R, C = length
        else:
            # Replicate GridAbstractionSet.sample_input's factor-splitting logic or simply use 1xN
            R = 1
            C = length
        target_len = R * C
    else:
        target_len = length
    
    for _ in range(200):
        majority_token = rng.choice(pool)
        majority_count = (target_len // 2) + 1
        seq = [majority_token] * majority_count
        remaining = target_len - majority_count
        
        non_majority_pool = [token for token in pool if token != majority_token]
        if not non_majority_pool:
            non_majority_pool = [majority_token]
            
        seq.extend(rng.choice(non_majority_pool) for _ in range(remaining))
        rng.shuffle(seq)
        
        if is_grid_abstraction:
            item = [seq[i * C:(i + 1) * C] for i in range(R)]
            item_hashable = tuple(tuple(r) for r in item)
        else:
            item = seq
            item_hashable = tuple(item)
             
        if item_hashable not in used_inputs:
            used_inputs.add(item_hashable)
            return item
    raise RuntimeError("Could not sample unique majority-valid sequences.")


def sample_unique_item_for_context(
    rng: random.Random,
    abstraction: AbstractionSet,
    pool: list[Any],
    context_family: str,
    target_value: str,
    allowed_sizes: list[Any],
    repetition_mode: str,
    used_inputs: set[Any],
    total_max_tries: int = 5000,
) -> Any:
    # Contexts are mainly for 1D sequences right now.
    for i in range(total_max_tries):
        length = rng.choice(allowed_sizes)
        
        # Heuristic: if we fail many times, relax repetition mode
        current_rep = repetition_mode
        if i > total_max_tries // 4:
            current_rep = "medium"
        if i > total_max_tries // 2:
            current_rep = "high"

        item = abstraction.sample_input(
            rng=rng,
            pool=pool,
            size_constraints=length,
            repetition_mode=current_rep,
        )
        key = _make_hashable(item)
        if key in used_inputs:
            continue
        if eval_context(item, context_family) == target_value:
            used_inputs.add(key)
            return item
    raise RuntimeError(
        f"Could not sample enough unique sequences for context_family={context_family}, target_value={target_value}"
    )