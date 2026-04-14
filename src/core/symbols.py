from __future__ import annotations

import random

from typing import Optional

"""String symbol generation utilities.

This module provides the minimal string-side primitives for generating pseudo-word
symbol pools that can be reused by higher-level task generators.
"""

# Allowed consonant-like chunks used in pseudo-word construction.
ONSETS = [
    "b", "d", "f", "g", "k", "l", "m", "n", "p", "r",
    "s", "t", "v", "z", "sh", "ch",
]

# Allowed vowel slots used in pseudo-word construction.
VOWELS = ["a", "e", "i", "o", "u"]

# Simple structural templates for generating pronounceable pseudo-words.
SYMBOL_PATTERNS = [
    "CVC",
    "CVCV",
    "CVCCV",
]


def make_rng(seed: int) -> random.Random:
    """Create a deterministic random number generator for reproducible sampling."""
    return random.Random(seed)


def distinct_symbol_subset(rng: random.Random, pool: list[str], n: int) -> list[str]:
    """Pick a diverse subset of symbols from a larger pool.

    The heuristic tries to avoid selecting symbols that share both the same first
    and last character pattern too often.
    """
    items = list(pool)
    rng.shuffle(items)

    chosen: list[str] = []
    firsts: set[str] = set()
    lasts: set[str] = set()

    for sym in items:
        if len(chosen) == n:
            break

        # Prefer symbols that introduce some edge-character diversity.
        if sym[0] in firsts and sym[-1] in lasts:
            continue

        chosen.append(sym)
        firsts.add(sym[0])
        lasts.add(sym[-1])

    if len(chosen) < n:
        # Backfill if the diversity heuristic was too strict.
        remaining = [x for x in items if x not in chosen]
        chosen.extend(remaining[: n - len(chosen)])

    return chosen