from __future__ import annotations
from typing import Literal, Any, Callable, Optional
from dataclasses import dataclass, field

import random

# Coarse difficulty label used for reporting and config lookup.
Difficulty = Literal["easy", "medium", "hard"]

# Benchmark capability modules. These describe what is being measured.
ModuleName = Literal[
    "symbolic_binding",
    "compositional_induction",
    "contextual_adaptation",
    "feedback_exploration",
]

# Surface representation. Keep this generic so later we can plug in numeric/spatial backends.
Representation = Literal["string", "number", "grid"]



# One input/output example pair.
@dataclass
class ExamplePair:
    inp: Any
    out: Any


@dataclass(frozen=True)
class RuleSpec:
    """Concrete instantiated rule used by generation, logging, and resolution."""

    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextSpec:
    """Abstract context family specification.

    `family` is representation-agnostic (for example `size_parity`), while
    `values` enumerates the allowed context buckets for that family.
    """

    family: str
    values: list[str]


@dataclass(frozen=True)
class RuleTemplate:
    """Sampleable rule family specification.

    A template defines:
    - the rule family,
    - how its parameters are sampled,
    - how a concrete sampled rule should be named.
    """

    name: str
    family: str
    sample_params: Callable[[random.Random], dict[str, Any]]



# One learning episode.
#
# support_pool / queries:
#   Pre-shift examples and evaluation queries.
#
# post_support_pool / post_queries:
#   Optional post-shift examples and evaluation queries after a rule change.
#
# shift_type:
#   A lightweight label describing what changed, if anything.
@dataclass
class Episode:
    module: ModuleName
    task_name: str
    representation: Representation
    difficulty: Difficulty

    support_pool: list[ExamplePair]
    queries: list[ExamplePair]
    probes: list[ExamplePair] = field(default_factory=list)

    shift_type: str | None = None
    post_support_pool: list[ExamplePair] = field(default_factory=list)
    post_queries: list[ExamplePair] = field(default_factory=list)
    post_probes: list[ExamplePair] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DIFFICULTY & CONFIGURATION
# =============================================================================

@dataclass
class DifficultyProfile:
    """Base configuration for benchmark task difficulty.
    
    Contains generic controls shared across all cognitive modules.
    Subclasses add module-specific knobs (e.g. chaining steps, hint flags).
    """
    # Difficulty name used in reports (easy, medium, hard).
    name: Difficulty

    # Prefix budgets for incremental support exposure, e.g. [1, 2, 3, 4, 5].
    # Essential for measuring learning curves.
    evidence_budgets: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    # Pre-shift task parameters
    support_pool_size: int = 5
    query_count: int = 1

    # Number of distinct symbols available for generating sequences/grids.
    num_active_symbols: int = 3
    
    # Number of latent rules or candidate transformations available in the search space.
    num_candidate_rules: int = 1

    # Sampling mode for repetition density (low, medium, high).
    repetition_mode: str = "low"

    # Allowed input sizes (e.g. sequence lengths or grid area).
    sequence_lengths: list[int] = field(default_factory=lambda: [3])

    # Rule shift 
    has_shift: bool = True
    post_support_pool_size: int = 5
    post_query_count: int = 1


@dataclass
class SymbolicBindingDifficultyProfile(DifficultyProfile):
    """Configuration for Module I: Symbolic Mapping tasks."""
    # Whether irrelevant/noisy symbols can appear in the input sequence.
    has_distractors: bool = False
    distractor_count: int = 0


@dataclass
class CompositionalInductionDifficultyProfile(DifficultyProfile):
    """Configuration for Module II: Rule-Chaining tasks."""
    # Number of atomic operations chained together in the latent program.
    num_steps: int = 1


@dataclass
class ContextualAdaptationDifficultyProfile(DifficultyProfile):
    """Configuration for Module III: Structural Context tasks."""
    # How many distinct context families are eligible to be sampled.
    num_context_families: Optional[int] = None

    # Number of distinct context labels mapped to rules (typically 2).
    num_context_values: int = 2
    
    # Contextual Adaptation sequence lengths are robust at [3, 4]
    sequence_lengths: list[int] = field(default_factory=lambda: [3, 4])


@dataclass
class FeedbackExplorationDifficultyProfile(DifficultyProfile):
    """Configuration for Module IV: Feedback-based exploration tasks."""
    # Granular hint controls for layered feedback.
    allow_shape_hints: bool = True
    allow_structure_hints: bool = True
    allow_semantic_hints: bool = True
    
    # Feedback Exploration sequence lengths are robust at [3, 4]
    sequence_lengths: list[int] = field(default_factory=lambda: [3, 4])