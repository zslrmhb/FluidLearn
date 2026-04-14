from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.representations.base import RuleSpec


@dataclass(frozen=True)
class FeedbackPolicy:
    """Controls how the environment responds to wrong answers during exploration.

    Attributes:
        max_rounds_per_shift: Maximum number of feedback rounds allowed before a rule shift or probe.
        binary_feedback_only: If True, only 'Correct' or 'Incorrect' is returned.
        allow_shape_hints: If True, feedback can include information about expected output length/dimensions.
        allow_structure_hints: If True, feedback can specify which indices or cells are correct.
        allow_semantic_hints: If True, feedback can provide hints about the rule family.
    """
    max_rounds_per_shift: int = 5
    binary_feedback_only: bool = False
    allow_shape_hints: bool = True
    allow_structure_hints: bool = True
    allow_semantic_hints: bool = True


@dataclass(frozen=True)
class FeedbackTaskConfig:
    """Static configuration for a feedback exploration task instance.

    Defines the initial state, the latent rules (pre/post shift), and the specific
    queries used for interaction and probing.
    """
    module: str
    task_name: str
    representation: str
    difficulty: str
    task_instruction: str
    symbol_pool: list[Any]

    rule_spec: RuleSpec
    shifted_rule_spec: Optional[RuleSpec] = None

    interaction_query: Any = None
    shifted_interaction_query: Any = None
    probe_query: Any = None
    shifted_probe_query: Any = None
    
    toolbox: dict[str, RuleSpec] = field(default_factory=dict)
    
    feedback_policy: FeedbackPolicy = field(default_factory=FeedbackPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackAction:
    """An agent's interaction with the environment."""
    answer: Any


@dataclass
class QueryState:
    """Tracking state for a specific query during interactive learning."""
    inp: Any
    gold: Any
    rounds_used: int = 0
    solved: bool = False
    last_feedback: Optional[str] = None
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FeedbackGameState:
    """Top-level state for the feedback exploration session."""
    phase: str = "pre_interaction"
    finished: bool = False
