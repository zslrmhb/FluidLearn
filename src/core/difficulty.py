from __future__ import annotations

from core.types import (
    SymbolicBindingDifficultyProfile,
    CompositionalInductionDifficultyProfile,
    ContextualAdaptationDifficultyProfile,
    FeedbackExplorationDifficultyProfile,
)


# =============================================================================
# MODULE I: SYMBOLIC BINDING
# -----------------------------------------------------------------------------
# Measures simple mapping between abstract tokens.
# =============================================================================
SYMBOLIC_BINDING_DIFFICULTY = {
    "easy": SymbolicBindingDifficultyProfile(
        name="easy",
        num_active_symbols=3,
        num_candidate_rules=2,
        repetition_mode="high",
        sequence_lengths=[4, 5],
    ),
    "medium": SymbolicBindingDifficultyProfile(
        name="medium",
        num_active_symbols=4,
        num_candidate_rules=4,
        repetition_mode="medium",
        sequence_lengths=[3, 4],
    ),
    "hard": SymbolicBindingDifficultyProfile(
        name="hard",
        num_active_symbols=5,
        num_candidate_rules=6,
        repetition_mode="low",
        sequence_lengths=[2, 3],
    ),
}


# =============================================================================
# MODULE II: COMPOSITIONAL INDUCTION
# -----------------------------------------------------------------------------
# Measures the ability to induce chained operations.
# =============================================================================
COMPOSITIONAL_INDUCTION_DIFFICULTY = {
    "easy": CompositionalInductionDifficultyProfile(
        name="easy",
        num_steps=1,
        num_active_symbols=3,
        repetition_mode="high",
        num_candidate_rules=2,
        sequence_lengths=[4, 5],
    ),
    "medium": CompositionalInductionDifficultyProfile(
        name="medium",
        num_steps=2,
        num_active_symbols=3,
        repetition_mode="medium",
        num_candidate_rules=3,
        sequence_lengths=[3, 4],
    ),
    "hard": CompositionalInductionDifficultyProfile(
        name="hard",
        num_steps=2,
        num_active_symbols=5,
        repetition_mode="low",
        num_candidate_rules=5,
        sequence_lengths=[3],
    ),
}


# =============================================================================
# MODULE III: CONTEXTUAL ADAPTATION
# -----------------------------------------------------------------------------
# Measures environmental change detection and structural context mapping.
# =============================================================================
CONTEXTUAL_ADAPTATION_DIFFICULTY = {
    "easy": ContextualAdaptationDifficultyProfile(
        name="easy",
        num_active_symbols=3,
        num_candidate_rules=3,
        repetition_mode="high",
    ),
    "medium": ContextualAdaptationDifficultyProfile(
        name="medium",
        num_active_symbols=4,
        num_candidate_rules=4,
        repetition_mode="medium",
    ),
    "hard": ContextualAdaptationDifficultyProfile(
        name="hard",
        num_active_symbols=5,
        num_candidate_rules=5,
        repetition_mode="low",
    ),
}


# =============================================================================
# MODULE IV: FEEDBACK EXPLORATION
# -----------------------------------------------------------------------------
# Measures active information gathering and rule refinement through feedback.
# =============================================================================
FEEDBACK_EXPLORATION_DIFFICULTY = {
    "easy": FeedbackExplorationDifficultyProfile(
        name="easy",
        num_candidate_rules=2,
        num_active_symbols=3,
        repetition_mode="high",
        allow_structure_hints=True,
        allow_semantic_hints=True,
    ),
    "medium": FeedbackExplorationDifficultyProfile(
        name="medium",
        num_candidate_rules=3,
        num_active_symbols=4,
        repetition_mode="medium",
        allow_structure_hints=True,
        allow_semantic_hints=False,
    ),
    "hard": FeedbackExplorationDifficultyProfile(
        name="hard",
        num_candidate_rules=4,
        num_active_symbols=5,
        repetition_mode="low",
        allow_structure_hints=False,
        allow_semantic_hints=False,
    ),
}


