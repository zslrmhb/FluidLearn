from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Callable

# Generic rule types
RuleFn = Callable[[Any], Any]


class RuleTemplate:
    """A template for sampling parameterized rules."""
    def __init__(self, name: str, family: str, sample_params: Callable[[random.Random], dict[str, Any]]):
        self.name = name
        self.family = family
        self.sample_params = sample_params


class RuleSpec:
    """A concrete instantiated rule specification."""
    def __init__(self, name: str, family: str, params: dict[str, Any]):
        self.name = name
        self.family = family
        self.params = params


class AbstractionSet(ABC):
    """Top-level abstract interface for the benchmark's structural abstraction set."""

    @property
    @abstractmethod
    def rule_templates(self) -> dict[str, RuleTemplate]:
        """Expose the sampleable rule templates for this representation."""
        raise NotImplementedError

    @abstractmethod
    def resolve_rule(self, spec: RuleSpec) -> RuleFn:
        """Resolve one concrete instantiated rule specification to a callable."""
        raise NotImplementedError

    @abstractmethod
    def build_rule(self, family: str, **params: Any) -> RuleFn:
        """Build a parameterized rule directly from a rule family plus arguments."""
        raise NotImplementedError

    def sample_rule_spec(self, template_name: str, rng: random.Random) -> RuleSpec:
        """Sample one concrete rule from a named rule template."""
        if template_name not in self.rule_templates:
            raise KeyError(f"Unknown rule template: {template_name}")
        template = self.rule_templates[template_name]
        params = template.sample_params(rng)
        
        if params:
            suffix = "_".join(f"{key}_{value}" for key, value in sorted(params.items()))
            name = f"{template.name}_{suffix}"
        else:
            name = template.name
            
        return RuleSpec(name=name, family=template.family, params=params)

    @abstractmethod
    def render(self, x: Any) -> str:
        """Render one representation-specific object into a human-readable string."""
        raise NotImplementedError

    @abstractmethod
    def sample_input(
        self,
        *,
        rng: random.Random,
        pool: list[Any],
        size_constraints: Any,
        **kwargs: Any
    ) -> Any:
        """Sample one input representation from a pool with given constraints."""
        raise NotImplementedError

    @abstractmethod
    def soft_score(self, gold: Any, pred: str) -> float:
        """Calculate a similarity score [0, 1] between the gold object and a predicted string."""
        raise NotImplementedError


class SequenceAbstractionSet(AbstractionSet):
    """Intermediate abstract base class for 1D sequence-like representations."""

    @abstractmethod
    def sequence_to_text(self, seq: list[Any]) -> str:
        """Render a sequence into a human-readable string."""
        raise NotImplementedError

    def render(self, x: Any) -> str:
        return self.sequence_to_text(x)