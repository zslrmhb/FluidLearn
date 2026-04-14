from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from core.types import ExamplePair
from core.representations.base import AbstractionSet
from core.taxonomy import get_benchmark_rule_space

@dataclass
class ValidationResult:
    is_valid: bool
    matched_rules: List[str]
    rejection_reason: str
    num_examples_needed: int = 0
    ood_divergence: bool = False

class AmbiguityValidator:
    def __init__(self, abstraction: AbstractionSet):
        self.abstraction = abstraction
        self.rule_space = get_benchmark_rule_space(abstraction)

    def validate_identifiability(
        self, 
        support_pool: List[ExamplePair], 
        probe: ExamplePair,
        max_budget: Optional[int] = None
    ) -> ValidationResult:
        """Verify that the support pool uniquely identifies the rule among the taxonomy.
        
        Args:
            support_pool: Examples of (inp, out) provided as evidence.
            probe: The held-out query to test prediction (should be OOD).
            max_budget: Optional constraint on how many examples to consider.
        """
        budget = max_budget if max_budget is not None else len(support_pool)
        sub_pool = support_pool[:budget]
        
        # 1. Find all rules that match the support pool
        matched_rules = []
        for rule_name, rule_fn in self.rule_space.items():
            try:
                # Rule matches only if it perfectly explains EVERY pair in sub_pool
                if all(self.abstraction.render(rule_fn(pair.inp)) == pair.out for pair in sub_pool):
                    matched_rules.append(rule_name)
            except Exception:
                continue # Ignore rules that crash on invalid inputs
        
        # 2. Check for functional equivalence (Rule Classes)
        # Some rules might match the support but differ on the probe (identifiability)
        if len(matched_rules) > 1:
            # Check if they all predict the same output for the probe
            probe_outputs = set()
            for name in matched_rules:
                try:
                    p_out = self.abstraction.render(self.rule_space[name](probe.inp))
                    probe_outputs.add(p_out)
                except Exception:
                    probe_outputs.add("ERROR")
            
            # If all matching rules predict the same probe output, it's 'weakly' identified
            # (they form an equivalence class on the current data manifold).
            # But the user requested STRICT uniqueness to maintain rigor.
            if len(probe_outputs) > 1:
                return ValidationResult(
                    is_valid=False,
                    matched_rules=matched_rules,
                    rejection_reason=f"Ambiguity: {len(matched_rules)} rules fit support, and divergent on probe.",
                    ood_divergence=True
                )
            else:
                # All match support AND probe identically. This is an equivalence class.
                return ValidationResult(
                    is_valid=False, # Still flagging as invalid for strict uniqueness
                    matched_rules=matched_rules,
                    rejection_reason=f"Ambiguity: {len(matched_rules)} rules are functionally identical on this data.",
                    ood_divergence=False
                )
        
        if len(matched_rules) == 0:
            return ValidationResult(
                is_valid=False,
                matched_rules=[],
                rejection_reason="Degenerate: No benchmark rules fit the support set."
            )
            
        return ValidationResult(is_valid=True, matched_rules=matched_rules, rejection_reason="")

    def find_minimal_budget(self, support_pool: List[ExamplePair], probe: ExamplePair) -> int:
        """Find the smallest N such that N examples uniquely identify the rule."""
        for n in range(1, len(support_pool) + 1):
            res = self.validate_identifiability(support_pool, probe, max_budget=n)
            if res.is_valid:
                return n
        return len(support_pool) + 1 # Never uniquely identified
