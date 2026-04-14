from __future__ import annotations

import random
from typing import Any

from core.representations.base import AbstractionSet, RuleSpec
from core.sampling import sample_unique_core_item, sample_symbol_pool
from core.types import Episode, ExamplePair, SymbolicBindingDifficultyProfile


BANNED_BINDING_FAMILIES = {
    "reduction",
}


def choose_candidate_template_names(
    profile: SymbolicBindingDifficultyProfile,
    rng: random.Random,
    abstraction: AbstractionSet,
) -> list[str]:
    all_template_names = [
        name
        for name, template in abstraction.rule_templates.items()
        if template.family not in BANNED_BINDING_FAMILIES
    ]
    k = min(profile.num_candidate_rules, len(all_template_names))
    if k <= 0:
        raise ValueError("num_candidate_rules must be positive")
    return rng.sample(all_template_names, k=k)


def build_examples(
    rng: random.Random,
    n_examples: int,
    active_symbols: list[Any],
    repetition_mode: str,
    rule_spec: RuleSpec,
    used_inputs: set[Any],
    abstraction: AbstractionSet,
    size_constraints: Any,
) -> list[ExamplePair]:
    rule_fn = abstraction.resolve_rule(rule_spec)
    examples: list[ExamplePair] = []

    for _ in range(n_examples):
        for _attempt in range(100):
            core_x = sample_unique_core_item(
                rng=rng,
                abstraction=abstraction,
                pool=active_symbols,
                size_constraints=size_constraints,
                repetition_mode=repetition_mode,
                used_inputs=used_inputs,
            )
            try:
                core_y = rule_fn(core_x)
                examples.append(
                    ExamplePair(
                        inp=abstraction.render(core_x),
                        out=abstraction.render(core_y),
                    )
                )
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not sample enough valid symbolic-binding examples.")

    return examples


def generate_symbolic_binding_episode(
    profile: SymbolicBindingDifficultyProfile,
    seed: int,
    abstraction: AbstractionSet,
    representation_name: str = "string",
    pool_override: list[Any] | None = None,
    size_override: Any | None = None,
) -> Episode:
    # Initialize all potential episode components
    support_pool = []
    queries = []
    probes = []
    post_support_pool = []
    post_queries = []
    post_probes = []
    rule_spec = None
    shifted_rule_spec = None
    candidate_template_names = []
    active_symbols = []
    size_constraints = []
    old_rule_post_query_output = None

    for _gen_attempt in range(50):
        attempt_rng = random.Random(seed + _gen_attempt)
        
        if pool_override is not None:
            symbol_pool = pool_override
        else:
            symbol_pool = sample_symbol_pool(seed=seed + _gen_attempt, n=max(profile.num_active_symbols, 8))
            
        active_symbols = symbol_pool[: profile.num_active_symbols]

        size_constraints = size_override if size_override is not None else profile.sequence_lengths
        if isinstance(size_constraints, int):
            support_lengths = [size_constraints]
        else:
            support_lengths = list(size_constraints)

        # Ensure OOD Probe Length
        all_possible_lengths = list(range(max(1, min(support_lengths)-1), max(support_lengths)+3))
        potential_probe_lengths = [slen for slen in all_possible_lengths if slen not in support_lengths]
        probe_length = attempt_rng.choice(potential_probe_lengths) if potential_probe_lengths else (max(support_lengths) + 1)

        candidate_template_names = choose_candidate_template_names(profile, attempt_rng, abstraction)
        pre_template_name = attempt_rng.choice(candidate_template_names)
        rule_spec = abstraction.sample_rule_spec(pre_template_name, attempt_rng)

        pre_used_inputs: set[Any] = set()
        post_used_inputs: set[Any] = set()

        try:
            support_pool = build_examples(
                rng=attempt_rng,
                n_examples=profile.support_pool_size,
                active_symbols=active_symbols,
                repetition_mode=profile.repetition_mode,
                rule_spec=rule_spec,
                used_inputs=pre_used_inputs,
                abstraction=abstraction,
                size_constraints=support_lengths,
            )

            queries = build_examples(
                rng=attempt_rng,
                n_examples=profile.query_count,
                active_symbols=active_symbols,
                repetition_mode=profile.repetition_mode,
                rule_spec=rule_spec,
                used_inputs=pre_used_inputs,
                abstraction=abstraction,
                size_constraints=support_lengths,
            )

            probes = build_examples(
                rng=attempt_rng,
                n_examples=1,
                active_symbols=active_symbols,
                repetition_mode=profile.repetition_mode,
                rule_spec=rule_spec,
                used_inputs=set(),
                abstraction=abstraction,
                size_constraints=probe_length,
            )

            if profile.has_shift:
                potential_shifted_templates = [t for t in candidate_template_names if t != pre_template_name]
                if not potential_shifted_templates:
                    potential_shifted_templates = candidate_template_names
                
                shifted_template_name = attempt_rng.choice(potential_shifted_templates)
                shifted_rule_spec = abstraction.sample_rule_spec(shifted_template_name, attempt_rng)

                post_support_pool = build_examples(
                    rng=attempt_rng,
                    n_examples=profile.post_support_pool_size,
                    active_symbols=active_symbols,
                    repetition_mode=profile.repetition_mode,
                    rule_spec=shifted_rule_spec,
                    used_inputs=post_used_inputs,
                    abstraction=abstraction,
                    size_constraints=support_lengths,
                )

                post_queries = build_examples(
                    rng=attempt_rng,
                    n_examples=profile.post_query_count,
                    active_symbols=active_symbols,
                    repetition_mode=profile.repetition_mode,
                    rule_spec=shifted_rule_spec,
                    used_inputs=post_used_inputs,
                    abstraction=abstraction,
                    size_constraints=support_lengths,
                )

                post_probes = build_examples(
                    rng=attempt_rng,
                    n_examples=1,
                    active_symbols=active_symbols,
                    repetition_mode=profile.repetition_mode,
                    rule_spec=shifted_rule_spec,
                    used_inputs=set(),
                    abstraction=abstraction,
                    size_constraints=probe_length,
                )
            # SUCCESS
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("Could not sample a symbolic binding episode.")

    return Episode(
        module="symbolic_binding",
        task_name=f"symbolic_binding_{representation_name}",
        representation=representation_name,  # type: ignore
        difficulty=profile.name,
        support_pool=support_pool,
        queries=queries,
        probes=probes,
        shift_type="full_rule_remap" if shifted_rule_spec else None,
        post_support_pool=post_support_pool,
        post_queries=post_queries,
        post_probes=post_probes,
        metadata={
            "evidence_budgets": profile.evidence_budgets,
            "num_active_symbols": profile.num_active_symbols,
            "candidate_template_names": candidate_template_names,
            "template": rule_spec.name if rule_spec else None,
            "template_family": rule_spec.family if rule_spec else None,
            "template_params": rule_spec.params if rule_spec else None,
            "shifted_template": shifted_rule_spec.name if shifted_rule_spec else None,
            "shifted_template_family": shifted_rule_spec.family if shifted_rule_spec else None,
            "shifted_template_params": shifted_rule_spec.params if shifted_rule_spec else None,
            "has_distractors": getattr(profile, "has_distractors", False),
            "repetition_mode": profile.repetition_mode,
            "old_rule_post_query_output": old_rule_post_query_output,
        },
    )
