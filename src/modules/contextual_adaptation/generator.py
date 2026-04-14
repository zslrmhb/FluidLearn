from __future__ import annotations

import random
from typing import Any

from core.contexts import CONTEXT_REGISTRY
from core.representations.base import AbstractionSet, RuleSpec
from core.sampling import sample_unique_item_for_context, sample_symbol_pool
from core.types import (
    ContextSpec,
    ContextualAdaptationDifficultyProfile,
    Episode,
    ExamplePair,
)


BANNED_CONTEXTUAL_RULE_FAMILIES = {
    "reduction",
}


def choose_context_spec(
    rng: random.Random,
    exclude_family: str | None = None
) -> ContextSpec:
    registered_families = list(CONTEXT_REGISTRY.keys())
    if exclude_family:
        registered_families = [f for f in registered_families if f != exclude_family]
        
    chosen_family = rng.choice(registered_families)
    return CONTEXT_REGISTRY[chosen_family]


def choose_candidate_template_names(
    profile: ContextualAdaptationDifficultyProfile,
    rng: random.Random,
    abstraction: AbstractionSet,
) -> list[str]:
    all_template_names = [
        name
        for name, template in abstraction.rule_templates.items()
        if template.family not in BANNED_CONTEXTUAL_RULE_FAMILIES
    ]
    k = min(profile.num_candidate_rules, len(all_template_names))
    if k <= 0:
        raise ValueError("num_candidate_rules must be positive")
    return rng.sample(all_template_names, k=k)


def build_context_to_rule_map(
    context_spec: ContextSpec,
    candidate_template_names: list[str],
    rng: random.Random,
    abstraction: AbstractionSet,
) -> dict[str, RuleSpec]:
    context_values = list(context_spec.values)
    if len(candidate_template_names) < len(context_values):
        raise ValueError(
            f"Need at least {len(context_values)} candidate rules for context family {context_spec.family}, "
            f"but got only {len(candidate_template_names)}"
        )

    chosen_templates = rng.sample(candidate_template_names, k=len(context_values))
    mapping: dict[str, RuleSpec] = {}
    used_families: set[str] = set()

    for context_value, template_name in zip(context_values, chosen_templates):
        spec = abstraction.sample_rule_spec(template_name, rng)
        if spec.family in used_families:
            raise RuntimeError("Context-to-rule mapping should use distinct rule families.")
        used_families.add(spec.family)
        mapping[context_value] = spec

    return mapping


def mutate_context_to_rule_map(
    context_spec: ContextSpec,
    context_to_rule: dict[str, RuleSpec],
    candidate_template_names: list[str],
    rng: random.Random,
    abstraction: AbstractionSet,
) -> dict[str, RuleSpec]:
    if not context_to_rule:
        return context_to_rule

    context_values = list(context_spec.values)
    new_mapping = dict(context_to_rule)
    changed_context = rng.choice(context_values)
    old_spec = new_mapping[changed_context]

    used_families_except_changed = {
        spec.family for key, spec in new_mapping.items() if key != changed_context
    }

    replacement_candidates: list[RuleSpec] = []
    for template_name in candidate_template_names:
        candidate_spec = abstraction.sample_rule_spec(template_name, rng)

        if candidate_spec.name == old_spec.name:
            continue
        if candidate_spec.family in used_families_except_changed:
            continue

        replacement_candidates.append(candidate_spec)

    if not replacement_candidates:
        raise RuntimeError("Could not sample a valid shifted context-to-rule mapping.")

    new_mapping[changed_context] = rng.choice(replacement_candidates)
    return new_mapping


def sample_shifted_environment(
    profile: ContextualAdaptationDifficultyProfile,
    context_spec: ContextSpec,
    context_to_rule: dict[str, RuleSpec],
    rng: random.Random,
    abstraction: AbstractionSet,
) -> tuple[ContextSpec, dict[str, RuleSpec], str]:
    profile_name = profile.name.lower()

    if profile_name == "easy":
        all_template_names = [
            name
            for name, template in abstraction.rule_templates.items()
            if template.family not in BANNED_CONTEXTUAL_RULE_FAMILIES
        ]
        shifted_context_to_rule = mutate_context_to_rule_map(
            context_spec=context_spec,
            context_to_rule=context_to_rule,
            candidate_template_names=all_template_names,
            rng=rng,
            abstraction=abstraction,
        )
        return context_spec, shifted_context_to_rule, "single_rule_remap_same_context"

    if profile_name == "medium":
        candidate_template_names = choose_candidate_template_names(profile, rng, abstraction)
        shifted_context_to_rule = build_context_to_rule_map(
            context_spec=context_spec,
            candidate_template_names=candidate_template_names,
            rng=rng,
            abstraction=abstraction,
        )
        return context_spec, shifted_context_to_rule, "full_rule_remap_same_context"

    if profile_name == "hard":
        shifted_context_spec = choose_context_spec(rng, exclude_family=context_spec.family)
        candidate_template_names = choose_candidate_template_names(profile, rng, abstraction)
        shifted_context_to_rule = build_context_to_rule_map(
            context_spec=shifted_context_spec,
            candidate_template_names=candidate_template_names,
            rng=rng,
            abstraction=abstraction,
        )
        return shifted_context_spec, shifted_context_to_rule, "full_context_and_rule_remap"

    raise ValueError(f"Unknown contextual-adaptation difficulty level: {profile.name}")


def build_examples(
    rng: random.Random,
    n_examples: int,
    active_symbols: list[Any],
    context_spec: ContextSpec,
    context_to_rule: dict[str, RuleSpec],
    repetition_mode: str,
    used_inputs: set[Any],
    abstraction: AbstractionSet,
    size_constraints: Any,
    return_core_inputs: bool = False,
) -> list[ExamplePair] | tuple[list[ExamplePair], list[Any]]:
    context_values = list(context_spec.values)
    examples: list[ExamplePair] = []
    core_inputs: list[Any] = []
    
    allowed_sizes = size_constraints if isinstance(size_constraints, list) else [size_constraints]

    for i in range(n_examples):
        target_context = context_values[i % len(context_values)]
        rule_spec = context_to_rule[target_context]
        rule_fn = abstraction.resolve_rule(rule_spec)

        for _attempt in range(100):
            core_x = sample_unique_item_for_context(
                rng=rng,
                abstraction=abstraction,
                pool=active_symbols,
                context_family=context_spec.family,
                target_value=target_context,
                allowed_sizes=allowed_sizes,
                repetition_mode=repetition_mode,
                used_inputs=used_inputs,
            )

            try:
                core_y = rule_fn(core_x)
                if isinstance(core_y, list) and len(core_y) == 0:
                    continue
                examples.append(
                    ExamplePair(
                        inp=abstraction.render(core_x),
                        out=abstraction.render(core_y),
                    )
                )
                core_inputs.append(core_x)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not sample enough valid contextual-adaptation examples.")

    if return_core_inputs:
        return examples, core_inputs
    return examples


def _serialize_rule_map(rule_map: dict[str, RuleSpec] | None) -> dict[str, dict[str, object]] | None:
    if rule_map is None:
        return None
    return {
        context_value: {
            "name": spec.name,
            "family": spec.family,
            "params": spec.params,
        }
        for context_value, spec in rule_map.items()
    }


def generate_contextual_adaptation_episode(
    profile: ContextualAdaptationDifficultyProfile,
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
    context_to_rule = {}
    shifted_context_to_rule = None
    shift_type = None
    context_spec = None
    shifted_context_spec = None
    candidate_template_names = []
    active_symbols = []
    size_constraints = []
    old_rule_post_query_output = None

    for _gen_attempt in range(50):
        attempt_rng = random.Random(seed + _gen_attempt)
        
        context_spec = choose_context_spec(attempt_rng)
        candidate_template_names = choose_candidate_template_names(profile, attempt_rng, abstraction)
        context_to_rule = build_context_to_rule_map(
            context_spec=context_spec,
            candidate_template_names=candidate_template_names,
            rng=attempt_rng,
            abstraction=abstraction,
        )

        shifted_context_spec = context_spec
        shifted_context_to_rule = None
        shift_type = None
        if profile.has_shift:
            shifted_context_spec, shifted_context_to_rule, shift_type = sample_shifted_environment(
                profile=profile,
                context_spec=context_spec,
                context_to_rule=context_to_rule,
                rng=attempt_rng,
                abstraction=abstraction,
            )

        if pool_override is not None:
            symbol_pool = pool_override
        else:
            symbol_pool = sample_symbol_pool(seed=seed + _gen_attempt, n=max(profile.num_active_symbols, 8))
            
        active_symbols = symbol_pool[: profile.num_active_symbols]

        pre_used_inputs: set[Any] = set()
        post_used_inputs: set[Any] = set()

        size_constraints = size_override if size_override is not None else profile.sequence_lengths
        if isinstance(size_constraints, int):
            support_lengths = [size_constraints]
        else:
            support_lengths = list(size_constraints)

        # Ensure OOD Probe Length
        all_possible_lengths = list(range(max(1, min(support_lengths)-1), max(support_lengths)+3))
        potential_probe_lengths = [slen for slen in all_possible_lengths if slen not in support_lengths]
        probe_length = attempt_rng.choice(potential_probe_lengths) if potential_probe_lengths else (max(support_lengths) + 1)

        try:
            support_pool = build_examples(
                rng=attempt_rng,
                n_examples=profile.support_pool_size,
                active_symbols=active_symbols,
                context_spec=context_spec,
                context_to_rule=context_to_rule,
                repetition_mode=profile.repetition_mode,
                used_inputs=pre_used_inputs,
                abstraction=abstraction,
                size_constraints=support_lengths,
            )

            queries = build_examples(
                rng=attempt_rng,
                n_examples=profile.query_count,
                active_symbols=active_symbols,
                context_spec=context_spec,
                context_to_rule=context_to_rule,
                repetition_mode=profile.repetition_mode,
                used_inputs=pre_used_inputs,
                abstraction=abstraction,
                size_constraints=support_lengths,
            )

            probes = build_examples(
                rng=attempt_rng,
                n_examples=1,
                active_symbols=active_symbols,
                context_spec=context_spec,
                context_to_rule=context_to_rule,
                repetition_mode=profile.repetition_mode,
                used_inputs=set(),
                abstraction=abstraction,
                size_constraints=probe_length,
            )

            if shifted_context_to_rule is not None:
                post_support_pool = build_examples(
                    rng=attempt_rng,
                    n_examples=profile.post_support_pool_size,
                    active_symbols=active_symbols,
                    context_spec=shifted_context_spec,
                    context_to_rule=shifted_context_to_rule,
                    repetition_mode=profile.repetition_mode,
                    used_inputs=post_used_inputs,
                    abstraction=abstraction,
                    size_constraints=support_lengths,
                )

                post_queries, post_query_cores = build_examples(
                    rng=attempt_rng,
                    n_examples=profile.post_query_count,
                    active_symbols=active_symbols,
                    context_spec=shifted_context_spec,
                    context_to_rule=shifted_context_to_rule,
                    repetition_mode=profile.repetition_mode,
                    used_inputs=post_used_inputs,
                    abstraction=abstraction,
                    size_constraints=support_lengths,
                    return_core_inputs=True,
                )

                post_probes = build_examples(
                    rng=attempt_rng,
                    n_examples=1,
                    active_symbols=active_symbols,
                    context_spec=shifted_context_spec,
                    context_to_rule=shifted_context_to_rule,
                    repetition_mode=profile.repetition_mode,
                    used_inputs=set(),
                    abstraction=abstraction,
                    size_constraints=probe_length,
                )

                if post_queries:
                    from core.contexts import eval_context
                    try:
                        old_context_val = eval_context(post_query_cores[0], context_spec.family)
                        old_rule_spec = context_to_rule[old_context_val]
                        old_rule_fn = abstraction.resolve_rule(old_rule_spec)
                        old_core_y = old_rule_fn(post_query_cores[0])
                        old_rule_post_query_output = abstraction.render(old_core_y)
                    except Exception:
                        old_rule_post_query_output = None
            
            # SUCCESS
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("Could not sample a contextual adaptation episode.")

    return Episode(
        module="contextual_adaptation",
        task_name=f"contextual_adaptation_{representation_name}",
        representation=representation_name,  # type: ignore
        difficulty=profile.name,
        support_pool=support_pool,
        queries=queries,
        probes=probes,
        shift_type=shift_type,
        post_support_pool=post_support_pool,
        post_queries=post_queries,
        post_probes=post_probes,
        metadata={
            "evidence_budgets": profile.evidence_budgets,
            "context_family": context_spec.family if context_spec else None,
            "context_values": list(context_spec.values) if context_spec else [],
            "candidate_template_names": candidate_template_names,
            "context_to_rule": _serialize_rule_map(context_to_rule),
            "shifted_context_family": shifted_context_spec.family if shifted_context_spec else None,
            "shifted_context_to_rule": _serialize_rule_map(shifted_context_to_rule),
            "num_active_symbols": profile.num_active_symbols,
            "num_candidate_rules": profile.num_candidate_rules,
            "num_context_families": profile.num_context_families,
            "num_context_values": profile.num_context_values,
            "repetition_mode": profile.repetition_mode,
            "old_rule_post_query_output": old_rule_post_query_output,
        },
    )
