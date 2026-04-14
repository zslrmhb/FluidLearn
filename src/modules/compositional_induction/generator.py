from __future__ import annotations

import random
from typing import Any

from core.representations.base import AbstractionSet, RuleSpec
from core.sampling import sample_unique_core_item, sample_symbol_pool
from core.types import Episode, ExamplePair, CompositionalInductionDifficultyProfile


BANNED_COMPOSITION_FAMILIES = {
    "reduction",
}


def choose_candidate_template_names(
    profile: CompositionalInductionDifficultyProfile,
    rng: random.Random,
    abstraction: AbstractionSet,
) -> list[str]:
    all_template_names = [
        name
        for name, template in abstraction.rule_templates.items()
        if template.family not in BANNED_COMPOSITION_FAMILIES
    ]
    k = min(profile.num_candidate_rules, len(all_template_names))
    if k <= 0:
        raise ValueError("num_candidate_rules must be positive")
    return rng.sample(all_template_names, k=k)


def sample_op_sequence(
    candidate_template_names: list[str],
    num_steps: int,
    rng: random.Random,
    abstraction: AbstractionSet,
) -> list[RuleSpec]:
    specs = []
    # Avoid picking the same template twice in a row if possible
    last_template = None
    for _ in range(num_steps):
        choices = [t for t in candidate_template_names if t != last_template]
        if not choices:
            choices = candidate_template_names
        t_name = rng.choice(choices)
        specs.append(abstraction.sample_rule_spec(t_name, rng))
        last_template = t_name
    return specs


def apply_composition(
    x: Any, specs: list[RuleSpec], abstraction: AbstractionSet
) -> Any:
    current_x = x
    for spec in specs:
        rule_fn = abstraction.resolve_rule(spec)
        current_x = rule_fn(current_x)
    return current_x


def mutate_op_sequence(
    op_specs: list[RuleSpec],
    candidate_template_names: list[str],
    rng: random.Random,
    abstraction: AbstractionSet,
) -> list[RuleSpec]:
    if not op_specs:
        return op_specs
    
    new_specs = list(op_specs)
    idx_to_change = rng.randrange(len(new_specs))
    old_spec = new_specs[idx_to_change]
    
    # Simple mutation: replace one op with a different one
    potential_templates = [t for t in candidate_template_names if t != old_spec.name]
    if not potential_templates:
        potential_templates = candidate_template_names
    
    new_template = rng.choice(potential_templates)
    new_specs[idx_to_change] = abstraction.sample_rule_spec(new_template, rng)
    return new_specs


def build_examples(
    rng: random.Random,
    n_examples: int,
    active_symbols: list[Any],
    repetition_mode: str,
    op_specs: list[RuleSpec],
    used_inputs: set[Any],
    abstraction: AbstractionSet,
    size_constraints: Any,
    return_core_inputs: bool = False,
) -> list[ExamplePair] | tuple[list[ExamplePair], list[Any]]:
    examples: list[ExamplePair] = []
    core_inputs: list[Any] = []

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
                core_y = apply_composition(core_x, op_specs, abstraction)
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
            raise RuntimeError("Could not sample enough valid compositional-induction examples.")

    if return_core_inputs:
        return examples, core_inputs
    return examples


def generate_compositional_induction_episode(
    profile: CompositionalInductionDifficultyProfile,
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
    op_specs = []
    candidate_template_names = []
    post_support_pool = []
    post_queries = []
    post_probes = []
    shifted_op_specs = None
    old_rule_post_query_output = None
    active_symbols = []
    size_constraints = []

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
        potential_probe_lengths = [length for length in all_possible_lengths if length not in support_lengths]
        probe_length = attempt_rng.choice(potential_probe_lengths) if potential_probe_lengths else (max(support_lengths) + 1)

        candidate_template_names = choose_candidate_template_names(profile, attempt_rng, abstraction)
        op_specs = sample_op_sequence(
            candidate_template_names=candidate_template_names,
            num_steps=profile.num_steps,
            rng=attempt_rng,
            abstraction=abstraction,
        )

        pre_used_inputs: set[Any] = set()
        post_used_inputs: set[Any] = set()

        try:
            support_pool = build_examples(
                rng=attempt_rng,
                n_examples=profile.support_pool_size,
                active_symbols=active_symbols,
                repetition_mode=profile.repetition_mode,
                op_specs=op_specs,
                used_inputs=pre_used_inputs,
                abstraction=abstraction,
                size_constraints=support_lengths,
            )

            queries = build_examples(
                rng=attempt_rng,
                n_examples=profile.query_count,
                active_symbols=active_symbols,
                repetition_mode=profile.repetition_mode,
                op_specs=op_specs,
                used_inputs=pre_used_inputs,
                abstraction=abstraction,
                size_constraints=support_lengths,
            )

            probes = build_examples(
                rng=attempt_rng,
                n_examples=1,
                active_symbols=active_symbols,
                repetition_mode=profile.repetition_mode,
                op_specs=op_specs,
                used_inputs=set(),
                abstraction=abstraction,
                size_constraints=probe_length,
            )

            if profile.has_shift:
                shifted_op_specs = mutate_op_sequence(
                    op_specs=op_specs,
                    candidate_template_names=candidate_template_names,
                    rng=attempt_rng,
                    abstraction=abstraction,
                )

                post_support_pool = build_examples(
                    rng=attempt_rng,
                    n_examples=profile.post_support_pool_size,
                    active_symbols=active_symbols,
                    repetition_mode=profile.repetition_mode,
                    op_specs=shifted_op_specs,
                    used_inputs=post_used_inputs,
                    abstraction=abstraction,
                    size_constraints=support_lengths,
                )

                post_queries, post_query_cores = build_examples(
                    rng=attempt_rng,
                    n_examples=profile.post_query_count,
                    active_symbols=active_symbols,
                    repetition_mode=profile.repetition_mode,
                    op_specs=shifted_op_specs,
                    used_inputs=post_used_inputs,
                    abstraction=abstraction,
                    size_constraints=support_lengths,
                    return_core_inputs=True,
                )

                post_probes = build_examples(
                    rng=attempt_rng,
                    n_examples=1,
                    active_symbols=active_symbols,
                    repetition_mode=profile.repetition_mode,
                    op_specs=shifted_op_specs,
                    used_inputs=set(),
                    abstraction=abstraction,
                    size_constraints=probe_length,
                )

                if post_queries:
                    try:
                        old_core_y = apply_composition(post_query_cores[0], op_specs, abstraction)
                        old_rule_post_query_output = abstraction.render(old_core_y)
                    except Exception:
                        old_rule_post_query_output = None
            
            # SUCCESS
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("Could not sample a compositional induction episode.")

    return Episode(
        module="compositional_induction",
        task_name=f"compositional_induction_{representation_name}",
        representation=representation_name,  # type: ignore
        difficulty=profile.name,
        support_pool=support_pool,
        queries=queries,
        probes=probes,
        shift_type="single_op_shift" if shifted_op_specs else None,
        post_support_pool=post_support_pool,
        post_queries=post_queries,
        post_probes=post_probes,
        metadata={
            "evidence_budgets": profile.evidence_budgets,
            "num_active_symbols": profile.num_active_symbols,
            "candidate_template_names": candidate_template_names,
            "op_names": [spec.name for spec in op_specs],
            "op_families": [spec.family for spec in op_specs],
            "op_params": [spec.params for spec in op_specs],
            "shifted_op_names": [spec.name for spec in shifted_op_specs] if shifted_op_specs else None,
            "shifted_op_families": [spec.family for spec in shifted_op_specs] if shifted_op_specs else None,
            "shifted_op_params": [spec.params for spec in shifted_op_specs] if shifted_op_specs else None,
            "num_steps": profile.num_steps,
            "repetition_mode": profile.repetition_mode,
            "old_rule_post_query_output": old_rule_post_query_output,
        },
    )
