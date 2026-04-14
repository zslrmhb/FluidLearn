from __future__ import annotations

import random
from typing import Any

from core.representations.base import AbstractionSet, RuleSpec
from core.sampling import sample_symbol_pool
from core.types import Episode, ExamplePair, FeedbackExplorationDifficultyProfile
from modules.feedback_exploration.types import FeedbackPolicy, FeedbackTaskConfig


BANNED_FEEDBACK_FAMILIES = {
    "reduction",
}


def _sample_distinct_rule_specs(
    rng: random.Random, n: int, abstraction: AbstractionSet
) -> list[RuleSpec]:
    all_template_names = [
        name
        for name, template in abstraction.rule_templates.items()
        if template.family not in BANNED_FEEDBACK_FAMILIES
    ]
    if len(all_template_names) < n:
        chosen_templates = rng.choices(all_template_names, k=n)
    else:
        chosen_templates = rng.sample(all_template_names, k=n)

    specs = []
    for t_name in chosen_templates:
        specs.append(abstraction.sample_rule_spec(t_name, rng))
    return specs


def _find_good_query_pair_and_pruned_specs(
    rng: random.Random,
    pool: list[Any],
    support_lengths: list[int],
    probe_length: int,
    repetition_mode: str,
    specs: list[RuleSpec],
    abstraction: AbstractionSet,
    fixed_specs: list[RuleSpec],
    max_attempts: int = 200,
) -> tuple[Any, Any, list[RuleSpec]]:
    """
    Find a pair of queries (interaction, probe) and return the maximal subset of `specs`
    that are mutually distinguishable. `fixed_specs` MUST be in the returned subset.
    Raises RuntimeError if `fixed_specs` themselves collide or if we can't get at least 3 total.
    """
    from core.sampling import sample_unique_core_item, _make_hashable

    best_queries = None
    best_specs = []

    for _ in range(max_attempts):
        used_inputs: set[Any] = set()
        
        try:
            interaction_query = sample_unique_core_item(
                rng=rng,
                abstraction=abstraction,
                pool=pool,
                size_constraints=support_lengths,
                repetition_mode=repetition_mode,
                used_inputs=used_inputs,
            )
            used_inputs.add(_make_hashable(interaction_query))

            probe_query = sample_unique_core_item(
                rng=rng,
                abstraction=abstraction,
                pool=pool,
                size_constraints=probe_length,
                repetition_mode=repetition_mode,
                used_inputs=used_inputs,
            )

            signatures = set()
            valid_specs = []
            
            # Step 1: Ensure fixed_specs are distinguishable
            fixed_valid = True
            for spec in fixed_specs:
                rule_fn = abstraction.resolve_rule(spec)
                try:
                    i_out = abstraction.render(rule_fn(interaction_query))
                    p_out = abstraction.render(rule_fn(probe_query))
                    sig = (i_out, p_out)
                    if sig in signatures:
                        fixed_valid = False
                        break
                    signatures.add(sig)
                    valid_specs.append(spec)
                except Exception:
                    fixed_valid = False
                    break
                    
            if not fixed_valid:
                continue
                
            # Step 2: Add as many optional specs as possible
            for spec in specs:
                if spec in fixed_specs:
                    continue
                rule_fn = abstraction.resolve_rule(spec)
                try:
                    i_out = abstraction.render(rule_fn(interaction_query))
                    p_out = abstraction.render(rule_fn(probe_query))
                    sig = (i_out, p_out)
                    if sig not in signatures:
                        signatures.add(sig)
                        valid_specs.append(spec)
                except Exception:
                    continue

            if len(valid_specs) >= 3:
                if len(valid_specs) > len(best_specs):
                    best_specs = valid_specs
                    best_queries = (interaction_query, probe_query)
                if len(valid_specs) == len(specs):
                    break # Max possible

        except Exception as e:
            if not isinstance(e, RuntimeError) or "unique core items" not in str(e):
                import traceback
                traceback.print_exc()
            continue
            
    if best_queries is not None and len(best_specs) >= 3:
        return best_queries[0], best_queries[1], best_specs
        
    raise RuntimeError(f"Could not find queries resolving at least 3 distinct operators. Best size: {len(best_specs)}. Fixed specs: {[s.name for s in fixed_specs]}")


def generate_feedback_exploration_episode(
    profile: FeedbackExplorationDifficultyProfile,
    seed: int,
    abstraction: AbstractionSet,
    representation_name: str = "string",
    pool_override: list[Any] | None = None,
    size_override: Any | None = None,
    return_config: bool = False,
) -> Episode | tuple[Episode, FeedbackTaskConfig]:
    # Initialize all potential episode components
    candidate_template_names = [
        name for name, template in abstraction.rule_templates.items()
        if template.family not in BANNED_FEEDBACK_FAMILIES
    ]

    interaction_query = None
    probe_query = None
    shifted_interaction_query = None
    shifted_probe_query = None
    pre_rule_spec = None
    shifted_rule_spec = None
    symbol_pool = []
    toolbox = {}

    for _gen_attempt in range(50):
        attempt_rng = random.Random(seed + _gen_attempt)
        
        pre_template_name = attempt_rng.choice(candidate_template_names)
        pre_rule_spec = abstraction.sample_rule_spec(pre_template_name, attempt_rng)

        size_constraints = size_override if size_override is not None else profile.sequence_lengths
        if isinstance(size_constraints, int):
            support_lengths = [size_constraints]
        else:
            support_lengths = list(size_constraints)

        all_possible_lengths = list(range(max(1, min(support_lengths)-1), max(support_lengths)+3))
        potential_probe_lengths = [slen for slen in all_possible_lengths if slen not in support_lengths]
        probe_length = attempt_rng.choice(potential_probe_lengths) if potential_probe_lengths else (max(support_lengths) + 1)

        symbol_pool = sample_symbol_pool(seed=seed + _gen_attempt, n=max(profile.num_active_symbols, 2))

        fixed_specs = [pre_rule_spec]
        toolbox_specs = [pre_rule_spec]
        if profile.has_shift:
            shifted_template_name = attempt_rng.choice([t for t in candidate_template_names if t != pre_template_name])
            shifted_rule_spec = abstraction.sample_rule_spec(shifted_template_name, attempt_rng)
            toolbox_specs.append(shifted_rule_spec)
            fixed_specs.append(shifted_rule_spec)
        else:
            shifted_rule_spec = None
            
        target_toolbox_size = max(3, profile.num_candidate_rules + (2 if profile.has_shift else 1))
        for spec in _sample_distinct_rule_specs(attempt_rng, 15, abstraction):
            if len(toolbox_specs) >= target_toolbox_size:
                break
            if not any(s.name == spec.name for s in toolbox_specs):
                toolbox_specs.append(spec)

        try:
            interaction_query, probe_query, pruned_specs = _find_good_query_pair_and_pruned_specs(
                rng=attempt_rng,
                pool=symbol_pool,
                support_lengths=support_lengths,
                probe_length=probe_length,
                repetition_mode=profile.repetition_mode,
                specs=toolbox_specs,
                abstraction=abstraction,
                fixed_specs=fixed_specs,
            )
        except RuntimeError:
            continue

        if profile.has_shift and shifted_rule_spec:
            try:
                shifted_interaction_query, shifted_probe_query, post_pruned_specs = _find_good_query_pair_and_pruned_specs(
                    rng=attempt_rng,
                    pool=symbol_pool,
                    support_lengths=support_lengths,
                    probe_length=probe_length,
                    repetition_mode=profile.repetition_mode,
                    specs=pruned_specs, # Keep toolbox coherent across shift by using pre-pruned
                    abstraction=abstraction,
                    fixed_specs=fixed_specs,
                )
            except RuntimeError:
                continue
                
            # Intersect valid specs to ensure the toolbox is completely valid in both phases
            final_specs = [s for s in pruned_specs if s in post_pruned_specs]
            if len(final_specs) < 3 or not all(fs in final_specs for fs in fixed_specs):
                continue
        else:
            final_specs = pruned_specs
            
        attempt_rng.shuffle(final_specs)
        toolbox = {f"Op_{i+1}": spec for i, spec in enumerate(final_specs)}
        
        break
    else:
        raise RuntimeError("Could not sample a valid Feedback Exploration episode.")

    feedback_policy = FeedbackPolicy(
        max_rounds_per_shift=5,
        binary_feedback_only=False,
        allow_shape_hints=profile.allow_shape_hints,
        allow_structure_hints=profile.allow_structure_hints,
        allow_semantic_hints=profile.allow_semantic_hints,
    )

    old_rule_post_query_output = None
    if profile.has_shift and shifted_rule_spec and shifted_interaction_query:
        try:
            pre_rule_fn = abstraction.resolve_rule(pre_rule_spec)
            old_core_y = pre_rule_fn(shifted_interaction_query)
            old_rule_post_query_output = abstraction.render(old_core_y)
        except Exception:
            old_rule_post_query_output = None

    pre_rule_fn = abstraction.resolve_rule(pre_rule_spec)
    
    toolbox_outputs = {}
    for op_id, spec in toolbox.items():
        fn = abstraction.resolve_rule(spec)
        try:
            toolbox_outputs[op_id] = {
                "interaction": abstraction.render(fn(interaction_query)),
                "probe": abstraction.render(fn(probe_query)),
            }
        except Exception as e:
            toolbox_outputs[op_id] = {"error": str(e)}

    episode = Episode(
        module="feedback_exploration",
        task_name=f"feedback_exploration_{representation_name}",
        representation=representation_name,  # type: ignore
        difficulty=profile.name,
        support_pool=[ExamplePair(inp=abstraction.render(interaction_query), out=None)],
        queries=[], 
        probes=[ExamplePair(inp=abstraction.render(probe_query), out=abstraction.render(pre_rule_fn(probe_query)))],
        shift_type="full_rule_remap" if shifted_rule_spec else None,
        post_support_pool=[ExamplePair(inp=abstraction.render(shifted_interaction_query), out=None)] if shifted_interaction_query else [],
        post_queries=[],
        post_probes=[ExamplePair(inp=abstraction.render(shifted_probe_query), out=None)] if shifted_probe_query else [],
        metadata={
            "evidence_budgets": profile.evidence_budgets,
            "interaction_query": abstraction.render(interaction_query),
            "probe_query": abstraction.render(probe_query),
            "shifted_interaction_query": abstraction.render(shifted_interaction_query) if shifted_interaction_query else None,
            "shifted_probe_query": abstraction.render(shifted_probe_query) if shifted_probe_query else None,
            "toolbox": {k: {"name": v.name, "family": v.family, "params": v.params} for k, v in toolbox.items()},
            "toolbox_outputs": toolbox_outputs,
            "correct_operator_pre": [k for k, v in toolbox.items() if v.name == pre_rule_spec.name][0],
            "correct_operator_post": [k for k, v in toolbox.items() if v.name == (shifted_rule_spec.name if shifted_rule_spec else None)] or [None] if shifted_rule_spec else None,
            "module_design": "interaction_plus_probe",
            "old_rule_post_query_output": old_rule_post_query_output,
            "allow_shape_hints": profile.allow_shape_hints,
            "allow_structure_hints": profile.allow_structure_hints,
            "allow_semantic_hints": profile.allow_semantic_hints,
        },
    )

    if return_config:
        config = FeedbackTaskConfig(
            module="feedback_exploration",
            task_name=episode.task_name,
            representation=representation_name,
            difficulty=profile.name,
            task_instruction="Derive the hidden operator.",
            symbol_pool=symbol_pool,
            rule_spec=pre_rule_spec,
            shifted_rule_spec=shifted_rule_spec,
            interaction_query=interaction_query,
            shifted_interaction_query=shifted_interaction_query,
            probe_query=probe_query,
            shifted_probe_query=shifted_probe_query,
            toolbox=toolbox,
            feedback_policy=feedback_policy,
            metadata=episode.metadata,
        )
        return episode, config

    return episode
