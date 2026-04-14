from __future__ import annotations

import pandas as pd
import random
from core.io import dataclass_to_json
from core.difficulty import CONTEXTUAL_ADAPTATION_DIFFICULTY
from core.representations.string import StringAbstractionSet
from core.representations.number import NumberAbstractionSet
from core.representations.grid import GridAbstractionSet
from modules.contextual_adaptation.generator import generate_contextual_adaptation_episode


def generate_contextual_adaptation_dataset(
    representation: str = "string",
    total_problems: int = 150,
    difficulty_counts: dict[str, int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    if difficulty_counts is None:
        difficulty_counts = {
            "easy": total_problems // 3,
            "medium": total_problems // 3,
            "hard": total_problems - 2 * (total_problems // 3),
        }

    if sum(difficulty_counts.values()) != total_problems:
        raise ValueError("Sum of difficulty_counts must equal total_problems.")

    if representation == "string":
        abstraction = StringAbstractionSet()
        fallback_pool = None
        fallback_size = None
    elif representation == "number":
        abstraction = NumberAbstractionSet()
        fallback_pool = [str(i) for i in range(10)]
        fallback_size = None
    elif representation == "grid":
        abstraction = GridAbstractionSet()
        fallback_pool = list(range(10))
        fallback_size = None
    else:
        raise ValueError(f"Unknown representation {representation}")

    rng = random.Random(seed)
    rows: list[dict] = []
    instance_id = 0

    for difficulty, count in difficulty_counts.items():
        if difficulty not in CONTEXTUAL_ADAPTATION_DIFFICULTY:
            raise ValueError(f"Unknown difficulty level: {difficulty}")

        profile = CONTEXTUAL_ADAPTATION_DIFFICULTY[difficulty]

        for _ in range(count):
            instance_seed = rng.randint(0, 10**9)
            episode = generate_contextual_adaptation_episode(
                profile=profile,
                seed=instance_seed,
                abstraction=abstraction,
                representation_name=representation,
                pool_override=fallback_pool,
                size_override=fallback_size,
            )

            rows.append(
                {
                    "instance_id": instance_id,
                    "instance_seed": instance_seed,
                    "module": episode.module,
                    "task_name": episode.task_name,
                    "difficulty": episode.difficulty,
                    "representation": episode.representation,
                    "shift_type": episode.shift_type,
                    "num_active_symbols": profile.num_active_symbols,
                    "num_candidate_rules": profile.num_candidate_rules,
                    "instance_json": dataclass_to_json(episode),
                }
            )
            instance_id += 1

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
