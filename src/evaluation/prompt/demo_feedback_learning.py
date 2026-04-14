import argparse
import random
import sys
from typing import Any

from core.representations.string import StringAbstractionSet
from core.representations.number import NumberAbstractionSet
from core.representations.grid import GridAbstractionSet

from modules.feedback_exploration.generator import generate_feedback_exploration_episode
from modules.feedback_exploration.env import FeedbackGameEnv
from modules.feedback_exploration.backends import StringFeedbackBackend, NumberFeedbackBackend, GridFeedbackBackend
from modules.feedback_exploration.types import FeedbackAction

from core.difficulty import FEEDBACK_EXPLORATION_DIFFICULTY

def grid_to_human_readable(data: Any) -> str:
    """Pretty prints 2D data for user visibility."""
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            return "\n".join(" ".join(str(cell) for cell in row) for row in data)
        return " ".join(str(item) for item in data)
    return str(data)

def format_item(item: Any, is_grid: bool) -> str:
    if is_grid:
        return f"\n{grid_to_human_readable(item)}"
    return str(item)

def play_game(env: FeedbackGameEnv, is_grid: bool):
    """Interactive loop for agent-like exploration and learning."""
    while not env.is_done():
        print("\n" + "="*50)
        print(env.get_state_representation())
        print("="*50)
        
        user_input = input("\nYour answer: ").strip()
        if not user_input:
            continue
            
        action = FeedbackAction(answer=user_input)
        result = env.step(action)
        
        print("\n--- Feedback ---")
        print(result.feedback)
        
        if not result.correct:
            print(f"Match Similarity (Soft Score): {result.soft_score:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Interactive FluidLearn Feedback Game")
    parser.add_argument("--representation", type=str, default="string", choices=["string", "number", "grid"])
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000))
    args = parser.parse_args()

    # Setup representation and backend
    if args.representation == "string":
        abstraction = StringAbstractionSet()
        backend = StringFeedbackBackend(abstraction)
    elif args.representation == "number":
        abstraction = NumberAbstractionSet()
        backend = NumberFeedbackBackend(abstraction)
    else:
        abstraction = GridAbstractionSet()
        backend = GridFeedbackBackend(abstraction)

    print(f"Generating Feedback Exploration ({args.representation}) at {args.difficulty} difficulty (Seed: {args.seed})...")
    
    # Direct profile lookup from standardized registry
    profile = FEEDBACK_EXPLORATION_DIFFICULTY[args.difficulty]
    
    episode, config = generate_feedback_exploration_episode(
        profile=profile,
        seed=args.seed,
        abstraction=abstraction,
        representation_name=args.representation,
        return_config=True
    )

    env = FeedbackGameEnv(config=config, backend=backend)
    is_grid = (args.representation == "grid")
    
    play_game(env, is_grid)
    
    print("\nBenchmark episode complete.")

if __name__ == "__main__":
    main()
