import argparse
import random
from typing import Any

from core.representations.string import StringAbstractionSet
from core.representations.number import NumberAbstractionSet
from core.representations.grid import GridAbstractionSet

from modules.symbolic_binding.generator import generate_symbolic_binding_episode
from modules.compositional_induction.generator import generate_compositional_induction_episode
from modules.contextual_adaptation.generator import generate_contextual_adaptation_episode

from core.difficulty import (
    SYMBOLIC_BINDING_DIFFICULTY,
    COMPOSITIONAL_INDUCTION_DIFFICULTY,
    CONTEXTUAL_ADAPTATION_DIFFICULTY,
)

def grid_to_human_readable(data: Any) -> str:
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            return "\n".join(" ".join(str(cell) for cell in row) for row in data)
        return " ".join(str(item) for item in data)
    return str(data)

def format_item(item: Any, is_grid: bool) -> str:
    if is_grid:
        return f"\n{grid_to_human_readable(item)}"
    return str(item)

def run_interactive_round(episode, abstraction, phase: str, budgets: list[int], is_grid: bool):
    print(f"\n{'='*20} PHASE: {phase.upper()} {'='*20}")
    print(f"Task: {episode.task_name} | Difficulty: {episode.difficulty}")
    previous_budget = 0
    
    if phase == "pre":
        support_pool = episode.support_pool
        queries = episode.queries
    else:
        support_pool = episode.post_support_pool
        queries = episode.post_queries

    for budget in budgets:
        print(f"\n--- Budget: {budget} (Showing first {budget} examples) ---")
        new_support = support_pool[previous_budget:budget]
        
        print("Support Examples:")
        for ex in new_support:
            print(f"  {format_item(ex.inp, is_grid)} -> {format_item(ex.out, is_grid)}")
        
        print("\nQueries:")
        all_correct = True
        for i, q in enumerate(queries):
            # For grids, we want the prompt on a new line after the grid
            if is_grid:
                print(f"  Query {i+1}: {format_item(q.inp, is_grid)}")
                user_ans = input("  ? Your answer: ").strip()
            else:
                user_ans = input(f"  Query {i+1}: {format_item(q.inp, is_grid)} -> ? ").strip()
            
            # Ground truth rendering
            gold_str = abstraction.render(q.out).strip()
            
            # If not a grid, normalize spaces for cleaner comparison check
            if not is_grid:
                # Collapse internal whitespace for both so user isn't penalized for double spaces
                import re
                gold_display = re.sub(r'\s+', ' ', gold_str)
                user_normalized = re.sub(r'\s+', ' ', user_ans)
                correct = (user_normalized == gold_display)
            else:
                gold_display = gold_str
                # For grids, try JSON match first, fallback to string exact
                correct = (user_ans == gold_str)
                if not correct:
                    try:
                        import json
                        correct = (json.loads(user_ans) == json.loads(gold_str))
                    except Exception:
                        pass
            
            soft_score = abstraction.soft_score(q.out, user_ans)
            
            if correct:
                print(f"  [CORRECT] Soft Score: {soft_score:.2f}")
            else:
                print(f"  [WRONG] Expected: {gold_display}")
                print(f"  [SCORE] Binary: 0.00 | Soft: {soft_score:.2f}")
                all_correct = False
        
        previous_budget = budget
        if all_correct and budget < budgets[-1]:
            cont = input("\nAll correct! Continue to next budget? (y / n): ")
            if cont.lower() != 'y':
                break

def main():
    parser = argparse.ArgumentParser(description="Interactive FluidLearn Agent")
    parser.add_argument("--module", type=str, required=True, choices=["symbolic_binding", "compositional_induction", "contextual_adaptation"])
    parser.add_argument("--representation", type=str, default="string", choices=["string", "number", "grid"])
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000))
    args = parser.parse_args()

    # Setup representation
    if args.representation == "string":
        abstraction = StringAbstractionSet()
    elif args.representation == "number":
        abstraction = NumberAbstractionSet()
    else:
        abstraction = GridAbstractionSet()

    # Generate episode
    print(f"Generating {args.module} ({args.representation}) at {args.difficulty} difficulty (Seed: {args.seed})...")
    
    if args.module == "symbolic_binding":
        profile = SYMBOLIC_BINDING_DIFFICULTY[args.difficulty]
        episode = generate_symbolic_binding_episode(profile, args.seed, abstraction, args.representation)
    elif args.module == "compositional_induction":
        profile = COMPOSITIONAL_INDUCTION_DIFFICULTY[args.difficulty]
        episode = generate_compositional_induction_episode(profile, args.seed, abstraction, args.representation)
    else:
        profile = CONTEXTUAL_ADAPTATION_DIFFICULTY[args.difficulty]
        episode = generate_contextual_adaptation_episode(profile, args.seed, abstraction, args.representation)

    is_grid = (args.representation == "grid")
    
    budgets = episode.metadata.get("evidence_budgets", [1, 2, 3, 4, 5])
    
    run_interactive_round(episode, abstraction, "pre", budgets, is_grid)
    
    if episode.shift_type:
        print(f"\n[SHIFT DETECTED: {episode.shift_type}]")
        run_interactive_round(episode, abstraction, "post", budgets, is_grid)

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
