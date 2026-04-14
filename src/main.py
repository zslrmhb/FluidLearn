from pathlib import Path
from core.io import export_csv

from datasets.build_symbolic_binding import generate_symbolic_binding_dataset
from datasets.build_compositional_induction import generate_compositional_induction_dataset
from datasets.build_contextual_adaptation import generate_contextual_adaptation_dataset
from datasets.build_feedback_exploration import generate_feedback_exploration_dataset

def main():
    print("FluidLearn Benchmark Dataset Generator")
    print("=========================================")
    
    out_dir = Path("datasets_output")
    out_dir.mkdir(exist_ok=True)
    
    modules = [
        ("symbolic_binding", generate_symbolic_binding_dataset),
        ("compositional_induction", generate_compositional_induction_dataset),
        ("contextual_adaptation", generate_contextual_adaptation_dataset),
        ("feedback_exploration", generate_feedback_exploration_dataset),
    ]
    representations = ["string", "number", "grid"]
    
    for mod_name, gen_fn in modules:
        for rep in representations:
            print(f"Generating [{mod_name}] - [{rep}]...")
            df = gen_fn(representation=rep, total_problems=100, difficulty_counts={"easy": 20, "medium": 40, "hard": 40}) 
            # Note: For production use scale this to 150+. Used 15 to make local generation fast.
            
            output_file = out_dir / f"{mod_name}_{rep}.csv"
            export_csv(df, output_file)
            print(f"  -> Saved {len(df)} problems to {output_file}")
            
    print("=========================================")
    print("All datasets generated successfully.")

if __name__ == "__main__":
    main()