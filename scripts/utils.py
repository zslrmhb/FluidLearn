"""
Shared utility functions for the FluidLearn benchmark analysis pipeline.
"""
import os
import pandas as pd
import re

def identify_brand(model_info: str) -> str:
    """Identify the parent brand based on model name substrings."""
    m = model_info.lower()
    if any(x in m for x in ["anthropic", "claude"]): return "Anthropic"
    if any(x in m for x in ["openai", "gpt"]): return "OpenAI"
    if any(x in m for x in ["google", "gemini"]): return "Google"
    if "deepseek" in m: return "DeepSeek"
    return "Other"

def safe_read_csv(path: str) -> pd.DataFrame:
    """Read CSV and drop common junk columns."""
    try:
        df = pd.read_csv(path)
        # Drop junk Unnamed columns
        df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')
        return df
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}")
        return pd.DataFrame()

def parse_benchmark_filename(filename: str, module: str, representation: str) -> dict:
    """
    Parses hierarchical filenames to extract core model and report type.
    Pattern: {module}_{rep}_{model}_{type}.csv
    """
    report_types = [
        "overall_report", "by_difficulty_report", "by_rule_family_report", 
        "mistakes", "schema_failures", "by_shift_type", "by_rule_report"
    ]
    
    matched_type = None
    core_model = None
    
    # Sort types by length descending to avoid partial matches (e.g. by_rule matches by_rule_family)
    for rt in sorted(report_types, key=len, reverse=True):
        if f"_{rt}.csv" in filename:
            matched_type = rt
            # prefix is {module}_{representation}_
            prefix = f"{module}_{representation}_"
            if filename.startswith(prefix):
                # model is the middle part
                core_model = filename[len(prefix):-(len(rt) + 5)] # +5 for .csv and underscore
            break
            
    return {
        "model": core_model,
        "type": matched_type
    }
