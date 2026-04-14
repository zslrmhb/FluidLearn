"""
Aggregate Results 
-----------------------------------------------
This script consolidates benchmark CSVs into a single research JSON.
It groups results by core model identity and implements a sophisticated 
Universal vs. Unique failure sampling strategy for deep diagnostics.
"""
import os
import json
import glob
import re
from collections import defaultdict
from pathlib import Path
from utils import identify_brand, safe_read_csv, parse_benchmark_filename

class FailureAnalyzer:
    def __init__(self, universal_failures: set, model_names: list):
        self.universal_failures = universal_failures
        self.num_models = len(model_names)

    def get_diagnosis(self, row: dict, module: str, representation: str) -> dict:
        """
        Apply the Failure Taxonomy to a mistake record and provide a human-readable 
        diagnosis and heuristic explanation.
        """
        labels = []
        
        pred_post = str(row.get("pred_post_json", ""))
        error_msg = str(row.get("error_message", ""))
        
        # Threshold Reasoning:
        # score < 0.2: Floor performance. In 0-1 range, <20% is usually noise or total logic failure.
        # acquisition > 0.6: Signal for "learned the rule". 60%+ means the model isn't just guessing.
        # adaptation < 0.2: Shift shock. Clear failure to adjust after the change.
        # probe_score < 0.3: Generalization check. Failure to transfer logic to new contexts.
        score = float(row.get("final_score", 0))
        acq_score = float(row.get("acquisition_score", 0))
        ada_score = float(row.get("adaptation_score", 0))
        prob_score = float(row.get("post_probe_score") or row.get("post_probe_soft_score") or 0)
        
        # 1. Perseveration (Logic Rigidity)
        meta = {}
        try: meta = json.loads(row.get("metadata_json", "{}"))
        except: pass
        
        old_out = str(meta.get("old_rule_post_query_output", ""))
        def normalize(s): return re.sub(r'[\s\[\],]', '', str(s))
        
        if old_out and normalize(old_out) == normalize(pred_post) and score < 0.2:
            labels.append("Perseveration")
        
        # 2. Empty or Broken Output (formerly Structural Collapse)
        if re.search(r'\[\s*\]|\[\s*\[\s*\]\s*\]', pred_post) or row.get("schema_success") == False:
            labels.append("Empty or Broken Output")
            
        # 3. Schema / Format Violation
        if "RoundOutputSchema" in error_msg or "ValidationError" in str(row.get("error_type", "")):
            labels.append("Schema / Format Violation")
            
        # 4. Type Mismatch
        if re.search(r'input_type|should be a valid|type=', error_msg.lower()):
            labels.append("Type Mismatch")
            
        # 5. Learned Before Shift, Failed After Shift (formerly Adaptation Failure)
        if acq_score > 0.6 and ada_score < 0.2 and "Perseveration" not in labels:
            labels.append("Learned Before Shift, Failed After Shift")
            
        # 6. Failed Both Before and After Shift (formerly Broad Failure)
        if acq_score < 0.2 and ada_score < 0.2 and "Empty or Broken Output" not in labels:
            labels.append("Failed Both Before and After Shift")
            
        # 7. Solved Training Pattern, Failed Transfer (formerly Probe / Generalization Failure)
        if ada_score > 0.5 and prob_score < 0.3:
            labels.append("Solved Training Pattern, Failed Transfer")
            
        # 8. Hard for Almost Every Model (formerly Universal Failure Collision)
        tid = self._make_id(row, module, representation)
        if tid in self.universal_failures:
            labels.append("Hard for Almost Every Model")
            
        # 9. Inconsistent Rule Application (New)
        if 0.2 < score < 0.8:
            labels.append("Inconsistent Rule Application")

        # 10. Pre-training Over-reliance (New - Heuristic)
        if acq_score < 0.2 and "Perseveration" not in labels and "Empty or Broken Output" not in labels:
            labels.append("Pre-training Over-reliance")

        # Determine Primary Diagnosis (Priority Order)
        priority = ["Empty or Broken Output", "Schema / Format Violation", "Type Mismatch", 
                    "Perseveration", "Failed Both Before and After Shift", "Learned Before Shift, Failed After Shift", 
                    "Solved Training Pattern, Failed Transfer", "Hard for Almost Every Model",
                    "Inconsistent Rule Application", "Pre-training Over-reliance"]
        
        primary = "Unclassified Failure"
        for p in priority:
            if p in labels:
                primary = p
                break
        
        # Heuristic "Why it Failed" explanations for notebook storytelling
        explanations = {
            "Perseveration": "Copied old rule after shift",
            "Empty or Broken Output": "Produced empty structured output",
            "Schema / Format Violation": "Output violated the required JSON schema or format",
            "Type Mismatch": "Produced incorrect data types in specific fields",
            "Learned Before Shift, Failed After Shift": "Got the logic initially but couldn't let go or adapt",
            "Failed Both Before and After Shift": "Failed basic task logic even without shift",
            "Solved Training Pattern, Failed Transfer": "Solved post-shift query but failed transfer",
            "Hard for Almost Every Model": "Inherent task difficulty high for all tested models",
            "Inconsistent Rule Application": "Identified the shift correctly but applied it inconsistently",
            "Pre-training Over-reliance": "Model ignored the new rule and fell back to pre-trained habits",
            "Unclassified Failure": "Failure occurred but didn't match specific known patterns"
        }
        
        return {
            "labels": labels, 
            "primary_failure_mode": primary,
            "why_it_failed": explanations.get(primary, "Unknown error reason")
        }

    def _make_id(self, row, mod, rep):
        # Stable unique task ID
        # Fields: module, rep, difficulty, shift_type, task_name, chain
        diff = str(row.get("difficulty", "unknown"))
        shift = str(row.get("shift_type", "no_shift"))
        name = str(row.get("task_name") or row.get("chain", "unknown"))
        return f"{mod}|{rep}|{diff}|{shift}|{name}"

def main():
    base_dir = Path(__file__).resolve().parent.parent / "run_results"
    output_path = Path(__file__).resolve().parent / "aggregated_results.json"
    
    csv_files = list(base_dir.rglob("*.csv"))
    print(f"Discovered {len(csv_files)} files.")
    
    # -------------------------------------------------------------------------
    # STAGE 0: Identity Discovery & Universal Failure Analysis
    # -------------------------------------------------------------------------
    # We first scan all files to identify unique models and detect "Universal Failures"
    # (tasks that every model failed), which helps anchor our difficulty analysis.
    model_identities = set()
    global_task_stats = defaultdict(lambda: {"models_failed": set(), "scores": [], "meta": {}})
    
    for f in csv_files:
        mod, rep = f.parent.parent.name, f.parent.name
        parsed = parse_benchmark_filename(f.name, mod, rep)
        m_name = parsed["model"]
        if not m_name: continue
        model_identities.add(m_name)
        
        if parsed["type"] == "mistakes":
            df = safe_read_csv(str(f))
            if df is None or df.empty: continue
            for _, row in df.iterrows():
                tid = f"{mod}|{rep}|{row.get('difficulty','un')}|{row.get('shift_type','un')}|{row.get('task_name') or row.get('chain','un')}"
                if float(row.get("final_score", 1.0)) < 0.2:
                    global_task_stats[tid]["models_failed"].add(m_name)
                global_task_stats[tid]["scores"].append(float(row.get("final_score", 0)))
                # Store rich meta once
                if not global_task_stats[tid]["meta"]:
                    global_task_stats[tid]["meta"] = {
                        "module": mod, "rep": rep, "diff": row.get("difficulty"),
                        "shift": row.get("shift_type"), "rule": row.get("rule_family") or row.get("chain_families")
                    }

    num_models = len(model_identities)
    print(f"Tracking {num_models} core models.")
    
    universal_failures = {t for t, s in global_task_stats.items() if len(s["models_failed"]) == num_models}
    print(f"Identified {len(universal_failures)} Universal Failures.")
    
    analyzer = FailureAnalyzer(universal_failures, list(model_identities))
    
    # -------------------------------------------------------------------------
    # STAGE 1: Multi-Pass Aggregation
    # -------------------------------------------------------------------------
    # We iterate through the CSV results to group performance metrics and mistake 
    # records by model. This pass builds the foundation for the "Fluid Wrapped" reports.
    results = {}
    
    for f in csv_files:
        mod, rep = f.parent.parent.name, f.parent.name
        parsed = parse_benchmark_filename(f.name, mod, rep)
        m_name = parsed["model"]
        if not m_name: continue
        
        if m_name not in results:
            results[m_name] = {
                "brand": identify_brand(m_name),
                "global_stats": defaultdict(list),
                "breakdowns": {
                    "module": defaultdict(lambda: defaultdict(list)),
                    "representation": defaultdict(lambda: defaultdict(list)),
                    "difficulty": defaultdict(lambda: defaultdict(list)),
                    "rule_family": defaultdict(lambda: defaultdict(list)),
                    "shift_type": defaultdict(lambda: defaultdict(list)),
                },
                "curves": {
                    "global": {"pre": [], "post": []},
                    "by_module": defaultdict(lambda: {"pre": [], "post": []})
                },
                "failures": {
                    "taxonomy_counts": defaultdict(int),
                    "failed_task_records": [],
                    "schema_failure_records": []
                },
                "audit": {"report_files": 0, "mistake_rows": 0, "schema_failure_row_count": 0}
            }
            
        m_data = results[m_name]
        rtype = parsed["type"]

        if rtype in ["overall_report", "by_shift_type", "by_rule_report"]:
            df = safe_read_csv(str(f))
            if df is None or df.empty: continue
            
            for _, row in df.iterrows():
                # For overall_report, we take the mean
                # For others, we group by a key
                key = None
                bd_type = None
                if rtype == "by_shift_type":
                    key, bd_type = row.get("shift_type"), "shift_type"
                elif rtype == "by_rule_report":
                    key, bd_type = row.get("chain") or row.get("rule") or row.get("task_name"), "rule_family"
                
                stats = row.to_dict()
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        if rtype == "overall_report":
                            m_data["global_stats"][k].append(v)
                            m_data["breakdowns"]["module"][mod][k].append(v)
                            m_data["breakdowns"]["representation"][rep][k].append(v)
                        if bd_type and key:
                            m_data["breakdowns"][bd_type][key][k].append(v)

                if rtype == "overall_report":
                    m_data["audit"]["report_files"] += 1
                    pre = [stats.get(f"mean_pre_query_score_{i}", 0) for i in range(1, 6)]
                    post = [stats.get(f"mean_post_query_score_{i}", 0) for i in range(1, 6)]
                    m_data["curves"]["global"]["pre"].append(pre)
                    m_data["curves"]["global"]["post"].append(post)
                    m_data["curves"]["by_module"][mod]["pre"].append(pre)
                    m_data["curves"]["by_module"][mod]["post"].append(post)

        elif rtype in ["by_difficulty_report", "by_rule_family_report"]:
            df = safe_read_csv(str(f))
            if df is None or df.empty: continue
            dest = "difficulty" if "difficulty" in rtype else "rule_family"
            for _, row in df.iterrows():
                key = row.get("difficulty") or row.get("rule_family") or row.get("chain_families")
                for k, v in row.to_dict().items():
                    if isinstance(v, (int, float)):
                        m_data["breakdowns"][dest][key][k].append(v)

        elif rtype == "mistakes":
            df = safe_read_csv(str(f))
            if df is None or df.empty: continue
            m_data["audit"]["mistake_rows"] += len(df)
            for _, row in df.iterrows():
                score = float(row.get("final_score", 0))
                # We skip records with score >= 0.9 because this analyzer focuses on 
                # diagnostic mistaking behavior; >90% is considered a successful mastery.
                if score >= 0.9: continue
                
                diag = analyzer.get_diagnosis(row.to_dict(), mod, rep)
                m_data["failures"]["taxonomy_counts"][diag["primary_failure_mode"]] += 1
                
                # Bundle storytelling evidence with clear, descriptive keys for the notebook
                bundle = {
                    "support_set_examples": row.get("support_pool_json", "[]")[:500],
                    "pre_shift_ground_truth": row.get("queries_json", "[]")[:500],
                    "pre_shift_prediction": row.get("pred_pre_json", "[]")[:500],
                    "post_shift_ground_truth": row.get("post_queries_json", "[]")[:500],
                    "post_shift_prediction": row.get("pred_post_json", "[]")[:500]
                }
                
                record = {
                    "task_id": analyzer._make_id(row, mod, rep),
                    "module": mod, 
                    "representation": rep, 
                    "difficulty": row.get("difficulty"),
                    "final_score": score, 
                    "diagnosis": diag, 
                    "example_bundle": bundle
                }
                m_data["failures"]["failed_task_records"].append(record)

        elif rtype == "schema_failures":
            df = safe_read_csv(str(f))
            if df is None or df.empty: continue
            m_data["audit"]["schema_failure_row_count"] += len(df)
            for _, row in df.iterrows():
                m_data["failures"]["schema_failure_records"].append({
                    "module": mod, "representation": rep, "difficulty": row.get("difficulty"),
                    "error_type": row.get("error_type"), "error_message": row.get("error_message")
                })

    # -------------------------------------------------------------------------
    # STAGE 2: Formatting & Stratified Sampling
    # -------------------------------------------------------------------------
    # We condense the raw data into averages and implement a sophisticated 
    # sampling strategy to build a "Failure Gallery" that highlights the most 
    # interesting edge cases without overwhelming the final JSON size.
    final_json = {"models": {}, "popular_mistakes": []}
    
    for m_name, d in results.items():
        def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0
        
        m_proc = {
            "brand": d["brand"],
            "global_stats": {k: avg(v) for k, v in d["global_stats"].items()},
            "breakdowns": {
                lvl: {k: {stat: avg(vals) for stat, vals in v.items()} for k, v in lvls.items()}
                for lvl, lvls in d["breakdowns"].items()
            },
            "curves": {
                "global": {
                    "pre": [avg(x) for x in zip(*d["curves"]["global"]["pre"])] if d["curves"]["global"]["pre"] else [],
                    "post": [avg(x) for x in zip(*d["curves"]["global"]["post"])] if d["curves"]["global"]["post"] else []
                },
                "by_module": {
                    mod: {
                        "pre": [avg(x) for x in zip(*cv["pre"])] if cv["pre"] else [],
                        "post": [avg(x) for x in zip(*cv["post"])] if cv["post"] else []
                    } for mod, cv in d["curves"]["by_module"].items()
                }
            },
            "audit": d["audit"]
        }
        
        # Pivot Stats
        pre, post = m_proc["curves"]["global"]["pre"], m_proc["curves"]["global"]["post"]
        if pre and post:
            m_proc["pivot"] = {
                "shift_shock": round(pre[-1] - post[0], 4), "recovery_gain": round(post[-1] - post[0], 4)
            }
        
        # ---------------------------------------------------------------------
        # STRATIFIED SAMPLING STRATEGY
        # ---------------------------------------------------------------------
        # The goal is to produce a "Failure Gallery" for the notebook that is:
        # 1. Representative of common issues (Universal Failures)
        # 2. Informative about model-specific quirks (Unique Failures)
        # 3. Diverse in error types (Filling Diagnosis Gaps)
        # 4. Compact and performant for notebook rendering (Capping Size)
        # ---------------------------------------------------------------------
        all_records = d["failures"]["failed_task_records"]
        kept = []
        u_ids = set()
        
        # Universal
        for r in all_records:
            if r["task_id"] in universal_failures:
                kept.append(r)
                u_ids.add(r["task_id"])
        
        # Unique
        for r in all_records:
            if len(global_task_stats[r["task_id"]]["models_failed"]) == 1:
                kept.append(r)
                u_ids.add(r["task_id"])
                
        # Fill Diagnosis gaps (up to 3 more per label if missing)
        # We target labels like "Perseveration" to ensure they appear in the gallery.
        for diag_label in ["Perseveration", "Empty or Broken Output", "Type Mismatch", "Learned Before Shift, Failed After Shift"]:
            found = [r for r in all_records if diag_label in r["diagnosis"]["labels"] and r["task_id"] not in u_ids]
            kept.extend(found[:3])
            
        m_proc["failures"] = {
            "taxonomy_counts": d["failures"]["taxonomy_counts"],
            "failed_task_records": kept[:100], # Cap at 100 gallery items
            "schema_failure_records": d["failures"]["schema_failure_records"][:20]
        }
        
        final_json["models"][m_name] = m_proc

    # Populate popular_mistakes with rich metadata for global analysis
    for tid in sorted(universal_failures):
        stats = global_task_stats[tid]
        meta = stats["meta"]
        final_json["popular_mistakes"].append({
            "task_id": tid, 
            "fail_rate": round(len(stats["models_failed"]) / num_models, 2),
            "avg_score": avg(stats["scores"]), 
            "module": meta["module"], 
            "representation": meta["rep"], # Renamed for JSON clarity
            "difficulty": meta["diff"],     # Renamed for JSON clarity
            "rule": meta["rule"]
        })

    # -------------------------------------------------------------------------
    # STAGE 3: Final Serialization
    # -------------------------------------------------------------------------
    with open(output_path, "w") as jf:
        json.dump(final_json, jf, indent=2)
    print(f"Aggregation Complete. JSON saved to {output_path}")

if __name__ == "__main__":
    main()
