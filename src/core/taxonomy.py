from __future__ import annotations

from typing import Dict, Any, List, Callable
from core.representations.base import AbstractionSet, RuleSpec

RuleFn = Callable[[Any], Any]

def get_benchmark_rule_space(abstraction: AbstractionSet) -> Dict[str, RuleFn]:
    """Enumerate all standard rule parameterizations for a given representation.
    
    This forms the 'Taxonomy' against which we check for ambiguity.
    """
    rule_space: Dict[str, RuleFn] = {}
    
    templates = abstraction.rule_templates
    
    for name, template in templates.items():
        # Instantiate standard parameter ranges for each family
        params_list: List[Dict[str, Any]] = []
        
        if template.family == "reindexing":
            # For sequence/number: rotate by -1, 1
            # For grid: axis 0/1, shift -1, 1
            if "axis" in template.sample_params.__code__.co_varnames or template.name == "reindexing_spatial_shift":
                for axis in [0, 1]:
                    for shift in [-1, 1]:
                        params_list.append({"axis": axis, "shift": shift})
            else:
                for k in [-1, 1]:
                    params_list.append({"k": k})
                    
        elif template.family == "partitioning":
            # For sequence/number: offset 0, 1
            # For grid: quadrant 1, 2, 3, 4
            if "quadrant" in template.sample_params.__code__.co_varnames or template.name == "partitioning_crop":
                for q in [1, 2, 3, 4]:
                    params_list.append({"quadrant": q})
            else:
                for offset in [0, 1]:
                    params_list.append({"offset": offset})
                    
        elif template.family == "extension":
            # For sequence/number: side first, last
            # For grid: pad_val 0, 1, 9
            if "pad_val" in template.sample_params.__code__.co_varnames or template.name == "extension_pad":
                for v in [0, 1, 9]:
                    params_list.append({"pad_val": v})
            else:
                for side in ["first", "last"]:
                    params_list.append({"side": side})
        
        elif template.family == "iteration":
            # For grid: axis 0, 1
            if "axis" in template.sample_params.__code__.co_varnames or template.name == "iteration_tile":
                for axis in [0, 1]:
                    params_list.append({"axis": axis})
            else:
                params_list.append({}) # Duplicate (no params)
                
        else:
            # Families with no params (permutation, reduction)
            params_list.append({})

        # Register every instantiated rule
        for params in params_list:
            if params:
                suffix = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
                rule_name = f"{template.name}_{suffix}"
            else:
                rule_name = template.name
                
            spec = RuleSpec(name=rule_name, family=template.family, params=params)
            try:
                rule_space[rule_name] = abstraction.resolve_rule(spec)
            except Exception:
                continue # Skip rules that are not resolvable for this rep
                
    return rule_space
