from __future__ import annotations

import json
import random
from functools import partial
from typing import Any

from core.representations.base import AbstractionSet, RuleTemplate, RuleSpec

# --------------------------------------------------
# Grid-specific primitive implementations
# --------------------------------------------------

def grid_to_text(grid: list[list[int]]) -> str:
    """Render a grid as a JSON string for LLM parsing."""
    return json.dumps(grid)


# P1: Permutation
def permutation_transpose(grid: list[list[int]]) -> list[list[int]]:
    """Transpose the 2D grid."""
    if not grid or not grid[0]:
        return grid
    return [list(row) for row in zip(*grid)]


# P2: Reindexing
def reindexing_spatial_shift(grid: list[list[int]], axis: int = 0, shift: int = 1) -> list[list[int]]:
    """Cyclic shift along rows (axis=0) or cols (axis=1)."""
    if not grid or not grid[0]:
        return grid
    if axis == 0:
        k = shift % len(grid)
        return grid[-k:] + grid[:-k] if k else [list(row) for row in grid]
    if axis == 1:
        cols = len(grid[0])
        k = shift % cols
        return [row[-k:] + row[:-k] if k else list(row) for row in grid]
    raise ValueError("Axis must be 0 or 1.")


# P3: Iteration
def iteration_tile(grid: list[list[int]], axis: int = 0) -> list[list[int]]:
    """Tile the grid (repeat along an axis)."""
    if not grid or not grid[0]:
        return grid
    if axis == 0:
        return [list(row) for row in grid] + [list(row) for row in grid]
    elif axis == 1:
        return [row + row for row in grid]
    raise ValueError("Axis must be 0 or 1.")


# P4: Partitioning
def partitioning_crop(grid: list[list[int]], quadrant: int = 1) -> list[list[int]]:
    """Keep only one quadrant of the grid."""
    if not grid or not grid[0]:
        return grid
    R_half = max(1, len(grid) // 2)
    C_half = max(1, len(grid[0]) // 2)
    
    if quadrant == 1: # Top-left
        return [row[:C_half] for row in grid[:R_half]]
    if quadrant == 2: # Top-right
        return [row[C_half:] for row in grid[:R_half]]
    if quadrant == 3: # Bottom-left
        return [row[:C_half] for row in grid[R_half:]]
    if quadrant == 4: # Bottom-right
        return [row[C_half:] for row in grid[R_half:]]
    raise ValueError("Quadrant must be 1, 2, 3, or 4.")


# P5: Extension
def extension_pad(grid: list[list[int]], pad_val: int = 0) -> list[list[int]]:
    """Pad the grid boundaries with a constant value."""
    if not grid or not grid[0]:
        return grid
    C = len(grid[0])
    new_grid = [[pad_val] * (C + 2)]
    for row in grid:
        new_grid.append([pad_val] + row + [pad_val])
    new_grid.append([pad_val] * (C + 2))
    return new_grid


# P6: Reduction
def reduction_flatten(grid: list[list[int]]) -> list[list[int]]:
    """Flatten into a 1D grid (1xN representation)."""
    if not grid or not grid[0]:
        return grid
    flat = []
    for row in grid:
        flat.extend(row)
    return [flat]


def resolve_grid_rule(spec: RuleSpec):
    if spec.family == "permutation":
        return permutation_transpose
    if spec.family == "reindexing":
        return partial(reindexing_spatial_shift, **spec.params)
    if spec.family == "iteration":
        return partial(iteration_tile, **spec.params)
    if spec.family == "partitioning":
        return partial(partitioning_crop, **spec.params)
    if spec.family == "extension":
        return partial(extension_pad, **spec.params)
    if spec.family == "reduction":
        return reduction_flatten
    raise KeyError(f"Unknown grid rule family: {spec.family}")


GRID_RULE_TEMPLATES: dict[str, RuleTemplate] = {
    "permutation": RuleTemplate(
        name="permutation_transpose",
        family="permutation",
        sample_params=lambda rng: {},
    ),
    "reindexing": RuleTemplate(
        name="reindexing_spatial_shift",
        family="reindexing",
        sample_params=lambda rng: {
            "axis": rng.choice([0, 1]),
            "shift": rng.choice([-1, 1])
        },
    ),
    "iteration": RuleTemplate(
        name="iteration_tile",
        family="iteration",
        sample_params=lambda rng: {"axis": rng.choice([0, 1])},
    ),
    "partitioning": RuleTemplate(
        name="partitioning_crop",
        family="partitioning",
        sample_params=lambda rng: {"quadrant": rng.choice([1, 2, 3, 4])},
    ),
    "extension": RuleTemplate(
        name="extension_pad",
        family="extension",
        sample_params=lambda rng: {"pad_val": rng.choice([0, 1, 9])},
    ),
    "reduction": RuleTemplate(
        name="reduction_flatten",
        family="reduction",
        sample_params=lambda rng: {},
    ),
}


class GridAbstractionSet(AbstractionSet):
    """Concrete abstraction-set implementation for grid representations."""

    @property
    def rule_templates(self) -> dict[str, RuleTemplate]:
        return GRID_RULE_TEMPLATES

    def resolve_rule(self, spec: RuleSpec):
        return resolve_grid_rule(spec)

    def build_rule(self, family: str, **params: Any):
        spec = RuleSpec(name=f"custom_{family}", family=family, params=params)
        return resolve_grid_rule(spec)

    def grid_to_text(self, grid: list[list[int]]) -> str:
        return grid_to_text(grid)
    
    def render(self, x: Any) -> str:
        return self.grid_to_text(x)

    def soft_score(self, gold: list[list[int]], pred: str) -> float:
        """Calculate cell-match similarity between gold grid and predicted string."""
        if not pred or not isinstance(pred, str):
            return 0.0
            
        # Try to parse pred string as grid.
        # It could be JSON [[...], [...]] or newline-separated rows
        try:
            import json
            pred_grid = json.loads(pred)
        except:
            # Fallback: newline separated rows
            rows = pred.strip().split("\n")
            pred_grid = [row.split() for row in rows]

        if not pred_grid or not gold:
            return 0.0
            
        gold_rows = len(gold)
        gold_cols = len(gold[0]) if gold_rows > 0 else 0
        
        pred_rows = len(pred_grid)
        pred_cols = len(pred_grid[0]) if pred_rows > 0 else 0
        
        matches = 0
        for r in range(min(gold_rows, pred_rows)):
            for c in range(min(gold_cols, pred_cols)):
                g_val = str(gold[r][c])
                p_val = str(pred_grid[r][c])
                if g_val == p_val:
                    matches += 1
        
        total_cells = gold_rows * gold_cols
        if total_cells == 0:
            return 1.0
            
        accuracy = matches / total_cells
        # Penalty for shape mismatch
        shape_factor = (min(gold_rows, pred_rows) / max(gold_rows, pred_rows)) * \
                       (min(gold_cols, pred_cols) / max(gold_cols, pred_cols))
        
        return accuracy * shape_factor

    def sample_input(
        self,
        *,
        rng: random.Random,
        pool: list[int],
        size_constraints: Any,
        repetition_mode: str = "medium",
        **kwargs: Any
    ) -> list[list[int]]:
        if isinstance(size_constraints, int):
            # If constrained by total N, reliably map to distinct shapes so OOD probes use distinct dimensional checks
            N = max(1, size_constraints)
            if N == 1:
                R, C = 1, 1
            elif N == 2:
                R, C = 1, 2
            elif N == 3:
                R, C = 3, 1
            elif N == 4:
                R, C = 2, 2
            elif N == 5:
                R, C = 1, 5
            elif N == 6:
                R, C = 3, 2
            elif N == 7:
                R, C = 7, 1
            elif N == 8:
                R, C = 2, 4
            else:
                R, C = 1, N
        else:
            # Tuple (R, C)
            R, C = size_constraints
            
        if R <= 0 or C <= 0:
            return []
            
        return [[rng.choice(pool) for _ in range(C)] for _ in range(R)]
