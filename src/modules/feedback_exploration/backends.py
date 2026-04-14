from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Optional

from core.representations.base import AbstractionSet, RuleSpec
from modules.feedback_exploration.types import FeedbackPolicy


class FeedbackBackend(ABC):
    def __init__(self, abstraction: AbstractionSet):
        self.abstraction = abstraction

    def resolve_rule(self, spec: RuleSpec):
        return self.abstraction.resolve_rule(spec)

    def render_input(self, x: Any) -> str:
        return self.abstraction.render(x)

    def render_output(self, y: Any) -> str:
        return self.abstraction.render(y)

    @abstractmethod
    def parse_answer(self, raw_text: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_correct(self, pred: Any, gold: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def soft_score(self, pred: Any, gold: Any) -> float:
        raise NotImplementedError

    def check_toolbox_matches(self, pred: Any, metadata: Optional[dict]) -> Optional[str]:
        if not metadata or "toolbox_outputs" not in metadata:
            return None
        
        outputs = metadata["toolbox_outputs"]
        # Use the appropriate phase key (interaction or shifted_interaction)
        # In feedback exploration, the 'support_pool[0]' matches 'interaction' results.
        phase_key = "interaction" if "interaction" in outputs.get(next(iter(outputs), {}), {}) else "shifted_interaction"
        
        pred_rendered = self.render_output(pred)
        for op_name, op_outs in outputs.items():
            if pred_rendered == self.render_output(op_outs.get(phase_key)):
                return f"Incorrect. Your answer matches the output of the operator '{op_name}', but that is not the hidden rule."
        return None

    def feedback_message(
        self,
        pred: Any,
        gold: Any,
        policy: FeedbackPolicy,
        attempt: int = 1,
        rule_family: Optional[str] = None,
        metadata: Optional[dict] = None,
        inp: Optional[Any] = None,
    ) -> str:
        """Dispatches to interaction_hint (property-describing) for the interaction phase.
        Falls back to pred-vs-gold structural hints for the probe phase."""
        if self.is_correct(pred, gold):
            return "Correct."

        if policy.binary_feedback_only:
            return "Incorrect."

        # Interaction phase: describe gold output properties (cross-referenceable with toolbox)
        if inp is not None:
            return self.interaction_hint(inp, gold, attempt, metadata or {})

        # Probe phase: pred-vs-gold structural escalation
        shape = self.shape_hint(pred, gold)
        if shape:
            return shape
        if attempt == 1:
            return "Incorrect."
        if attempt == 2:
            h = self.structure_hint(pred, gold, 2)
            return h if h else "Incorrect structure."
        if attempt == 3:
            h = self.structure_hint(pred, gold, 3)
            return h if h else "Incorrect."
        if attempt == 4:
            h = self.semantic_hint(pred, gold, 4, rule_family)
            return h if h else "Incorrect symbols."
        if rule_family:
            return f"Hint: The rule belongs to the '{rule_family}' family."
        return "Incorrect."

    @abstractmethod
    def interaction_hint(self, inp: Any, gold: Any, attempt: int, metadata: dict) -> str:
        """Generate a hint describing a property of the gold output for hypothesis narrowing."""
        raise NotImplementedError

    # ---- Family description helpers (shared) ----
    _FAMILY_DESC = {
        "permutation": "rearranges the input elements without adding or removing any",
        "partitioning": "selects a subset of the input elements",
        "iteration": "repeats or extends the input elements",
        "reindexing": "shifts or rotates the position of elements",
        "extension": "appends additional elements to the input",
    }
    _GRID_FAMILY_DESC = {
        "permutation": "rearranges the grid structure (e.g. transpose, flip rows/cols)",
        "partitioning": "crops or selects a region of the grid",
        "iteration": "tiles or repeats the grid along an axis",
        "reindexing": "shifts grid elements spatially along an axis",
        "extension": "appends rows or columns to the grid",
    }

    @abstractmethod
    def shape_hint(self, pred: Any, gold: Any) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def structure_hint(self, pred: Any, gold: Any, attempt: int) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def semantic_hint(self, pred: Any, gold: Any, attempt: int, rule_family: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError

# Helper
def _prefix_match_len(p: list[Any], g: list[Any]) -> int:
    i = 0
    while i < len(p) and i < len(g) and p[i] == g[i]:
        i += 1
    return i

def _suffix_match_len(p: list[Any], g: list[Any]) -> int:
    i = 0
    p_rev, g_rev = p[::-1], g[::-1]
    while i < len(p_rev) and i < len(g_rev) and p_rev[i] == g_rev[i]:
        i += 1
    return i


class SequenceFeedbackBackendBase(FeedbackBackend):
    def parse_answer(self, raw_text: str) -> Any:
        raise NotImplementedError

    def is_correct(self, pred: Any, gold: Any) -> bool:
        """Binary check for exact matching."""
        return pred == gold

    def soft_score(self, pred: Any, gold: Any) -> float:
        """Grant granular progress based on position and token overlap."""
        if not gold:
            return 1.0 if not pred else 0.0
        if not pred:
            return 0.0
            
        same_pos = sum(1 for a, b in zip(pred, gold) if a == b)
        pos_score = same_pos / max(len(gold), 1)
        
        p_count = Counter(pred)
        g_count = Counter(gold)
        overlap = sum(min(p_count[k], g_count[k]) for k in (set(p_count) | set(g_count)))
        bag_score = overlap / max(len(gold), 1)
        
        # Combined score: arrangement (50%) + content (50%)
        return 0.5 * pos_score + 0.5 * bag_score

    def shape_hint(self, pred: Any, gold: Any) -> Optional[str]:
        if len(pred) != len(gold):
            if len(pred) < len(gold):
                return "Incorrect. Output is too short."
            return "Incorrect. Output is too long."
        return None

    def structure_hint(self, pred: Any, gold: Any, attempt: int) -> Optional[str]:
        if len(pred) != len(gold):
            return None
        if pred == gold:
            return None

        # --- Round-aware progression ---
        
        # Level 1: Arrangement / Histogram check (if content is right but positions wrong)
        if Counter(pred) == Counter(gold):
            if attempt < 3:
                return "Incorrect. Symbols correct but arrangement wrong."
            else:
                first_bad = next((i for i, (a, b) in enumerate(zip(pred, gold)) if a != b), None)
                if first_bad is not None:
                    return f"Incorrect. Symbols correct but arrangement wrong. First mismatch at position {first_bad + 1}."

        # Level 2: Prefix/Suffix checking (Early hints)
        if attempt < 3:
            prefix_len = _prefix_match_len(pred, gold)
            suffix_len = _suffix_match_len(pred, gold)
            if prefix_len > 0 and suffix_len == 0:
                return f"Incorrect. First {prefix_len} symbols were correct."
            if suffix_len > 0 and prefix_len == 0:
                return f"Incorrect. Last {suffix_len} symbols were correct."
            if prefix_len > 0 and suffix_len > 0:
                return f"Incorrect. Some symbols at the start and end are correct."
            # If we are in an early round and none of the above specific hints apply:
            return "Incorrect structure."

        # Level 3: Precise position (Late hints: attempt >= 3)
        wrong_positions = sum(1 for a, b in zip(pred, gold) if a != b)
        if wrong_positions == 1:
            return "Incorrect. Exactly one position is wrong."

        first_bad = next((i for i, (a, b) in enumerate(zip(pred, gold)) if a != b), None)
        if first_bad is not None:
            return f"Incorrect. The symbol at position {first_bad + 1} is wrong."

        return "Incorrect structure."


class StringFeedbackBackend(SequenceFeedbackBackendBase):
    def parse_answer(self, raw_text: str) -> list[str]:
        s = raw_text.strip()
        if not s:
            return []
        normalized = s.replace(",", " ").replace("|", " ")
        return [tok.strip() for tok in normalized.split() if tok.strip()]

    def semantic_hint(self, pred: Any, gold: Any, attempt: int, rule_family: Optional[str] = None) -> Optional[str]:
        if not pred or not gold:
            return None
        p_count, g_count = Counter(pred), Counter(gold)
        mismatched = [t for t in g_count if p_count[t] != g_count[t]]
        if mismatched:
            tok = mismatched[0]
            return f"Incorrect. Token '{tok}' should appear {'more' if p_count[tok] < g_count[tok] else 'less'} often."
        if len(pred) == len(gold):
            if pred[0] != gold[0] and pred[-1] == gold[-1]:
                return "Incorrect. The first token is wrong."
            if pred[-1] != gold[-1] and pred[0] == gold[0]:
                return "Incorrect. The last token is wrong."
        return None

    def interaction_hint(self, inp: Any, gold: Any, attempt: int, metadata: dict) -> str:
        """4-tier: shape → composition → boundary → family"""
        inp_toks  = inp.split()  if isinstance(inp, str)  else [str(x) for x in inp]
        gold_toks = gold.split() if isinstance(gold, str) else [str(x) for x in gold]

        if attempt == 1:  # H1 — shape
            return f"The output has {len(gold_toks)} token(s)."

        if attempt == 2:  # H2 — composition
            if Counter(gold_toks) == Counter(inp_toks):
                return "The output contains the same symbols as the input, just rearranged."
            if len(gold_toks) < len(inp_toks):
                return "The output contains fewer tokens than the input."
            if len(gold_toks) > len(inp_toks):
                return "The output contains more tokens than the input (elements are repeated)."
            return "The output contains different symbols than the input."

        if attempt == 3:  # H3 — boundary relationship
            clues = []
            if inp_toks and gold_toks:
                if gold_toks[0] == inp_toks[0]:
                    clues.append("The first output token matches the first input token.")
                elif gold_toks[0] == inp_toks[-1]:
                    clues.append("The first output token matches the last input token.")
                if gold_toks[-1] == inp_toks[-1]:
                    clues.append("The last output token matches the last input token.")
                elif gold_toks[-1] == inp_toks[0]:
                    clues.append("The last output token matches the first input token.")
            return " ".join(clues) if clues else "No boundary tokens are preserved in their original positions."

        # H4 — family reveal
        fam = metadata.get("template_family") or metadata.get("shifted_template_family") or ""
        desc = self._FAMILY_DESC.get(fam, "applies a structural transformation")
        return f"Hint: The rule belongs to the '{fam}' family — it {desc}." if fam else "Hint: Focus on the structural relationship between input and output."


class NumberFeedbackBackend(SequenceFeedbackBackendBase):
    def parse_answer(self, raw_text: str) -> list[str]:
        s = raw_text.strip()
        if not s:
            return []
        normalized = s.replace(",", " ").replace("|", " ")
        return [tok.strip() for tok in normalized.split() if tok.strip()]

    def semantic_hint(self, pred: Any, gold: Any, attempt: int, rule_family: Optional[str] = None) -> Optional[str]:
        if not pred or not gold:
            return None
        p_count, g_count = Counter(pred), Counter(gold)
        mismatched = [t for t in g_count if p_count[t] != g_count[t]]
        if mismatched:
            tok = mismatched[0]
            return f"Incorrect. Number '{tok}' should appear {'more' if p_count[tok] < g_count[tok] else 'less'} often."
        if len(pred) == len(gold):
            if pred[0] != gold[0] and pred[-1] == gold[-1]:
                return "Incorrect. The first number is wrong."
            if pred[-1] != gold[-1] and pred[0] == gold[0]:
                return "Incorrect. The last number is wrong."
        return None

    def interaction_hint(self, inp: Any, gold: Any, attempt: int, metadata: dict) -> str:
        """Numbers treated as structural tokens — same 4-tier logic as String."""
        inp_toks  = inp.split()  if isinstance(inp, str)  else [str(x) for x in inp]
        gold_toks = gold.split() if isinstance(gold, str) else [str(x) for x in gold]

        if attempt == 1:
            return f"The output has {len(gold_toks)} number(s)."

        if attempt == 2:
            if Counter(gold_toks) == Counter(inp_toks):
                return "The output contains the same numbers as the input, just rearranged."
            if len(gold_toks) < len(inp_toks):
                return "The output contains fewer numbers than the input."
            if len(gold_toks) > len(inp_toks):
                return "The output contains more numbers than the input (elements are repeated)."
            return "The output contains different numbers than the input."

        if attempt == 3:
            clues = []
            if inp_toks and gold_toks:
                if gold_toks[0] == inp_toks[0]:
                    clues.append("The first output number matches the first input number.")
                elif gold_toks[0] == inp_toks[-1]:
                    clues.append("The first output number matches the last input number.")
                if gold_toks[-1] == inp_toks[-1]:
                    clues.append("The last output number matches the last input number.")
                elif gold_toks[-1] == inp_toks[0]:
                    clues.append("The last output number matches the first input number.")
            return " ".join(clues) if clues else "No boundary positions are preserved from the input."

        fam = metadata.get("template_family") or metadata.get("shifted_template_family") or ""
        desc = self._FAMILY_DESC.get(fam, "applies a structural transformation")
        return f"Hint: The rule belongs to the '{fam}' family — it {desc}." if fam else "Hint: Consider how the position of each number relates to its position in the input."


class GridFeedbackBackend(FeedbackBackend):
    def parse_answer(self, raw_text: str) -> list[list[int]]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return []

    def is_correct(self, pred: Any, gold: Any) -> bool:
        return pred == gold

    def soft_score(self, pred: Any, gold: Any) -> float:
        if not gold or not gold[0]:
            return 1.0 if not pred else 0.0
        if not pred or not isinstance(pred, list) or not isinstance(pred[0], list):
            return 0.0
        
        # very simple soft scoring for grids:
        if len(pred) == len(gold) and len(pred[0]) == len(gold[0]):
            correct = 0
            total = len(gold) * len(gold[0])
            for r in range(len(gold)):
                for c in range(len(gold[0])):
                    if pred[r][c] == gold[r][c]:
                        correct += 1
            return correct / total
        return 0.0

    def shape_hint(self, pred: Any, gold: Any) -> Optional[str]:
        if not isinstance(pred, list):
            return "Incorrect format. Expecting a 2D array."
        if len(pred) != len(gold):
            return f"Incorrect. Expected {len(gold)} rows, but got {len(pred)} rows."
        if len(pred) > 0 and len(gold) > 0:
            if not isinstance(pred[0], list):
                return "Incorrect format. Expecting a 2D array of items."
            if len(pred[0]) != len(gold[0]):
                return f"Incorrect. Expected {len(gold[0])} columns, but got {len(pred[0])} columns."
        return None

    def structure_hint(self, pred: Any, gold: Any, attempt: int) -> Optional[str]:
        if not isinstance(pred, list) or not isinstance(gold, list):
            return None
        if len(pred) != len(gold) or (len(pred) > 0 and len(pred[0]) != len(gold[0])):
            return None

        # Level 1: Row/Col correctness (Moderate hints)
        correct_rows = []
        for r_idx, (p_row, g_row) in enumerate(zip(pred, gold)):
            if p_row == g_row:
                correct_rows.append(r_idx + 1)
        
        if correct_rows and len(correct_rows) < len(gold):
            return f"Incorrect. Row(s) {', '.join(map(str, correct_rows))} are correct."

        # Level 2: Histogram/Permutation check
        pred_flat = [cell for row in pred for cell in row]
        gold_flat = [cell for row in gold for cell in row]
        if Counter(pred_flat) == Counter(gold_flat):
            if attempt < 3:
                return "Incorrect. All elements are correct, but they are in the wrong positions."
            else:
                # Find first cell mismatch
                for r in range(len(gold)):
                    for c in range(len(gold[0])):
                        if pred[r][c] != gold[r][c]:
                            return f"Incorrect. Mismatch at row {r+1}, column {c+1}."

        # Level 3: Partial cell hint (Late game)
        if attempt >= 3:
            for r in range(len(gold)):
                for c in range(len(gold[0])):
                    if pred[r][c] != gold[r][c]:
                        return f"Incorrect. At row {r+1}, column {c+1}, the expected value is {gold[r][c]}."

        return "Incorrect spatial arrangement."

    def semantic_hint(self, pred: Any, gold: Any, attempt: int, rule_family: Optional[str] = None) -> Optional[str]:
        try:
            pred_flat = [cell for row in pred for cell in row]
            gold_flat = [cell for row in gold for cell in row]
            p_count, g_count = Counter(pred_flat), Counter(gold_flat)
            mismatched = [v for v in g_count if p_count[v] != g_count[v]]
            if mismatched:
                val = mismatched[0]
                return f"Incorrect. Element {val} should appear {'more' if p_count[val] < g_count[val] else 'fewer'} times."
        except Exception:
            pass
        return None

    def interaction_hint(self, inp: Any, gold: Any, attempt: int, metadata: dict) -> str:
        """4-tier: dimensions → composition → spatial relationship → family"""
        try:
            inp_grid  = json.loads(inp)  if isinstance(inp, str)  else inp
            gold_grid = json.loads(gold) if isinstance(gold, str) else gold

            inp_rows  = len(inp_grid)
            inp_cols  = len(inp_grid[0])  if inp_grid and inp_grid[0]  else 0
            gold_rows = len(gold_grid)
            gold_cols = len(gold_grid[0]) if gold_grid and gold_grid[0] else 0

            if attempt == 1:  # H1 — dimensions
                return f"The output is a {gold_rows}×{gold_cols} grid."

            if attempt == 2:  # H2 — cell composition
                inp_flat  = [c for r in inp_grid  for c in r]
                gold_flat = [c for r in gold_grid for c in r]
                if Counter(inp_flat) == Counter(gold_flat):
                    return "The output contains the same cell values as the input, just in different positions."
                if len(gold_flat) < len(inp_flat):
                    return "The output has fewer cells than the input (a region was selected)."
                if len(gold_flat) > len(inp_flat):
                    return "The output has more cells than the input (elements were repeated)."
                return "The output has the same number of cells but different value distribution."

            if attempt == 3:  # H3 — spatial relationship
                if gold_rows == inp_cols and gold_cols == inp_rows and gold_rows != gold_cols:
                    return "The rows and columns of the output are swapped relative to the input (transposed)."
                if gold_rows == inp_rows and gold_cols == inp_cols:
                    return "The output has the same shape as the input, but elements are in different positions."
                if gold_rows == 1:
                    return "The output is a single row."
                if gold_cols == 1:
                    return "The output is a single column."
                return "The output dimensions differ significantly from the input."

            # H4 — family reveal
            fam = metadata.get("template_family") or metadata.get("shifted_template_family") or ""
            desc = self._GRID_FAMILY_DESC.get(fam, "applies a structural grid transformation")
            return f"Hint: The rule belongs to the '{fam}' family — it {desc}." if fam else "Hint: Consider how rows and columns of the input relate to the output grid."

        except Exception:
            fam = metadata.get("template_family", "")
            return f"Hint: The rule belongs to the '{fam}' family." if fam else "Incorrect."
