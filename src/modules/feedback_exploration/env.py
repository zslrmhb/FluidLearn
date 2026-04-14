from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.feedback_exploration.backends import FeedbackBackend
from modules.feedback_exploration.types import (
    FeedbackAction,
    FeedbackGameState,
    FeedbackTaskConfig,
    QueryState,
)


@dataclass
class StepResult:
    correct: bool
    soft_score: float
    feedback: str
    done: bool
    phase: str


class FeedbackGameEnv:
    """Single-player interactive environment for feedback-based learning."""

    def __init__(self, config: FeedbackTaskConfig, backend: FeedbackBackend):
        self.config = config
        self.backend = backend
        self.state = FeedbackGameState()

        self._pre_rule = self.backend.resolve_rule(config.rule_spec)
        self._post_rule = (
            self.backend.resolve_rule(config.shifted_rule_spec)
            if config.shifted_rule_spec is not None
            else None
        )

        self._pre_interaction = QueryState(
            inp=config.interaction_query,
            gold=self._pre_rule(config.interaction_query),
        )
        self._pre_probe = (
            QueryState(inp=config.probe_query, gold=self._pre_rule(config.probe_query))
            if config.probe_query is not None
            else None
        )

        self._post_interaction = (
            QueryState(
                inp=config.shifted_interaction_query,
                gold=self._post_rule(config.shifted_interaction_query),
            )
            if self._post_rule is not None and config.shifted_interaction_query is not None
            else None
        )
        self._post_probe = (
            QueryState(
                inp=config.shifted_probe_query,
                gold=self._post_rule(config.shifted_probe_query),
            )
            if self._post_rule is not None and config.shifted_probe_query is not None
            else None
        )

    def _phase_label(self) -> str:
        labels = {
            "pre_interaction": "PRE-SHIFT interaction",
            "pre_probe": "PRE-SHIFT probe",
            "post_interaction": "POST-SHIFT interaction",
            "post_probe": "POST-SHIFT probe",
            "done": "DONE",
        }
        return labels.get(self.state.phase, "DONE")

    def _current_query(self) -> Optional[QueryState]:
        if self.state.phase == "pre_interaction":
            return self._pre_interaction
        if self.state.phase == "pre_probe":
            return self._pre_probe
        if self.state.phase == "post_interaction":
            return self._post_interaction
        if self.state.phase == "post_probe":
            return self._post_probe
        return None

    def _advance_after_pre(self) -> str:
        if self._pre_probe is not None and self.state.phase == "pre_interaction":
            self.state.phase = "pre_probe"
            return "Moving to pre-shift probe."

        if self._post_interaction is not None:
            self.state.phase = "post_interaction"
            return "Moving to post-shift interaction. The underlying rule may have changed."

        self.state.phase = "done"
        self.state.finished = True
        return "Episode complete."

    def _advance_after_post(self) -> str:
        if self._post_probe is not None and self.state.phase == "post_interaction":
            self.state.phase = "post_probe"
            return "Moving to post-shift probe."

        self.state.phase = "done"
        self.state.finished = True
        return "Episode complete."

    def _advance_phase(self) -> str:
        if self.state.phase in {"pre_interaction", "pre_probe"}:
            if self.state.phase == "pre_probe" or self._pre_probe is None:
                if self._post_interaction is not None:
                    self.state.phase = "post_interaction"
                    return "Moving to post-shift interaction. The underlying rule may have changed."
                self.state.phase = "done"
                self.state.finished = True
                return "Episode complete."
            return self._advance_after_pre()

        if self.state.phase in {"post_interaction", "post_probe"}:
            if self.state.phase == "post_probe" or self._post_probe is None:
                self.state.phase = "done"
                self.state.finished = True
                return "Episode complete."
            return self._advance_after_post()

        self.state.finished = True
        self.state.phase = "done"
        return "Episode complete."

    def get_state_representation(self) -> str:
        current_query = self._current_query()
        if current_query is None:
            return "Episode complete."

        lines: list[str] = []
        lines.append(f"Phase: {self._phase_label()}")
        lines.append("")
        lines.append(self.config.task_instruction)
        lines.append("")

        if self.state.phase.startswith("post"):
            lines.append("The underlying rule may have changed.")
            lines.append("")

        if self.state.phase.endswith("interaction"):
            remaining = self.config.feedback_policy.max_rounds_per_shift - current_query.rounds_used
            lines.append(
                f"Interactive query ({remaining} round(s) remaining in this shift):"
            )
        else:
            lines.append("One-shot probe query:")

        lines.append(f"- {self.backend.render_input(current_query.inp)} -> ?")

        lines.append("")
        lines.append("--- Toolbox (Candidate Operations) ---")
        for op_id, spec in self.config.toolbox.items():
            try:
                op_fn = self.backend.resolve_rule(spec)
                out_raw = op_fn(current_query.inp)
                out_str = self.backend.render_output(out_raw)
                
                input_str = self.backend.render_input(current_query.inp)
                if "\n" in input_str or "\n" in out_str:
                    lines.append(f"[{op_id}] {spec.name}\nInput:\n{input_str}\nOutput:\n{out_str}\n")
                else:
                    lines.append(f"[{op_id}] {spec.name} | Output: {out_str}")
            except Exception:
                lines.append(f"[{op_id}] {spec.name} | Output: [Error]")
        lines.append("--------------------------------------")

        if current_query.last_feedback is not None:
            lines.append("")
            lines.append(f"Last feedback: {current_query.last_feedback}")

        lines.append("")
        lines.append('Select an operation by responding with JSON, e.g., {"operator": "Op_1"}')
        return "\n".join(lines)

    def step(self, action: FeedbackAction) -> StepResult:
        query = self._current_query()
        if query is None:
            return StepResult(
                correct=False,
                soft_score=0.0,
                feedback="Episode complete.",
                done=True,
                phase="done" if self.state.finished else self.state.phase,
            )

        import json
        selected_op = None
        if isinstance(action.answer, str):
            try:
                parsed = json.loads(action.answer)
                if isinstance(parsed, dict) and "operator" in parsed:
                    selected_op = parsed.get("operator")
            except Exception:
                pass

        if selected_op and selected_op in self.config.toolbox:
            chosen_spec = self.config.toolbox[selected_op]
            chosen_rule_fn = self.backend.resolve_rule(chosen_spec)
            try:
                pred_raw = chosen_rule_fn(query.inp)
                pred = pred_raw
                pred_rendered = self.backend.render_output(pred_raw)
            except Exception:
                pred = self.backend.parse_answer(action.answer) if isinstance(action.answer, str) else action.answer
                pred_rendered = "Error"
        else:
            pred = self.backend.parse_answer(action.answer) if isinstance(action.answer, str) else action.answer
            try:
                pred_rendered = self.backend.render_output(pred)
            except Exception:
                pred_rendered = str(pred)

        correct = self.backend.is_correct(pred, query.gold)
        soft = self.backend.soft_score(pred, query.gold)

        query.rounds_used += 1
        query.history.append(
            {
                "prediction": pred,
                "correct": correct,
                "soft_score": soft,
                "phase": "done" if self.state.finished else self.state.phase,
                "round": query.rounds_used,
            }
        )

        if self.state.phase.endswith("probe"):
            query.solved = correct
            query.last_feedback = "Correct." if correct else "Incorrect."
            transition = self._advance_phase()
            return StepResult(
                correct=correct,
                soft_score=soft,
                feedback=query.last_feedback if transition == "Episode complete." else f"{query.last_feedback} {transition}",
                done=self.state.finished,
                phase="done" if self.state.finished else self.state.phase,
            )

        if correct:
            query.solved = True
            query.last_feedback = "Correct."
            transition = self._advance_phase()
            return StepResult(
                correct=True,
                soft_score=soft,
                feedback=query.last_feedback if transition == "Episode complete." else f"{query.last_feedback} {transition}",
                done=self.state.finished,
                phase="done" if self.state.finished else self.state.phase,
            )

        current_rule_family = self.config.rule_spec.family
        if self.state.phase.startswith("post") and self.config.shifted_rule_spec:
            current_rule_family = self.config.shifted_rule_spec.family

        # Pass rendered inp during interaction so backends can generate cross-referenceable hints
        is_interaction = self.state.phase.endswith("interaction")
        rendered_inp = self.backend.render_input(query.inp) if is_interaction else None
        
        feedback = self.backend.feedback_message(
            pred,
            query.gold,
            self.config.feedback_policy,
            attempt=query.rounds_used,
            rule_family=current_rule_family,
            metadata=self.config.metadata,
            inp=rendered_inp,
        )
        
        if selected_op:
            feedback = f"Result of applying {selected_op}: {pred_rendered}\n{feedback}"
            
        query.last_feedback = feedback

        if query.rounds_used >= self.config.feedback_policy.max_rounds_per_shift:
            transition = self._advance_phase()
            return StepResult(
                correct=False,
                soft_score=soft,
                feedback=f"{feedback} {transition}",
                done=self.state.finished,
                phase="done" if self.state.finished else self.state.phase,
            )

        return StepResult(
            correct=False,
            soft_score=soft,
            feedback=feedback,
            done=False,
            phase="done" if self.state.finished else self.state.phase,
        )

    def is_done(self) -> bool:
        return self.state.finished
