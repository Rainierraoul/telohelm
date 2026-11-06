"""A task's goal in parts (slice 2(a) of the task pack; paper2/notes/SLICE-2-INVENTORY.md).

The monitor judges an episode's goal as a few parts, each read from one true-or-false field of the outcome. When the
goal is not met, the learner is told which part failed by a label, and a learning attempt is queued only if a failed
part depends on a decision being learned. For the tanker these rules were written inline in learning.py (after_episode
and the floor) and runtime/episode_loop.py (the feedback at episode end); this object states them once, for any task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class GoalPart:
    field: str                                     # the outcome field that is true when this part is met
    label: str                                     # what the learner is told when only this part failed
    decisions: frozenset                           # the decisions this part depends on

    def __post_init__(self):
        object.__setattr__(self, "decisions", frozenset(self.decisions))
        for name in ("field", "label"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be text")


@dataclass(frozen=True)
class GoalParts:
    parts: tuple[GoalPart, ...]
    all_failed: str                                # the label when every part failed

    def __post_init__(self):
        object.__setattr__(self, "parts", tuple(self.parts))
        if not self.parts or not all(isinstance(p, GoalPart) for p in self.parts):
            raise TypeError("a goal has one or more parts")
        labels = [p.label for p in self.parts] + [self.all_failed]
        fields = [p.field for p in self.parts]
        if len(set(labels)) != len(labels) or len(set(fields)) != len(fields):
            raise ValueError("each part needs its own field and label, and the all-failed label must differ")

    def failed(self, outcome: Mapping) -> str | None:
        """The label for the parts this outcome failed: None if none failed, a part's own label if only it failed,
        the all-failed label if every part failed. A goal of three or more parts would need a label for the other
        combinations; this refuses them rather than guess one."""
        missed = [p for p in self.parts if not outcome[p.field]]
        if not missed:
            return None
        if len(missed) == 1:
            return missed[0].label
        if len(missed) == len(self.parts):
            return self.all_failed
        raise ValueError(f"no label for failing {[p.field for p in missed]} together")

    def labels(self) -> tuple:
        """Every label the learner can be told at episode end, None for a goal met (the broker's feedback schema)."""
        return tuple(p.label for p in self.parts) + (self.all_failed, None)

    def concerns(self, label: str | None) -> frozenset:
        """The decisions a failure label points to: a part's own, or every part's for the all-failed label. Nothing
        for None or a label that is not a part's, such as the floor's."""
        if label == self.all_failed:
            return frozenset().union(*(p.decisions for p in self.parts))
        return next((p.decisions for p in self.parts if p.label == label), frozenset())

    def met_for(self, goal_met: bool, failed: str | None, learned: Iterable[str]) -> bool:
        """Whether an episode counts as met for the floor when only the decisions in learned are being learned: it met
        the goal, or it failed only in parts that depend on none of them. Help is judged by the caller."""
        return bool(goal_met) or (failed is not None and not (self.concerns(failed) & set(learned)))
