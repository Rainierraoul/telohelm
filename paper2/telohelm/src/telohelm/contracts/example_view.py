"""Trusted task interpretation of recorded examples; verification and evidence ownership remain in the core."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

METHOD = {"procedure": "stored procedure", "small_model": "small model", "reasoning_model": "reasoning model",
          "none": "no method"}


def applied_at(t: int, steps: dict) -> bool | None:
    """Whether the world accepted this action; absent acceptance remains unknown."""
    return None if t not in steps else steps[t].get("accepted") is True


@dataclass(frozen=True)
class ExamplePolicy:
    """Authored host callbacks, not learned tools. They interpret already recorded task data.

    points(decisions, failed, outcome, goal) returns recorded seconds and omission reasons.
    project(context) returns the displayed context and task-specific receipt statistics.
    details(t, outcome, actions, sources, steps, goal_changes) returns public event annotations.
    rules describes these examples to the learner. The monitor owns private outcome data; each task's projection
    must be reviewed for what it exposes. This interface is not a new security boundary.
    """
    rules: str
    points: Callable
    project: Callable
    details: Callable

    def __post_init__(self):
        if not isinstance(self.rules, str) or not self.rules:
            raise ValueError("example rules must be nonempty text")
        if not all(callable(getattr(self, name)) for name in ("points", "project", "details")):
            raise TypeError("example policy callbacks must be callable")
