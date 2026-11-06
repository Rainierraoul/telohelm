"""How a task's decisions behave each second (slice 2(b) of the task pack; paper2/notes/SLICE-2-INVENTORY.md).

A task names its decisions in the order they are asked each second, the one that moves the vehicle (the help rule and
the attribution of each second follow it), the ones that act once per goal and are not asked again after acting, and
the context fields a decision's procedure needs before it runs; with one missing, the decision climbs to a model on
the reason of missing or stale evidence. For the tanker these were written inline in runtime/episode_loop.py and
runtime/chooser.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping


@dataclass(frozen=True)
class DecisionRoles:
    order: tuple[str, ...]                         # the decisions asked each second, in this order
    flying: str                                    # the decision that moves the vehicle
    once_per_goal: frozenset                       # not asked again once it has acted under the goal version
    needs: Mapping[str, tuple]                     # context fields each decision's procedure needs
    completion_fields: Mapping[str, frozenset] = field(default_factory=dict)
    # For each once-per-goal decision, action fields whose issuance completes it; deferred setup fields are omitted.

    def __post_init__(self):
        completion = dict(self.completion_fields)
        if any(isinstance(v, str) or not v or any(not isinstance(k, str) or not k for k in v)
               for v in completion.values()):
            raise ValueError("completion fields must be nonempty collections of action field names")
        object.__setattr__(self, "completion_fields", MappingProxyType({k: frozenset(v) for k, v in completion.items()}))
        object.__setattr__(self, "order", tuple(self.order))
        object.__setattr__(self, "once_per_goal", frozenset(self.once_per_goal))
        object.__setattr__(self, "needs", MappingProxyType({k: tuple(v) for k, v in dict(self.needs).items()}))
        if self.flying not in self.order or not self.once_per_goal <= set(self.order):
            raise ValueError("the flying and once-per-goal decisions must be among the decisions asked")
        if set(self.completion_fields) != self.once_per_goal:
            raise ValueError("each once-per-goal decision needs exactly one completion-field entry")
        if not set(self.order) <= set(self.needs):
            raise ValueError("every decision asked needs an entry in needs, even an empty one")

    def asked(self, acted: Iterable[str] = ()) -> tuple[str, ...]:
        """The decisions to ask this second, leaving out once-per-goal decisions that have acted under this goal."""
        done = set(acted) & self.once_per_goal
        return tuple(d for d in self.order if d not in done)


@dataclass(frozen=True)
class HelpEvidence:
    """When the evidence for the decision that flies is too old to fly on, the runtime may ask the person (one
    request an episode, with the reason missing or stale evidence). The task names the context field holding the
    evidence's age, the age above which it is stale, and the request's text."""
    age_field: str
    stale_after_s: float
    text: str

    def stale(self, ctx: Mapping) -> bool:
        age = ctx.get(self.age_field)
        return age is None or age > self.stale_after_s
