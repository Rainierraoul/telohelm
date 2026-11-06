"""How a task's evidence can disagree with what Telohelm was told (opt-in; paper2/notes/HANDOVER-SIGNALS.md).

A task pack names one public measurement of disagreement: how far the newest observation in the context is from what
the briefing alone predicts for it. It also says how large a gap it tolerates, which decisions' procedures read that
evidence, what part withdraws a standing order a withheld procedure left in the world, and the sentence the flight
models are given when a procedure is withheld. Nothing here knows which disturbances a task has (D8): the gap is
computed from the briefing and the observations Telohelm received.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class EvidenceView:
    gap: Callable[[Mapping, Mapping], float | None]   # (briefing, context) -> the gap, or None with no observation
    tolerance: float                                   # a gap above this opens an incident
    unit: str                                          # the gap's unit, as the models are told it
    affects: frozenset                                 # decisions whose procedures read this evidence
    withdraw: Mapping[str, Mapping[str, Any]]          # decision -> the part that withdraws its standing order
    meaning: str                                       # what the gap is, for the flight models

    def __post_init__(self):
        object.__setattr__(self, "affects", frozenset(self.affects))
        object.__setattr__(self, "withdraw", MappingProxyType({k: MappingProxyType(dict(v))
                                                               for k, v in dict(self.withdraw).items()}))
        if not callable(self.gap) or not isinstance(self.meaning, str) or not isinstance(self.unit, str):
            raise TypeError("an evidence view needs a gap function, a unit and a meaning")
        if not self.tolerance > 0:
            raise ValueError("the tolerance must be above 0")
