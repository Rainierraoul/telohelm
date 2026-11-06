"""Task packs: what Telohelm knows about one task, behind one object (paper2/notes/REUSABILITY-AUDIT.md,
TASK-PACK-REVIEW-CODEX.md, COMPONENT-MAP-COMPARISON.md).

Slice 1 (22 October 2026) holds a pack's public contract only: its decisions and their output fields, the context and
output schemas, and the exact task texts the learner and the in-flight models receive. A public contract gives no
handle to the world, the monitor, the reference pilot, the certificates or the disturbance prior; those belong to the
pack's execution and protected-monitor views, which later slices add. The composition root (run_stream.py) chooses the
concrete pack. Telohelm's own texts, the parts that are the same for every task, stay in the core and are joined to
the pack's parts there.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .example_view import ExamplePolicy


@dataclass(frozen=True)
class PublicContract:
    pack_id: str
    schema_version: int
    decisions: tuple[str, ...]                     # the decisions a procedure can be learned for
    decision_fields: Mapping[str, frozenset]       # the action fields each decision writes
    pair_decisions: tuple[str, ...]                # decisions a procedure and its check may be proposed for together
    context_fields: frozenset                      # what a procedure or check may read
    output_fields: frozenset                       # what a procedure may return
    learner_purpose: str                           # completes "You improve a library of Python procedures and checks that ..."
    learner_actions: str                           # each decision's action format
    learner_goal: str                              # how the goal is judged
    learner_tools: Callable[[Mapping], str]        # the task's tools, given the core's limits
    learner_trial_rule: str                        # what passing a trial means for each decision
    learner_attribute_example: str                 # how to read ctx and out, by example
    learner_decision_choices: str                  # the decisions as the reply format names them
    flight_system: str                             # the in-flight models' system prompt
    flight_ask: Mapping[str, str]                  # the in-flight models' question for each decision

    def __post_init__(self):
        """Take copies and freeze them, so neither the caller's objects nor anyone holding the pack can change it
        after it is made (Codex's TURN225 witnesses: the mappings were writable despite frozen=True)."""
        def fix(name, value):
            object.__setattr__(self, name, value)
        fix("decisions", tuple(self.decisions))
        fix("pair_decisions", tuple(self.pair_decisions))
        fix("context_fields", frozenset(self.context_fields))
        fix("output_fields", frozenset(self.output_fields))
        fix("decision_fields", MappingProxyType({k: frozenset(v) for k, v in dict(self.decision_fields).items()}))
        fix("flight_ask", MappingProxyType({k: str(v) for k, v in dict(self.flight_ask).items()}))
        for name in ("pack_id", "learner_purpose", "learner_actions", "learner_goal", "learner_trial_rule",
                     "learner_attribute_example", "learner_decision_choices", "flight_system"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be text")
        if not callable(self.learner_tools):
            raise TypeError("learner_tools must be callable")

    def fingerprint(self, limits: Mapping) -> str:
        """SHA-256 over every public value, with the tool text rendered under the given limits."""
        body = {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in self.__dict__.items()
                if k not in ("learner_tools", "decision_fields", "flight_ask")}
        body["decision_fields"] = {k: sorted(v) for k, v in self.decision_fields.items()}
        body["flight_ask"] = dict(self.flight_ask)
        body["learner_tools"] = self.learner_tools(limits)
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class TaskPack:
    """Everything Telohelm's core needs to know about one task, in one object (paper2/notes/PLAN-23-OCTOBER.md, "How
    Telohelm slots into the world"). The composition root hands it to the core; a new task is a new pack.

    public: what the learner and the in-flight models are told, and the decisions' fields (PublicContract).
    goal: how the goal splits into parts and which decisions each part points to (goal_parts.GoalParts).
    roles: the decisions asked each second, the one that flies, the once-per-goal ones and the fields each needs
        (decisions.DecisionRoles).
    help: when stale evidence may prompt a request for help (decisions.HelpEvidence).
    part_error: the public rules for the values of the task's action fields; returns the reason a part may not act.
    context_builder: makes, from the briefing and each second's observations, what Telohelm sees.
    evidence: the opt-in briefing check's measurement of disagreement (evidence.EvidenceView), or None.
    hidden: context fields the runtime's own rules read but procedures, checks and models do not see. Empty by
        default; the reduced-feature option hides the fields the context builder derives, so Telohelm's own code
        must work them out (paper2/notes/PLAN-23-OCTOBER.md, "Orientation Telohelm builds itself").
    """
    public: PublicContract
    goal: Any
    roles: Any
    help: Any
    part_error: Callable[[Mapping], str | None]
    context_builder: Callable[[Mapping], Any]
    evidence: Any = None
    hidden: frozenset = frozenset()
    example_policy: ExamplePolicy | None = None    # recorded moments, projection, event details and public rules

    environment: Any = None    # trusted host simulator/trial view, outside the public contract
    model_precondition: Callable | None = None    # (decision, public context, own part) -> {reason, evidence} or None
