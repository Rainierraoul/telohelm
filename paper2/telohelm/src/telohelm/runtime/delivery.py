"""Delayed delivery of model replies inside an episode (DISTURBANCES.md section 8; paper2/notes/MODEL-LAYER.md).

A model call made at second s with modelled latency L is charged when it is made, through the gateway, and its reply,
or its failure, is held until second s + ceil(L). The episode loop asks for due replies in the reply phase of each
second: after the world's termination checks, goal changes and reports for that second, and before the condition's
actions. A reply keeps the tags it was issued with: request identifier, decision, goal version, issue second and step,
and the SHA-256 of the context it was given. If the goal version has changed by delivery, the reply is delivered
marked out of date and its proposed action is refused. While a call for a decision is pending, no second call is made
for that decision. A reply due after the episode has ended is never delivered; its call is still charged and the
ledger lists it as straddling that ending.

Zero latency (Codex's TURN 67 condition): a call issued during the decisions of second t with L = 0 cannot use the
reply phase of t, which has already passed. It gets an explicit post-issuance subphase in the same second: the reply is
delivered straight after the call returns and can shape that second's action. This is a declared engineering boundary,
not a measured timing.

This module uses stub providers only through the gateway; parsing a reply into an action part is supplied by the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from ..taskpacks.tanker.public import TANKER
from .models import ModelGateway

# The fields each decision may write, so one decision's part cannot overwrite another's (Codex's MS76.ACCEPT).
DECISION_FIELDS = dict(TANKER.decision_fields)        # from the task's public contract
# The action keys a model reply may propose, every decision's fields together; the world validates their values.
ACTION_KEYS = set().union(*DECISION_FIELDS.values())


class ReplyRefused(ValueError):
    """Raised by a parser for a reply it cannot turn into an action part. Any ValueError (a JSON error included) is
    treated the same way; any other exception is an unrelated fault and is not masked (Codex's DL70.PARSE)."""


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Pending:
    """One issued call and its reply, with the tags it was issued under. Immutable once issued."""
    order: int
    decision: str
    role: str                 # small_model or reasoning_model, the source role of any part it gives
    request_id: str
    goal_version: int
    issued_second: int
    issued_step: int
    available_second: int
    context_sha256: str
    status: str
    text: str | None
    reportable: bool


@dataclass
class DelayedModels:
    """The trusted scheduler between the episode loop and one episode's model gateway.

    role_for(decision, reason) picks the role; messages_for(decision, ctx, reason) builds the messages from the context
    (it must not reach anything but the context); latency_s[role] is the modelled latency; parse(decision, text)
    turns a reply into an action part or None."""
    gateway: ModelGateway
    role_for: Callable[[str, str], str]
    messages_for: Callable[[str, dict, str], list]
    latency_s: dict[str, float]
    parse: Callable[[str, str], dict | None]
    max_output_tokens: int | dict[str, int] = 1024    # one limit for every role, or one per role (decision D26)
    # may_issue(role, decision, t), if given, is asked before a call is sent; False makes no call and charges nothing
    # here (the flight ladder uses it to draw reasoning calls from the episode's shared allowance, D26).
    may_issue: Callable[[str, str, int], bool] | None = None
    wait_decisions: frozenset[str] = field(default_factory=frozenset)
    # Opt-in host contract: an empty dictionary is a deliberate no-action answer for these decisions only.
    _pending: list[Pending] = field(default_factory=list)
    _order: int = 0

    def __post_init__(self):
        if isinstance(self.wait_decisions, str) or any(not isinstance(d, str) or not d for d in self.wait_decisions):
            raise ValueError("wait decisions must be a collection of decision names")
        self.wait_decisions = frozenset(self.wait_decisions)

    def pending_for(self, decision: str) -> bool:
        return any(p.decision == decision for p in self._pending)

    def issue(self, decision: str, ctx: dict, reason: str, t: int, action_step: int) -> Pending | None:
        """Make the call for a decision that climbed, unless one is already pending for it. Returns the pending record,
        or None if one was already pending or the role's allowance is used up (refused calls are not charged)."""
        if self.pending_for(decision):
            return None
        role = self.role_for(decision, reason)
        if self.may_issue is not None and not self.may_issue(role, decision, t):
            return None
        limit = self.max_output_tokens[role] if isinstance(self.max_output_tokens, dict) else self.max_output_tokens
        out = self.gateway.submit(role, self.messages_for(decision, ctx, reason), limit,
                                  ctx["goal_version"], t, action_step, self.latency_s[role])
        if out["status"] == "refused":
            return None
        self._order += 1
        p = Pending(self._order, decision, role, out["request_id"], ctx["goal_version"], t, action_step,
                    out["available_second"], _sha(ctx), out["status"], out["text"], out["reportable"])
        self._pending.append(p)
        return p

    def due(self, t: int) -> list[Pending]:
        """Replies whose second has come, in the order their calls were made, removed from the pending list. A reply
        is never handed over before its second; a pending reply with an earlier second would be a fault."""
        if any(p.available_second < t for p in self._pending):
            raise RuntimeError("a reply was not delivered at its second")
        ready = [p for p in self._pending if p.available_second == t]
        self._pending = [p for p in self._pending if p.available_second != t]
        return sorted(ready, key=lambda p: p.order)

    def undelivered(self) -> list[Pending]:
        """At the end of the episode: replies that will never be delivered."""
        left, self._pending = self._pending, []
        return left

    def part_for(self, p: Pending, current_goal_version: int, *,
                 action_keys: frozenset[str] | None = None) -> tuple[dict | None, bool, str | None]:
        """The action part a delivered reply proposes, whether it is out of date, and the public refusal, if any.

        action_keys comes from the selected task's decision fields. Omitting it preserves the earlier tanker API.
        Parser contract (Codex's DL70.PARSE): parse(decision, text) returns an action part, a dict whose keys are
        action keys, or None for no action; for a reply it cannot use it raises ReplyRefused or another ValueError.
        Those become the refusal 'unparseable', and a part of the wrong shape becomes 'invalid_part'; no error text
        is passed on. Any other exception propagates, so an unrelated fault is never hidden as a bad reply."""
        stale = p.goal_version != current_goal_version
        if stale:
            return None, True, "out_of_date"
        if p.status != "answered" or p.text is None:
            return None, False, None
        try:
            part = self.parse(p.decision, p.text)
        except ValueError:
            return None, False, "unparseable"
        if part is None:
            return None, False, None
        allowed = ACTION_KEYS if action_keys is None else action_keys
        if (not isinstance(part, dict) or (not part and p.decision not in self.wait_decisions)
                or not set(part) <= allowed):
            return None, False, "invalid_part"
        try:                     # values must be finite JSON that can be written as UTF-8, or logging the part
            json.dumps(part, ensure_ascii=False, allow_nan=False).encode("utf-8")   # would fail (DL72.JSON, DL74.UTF8)
        except (TypeError, ValueError, OverflowError):
            return None, False, "invalid_part"
        return part, False, None


def delivery_record(p: Pending, stale: bool, part: dict | None, subphase: str, refusal: str | None = None) -> dict:
    """The public checkpoint for a delivery: the original tags, the reply's SHA-256, the outcome, any public refusal and
    where in the second it happened."""
    return {"request_id": p.request_id, "decision": p.decision, "role": p.role, "goal_version": p.goal_version,
            "issued_second": p.issued_second, "issued_step": p.issued_step, "available_second": p.available_second,
            "context_sha256": p.context_sha256, "status": p.status,
            "reply_sha256": None if p.text is None else hashlib.sha256(p.text.encode("utf-8", "surrogatepass")).hexdigest(),
            "stale": stale, "part": part, "refusal": refusal, "reportable": p.reportable, "subphase": subphase}
