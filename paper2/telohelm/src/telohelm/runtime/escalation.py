"""The method ladder in flight (backlog B36, decision D26 amendments 1, 3 and 4; paper2/notes/B36-FLIGHT-LADDER-PLAN.md).

A decision with no accepted stored procedure climbs to the small model. One unusable small-model reply for the current
goal version (a provider failure, a reply refused at delivery, or a reply an applicable adopted check rejects) latches
that decision to the reasoning model until it has an accepted procedure output or a usable current reply, or the goal
changes. The small model's per-episode allowance running out also sends climbs to the reasoning model. A reasoning call
made in flight draws on the original episode's shared reasoning allowance (ladder.ReasoningAllowance), the same one the
rewrite attempt after the episode draws on, so flight and learning together never exceed 4 for that episode.

Help: at most one request per episode, with one declared reason, when a decision has no usable method and no reasoning
allowance is left for it, or when the task's evidence is stale (for the tanker, the goal tanker's newest report is
over 10 s old or missing) and no accepted stored procedure flew the decision that flies. The task's parts of these
rules, the stale-evidence rule and its fields' value rules, come from its pack (packs/tanker_execution.py). A request whose reply (30 s later) could not arrive strictly before the goal's deadline or
the horizon is suppressed, and the suppression recorded once per reason.

Version 0.11 (proposed; Simon's decision of 15 October, D39): flight may use at most FLIGHT_REASONING_LIMIT of the
episode's reasoning allocations, so the rewrite attempt after it keeps at least three. The runner passes it; a ladder
built without flight_limit keeps the D26 rule, under which flight could use all four.

FlightLadder holds only this episode's state. The episode loop tells it the goal version each second and what each
decision's method produced; it answers which role a climb goes to and whether a reasoning call may be made.
"""
from __future__ import annotations

from typing import Any

from ..taskpacks.tanker.public import TANKER
from ..taskpacks.tanker.execution import TANKER_HELP; from ..taskpacks.tanker.execution import tanker_part_error

STALE_S = TANKER_HELP.stale_after_s        # kept for the comparison conditions that read it
FLIGHT_REASONING_LIMIT = 1
OWN_FIELDS = TANKER.decision_fields       # the fields each decision may write (delivery.DECISION_FIELDS)


def own_part(decision: str, part: dict, *, fields: Any = OWN_FIELDS) -> dict:
    """Keep the fields this decision may write under the supplied task; default to the earlier tanker API."""
    return {k: v for k, v in part.items() if k in fields[decision]}


def part_error(decision: str, part: dict, task_rules: Any = tanker_part_error) -> str | None:
    """The public action rules for a decision's own fields, as stated to the models and enforced by the world: the
    task's rules for its fields' values (for the tanker, packs/tanker_execution.py), then the help request's form.
    None if the part may act (LAD163.ACTION)."""
    why = task_rules(part)
    if why is not None:
        return why
    if "help" in part:
        h = part["help"]
        if (not isinstance(h, dict) or set(h) != {"reason", "text"} or h["reason"] not in HELP_REASONS
                or not isinstance(h["text"], str) or len(h["text"]) > 500):
            return "help must name one declared reason and text of at most 500 characters"
    return None


class FlightLadder:
    def __init__(self, gateway: Any, allowance: Any = None, key: str | None = None, help_delay_s: int = 30,
                 horizon_s: int = 3600, flight_limit: int | None = None, help_evidence: Any = TANKER_HELP):
        self.gateway, self.allowance, self.key = gateway, allowance, key
        self.flight_limit, self.flight_allocated = flight_limit, 0
        self.help_delay_s, self.horizon_s = help_delay_s, horizon_s
        self.help_evidence = help_evidence     # when the task's evidence is too old to fly on, and the request's text
        self.goal_version: int | None = None
        self.latched: set[tuple[str, int]] = set()        # (decision, goal version) sent to the reasoning model
        self.no_reasoning: set[tuple[str, int]] = set()   # (decision, goal version) refused a reasoning call
        self.help_asked = False
        self._suppressed: set[str] = set()
        self.events: list[dict] = []

    # -- what the loop tells the ladder --------------------------------------------------------------------------
    def at(self, goal_version: int) -> None:
        self.goal_version = goal_version

    def unusable(self, decision: str, role: str, goal_version: int, why: str, t: int) -> None:
        """A due reply for this decision could not be used. Only a current-goal small-model reply latches; a reply to
        an old goal never escalates (D26 amendment 1)."""
        if role == "small_model" and goal_version == self.goal_version and (decision, goal_version) not in self.latched:
            self.latched.add((decision, goal_version))
            self.events.append({"kind": "escalated", "t": t, "decision": decision, "goal_version": goal_version,
                                "why": why})

    def usable(self, decision: str, goal_version: int, t: int, by: str) -> None:
        """An accepted procedure output or a usable current reply: the decision goes back to the cheaper order."""
        if (decision, goal_version) in self.latched:
            self.latched.discard((decision, goal_version))
            self.events.append({"kind": "released", "t": t, "decision": decision, "goal_version": goal_version,
                                "by": by})

    # -- what the ladder answers ---------------------------------------------------------------------------------
    def small_spent(self) -> bool:
        return self.gateway.used["small_model"] >= self.gateway.limits["small_model"]

    def role_for(self, decision: str, reason: str) -> str:
        if (decision, self.goal_version) in self.latched or self.small_spent():
            return "reasoning_model"
        return "small_model"

    def may_issue(self, role: str, decision: str, t: int) -> bool:
        """Before a call is sent: a reasoning call needs an allocation from the episode's shared allowance, and flight
        may hold at most flight_limit of them."""
        if role != "reasoning_model":
            return True
        capped = self.flight_limit is not None and self.flight_allocated >= self.flight_limit
        if capped or self.gateway.used["reasoning_model"] >= self.gateway.limits["reasoning_model"]:
            ok = False
        elif self.allowance is None:
            ok = True
        else:
            n = self.gateway.used["reasoning_model"] + 1
            ok = self.allowance.allocate(self.key, f"flight/call{n}", "flight", "provider") is not None
        if ok:
            self.flight_allocated += 1
        elif (decision, self.goal_version) not in self.no_reasoning:
            self.no_reasoning.add((decision, self.goal_version))
            self.events.append({"kind": "no_reasoning_allowance", "t": t, "decision": decision,
                                "goal_version": self.goal_version, **({"why": "flight_limit"} if capped else {})})
        return ok

    def help(self, ctx: dict, unresolved: dict[str, str], heading_by_procedure: bool) -> dict | None:
        """The help action to send this second, or None. unresolved maps each decision with no usable part this
        second, no pending call and no reasoning allowance left to its climb reason."""
        if self.help_asked:
            return None
        reason, text = None, None
        if unresolved:
            decision = sorted(unresolved)[0]
            reason = unresolved[decision] if unresolved[decision] in HELP_REASONS else "missing_tool"
            text = f"no usable method for the {decision} decision and no reasoning calls left"
        elif ((not heading_by_procedure or ctx.get("deadline") is not None)
              and self.help_evidence.stale(ctx)):
            # A learned procedure with no conditions always applies, so before 20 October it switched this rule off and
            # Telohelm never asked in a help case (paper2/notes/STACK-WALKTHROUGH.md). When the goal has a deadline,
            # waiting for the reports to come back risks the goal, so the runtime asks whatever flies the heading.
            # Without a deadline it waits, as before, so dropouts and late reports do not cause avoidable requests.
            reason, text = "missing_or_stale_evidence", self.help_evidence.text
        if reason is None:
            return None
        t, deadline = ctx["t"], ctx.get("deadline")
        reply_at = t + self.help_delay_s
        if reply_at >= self.horizon_s or (deadline is not None and reply_at >= deadline):
            if reason not in self._suppressed:
                self._suppressed.add(reason)
                self.events.append({"kind": "help_suppressed", "t": t, "reason": reason,
                                    "reply_at": reply_at, "deadline": deadline})
            return None
        return {"reason": reason, "text": text}         # proposed; it counts only once the world accepts it

    def help_outcome(self, t: int, help: dict, by: str, decision: str | None, accepted: bool) -> None:
        """Record a help request after the world's verdict on the whole action. Only an accepted request uses the
        episode's one request; a refused action carried no request (HELP167.ATOMIC)."""
        if accepted:
            self.help_asked = True
            self.events.append({"kind": "help_asked" if by == "ladder" else "help_by_model", "t": t,
                                "reason": help.get("reason"), "decision": decision})
        else:
            self.events.append({"kind": "help_refused_with_action", "t": t, "reason": help.get("reason"),
                                "by": by, "decision": decision})


HELP_REASONS = ("missing_tool", "missing_or_stale_evidence", "unfamiliar_but_solvable", "rules_changed")
