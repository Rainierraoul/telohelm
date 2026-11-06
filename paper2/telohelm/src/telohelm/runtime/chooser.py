"""Choosing the method for one decision: the bottom of the ladder (plan section 5, backlog note B7).

The rule is fixed and simple. Take the adopted procedures for the decision whose declared conditions hold,
in the order they were added. Run each in the sandbox; the first whose output every adopted check for
that decision accepts is used. Otherwise climb, with a reason:
- 'missing_or_stale_evidence' when a field the decision needs is missing;
- 'missing_tool' when no adopted procedure fits, or the decision has no adopted check;
- 'unfamiliar_but_solvable' when procedures ran but none produced an accepted output.
'rules_changed' is never chosen here; only the reasoning model's reply can give it.
The rungs above, the small model, the reasoning model and the person, belong to the runtime above this.
"""
from __future__ import annotations

from typing import Any, Callable

from . import sandbox as S
from .library import Library
from .procedures import conditions_hold
from ..taskpacks.tanker.execution import TANKER_ROLES

NEEDED = TANKER_ROLES.needs       # the context fields each decision needs, from the task pack (slice 2(b))


def choose(decision: str, context: dict[str, Any], library: Library,
           run: Callable[..., S.SandboxResult] = S.run, broker: Any = None,
           needs: Any = NEEDED, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every attempt is recorded: for each procedure tried, its status, output and time, and for each applicable
    check its identity, status, verdict and time (Codex's R22.8: failed procedures used to lose these). A check
    applies only where its own stored conditions hold.

    With a broker (TURN 56), every procedure and check runs with the broker channel, as its own broker call named
    after the decision and entry, so it can call predict() within the episode's allowance."""
    if broker is not None:
        def run(code, entry, payload, context_args=False):
            return S.run_with_broker(code, entry, payload, broker, f"{decision}:{entry}", context_args=context_args)
    # Whether the evidence a decision needs is there is the runtime's rule, read from the full context (evidence);
    # the procedures and checks receive context, which may hide derived fields (the reduced-feature option).
    missing = [f for f in needs[decision] if (context if evidence is None else evidence).get(f) is None]
    if missing:
        return {"climb": "missing_or_stale_evidence", "missing": missing, "tried": []}
    order = {e["entry"]: e["seq"] for e in library.events if e["kind"] == "add"}
    checks = sorted((c for c in library.adopted(decision, "check")
                     if conditions_hold(library.entry(c)["conditions"], context)), key=order.get)
    procedures = sorted((p for p in library.adopted(decision, "procedure")
                         if conditions_hold(library.entry(p)["conditions"], context)), key=order.get)
    if not procedures or not checks:
        return {"climb": "missing_tool", "tried": [], "applicable_checks": checks}
    tried = []
    for sha in procedures:
        r = run(library.entry(sha)["source"], "decide", [context], context_args=True)
        attempt: dict[str, Any] = {"procedure": sha, "status": r.status, "output": r.result, "error": r.error,
                                   "wall_seconds": r.wall_seconds, "cpu_seconds": r.cpu_seconds, "checks": []}
        if r.status == "ok":
            for check in checks:
                c = run(library.entry(check)["source"], "check", [context, r.result], context_args=True)
                attempt["checks"].append({"check": check, "status": c.status, "verdict": c.result, "error": c.error,
                                          "wall_seconds": c.wall_seconds, "cpu_seconds": c.cpu_seconds,
                                          "accepted": c.status == "ok" and c.result == "accept"})
        attempt["accepted"] = r.status == "ok" and all(c["accepted"] for c in attempt["checks"])
        tried.append(attempt)
        if attempt["accepted"]:
            return {"use": sha, "output": r.result, "checks": checks, "tried": tried}
    return {"climb": "unfamiliar_but_solvable", "tried": tried, "checks": checks}


def check_output(decision: str, context: dict[str, Any], output: dict[str, Any], library: Library,
                 run: Callable[..., S.SandboxResult] = S.run, broker: Any = None) -> dict[str, Any]:
    """Run every applicable adopted check of a decision on an output that did not come from a stored procedure (a
    model's reply), as the chooser does for procedures (B36: every output passes the checks that apply to it). With no
    applicable adopted check, the output is accepted by the fixed gate alone, and that is recorded."""
    if broker is not None:
        def run(code, entry, payload, context_args=False):
            return S.run_with_broker(code, entry, payload, broker, f"{decision}:{entry}", context_args=context_args)
    order = {e["entry"]: e["seq"] for e in library.events if e["kind"] == "add"}
    checks = sorted((c for c in library.adopted(decision, "check")
                     if conditions_hold(library.entry(c)["conditions"], context)), key=order.get)
    records = []
    for check in checks:
        c = run(library.entry(check)["source"], "check", [context, output], context_args=True)
        records.append({"check": check, "status": c.status, "verdict": c.result, "error": c.error,
                        "wall_seconds": c.wall_seconds, "cpu_seconds": c.cpu_seconds,
                        "accepted": c.status == "ok" and c.result == "accept"})
    return {"accepted": all(r["accepted"] for r in records), "checks": records}
