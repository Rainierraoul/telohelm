"""D34 report predicates and trigger transitions, never authoritative B8 labels.

Inputs are supplied facts, not independently protected originals. This pure
component generates no trajectory or corpus and cannot certify a whole goal.
Constants/text semantics are pinned to world/episode.py at a068792.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import re

from .monitor_labels import Original


@dataclass(frozen=True)
class ReportGoal:
    version: int
    tanker: str
    ordered_nm: float
    control: str = "Wedgetail"


@dataclass(frozen=True)
class Trigger:
    distance_nm: float
    goal_version: int
    armed_at: int
    previous_range_nm: float | None


@dataclass(frozen=True)
class TriggerStep:
    state: Trigger | None
    text: str | None
    via_trigger: bool
    reason: str
    qualification: str = "conditional_supplied_originals_and_public_action"
    admissible: bool = False


@dataclass(frozen=True)
class ReportComponent:
    verdict: str
    reason: str
    original: Original
    evidence_json: str
    fingerprint: str
    qualification: str = "conditional_supplied_originals_and_numeric_facts"
    admissible: bool = False


def _finite(v):
    try:
        return type(v) in (float, int) and math.isfinite(v)
    except OverflowError:
        return False


def _second(v):
    return type(v) is int and v >= 0


def _goal(g):
    return (isinstance(g, ReportGoal) and _second(g.version) and
            isinstance(g.tanker, str) and bool(g.tanker) and
            isinstance(g.control, str) and bool(g.control) and
            _finite(g.ordered_nm) and 0 < g.ordered_nm <= 1000)


def _text(v):
    return isinstance(v, str) and bool(v.strip()) and len(v) <= 200


def advance_trigger(goal, second, state, observed_nm, command):
    """Report-field component of an already publicly accepted atomic action.

command contains only effective report/report_trigger_nm fields plus its
original goal_version. Full composite public admission must precede this API;
unknown fields are refused here rather than dropped. No heading/help effects.
Returns immutable state: old-goal triggers persist but cannot fire/retag.
"""
    if not _goal(goal) or not _second(second):
        raise ValueError("invalid goal or second")
    if observed_nm is not None and not (_finite(observed_nm) and observed_nm >= 0):
        raise ValueError("invalid observed range")
    if state is not None and not (isinstance(state, Trigger) and
            _finite(state.distance_nm) and 0 < state.distance_nm <= 1000 and
            _second(state.goal_version) and _second(state.armed_at) and
            state.armed_at <= second and (state.previous_range_nm is None or
            (_finite(state.previous_range_nm) and state.previous_range_nm >= 0))):
        raise ValueError("invalid retained trigger")
    if not isinstance(command, dict):
        raise ValueError("invalid report command")
    def refused(reason):
        # A refused action has no action effects; the ordinary trigger check
        # still runs later in the same world step on the retained state.
        step = advance_trigger(goal, second, state, observed_nm, {})
        return replace(step, reason=reason + "; " + step.reason)
    if command:
        if set(command) - {"goal_version", "report", "report_trigger_nm"}:
            raise ValueError("requires effective report component after atomic admission")
        if type(command.get("goal_version")) is not int or command["goal_version"] != goal.version:
            return refused("old or malformed goal action refused")
        if "report" in command and not _text(command["report"]):
            return refused("invalid direct report refused")
        if "report_trigger_nm" in command:
            d = command["report_trigger_nm"]
            if d is not None and not (_finite(d) and 0 < d <= 1000):
                return refused("invalid trigger refused")
            state = None if d is None else Trigger(d, goal.version, second, observed_nm)
        if "report" in command:
            return TriggerStep(None, command["report"], False, "direct report clears trigger")
    if state is None or state.goal_version != goal.version or second <= state.armed_at or observed_nm is None:
        return TriggerStep(state, None, False, "waiting or different goal")
    decreasing = state.previous_range_nm is not None and observed_nm < state.previous_range_nm
    state = replace(state, previous_range_nm=observed_nm)
    if decreasing and observed_nm <= state.distance_nm:
        text = f"{goal.control}, aircraft 2, {goal.ordered_nm:g} NM from {goal.tanker}"
        return TriggerStep(None, text, True, "one-shot trigger fired")
    return TriggerStep(state, None, False, "observed range retained")


def direct_report(original, goal, text, true_range_nm, derivative_nm_s,
                  prior_count=0, numeric_allowance_nm=1e-10, numerical_flags=()):
    """Conditional correctness of one report at its original decision second.

correct_component is not a right B8 label or a whole-goal upper. A count above
one is an irreversible report failure only conditional on accurate original
history. Public refusals are invalid, not incorrect reports. Numeric facts and
source/input hashes remain unverified; flags are never erased by content.
"""
    if not isinstance(original, Original) or not isinstance(goal, ReportGoal):
        raise TypeError("Original and ReportGoal required")
    raw = {"goal": asdict(goal), "text": text, "true_range_nm": true_range_nm,
           "derivative_nm_s": derivative_nm_s, "prior_count": prior_count,
           "numeric_allowance_nm": numeric_allowance_nm, "numerical_flags": numerical_flags}
    def retained(v):
        if type(v) is float and not math.isfinite(v):
            return {"nonfinite_float": str(v)}
        if isinstance(v, dict): return {k: retained(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [retained(x) for x in v]
        return v
    def result(verdict, reason):
        evidence = json.dumps(retained(raw), sort_keys=True, separators=(",", ":"), allow_nan=False)
        fingerprint = hashlib.sha256(json.dumps({"rule": "D34.report.component.v1",
            "original": retained(asdict(original)), "facts": retained(raw)},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return ReportComponent(verdict, reason, original, evidence, fingerprint)
    valid_original = (isinstance(original.episode, str) and bool(original.episode) and
        _second(original.second) and _second(original.goal_version) and original.decision == "report" and
        original.interface == "action" and all(isinstance(getattr(original, k), str) and
        re.fullmatch(r"[0-9a-f]{64}", getattr(original, k)) for k in
        ("goal_sha256", "input_sha256", "context_sha256", "output_sha256", "producer_sha256")))
    if not valid_original or not _goal(goal) or goal.version != original.goal_version:
        return result("invalid", "original identity or goal mismatch")
    if not _text(text) or not _second(prior_count) or not (_finite(numeric_allowance_nm) and numeric_allowance_nm >= 0):
        return result("invalid", "malformed report, history or numerical allowance")
    if not isinstance(numerical_flags, tuple) or any(type(v) is not str for v in numerical_flags):
        return result("invalid", "malformed retained flags")
    if not (_finite(true_range_nm) and true_range_nm >= 0 and _finite(derivative_nm_s)):
        return result("unknown", "missing or nonfinite numeric evidence")
    range_error = abs(true_range_nm - goal.ordered_nm)
    flags = list(numerical_flags)
    if abs(range_error - .25) <= 1e-5 + numeric_allowance_nm: flags.append("range near tolerance")
    if abs(derivative_nm_s) <= 1e-9: flags.append("inbound near zero")
    lower = text.lower()
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", lower)]
    content = (goal.control.lower() in lower and "aircraft 2" in lower and goal.tanker.lower() in lower and
               any(abs(n - goal.ordered_nm) <= .01 for n in numbers))
    raw["predicates"] = {"numbers": numbers, "content_ok": content, "range_ok": range_error <= .25,
        "inbound": derivative_nm_s < 0, "count_after": prior_count + 1, "flags": flags}
    if flags:
        return result("unknown", "; ".join(flags))
    if prior_count > 0:
        return result("incorrect_component", "second or later report under original goal")
    if not content or range_error > .25 or derivative_nm_s >= 0:
        return result("incorrect_component", "content, true range or inbound predicate failed")
    return result("correct_component", "one nominal correct report; whole-goal witness still required")
