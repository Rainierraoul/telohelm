"""Prospective B8 heading dispositions, conditional on independently valid bounds.

This module generates no world, numerical bound, or admissible corpus case.
The trusted acquisition/store path must verify original inputs and producer
witnesses before treating a proposed disposition as evidence.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re


@dataclass(frozen=True)
class Original:
    episode: str
    goal_version: int
    second: int
    goal_sha256: str
    input_sha256: str
    context_sha256: str
    output_sha256: str
    producer_sha256: str
    decision: str = "heading"
    interface: str = "action"


@dataclass(frozen=True)
class Bounds:
    """All times are elapsed from Original.second, including the one-second prefix."""
    origin_second: int
    lower_after: float | None = None
    upper_best: float | None = None
    upper_after: float | None = None
    lower_best: float | None = None
    units: str = "elapsed_seconds"
    commitment_seconds: int = 1
    goal_changed: bool = False
    unsettled: bool = False
    complete_at_issue: bool = False
    numerical_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Proposal:
    disposition: str  # proposed, unknown, invalid
    conditional_label: str | None
    reason: str
    original: Original
    evidence_json: str
    fingerprint: str
    qualification: str = "unverified_bounds_and_witnesses"
    admissible: bool = False


def propose_heading(original: Original, bounds: Bounds) -> Proposal:
    """Apply the fixed B8 inequalities, never certify supplied bounds or hashes.

    Wrong: L_after - U_best > 10 s. Right: U_after - L_best < 5 s.
    Distances/flags are evaluated by a future reviewed producer, not here.
    A one-second band around either cutoff is left unknown. Missing or
    nonfinite bounds cannot establish either branch. No reference failure
    proves unreachability, and this API has no deadline-exclusion shortcut.
    """
    if not isinstance(original, Original) or not isinstance(bounds, Bounds):
        raise TypeError("Original and Bounds required")
    def retained(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite_float": str(value)}
        if isinstance(value, dict):
            return {k: retained(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [retained(v) for v in value]
        return value
    raw = retained(asdict(bounds))
    evidence = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(json.dumps(
        {"original": retained(asdict(original)), "bounds": raw, "rule": "B8.heading.action.v1"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def answer(state, label, reason):
        return Proposal(state, label, reason, original, evidence, fingerprint)

    if not isinstance(original.episode, str) or not original.episode:
        return answer("invalid", None, "missing original episode")
    for v in (original.goal_version, original.second, bounds.origin_second):
        if type(v) is not int or v < 0:
            return answer("invalid", None, "invalid original version or second")
    for k in ("goal_sha256", "input_sha256", "context_sha256", "output_sha256", "producer_sha256"):
        v = getattr(original, k)
        if not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None:
            return answer("invalid", None, "invalid original fingerprint")
    if original.decision != "heading" or original.interface != "action":
        return answer("invalid", None, "unsupported decision or interface")
    if bounds.units != "elapsed_seconds" or bounds.origin_second != original.second:
        return answer("invalid", None, "bounds have different units or time origin")
    if type(bounds.commitment_seconds) is not int or bounds.commitment_seconds != 1:
        return answer("invalid", None, "live action commits exactly one second")
    for v in (bounds.goal_changed, bounds.unsettled, bounds.complete_at_issue):
        if type(v) is not bool:
            return answer("invalid", None, "invalid state flag")
    if not isinstance(bounds.numerical_flags, tuple) or any(type(v) is not str for v in bounds.numerical_flags):
        return answer("invalid", None, "invalid numerical flags")
    values = (bounds.lower_after, bounds.upper_best, bounds.upper_after, bounds.lower_best)
    for v in values:
        if v is not None:
            try:
                valid = type(v) in (int, float) and not math.isnan(v) and v >= 0
            except (OverflowError, ValueError):
                valid = False
            if not valid:
                return answer("invalid", None, "invalid elapsed bound")
    if any(v is not None and not math.isfinite(v) for v in values):
        return answer("unknown", None, "nonfinite bound")
    if bounds.goal_changed or bounds.unsettled or bounds.complete_at_issue or bounds.numerical_flags:
        return answer("unknown", None, "goal, ending or numerical evidence unsettled")
    la, ub, ua, lb = values
    if la is not None and ua is not None and la > ua:
        return answer("invalid", None, "after-action lower exceeds upper")
    if lb is not None and ub is not None and lb > ub:
        return answer("invalid", None, "best-goal lower exceeds upper")
    if lb is not None and ua is not None and lb > ua:
        return answer("invalid", None, "global lower exceeds available after witness")
    # U_best is the best supplied feasible witness, including the after witness.
    if ua is not None and ub is not None and ub > ua:
        return answer("invalid", None, "best witness slower than available after witness")
    wrong = la - ub if la is not None and ub is not None else None
    right = ua - lb if ua is not None and lb is not None else None
    if (wrong is not None and abs(wrong - 10) <= 1) or (right is not None and abs(right - 5) <= 1):
        return answer("unknown", None, "within one second of a cutoff")
    if wrong is not None and wrong > 10:
        return answer("proposed", "wrong", "conditional lower-after minus upper-best exceeds 10 seconds")
    if right is not None and right < 5:
        return answer("proposed", "right", "conditional upper-after minus lower-best below 5 seconds")
    return answer("unknown", None, "available bounds do not settle either inequality")
