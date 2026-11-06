"""Prospective B8 contact relaxation; no world draws or corpus admission.

The bound is conditional on supplied motion functions, their speed bounds and
an empirical distance-error allowance. It is not validated interval arithmetic,
a whole-goal witness or an admissible label. Review this kernel before wiring
actual world motion, fixed-prefix scoring and independent witness verification.
"""
from dataclasses import dataclass
import math
from typing import Callable
from .monitor_labels import Original; from .monitor_labels import Bounds; from .monitor_labels import propose_heading

Point = tuple[float, float]
Motion = Callable[[float], Point]


@dataclass(frozen=True)
class Settings:
    resolution_seconds: float = 0.01
    distance_allowance_nm: float = 1e-5
    evaluation_cap: int = 4096
    contact_radius_nm: float = 0.25


@dataclass(frozen=True)
class Cell:
    phase: str
    left: float
    right: float
    midpoint: float
    q: float | None
    lower_margin: float | None
    disposition: str
    error: str | None = None


@dataclass(frozen=True)
class LowerEvidence:
    original: Original
    prefix_seconds: int
    deadline_second: float
    receiver_speed_bound_nm_s: float
    tanker_speed_bound_nm_s: float
    settings: Settings
    elapsed_lower_seconds: float
    state: str
    cells: tuple[Cell, ...]
    evaluations: int
    motion_calls: tuple[tuple[str, int], ...]
    qualification: str = "conditional_motion_speed_and_error_assumptions"
    admissible: bool = False


def _finite(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def contact_lower(original: Original, tanker_at: Motion, fixed_prefix_at: Motion,
                  receiver_speed_bound_nm_s: float, tanker_speed_bound_nm_s: float,
                  deadline_second: float, *, prefix_seconds: int = 1,
                  settings: Settings = Settings()) -> LowerEvidence:
    """Exclude contact from the original second, left to right, with a finite cap.

    First exclude contact on the actual committed prefix using distance(T,R).
    Then relax all continuations to a speed-bounded ball about its endpoint:
    q(s) = |T(s)-R(end)| - S*(s-end) - radius. Both q functions have Lipschitz
    bound V+S. An interval excludes contact only if its midpoint q minus
    (V+S)*halfwidth and the numerical allowance is strictly positive.

    Stop at the first unresolved interval, error or cap. Its left edge is a
    conditional lower bound, expressed as elapsed time from original.second.
    No callback sampled here proves its declared speed bound or source identity.
    """
    if not isinstance(original, Original) or not isinstance(settings, Settings):
        raise TypeError("Original and Settings required")
    # Reuse the reviewed identity/type contract; an unknown arithmetic proposal
    # is expected, and cannot authorise this result or any corpus case.
    p = propose_heading(original, Bounds(original.second))
    if p.disposition == "invalid":
        raise ValueError(p.reason)
    if not callable(tanker_at) or not callable(fixed_prefix_at):
        raise TypeError("motion functions required")
    if type(prefix_seconds) is not int or prefix_seconds not in (0, 1):
        raise ValueError("prefix is zero (best relaxation) or one live action second")
    start = _finite(original.second, "original second")
    if start > 3600:
        raise ValueError("original second must lie within the episode")
    end = start + prefix_seconds
    deadline = _finite(deadline_second, "deadline")
    S = _finite(receiver_speed_bound_nm_s, "receiver speed bound")
    V = _finite(tanker_speed_bound_nm_s, "tanker speed bound")
    resolution = _finite(settings.resolution_seconds, "resolution")
    allowance = _finite(settings.distance_allowance_nm, "distance allowance")
    radius = _finite(settings.contact_radius_nm, "contact radius")
    if S < 0 or V < 0 or not math.isfinite(S+V) or resolution <= 0 or allowance < 0 or radius < 0:
        raise ValueError("invalid speed or numerical settings")
    if deadline < end or deadline > 3600:
        raise ValueError("deadline must include the prefix and lie within the episode")
    if type(settings.evaluation_cap) is not int or settings.evaluation_cap < 1:
        raise ValueError("positive integer evaluation cap required")
    cells, counts, evaluations = [], {"tanker": 0, "prefix": 0}, 0

    def motion(kind, fn, second):
        counts[kind] += 1  # Retain attempted primitive queries, including failures.
        point = fn(second)
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("motion must return two coordinates")
        return tuple(_finite(v, "motion coordinate") for v in point)

    def distance(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def result(left, state):
        return LowerEvidence(original, prefix_seconds, deadline, S, V, settings,
                             max(0.0, left-start), state, tuple(cells), evaluations,
                             tuple(counts.items()))

    def search(phase, lo, hi, q_at):
        nonlocal evaluations
        pending = [(lo, hi)]
        while pending:
            left, right = pending.pop()
            if evaluations >= settings.evaluation_cap:
                cells.append(Cell(phase,left,right,(left+right)/2,None,None,"cap"))
                return left, "evaluation_cap"
            midpoint = left+(right-left)/2
            evaluations += 1
            try:
                q = _finite(q_at(midpoint), "q")
                # Last-operation outward rounding is extra conservatism, not a
                # proof that the motion library/other operations are enclosed.
                margin = math.nextafter(q-(S+V)*(right-left)/2-allowance, -math.inf)
                if not math.isfinite(margin):
                    raise ValueError("nonfinite interval margin")
            except Exception as error:
                cells.append(Cell(phase,left,right,midpoint,None,None,"error",
                                  f"{type(error).__name__}: {str(error)[:200]}"))
                return left, "error"
            if margin > 0:
                cells.append(Cell(phase,left,right,midpoint,q,margin,"excluded"))
            elif right-left <= resolution:
                cells.append(Cell(phase,left,right,midpoint,q,margin,"unresolved"))
                return left, "contact_not_excluded"
            else:
                cells.append(Cell(phase,left,right,midpoint,q,margin,"split"))
                pending.append((midpoint,right))
                pending.append((left,midpoint))
        return hi, "excluded"

    if prefix_seconds:
        def prefix_q(s):
            return distance(motion("tanker",tanker_at,s),motion("prefix",fixed_prefix_at,s))-radius
        left, state = search("fixed_prefix",start,end,prefix_q)
        if state != "excluded":
            return result(left,state)
    try:
        anchor = motion("prefix",fixed_prefix_at,end)
    except Exception as error:
        cells.append(Cell("prefix_end",end,end,end,None,None,"error",
                          f"{type(error).__name__}: {str(error)[:200]}"))
        return result(end,"error")
    def relaxed_q(s):
        return distance(motion("tanker",tanker_at,s),anchor)-S*(s-end)-radius
    left, state = search("relaxation",end,deadline,relaxed_q)
    return result(left,"deadline_contact_excluded" if state=="excluded" else state)
