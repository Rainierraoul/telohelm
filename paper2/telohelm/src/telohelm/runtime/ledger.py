"""The cost ledger for Experiment 2 (backlog note B10).

Each request gets one record. Allocated GPU hours are split into idle time, outside the union of active
request intervals, and busy time, which alone is shared to calls in proportion to measured wall-clock
time, so that shares plus idle equal the allocation. A call with unknown time cannot be given a share,
so per-call shares are then withheld. Costs of an alternative ending stop at its place in the event
sequence. Build costs are recorded once and spread only in the separate amortised view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

MODEL_KINDS = ("small_model", "reasoning_model")
TOOL_KINDS = ("prediction_tool", "tool")
KINDS = MODEL_KINDS + TOOL_KINDS
STATUSES = ("answered", "failed", "timed_out", "cut_off", "interrupted")


@dataclass(frozen=True)
class Call:
    request_id: str
    kind: str
    episode: str
    action_step: int                  # position of the issuing action in the episode's event sequence
    started_s: float | None           # wall-clock start, for the union of active intervals
    wall_s: float | None              # measured wall-clock time; None if unknown
    tokens_in: int | None
    tokens_out: int | None
    status: str
    scope: str = "running"            # 'running', 'build', 'planting', 'trial', 'replay' or 'setup'
    allocation: str | None = None     # the allocation (job) the request ran in; None when unknown (L24.2)
    gpus: tuple[int, ...] | None = None   # devices within that allocation; None when placement is unknown (R22.6)
    # Links added in TURN 56 (Codex's TURN 55 integration requirements). None means not recorded, never zero.
    cpu_s: float | None = None        # processor time of this request's own work, measured apart from wall time
    condition: str | None = None      # the condition that made the request
    stream: str | None = None         # the stream the episode belongs to
    sim_second: int | None = None     # simulated second at which the request was made
    answered_step: int | None = None  # action step at which its answer arrived (a later step than action_step
                                      # means it straddles any ending between them)
    recovers: str | None = None       # a recovery row: the row that carried this call's charge (TURN 66). Its wall
                                      # time is informational and is left out of time-based shares.

    def __post_init__(self):
        if self.kind not in KINDS or self.status not in STATUSES:
            raise ValueError(f"bad kind or status in {self.request_id}")
        if self.gpus is not None:
            # Types are checked before hashing, so an unhashable entry is refused, not raised (L26.1).
            if (not isinstance(self.gpus, tuple) or not self.gpus
                    or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in self.gpus)
                    or len(set(self.gpus)) != len(self.gpus)):
                raise ValueError(f"gpus must be distinct non-negative integers in {self.request_id}")
        if self.allocation is not None and not isinstance(self.allocation, str):
            raise ValueError(f"allocation must be a string in {self.request_id}")
        for name in ("condition", "stream", "recovers"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, str):
                raise ValueError(f"{name} must be a string in {self.request_id}")
        for name in ("sim_second", "answered_step"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                raise ValueError(f"{name} must be a non-negative integer in {self.request_id}")
        if self.answered_step is not None and self.answered_step < self.action_step:
            raise ValueError(f"answered before it was made in {self.request_id}")
        for name in ("started_s", "wall_s", "cpu_s"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
                raise ValueError(f"{name} must be a finite non-negative number in {self.request_id}")


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


@dataclass
class Ledger:
    calls: list[Call] = field(default_factory=list)

    def add(self, call: Call) -> None:
        if any(c.request_id == call.request_id for c in self.calls):
            raise ValueError(f"request {call.request_id} recorded twice")
        self.calls.append(call)

    def gpu_shares(self, allocated_hours: float,
                   allocation_windows: list[tuple[str, float, float, int]]) -> dict[str, Any]:
        """Split allocated GPU hours A into idle I and busy B = A − I, device by device, and share B to calls.

        allocation_windows: (allocation, start_s, end_s, gpus), with devices 0 .. gpus − 1 inside each allocation.
        A model call is active on its own devices in its own allocation. B, I and the shares are null if any model
        call lacks its start, its measured time, its allocation or its devices. Shares follow B10's declared
        convention: in proportion to measured wall time, whatever the number of devices (L24.1). They are shares of
        allocated time, not measured utilisation."""
        allocations: dict[str, int] = {}
        for alloc, a, b, g in allocation_windows:
            if (not isinstance(alloc, str) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b))
                    or b < a or isinstance(g, bool) or not isinstance(g, int) or g < 1):
                raise ValueError(f"bad allocation window {(alloc, a, b, g)}")
            if alloc in allocations and allocations[alloc] != g:
                raise ValueError(f"allocation {alloc} declared with different device counts")
            allocations[alloc] = g
        # Windows of one allocation must not overlap, or the same device-hour would be counted twice (L26.2).
        # Different allocations may overlap in time.
        for alloc in allocations:
            spans = sorted((a, b) for x, a, b, _ in allocation_windows if x == alloc)
            for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
                if a2 < b1:
                    raise ValueError(f"windows of allocation {alloc} overlap")
        window_h = sum((b - a) * g for _, a, b, g in allocation_windows) / 3600
        if not math.isfinite(allocated_hours) or abs(window_h - allocated_hours) > 1e-9 * max(1.0, allocated_hours):
            raise ValueError(f"allocated hours {allocated_hours} do not match the windows ({window_h})")
        model = [c for c in self.calls if c.kind in MODEL_KINDS and c.recovers is None]
        for c in model:
            if c.allocation is not None and c.allocation not in allocations:
                raise ValueError(f"{c.request_id} names an allocation that is not in the windows")
            if c.gpus is not None and c.allocation is not None and max(c.gpus) >= allocations[c.allocation]:
                raise ValueError(f"{c.request_id} names a device outside its allocation")
        unknown = [c.request_id for c in model
                   if c.wall_s is None or c.started_s is None or c.gpus is None or c.allocation is None]
        out: dict[str, Any] = {"allocated_h": allocated_hours, "idle_h": None, "busy_h": None,
                               "calls_unknown_time_or_placement": unknown, "shares_h": None}
        if unknown:
            out["reason"] = "B, I and shares withheld: some model calls have unknown time, allocation or devices"
            return out
        idle_s = 0.0
        for alloc, start, end, gpus in allocation_windows:
            for device in range(gpus):
                active = [(max(c.started_s, start), min(c.started_s + c.wall_s, end)) for c in model
                          if c.allocation == alloc and device in c.gpus
                          and c.started_s + c.wall_s > start and c.started_s < end]
                idle_s += (end - start) - union_length(active)
        out["idle_h"] = idle_s / 3600
        out["busy_h"] = allocated_hours - out["idle_h"]
        weight = sum(c.wall_s for c in model)
        if weight > 0:
            out["shares_h"] = {c.request_id: out["busy_h"] * c.wall_s / weight for c in model}
        return out

    def ending_calls(self, episode: str, ending_step: int) -> dict[str, list[str]]:
        """Calls counted under an alternative ending: those made at action steps before it. A call made
        before the ending counts in full even if its answer came later; such a call is also listed as
        straddling when its answered_step is at or after the ending (B10). A call whose answer step is unknown
        is listed under straddling_unknown, never assumed either way."""
        # Recovery rows reconcile an earlier call; they are not new calls and are not counted (Codex's MG67.COST.COUNT).
        mine = [c for c in self.calls if c.episode == episode and c.recovers is None]
        counted = [c for c in mine if c.action_step < ending_step]
        counted_ids = {c.request_id for c in counted}
        # The rows that carry the counted calls' charges: each counted call's own row, and any recovery row linked to
        # it that carried tokens (after an interruption before the charge was booked), so a charge carried by a
        # recovery cannot drop out of an ending's costs (Codex's MG70.CUT.CHARGE).
        carriers = [c.request_id for c in self.calls
                    if c.episode == episode and c.recovers in counted_ids
                    and (c.tokens_in is not None or c.tokens_out is not None)]
        return {"counted": [c.request_id for c in counted],
                "charge_ids": [c.request_id for c in counted] + carriers,
                "excluded": [c.request_id for c in mine if c.action_step >= ending_step],
                "straddling": [c.request_id for c in counted
                               if c.answered_step is not None and c.answered_step >= ending_step],
                "straddling_unknown": [c.request_id for c in counted if c.answered_step is None]}


def amortised(build_costs: dict[int, float], running_costs: list[float]) -> list[float]:
    """Decision D1's view: a build cost C spent in episode k (0-based) adds C/(N − k) to episodes k..N−1.
    Running costs must already exclude build costs. Exact fractions keep the sum equal to the raw total."""
    n = len(running_costs)
    charges = [Fraction(x).limit_denominator(10**12) for x in running_costs]
    for k, c in build_costs.items():
        if not 0 <= k < n:
            raise ValueError(f"build cost in episode {k} outside 0..{n - 1}")
        share = Fraction(c).limit_denominator(10**12) / (n - k)
        for i in range(k, n):
            charges[i] += share
    return [float(x) for x in charges]


def per_goal_met(total_cost: float, goals_met: int) -> float | None:
    """Total cost over all episodes in the set, divided by the number in which the goal was met."""
    return None if goals_met == 0 else total_cost / goals_met
