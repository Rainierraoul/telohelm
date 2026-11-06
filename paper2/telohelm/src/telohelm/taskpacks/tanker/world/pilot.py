"""The reference pilot: flies from observations only (DISTURBANCES.md draft 0.3, section 11; decision D8).

It sees the briefing and the delivered bundles, nothing else. It knows the
disturbance classes (class knowledge, decision D8): any orbit change starts at
60 s and follows the half-cosine form over 120 s with the tanker holding its
total ground speed. It does not know whether a given episode has a change, or
its size. It replans every second, reports from its own prediction of the tanker, and,
when allowed, asks for help once its newest report of the named tanker is more
than 10 seconds old.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import physics as P

PRED_STEP = 0.5


class OrbitModel:
    """The pilot's prediction of one tanker, rebuilt from its latest report."""

    def __init__(self, brief: dict[str, Any]):
        self.cx, self.cy = brief["centre"]
        self.r0, self.v = brief["radius"], brief["speed_nm_s"]
        self.latest: dict[str, Any] = {"x": brief["position_at_0"][0], "y": brief["position_at_0"][1], "measured_at": 0}
        self._cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def update(self, report: dict[str, Any]) -> None:
        if report["measured_at"] >= self.latest["measured_at"]:
            self.latest = report

    def inferred_delta(self) -> float:
        """Radius change implied by the latest report under the class rule; 0 before any change could start."""
        tm = self.latest["measured_at"]
        if tm <= P.ORBIT_CHANGE_START:
            return 0.0
        frac = 1.0 if tm >= P.ORBIT_CHANGE_START + P.ORBIT_CHANGE_DURATION else \
            (1 - math.cos(math.pi * (tm - P.ORBIT_CHANGE_START) / P.ORBIT_CHANGE_DURATION)) / 2
        observed = math.hypot(self.latest["x"] - self.cx, self.latest["y"] - self.cy)
        return (observed - self.r0) / frac if frac > 1e-12 else 0.0

    def _angles(self, delta: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(delta, 9)
        if key not in self._cache:
            orbit = P.Orbit(self.cx, self.cy, self.r0, self.v, 0.0, delta)
            ts = np.arange(0.0, P.ORBIT_CHANGE_START + P.ORBIT_CHANGE_DURATION + PRED_STEP, PRED_STEP)
            self._cache[key] = (ts, np.array([orbit.angle(t) for t in ts]))
        return self._cache[key]

    def _angle(self, delta: float, t: float) -> float:
        ts, angles = self._angles(delta)
        end = P.ORBIT_CHANGE_START + P.ORBIT_CHANGE_DURATION
        if t <= ts[-1]:
            return float(np.interp(t, ts, angles))
        return float(angles[-1]) + self.v * (t - ts[-1]) / (self.r0 + delta) if t > end else float(angles[-1])

    def position(self, t: float) -> tuple[float, float]:
        d = self.inferred_delta()
        tm = self.latest["measured_at"]
        theta_m = math.atan2(self.latest["x"] - self.cx, self.latest["y"] - self.cy)
        theta = theta_m + self._angle(d, t) - self._angle(d, tm)
        r = P.Orbit(self.cx, self.cy, self.r0, self.v, 0.0, d).radius(t)
        return self.cx + r * math.sin(theta), self.cy + r * math.cos(theta)


class ReferencePilot:
    def __init__(self, briefing: dict[str, Any], ask_help: bool, horizon: float = 3600.0):
        self.models = {b["callsign"]: OrbitModel(b) for b in briefing["tankers"]}
        self.goal = briefing["goal"]
        self.ask_help = ask_help
        self.asked = False
        self.reported_for: int | None = None
        self._previous_range: dict[int, float] = {}
        self.control = briefing["control_callsign"]
        self.horizon = horizon
        self._last_tau: float | None = None

    def act(self, bundle: dict[str, Any]) -> dict[str, Any]:
        if bundle["goal"]:
            self.goal = bundle["goal"]
        for r in bundle["reports"]:
            self.models[r["callsign"]].update(r)
        own, t = bundle["own"], bundle["t"]
        hdg, trk = math.radians(own["heading_deg"]), math.radians(own["track_deg"])
        air = (own["tas_nm_s"] * math.sin(hdg), own["tas_nm_s"] * math.cos(hdg))
        ground = (own["ground_speed_nm_s"] * math.sin(trk), own["ground_speed_nm_s"] * math.cos(trk))
        wind = (ground[0] - air[0], ground[1] - air[1])
        model = self.models[self.goal["tanker"]]
        action: dict[str, Any] = {"goal_version": self.goal["version"], "heading_deg": self._intercept_heading(own, wind, model, t)}
        # Report from the pilot's own prediction of the tanker, once per goal version, inbound.
        tx, ty = model.position(float(t))
        predicted = math.hypot(tx - own["x"], ty - own["y"])
        previous = self._previous_range.get(self.goal["version"])
        self._previous_range[self.goal["version"]] = predicted
        if (self.reported_for != self.goal["version"] and previous is not None and predicted < previous
                and predicted <= self.goal["report_at_nm"]):
            action["report"] = (f"{self.control}, aircraft 2, {self.goal['report_at_nm']:g} NM from {self.goal['tanker']}")
            self.reported_for = self.goal["version"]
        if self.ask_help and not self.asked and t - model.latest["measured_at"] > 10:
            action["help"] = {"reason": "missing_or_stale_evidence", "text": "missing or stale evidence"}
            self.asked = True
        return action

    def _intercept_heading(self, own: dict[str, Any], wind: tuple[float, float], model: OrbitModel, t: float) -> float:
        """Heading through the air that reaches the predicted tanker soonest in a straight line."""
        px, py, v = own["x"], own["y"], own["tas_nm_s"]

        def gap(tau: float) -> tuple[float, float, float]:
            dt = tau - t
            tx, ty = model.position(tau)
            ax, ay = tx - px - wind[0] * dt, ty - py - wind[1] * dt
            return math.hypot(ax, ay) - v * dt, ax, ay

        lo = float(t)
        if self._last_tau is not None and self._last_tau - 30.0 > lo and gap(self._last_tau - 30.0)[0] > 0:
            lo = self._last_tau - 30.0          # the intercept moves little from one second to the next
        tau = lo + 1.0
        g, ax, ay = gap(tau)
        while g > 0 and tau < self.horizon:
            lo, tau = tau, tau + 1.0
            g, ax, ay = gap(tau)
        hi = tau
        for _ in range(40):
            mid = (lo + hi) / 2
            gm, mx, my = gap(mid)
            if gm > 0:
                lo = mid
            else:
                hi, ax, ay = mid, mx, my
        self._last_tau = hi
        return math.degrees(math.atan2(ax, ay)) % 360.0
