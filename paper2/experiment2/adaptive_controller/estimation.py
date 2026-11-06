"""Observation-only state estimation for the independent Experiment 2 controller.

Uses the public bundle in DISTURBANCES.md at 135431e and the conditional orbit
prior in decision D8 at 4255f2b. No world, reference pilot, certificate or
Telohelm module is imported. This is a provisional component, not a flight
controller, solvability oracle or validated uncertainty enclosure.
"""
from __future__ import annotations

import math
from copy import deepcopy
import time
from dataclasses import dataclass, field
from typing import Any

from scipy.integrate import quad


class InterfaceError(ValueError):
    """Delivered data do not satisfy the declared observation/prior contract."""


def number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InterfaceError(f"{field_name} must be a finite number")
    return float(value)


def second(value: Any, field_name: str) -> int:
    v = number(value, field_name)
    if v < 0 or not v.is_integer():
        raise InterfaceError(f"{field_name} must be a nonnegative whole second")
    return int(v)


@dataclass(frozen=True)
class Goal:
    version: int
    tanker: str
    report_at_nm: float
    deadline: int | None

    @classmethod
    def read(cls, data: dict[str, Any]) -> Goal:
        version = second(data["version"], "goal version")
        distance = number(data["report_at_nm"], "report_at_nm")
        if version < 1 or distance <= 0 or not isinstance(data["tanker"], str):
            raise InterfaceError("invalid version, reporting distance or tanker")
        deadline = None if data["deadline"] is None else second(data["deadline"], "deadline")
        return cls(version, data["tanker"], distance, deadline)


@dataclass(frozen=True)
class Observation:
    measured_at: int
    received_at: int
    x: float
    y: float


@dataclass(frozen=True)
class OrbitPrior:
    onset: float = 60.0
    duration: float = 120.0
    maximum_fractional_change: float = 0.2

    def fraction(self, t: float) -> float:
        if t <= self.onset:
            return 0.0
        if t >= self.onset + self.duration:
            return 1.0
        return (1.0 - math.cos(math.pi * (t - self.onset) / self.duration)) / 2.0

    def radial_rate(self, t: float, change: float) -> float:
        if not self.onset < t < self.onset + self.duration:
            return 0.0
        return change * math.pi / (2.0 * self.duration) * math.sin(math.pi * (t - self.onset) / self.duration)


@dataclass
class TargetBelief:
    callsign: str
    centre: tuple[float, float]
    initial_radius: float
    speed_nm_s: float
    prior: OrbitPrior
    coordinate_allowance_nm: float
    latest: Observation
    change_low: float
    change_high: float
    observations: dict[int, Observation] = field(default_factory=dict)

    @property
    def no_change_consistent(self) -> bool:
        return self.change_low <= 0.0 <= self.change_high

    def incorporate(self, obs: Observation) -> bool:
        old = self.observations.get(obs.measured_at)
        if old is not None:
            if math.hypot(old.x - obs.x, old.y - obs.y) > self.coordinate_allowance_nm:
                raise InterfaceError("conflicting exact reports at one measurement time")
            return False
        radius = math.hypot(obs.x - self.centre[0], obs.y - self.centre[1])
        fraction = self.prior.fraction(float(obs.measured_at))
        low, high = self.change_low, self.change_high
        if fraction == 0.0:
            if abs(radius - self.initial_radius) > self.coordinate_allowance_nm:
                raise InterfaceError("pre-change report contradicts the briefed radius")
        else:
            # An explicit numerical allowance; its division by a small fraction exposes early-report sensitivity.
            low = max(low, (radius - self.initial_radius - self.coordinate_allowance_nm) / fraction)
            high = min(high, (radius - self.initial_radius + self.coordinate_allowance_nm) / fraction)
            if low > high:
                raise InterfaceError("reports contradict the declared orbit-change prior")
        self.change_low, self.change_high = low, high
        self.observations[obs.measured_at] = obs
        if obs.measured_at > self.latest.measured_at:
            self.latest = obs
        return True

    def forecasts(self, t: float) -> list[dict[str, float]]:
        """Endpoint/midpoint hypotheses, plus no-change when consistent. These do not enclose all future positions."""
        if t < self.latest.measured_at:
            raise InterfaceError("forecast precedes the latest measurement")
        candidates = {self.change_low, self.change_high, (self.change_low + self.change_high) / 2.0}
        if self.no_change_consistent:
            candidates.add(0.0)
        measured = float(self.latest.measured_at)
        phase = math.atan2(self.latest.x - self.centre[0], self.latest.y - self.centre[1])
        out = []
        for change in sorted(candidates):
            def angular_rate(u: float) -> float:
                radius = self.initial_radius + change * self.prior.fraction(u)
                radial = self.prior.radial_rate(u, change)
                if radius <= 0 or abs(radial) > self.speed_nm_s:
                    raise InterfaceError("orbit hypothesis has invalid radius or radial speed")
                return math.sqrt(self.speed_nm_s**2 - radial**2) / radius

            # Split at public form changes. quad's error remains an estimate, not a certificate.
            cuts = sorted({measured, t} | {u for u in (self.prior.onset, self.prior.onset + self.prior.duration)
                                          if measured < u < t})
            angle_increment = error = 0.0
            for a, b in zip(cuts, cuts[1:]):
                value, estimated_error = quad(angular_rate, a, b, epsabs=1e-11, epsrel=1e-11)
                angle_increment += value
                error += estimated_error
            radius = self.initial_radius + change * self.prior.fraction(t)
            angle = phase + angle_increment
            radial = self.prior.radial_rate(t, change)
            tangential = math.sqrt(self.speed_nm_s**2 - radial**2)
            out.append({"change_nm": change, "x": self.centre[0] + radius * math.sin(angle),
                        "y": self.centre[1] + radius * math.cos(angle),
                        "vx": radial * math.sin(angle) + tangential * math.cos(angle),
                        "vy": radial * math.cos(angle) - tangential * math.sin(angle),
                        "estimated_angle_error_rad": error})
        return out


class EpisodeEstimator:
    """Clear on reset; retain only delivered public data and disclosed conditional class knowledge."""

    def __init__(self, coordinate_allowance_nm: float = 1e-6):
        self.coordinate_allowance_nm = number(coordinate_allowance_nm, "coordinate allowance")
        if self.coordinate_allowance_nm <= 0:
            raise InterfaceError("coordinate allowance must be positive")
        self.targets: dict[str, TargetBelief] = {}
        self.goal: Goal | None = None
        self.t: int | None = None
        self.wind_nm_s: tuple[float, float] | None = None
        self.forecast_calls = self.cpu_ns = 0

    def reset(self, briefing: dict[str, Any]) -> None:
        targets: dict[str, TargetBelief] = {}
        goal = Goal.read(briefing["goal"])
        own_callsign = briefing["own_callsign"]
        control_callsign = briefing["control_callsign"]
        if not isinstance(own_callsign, str) or not isinstance(control_callsign, str):
            raise InterfaceError("callsigns must be strings")
        for item in briefing["tankers"]:
            callsign = item["callsign"]
            if not isinstance(callsign, str) or callsign in targets or item["direction"] != "clockwise":
                raise InterfaceError("duplicate tanker or unsupported direction")
            centre = tuple(number(v, "centre") for v in item["centre"])
            position = tuple(number(v, "position_at_0") for v in item["position_at_0"])
            radius, speed = number(item["radius"], "radius"), number(item["speed_nm_s"], "speed_nm_s")
            if len(centre) != 2 or len(position) != 2 or radius <= 0 or speed <= 0:
                raise InterfaceError("invalid tanker briefing")
            obs = Observation(0, 0, *position)
            prior = OrbitPrior()
            extent = prior.maximum_fractional_change * radius
            belief = TargetBelief(callsign, centre, radius, speed, prior, self.coordinate_allowance_nm,
                                  obs, -extent, extent)
            belief.incorporate(obs)
            targets[callsign] = belief
        if goal.tanker not in targets:
            raise InterfaceError("goal names an unbriefed tanker")
        self.targets, self.goal = targets, goal
        self.own_callsign, self.control_callsign = own_callsign, control_callsign
        self.t = None
        self.wind_nm_s = None
        self.forecast_calls = self.cpu_ns = 0

    def update(self, bundle: dict[str, Any]) -> None:
        started = time.process_time_ns()
        try:
            t = second(bundle["t"], "bundle time")
            if self.goal is None or self.t is not None and t < self.t:
                raise InterfaceError("reset required, or bundle time moved backwards")
            own = bundle["own"]
            heading, track = number(own["heading_deg"], "heading"), number(own["track_deg"], "track")
            tas, ground = number(own["tas_nm_s"], "TAS"), number(own["ground_speed_nm_s"], "ground speed")
            if not 0 <= heading < 360 or not 0 <= track < 360 or tas <= 0 or ground < 0:
                raise InterfaceError("invalid own-aircraft units or values")
            h, trk = math.radians(heading), math.radians(track)
            # Speeds in the bundle already have units NM/s. No second conversion from knots.
            wind = (ground * math.sin(trk) - tas * math.sin(h),
                              ground * math.cos(trk) - tas * math.cos(h))
            goal = self.goal
            targets = deepcopy(self.targets)
            if bundle["goal"] is not None:
                goal = Goal.read(bundle["goal"])
                if goal.version < self.goal.version or goal.tanker not in self.targets:
                    raise InterfaceError("goal regressed or names an unbriefed tanker")
                if goal.version == self.goal.version and goal != self.goal:
                    raise InterfaceError("goal changed without a new version")
            for report in bundle["reports"]:
                measured = second(report["measured_at"], "measurement time")
                received = second(report["received_at"], "receipt time")
                if measured > received or received != t or report["callsign"] not in self.targets:
                    raise InterfaceError("future, mistimed or unbriefed report")
                obs = Observation(measured, received, number(report["x"], "report x"), number(report["y"], "report y"))
                targets[report["callsign"]].incorporate(obs)
            # Invalid bundles cannot leave partially revised estimates or goals.
            self.targets, self.goal, self.wind_nm_s, self.t = targets, goal, wind, t
        finally:
            self.cpu_ns += time.process_time_ns() - started

    def forecast(self, callsign: str, t: float) -> list[dict[str, float]]:
        started = time.process_time_ns()
        try:
            future = number(t, "forecast time")
            if callsign not in self.targets:
                raise InterfaceError("forecast names an unbriefed tanker")
            if self.t is None or future < self.t:
                raise InterfaceError("forecast requires current observations and a non-past time")
            self.forecast_calls += 1
            return self.targets[callsign].forecasts(future)
        finally:
            self.cpu_ns += time.process_time_ns() - started
