"""Independent finite-candidate guidance kernel; provisional, awaiting cross-review.

Only public own navigation, goal and estimator forecasts enter the search.
Predicted trajectories are heuristics, not contact/report/solvability certificates.
This module does not send reports/help or run a scored episode.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .estimation import EpisodeEstimator, InterfaceError, number


@dataclass(frozen=True)
class OwnState:
    x: float
    y: float
    heading_deg: float
    commanded_heading_deg: float
    tas_nm_s: float

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            number(getattr(self, key), key)
        if self.tas_nm_s <= 0 or any(not 0 <= getattr(self, k) < 360
                                   for k in ('heading_deg', 'commanded_heading_deg')):
            raise InterfaceError('invalid own state')

    @classmethod
    def read(cls, own: dict) -> OwnState:
        values = {k: number(own[k], k) for k in cls.__dataclass_fields__}
        if values['tas_nm_s'] <= 0 or any(not 0 <= values[k] < 360
                                         for k in ('heading_deg', 'commanded_heading_deg')):
            raise InterfaceError('invalid own state')
        return cls(**values)


def heading_error(command: float, actual: float) -> float:
    """Shortest turn; exactly 180 degrees turns right, per the public contract."""
    error = (command - actual + 180.0) % 360.0 - 180.0
    return 180.0 if error == -180.0 else error


def _heading_degrees(angle_rad: float) -> float:
    """Canonical derived heading. Floating modulo can round a tiny negative
    angle to exactly 360, which is the same direction as 0 but outside the API.
    Correct that endpoint only; do not snap any neighbouring valid heading.
    """
    degrees = math.degrees(angle_rad) % 360.0
    return 0.0 if degrees == 360.0 else degrees


def propagate(own: OwnState, command: float, seconds: float,
              wind: tuple[float, float]) -> OwnState:
    """Analytic air-relative turn/straight motion plus steady wind drift.

    Integrates TAS*(sin(h), cos(h)) with constant h' until command is met,
    then a straight leg. TAS NM/s converts to m/s only for g*tan(bank)/TAS.
    """
    command = number(command, 'command'); seconds = number(seconds, 'duration')
    if not 0 <= command < 360 or seconds < 0:
        raise InterfaceError('invalid heading or duration')
    wx, wy = (number(v, 'wind') for v in wind)
    error = heading_error(command, own.heading_deg)
    rate = 9.80665 * math.tan(math.pi / 6) / (own.tas_nm_s * 1852)
    omega = math.copysign(rate, error)
    turning = min(seconds, abs(math.radians(error)) / rate)
    h0 = math.radians(own.heading_deg); h1 = h0 + omega * turning
    dx = own.tas_nm_s * (math.cos(h0) - math.cos(h1)) / omega
    dy = own.tas_nm_s * (math.sin(h1) - math.sin(h0)) / omega
    straight = seconds - turning
    dx += own.tas_nm_s * straight * math.sin(h1) + wx * seconds
    dy += own.tas_nm_s * straight * math.cos(h1) + wy * seconds
    return OwnState(own.x + dx, own.y + dy, _heading_degrees(h1),
                    command, own.tas_nm_s)


@dataclass(frozen=True)
class SearchSettings:
    horizon_s: int = 300
    sample_s: int = 1
    heading_spacing_deg: int = 10
    second_leg_at_s: int = 120

    def __post_init__(self):
        values = (self.horizon_s, self.sample_s, self.heading_spacing_deg, self.second_leg_at_s)
        if any(type(v) is not int or v <= 0 for v in values):
            raise InterfaceError('search settings must be positive integers')
        if (self.horizon_s > 600 or self.horizon_s % self.sample_s or
                360 % self.heading_spacing_deg or self.heading_spacing_deg < 5 or
                self.second_leg_at_s >= self.horizon_s or self.second_leg_at_s % self.sample_s):
            raise InterfaceError('unsupported search budget or grid')


@dataclass(frozen=True)
class Plan:
    first_heading_deg: float
    second_heading_deg: float
    second_leg_at_s: int
    rank: tuple
    predicted_report_possible: bool
    predicted_join_all_hypotheses: bool
    candidates: int
    samples_per_candidate: int
    horizon_s: int
    deadline_within_horizon: bool
    forecast_hypotheses: int
    cpu_ns: int


def chord_minimum(start: tuple[float, float], end: tuple[float, float],
                  tolerance: float = .25) -> tuple[float, float, float | None]:
    """Closest distance/fraction and first contact fraction on a relative chord.

    Exact for linear relative motion. Curved own/target trajectories are only
    approximated by the chord; no world-level contact certificate is returned.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    square = dx*dx + dy*dy
    centre = 0.0 if square == 0 else -(start[0]*dx + start[1]*dy)/square
    at = min(1.0, max(0.0, centre))
    distance = math.hypot(start[0] + at*dx, start[1] + at*dy)
    contact = None
    if math.hypot(*start) <= tolerance:
        contact = 0.0
    elif square and distance <= tolerance:
        # Computing the perpendicular vector avoids subtracting two large squares.
        px, py = start[0] + centre*dx, start[1] + centre*dy
        half_width = math.sqrt(max(0.0, tolerance*tolerance - px*px - py*py)/square)
        contact = min(1.0, max(0.0, centre - half_width))
    return distance, at, contact


def prediction_metrics(track: list[OwnState], target: list[dict], times: list[int],
                       report_distance: float, wind: tuple[float, float],
                       report_already_sent: bool) -> tuple[float, float, bool]:
    """Heuristic continuous chord minimum and report at a delivered-action second.

    Reports require predicted instantaneous closing, before first chord contact.
    The default one-second mesh follows the report interface; coarse settings
    remain available for synthetic sensitivity, with their aliasing limitation.
    """
    relative = [(own.x - point['x'], own.y - point['y']) for own, point in zip(track,target)]
    best = (math.hypot(*relative[0]), float(times[0]))
    first_contact = float(times[0]) if best[0] <= .25 else math.inf
    for i in range(1, len(times)):
        distance, at, contact = chord_minimum(relative[i-1], relative[i])
        duration = times[i] - times[i-1]
        best = min(best, (distance, times[i-1] + at*duration))
        if contact is not None:
            first_contact = min(first_contact, times[i-1] + contact*duration)
    report_possible = report_already_sent
    for own, point, relative_position, dt in zip(track, target, relative, times):
        distance = math.hypot(*relative_position)
        if dt >= first_contact or distance == 0 or abs(distance - report_distance) >= .20:
            continue
        heading = math.radians(own.heading_deg)
        rvx = own.tas_nm_s * math.sin(heading) + wind[0] - point['vx']
        rvy = own.tas_nm_s * math.cos(heading) + wind[1] - point['vy']
        rate = (relative_position[0]*rvx + relative_position[1]*rvy)/distance
        if rate < -1e-9:
            report_possible = True
            break
    return best[0], best[1], report_possible


class GuidanceSearch:
    """Rank a fixed two-leg family; execute only its first command externally.

    Lexicographic ranking: report opportunity in every sampled hypothesis,
    chord-predicted join in every hypothesis, worst miss, earliest worst closest
    time, then turn magnitude and deterministic candidate order. Goals
    outside the horizon remain unresolved. No truth labels are consulted.
    """
    def __init__(self, settings: SearchSettings = SearchSettings()):
        self.settings = settings
        self.cpu_ns = 0
        self.searches = 0

    def choose(self, estimator: EpisodeEstimator, own: OwnState,
               report_already_sent: bool = False) -> Plan:
        started = time.process_time_ns()
        try:
            if estimator.t is None or estimator.goal is None or estimator.wind_nm_s is None:
                raise InterfaceError('guidance requires a current public bundle')
            goal = estimator.goal; cfg = self.settings
            remaining = cfg.horizon_s if goal.deadline is None else goal.deadline - estimator.t
            if remaining <= 0:
                raise InterfaceError('goal deadline reached')
            horizon = min(remaining, cfg.horizon_s)
            times = list(range(0, horizon + 1, cfg.sample_s))
            if times[-1] != horizon:
                times.append(horizon)
            targets = [estimator.forecast(goal.tanker, estimator.t + dt) for dt in times]
            hypotheses = len(targets[0])
            identities = [point['change_nm'] for point in targets[0]]
            if not hypotheses or any([point['change_nm'] for point in points] != identities
                                     for points in targets):
                raise InterfaceError('forecast hypothesis identity changed during search')
            wind = estimator.wind_nm_s
            # Lead headings aim at a forecast position after subtracting wind drift.
            # A turn is evaluated afterwards; this construction itself is not an intercept solution.
            leads = []
            for dt, points in zip(times[1:], targets[1:]):
                if dt % 30 == 0 or dt == horizon:
                    p = points[hypotheses // 2]
                    leads.append(_heading_degrees(math.atan2(p['x'] - own.x - wind[0] * dt,
                                                             p['y'] - own.y - wind[1] * dt)))
            headings = sorted(set([own.commanded_heading_deg, *range(0, 360, cfg.heading_spacing_deg), *leads]))
            terminal = leads[-1]
            candidates = sorted({(h, h) for h in headings} | {(h, terminal) for h in headings})
            best = None
            for index, (first, second) in enumerate(candidates):
                current = own; previous_dt = 0; track = [own]
                for dt in times[1:]:
                    cut = cfg.second_leg_at_s
                    if previous_dt < cut < dt:
                        current = propagate(current, first, cut - previous_dt, wind)
                        current = propagate(current, second, dt - cut, wind)
                    else:
                        current = propagate(current, first if previous_dt < cut else second,
                                            dt - previous_dt, wind)
                    track.append(current); previous_dt = dt
                minima = []; nearest_times = []; reports = []
                for j in range(hypotheses):
                    distance, at, report = prediction_metrics(
                        track, [points[j] for points in targets], times,
                        goal.report_at_nm, wind, report_already_sent)
                    minima.append(distance); nearest_times.append(at); reports.append(report)
                report_possible = all(reports); join_all = all(distance <= .25 for distance in minima)
                rank = (not report_possible, not join_all, max(minima), max(nearest_times),
                        abs(heading_error(first, own.heading_deg)), index)
                if best is None or rank < best[0]:
                    best = rank, first, second, report_possible, join_all
            self.searches += 1
            elapsed = time.process_time_ns() - started
            return Plan(best[1], best[2], cfg.second_leg_at_s, best[0], best[3], best[4],
                        len(candidates), len(times), horizon,
                        goal.deadline is not None and remaining <= cfg.horizon_s, hypotheses, elapsed)
        finally:
            self.cpu_ns += time.process_time_ns() - started
