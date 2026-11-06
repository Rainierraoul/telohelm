"""Observation-only episode adapter. Decisions remain provisional heuristics.

No world, hidden label, outcome, reference pilot or Telohelm dependency.
Synthetic verification is not evidence of episode competence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import time

from .estimation import EpisodeEstimator; from .estimation import InterfaceError; from .estimation import number; from .estimation import second
from .guidance import GuidanceSearch; from .guidance import OwnState; from .guidance import SearchSettings


@dataclass(frozen=True)
class AdapterSettings:
    report_guard_nm: float = .20
    inbound_guard_nm_s: float = 1e-9
    stale_after_s: int = 10

    def __post_init__(self):
        if not 0 < number(self.report_guard_nm, 'report guard') < .25:
            raise InterfaceError('report guard must be inside the public tolerance')
        if number(self.inbound_guard_nm_s, 'inbound guard') <= 0:
            raise InterfaceError('inbound guard must be positive')
        if type(self.stale_after_s) is not int or self.stale_after_s < 0:
            raise InterfaceError('invalid stale threshold')


class AdaptiveController:
    """reset(briefing), then act(bundle) once per strictly increasing second.

    Only the first planned heading is executed, with a new search each second.
    A report needs agreement of the finite current hypotheses on distance and
    instantaneous inbound motion. This agreement is not a truth certificate.
    One stale-evidence help attempt is allowed per episode, never per version.
    The fixed 30-second reply delay is part of the public interface.
    """

    def __init__(self, search_settings: SearchSettings | None = None,
                 adapter_settings: AdapterSettings | None = None,
                 coordinate_allowance_nm: float = 1e-6):
        self.search_settings = search_settings or SearchSettings()
        self.settings = adapter_settings or AdapterSettings()
        self.estimator = EpisodeEstimator(coordinate_allowance_nm)
        self.search = GuidanceSearch(self.search_settings)
        self._ready = False
        self._last_t = None
        self._last_report_version = None
        self._reported_versions: set[int] = set()
        self._help_attempted = False
        self._trace: list[dict] = []
        self.cpu_ns = 0

    def reset(self, briefing: dict) -> None:
        self.estimator.reset(briefing)
        self.search = GuidanceSearch(self.search_settings)
        self._last_t = self._last_report_version = None
        self._reported_versions = set()
        self._help_attempted = False
        self._trace = []
        self.cpu_ns = 0
        self._ready = True

    @property
    def trace(self) -> list[dict]:
        """A copied public-data audit; callers cannot mutate policy state."""
        return deepcopy(self._trace)

    def act(self, bundle: dict) -> dict:
        started = time.process_time_ns()
        try:
            if not self._ready:
                raise InterfaceError('reset required')
            t = second(bundle['t'], 'bundle time')
            if self._last_t is not None and t <= self._last_t:
                raise InterfaceError('act needs a strictly increasing second')
            own = OwnState.read(bundle['own'])
            refusals = bundle['refusals']
            if not isinstance(refusals, list):
                raise InterfaceError('refusals must be a list')
            for refusal in refusals:
                if (not isinstance(refusal, dict) or self._last_t is None or
                        second(refusal['t'], 'refusal time') != self._last_t or
                        not isinstance(refusal['reason'], str)):
                    raise InterfaceError('refusal must describe the preceding action')
            # Estimator update is transactional. Roll back submission state only
            # after a valid bundle. The world refuses the whole action atomically.
            self.estimator.update(bundle)
            if refusals and self._last_report_version is not None:
                self._reported_versions.discard(self._last_report_version)
            goal = self.estimator.goal
            age = t - self.estimator.targets[goal.tanker].latest.measured_at
            actions: dict = {}
            audit = {'t': t, 'goal_version': goal.version, 'evidence_age_s': age,
                     'search_settings': asdict(self.search_settings),
                     'adapter_settings': asdict(self.settings),
                     'coordinate_allowance_nm': self.estimator.coordinate_allowance_nm,
                     'previous_action_refused': bool(refusals), 'report_hypotheses': []}
            sent = None
            if goal.deadline is not None and t >= goal.deadline:
                audit['decision'] = 'deadline_reached'
            else:
                hypotheses = self.estimator.forecast(goal.tanker, t)
                heading = math.radians(own.heading_deg)
                wx, wy = self.estimator.wind_nm_s
                vx = own.tas_nm_s * math.sin(heading) + wx
                vy = own.tas_nm_s * math.cos(heading) + wy
                for hyp in hypotheses:
                    rx, ry = hyp['x'] - own.x, hyp['y'] - own.y
                    distance = math.hypot(rx, ry)
                    rate = None if distance == 0 else (rx*(hyp['vx']-vx) + ry*(hyp['vy']-vy))/distance
                    audit['report_hypotheses'].append({'change_nm': hyp['change_nm'],
                                                      'range_nm': distance, 'range_rate_nm_s': rate})
                report_ok = bool(hypotheses) and all(
                    abs(h['range_nm'] - goal.report_at_nm) < self.settings.report_guard_nm and
                    h['range_rate_nm_s'] is not None and
                    h['range_rate_nm_s'] < -self.settings.inbound_guard_nm_s
                    for h in audit['report_hypotheses'])
                already = goal.version in self._reported_versions
                plan = self.search.choose(self.estimator, own, report_already_sent=already or report_ok)
                actions = {'goal_version': goal.version, 'heading_deg': plan.first_heading_deg}
                audit['plan'] = asdict(plan)
                if report_ok and not already:
                    distance_text = f'{goal.report_at_nm:.10f}'.rstrip('0').rstrip('.')
                    actions['report'] = (f'{self.estimator.control_callsign}, {self.estimator.own_callsign}, '
                                         f'{goal.tanker} at {distance_text} NM inbound')
                    sent = goal.version
                    self._reported_versions.add(sent)
                remaining = None if goal.deadline is None else goal.deadline - t
                if age > self.settings.stale_after_s and not self._help_attempted:
                    if remaining is None or remaining > 30:
                        actions['help'] = {'reason': 'missing_or_stale_evidence',
                                           'text': f'Request current position of {goal.tanker}; latest measurement is {age} s old.'}
                        self._help_attempted = True
                        audit['help_decision'] = 'stale_evidence_attempt'
                    else:
                        audit['help_decision'] = 'reply_cannot_arrive_before_deadline'
                else:
                    audit['help_decision'] = 'already_attempted' if self._help_attempted else 'evidence_within_age_threshold'
                audit['decision'] = 'public_observation_replan'
            self._last_t, self._last_report_version = t, sent
            audit['actions'] = deepcopy(actions)
            audit['cpu_ns'] = time.process_time_ns() - started
            self._trace.append(audit)
            return deepcopy(actions)
        finally:
            # Includes all nested estimator/search work and refused calls.
            # Do not add component counters to this total again.
            self.cpu_ns += time.process_time_ns() - started
