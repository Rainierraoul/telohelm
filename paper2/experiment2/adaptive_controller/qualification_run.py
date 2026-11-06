"""Candidate single-process qualification scheduler; no sampler or auto-launch.

Uses the shared comparator runner unchanged. A frozen reviewed supplied-case
package and independent scientific validation remain prerequisites to flights.
Unit controls use invented cases/stubs only. Never resume/overwrite implicitly.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import weakref

from . import qualification as Q
from .adapter import AdaptiveController; from .adapter import AdapterSettings
from .guidance import SearchSettings
from .. import comparators as C


class QualificationStop(RuntimeError):
    """Operational censoring by this wrapper, not an inference about competence."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _sources():
    here = Path(__file__).resolve().parent
    from telohelm.taskpacks.tanker.world import episode, physics
    from telohelm.runtime import trusted_log
    paths = {str(p.relative_to(here.parent)): p for p in sorted(here.glob('*.py'))}
    paths.update({'comparators.py': Path(C.__file__), 'world/episode.py': Path(episode.__file__),
                  'world/physics.py': Path(physics.__file__), 'runtime/trusted_log.py': Path(trusted_log.__file__)})
    return {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()}


def make_plan(cases, *, cells=None, budget_cpu_ns=72 * 3600 * 10**9, stop_at_unix_ns):
    """Fingerprint supplied scenarios, never generate or label a new one.

    cases: ordered sequence of (case_id, Scenario). All original/reference and
    certification evidence must be frozen separately before scientific reliance.
    A smaller test schedule must be described as such, never the full sensitivity.
    """
    _integer(budget_cpu_ns, 'CPU budget', 1)
    _integer(stop_at_unix_ns, 'absolute deadline', 1)
    if not cases or len({name for name, _ in cases}) != len(cases):
        raise ValueError('nonempty unique case ids required')
    if any(type(name) is not str or not name for name, _ in cases):
        raise ValueError('nonempty case strings required')
    cells = deepcopy(Q.policy_grid() if cells is None else cells)
    if not cells or len({c['id'] for c in cells}) != len(cells):
        raise ValueError('nonempty unique policy cells required')
    for cell in cells:
        if set(cell) != {'id', 'search', 'adapter', 'coordinate_allowance_nm'}:
            raise ValueError('exact policy settings required')
        if not isinstance(cell['id'], str) or not cell['id']:
            raise ValueError('nonempty policy id required')
        SearchSettings(**cell['search'])
        AdapterSettings(**cell['adapter'])
        # Validate the same constructor domain without building a controller.
        allowance = cell['coordinate_allowance_nm']
        if isinstance(allowance, bool) or not isinstance(allowance, (int, float)) or not 0 < allowance < 1:
            raise ValueError('invalid coordinate allowance')
    entries = [{'ordinal': i, 'cell': cell['id'], 'case_id': name,
                'scenario_sha256': Q.digest(asdict(scenario))}
               for i, (cell, name, scenario) in enumerate(
                   (cell, name, scenario) for cell in cells for name, scenario in cases)]
    body = {'stage': 'candidate-supplied-case-qualification', 'cells': cells,
            'entries': entries, 'source_sha256': _sources(),
            'budget_cpu_ns': budget_cpu_ns, 'stop_at_unix_ns': stop_at_unix_ns,
            'order': 'cell-major; supplied case order; one current process',
            'scope': 'no sampler, certification, independent labels or experiment clearance'}
    return body | {'body_sha256': Q.digest(body)}


class Budget:
    """Inclusive current-process CPU; cooperative stop with explicit unknowns.

    No child/process-tree accounting or arbitrary-operation preemption claim.
    Absolute wall-clock regression stops the run instead of extending a deadline.
    """
    def __init__(self, cpu_limit, stop_at, cpu_clock, wall_clock):
        self.limit, self.stop_at = cpu_limit, stop_at
        self.cpu_clock, self.wall_clock = cpu_clock, wall_clock
        self.origin = self.last_cpu = self.last_wall = None
        self.last_known_spent = 0
        self.spent = None
        self.reason = None
        self.samples = 0

    def check(self):
        if self.reason is not None:
            raise QualificationStop(self.reason)
        self.samples += 1
        try:
            cpu, wall = self.cpu_clock(), self.wall_clock()
            _integer(cpu, 'CPU clock')
            _integer(wall, 'wall clock')
            if (self.last_cpu is not None and cpu < self.last_cpu or
                    self.last_wall is not None and wall < self.last_wall):
                raise ValueError('clock regressed')
            if self.origin is None:
                self.origin = cpu
            self.last_cpu, self.last_wall = cpu, wall
            self.spent = self.last_known_spent = cpu - self.origin
        except Exception as exc:
            self.spent, self.reason = None, f'unknown-clock:{type(exc).__name__}'
            raise QualificationStop(self.reason) from exc
        if self.spent >= self.limit:
            self.reason = 'CPU-ceiling'
        elif wall >= self.stop_at:
            self.reason = 'absolute-deadline'
        if self.reason:
            raise QualificationStop(self.reason)

    def record(self):
        return {'cpu_ns': self.spent, 'known_cpu_lower_ns': self.last_known_spent,
                'overshoot_ns': None if self.spent is None else max(0, self.spent - self.limit),
                'last_wall_unix_ns': self.last_wall, 'samples': self.samples, 'stop_reason': self.reason,
                'scope': 'inclusive current-process through last sample; final serialization/close outside'}


class _CheckedPilot:
    def __init__(self, pilot, budget):
        self.pilot, self.budget = pilot, budget

    @property
    def cpu_ns(self):
        return self.pilot.cpu_ns

    def reset(self, briefing):
        self.budget.check()
        self.pilot.reset(briefing)
        self.budget.check()

    def act(self, bundle):
        self.budget.check()
        action = self.pilot.act(bundle)
        self.budget.check()  # a produced but over-budget action is never applied
        return action


def run(plan, cases, out, *, expected_plan_sha256, factory=AdaptiveController,
        episode_runner=C.run_pilot_episode, cpu_clock=time.process_time_ns,
        wall_clock=time.time_ns):
    """After independent review/freeze only; no work occurs on import.

    Uses exact shared comparator records. A wrapper stop is represented by that
    runner as PolicyFailed; this schedule explicitly retains its operational
    censoring cause. Raw records are never rewritten to look like a completion.
    Any controller/runner failure stops the campaign; remaining cases stay named.
    """
    body = {k: v for k, v in plan.items() if k != 'body_sha256'}
    if Q.digest(body) != plan.get('body_sha256') or plan['body_sha256'] != expected_plan_sha256:
        raise ValueError('independently retained plan fingerprint differs')
    rebuilt = make_plan(cases, cells=plan['cells'], budget_cpu_ns=plan['budget_cpu_ns'],
                        stop_at_unix_ns=plan['stop_at_unix_ns'])
    if rebuilt != plan:
        raise ValueError('supplied scenarios/settings/source differ from frozen plan')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(plan['budget_cpu_ns'], plan['stop_at_unix_ns'], cpu_clock, wall_clock)
    records, seen = [], weakref.WeakSet()
    current = [None]
    cases_by_id = dict(cases)
    cells = {c['id']: c for c in plan['cells']}
    with (out / 'schedule.jsonl').open('xb') as journal:
        def retain(row):
            journal.write(json.dumps(row, sort_keys=True, allow_nan=False).encode() + b'\n')
            journal.flush()
            os.fsync(journal.fileno())

        def check():
            try:
                budget.check()
                return True
            except QualificationStop:
                return False

        check()  # start before schedule export; export work is in later samples
        retain({'kind': 'plan', 'plan': plan})
        stopped = not check()
        failure = None
        for entry in plan['entries']:
            if stopped:
                break
            if not check():
                stopped = True
                break
            cell = cells[entry['cell']]
            current[0] = None

            def make_pilot():
                budget.check()  # constructor too is inside the measured process
                pilot = factory(SearchSettings(**cell['search']), AdapterSettings(**cell['adapter']),
                                cell['coordinate_allowance_nm'])
                if pilot in seen:
                    raise ValueError('fresh controller instance required for every episode')
                seen.add(pilot)
                current[0] = pilot
                return _CheckedPilot(pilot, budget)

            folder = out / f"case-{entry['ordinal']:05d}"
            retain({'kind': 'case-start', **entry, 'budget': budget.record()})
            status, summary, error, operational_stop = 'failed', None, None, False
            try:
                condition = C.Condition('controller-qualification', 'adaptive_controller', make_pilot,
                                        {k: v for k, v in cell.items() if k != 'id'})
                summary = episode_runner(cases_by_id[entry['case_id']], condition, folder,
                                         'controller-qualification')
                status = 'completed' if summary['complete'] else 'censored'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'[:500]
                operational_stop = budget.reason is not None
            if not check():
                stopped = True
            if operational_stop:
                status = 'censored'
            # A fully completed episode remains completed if later export
            # crosses the budget. No next episode is started in that case.
            trace_file = None
            if current[0] is not None:
                trace_file = folder / 'controller-trace.json'
                with trace_file.open('x') as f:
                    json.dump(current[0].trace, f, allow_nan=False)
                    f.flush()
                    os.fsync(f.fileno())
            summary_file = folder / 'summary.json'
            record = {**entry, 'status': status,
                      'summary': None if summary is None else {k: summary[k] for k in
                          ('complete', 'goal_met', 'joined', 'settled', 'action_steps') if k in summary},
                      'raw_runner_error': error,
                      'censor_reason': budget.reason if status == 'censored' else None,
                      'budget': budget.record(),
                      'records_directory': folder.name,
                      'summary_sha256': hashlib.sha256(summary_file.read_bytes()).hexdigest()
                          if summary_file.exists() else None,
                      'controller_trace_sha256': hashlib.sha256(trace_file.read_bytes()).hexdigest()
                          if trace_file else None}
            records.append(record)
            retain({'kind': 'case-result', **record})
            if status != 'completed':
                stopped, failure = True, error or 'incomplete episode'
        check()  # include retained result writes through this final sample
        result = {'plan_sha256': plan['body_sha256'], 'records': records,
                  'unattempted': plan['entries'][len(records):], 'budget': budget.record(),
                  'failure': failure, 'all_completed': len(records) == len(plan['entries']) and
                      all(r['status'] == 'completed' for r in records),
                  'scientific_admission': False,
                  'limitations': ['candidate runner requires exact cross-review and scientific package freeze',
                                  'cooperative stop; no bound on one uninterruptible operation',
                                  'raw shared-runner policy exception retained for operational censoring',
                                  'clock sample excludes final serialization/close; no nested CPU addition']}
        retain({'kind': 'campaign-end', 'budget': result['budget'], 'failure': failure,
                'completed': sum(r['status'] == 'completed' for r in records),
                'unattempted': len(result['unattempted'])})
    with (out / 'qualification-summary.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return result
