"""Candidate controlled late-act profiler. Importing performs no measurements.

Preload public estimator inputs only, with explicitly cold policy flags. These
are controlled cost states, not reached full-policy states. Exact review and a
frozen invocation/compute estimate precede actual campaign execution.
"""
from copy import deepcopy
import time

from . import history_profile as H; from . import qualification as Q; from . import stateful_profile as S
from .adapter import AdaptiveController; from .adapter import AdapterSettings
from .guidance import SearchSettings

SCOPE = 'cold policy with public estimator history; not a reached episode state'


def flags(pilot):
    return {'ready': pilot._ready, 'last_t': pilot._last_t,
            'last_report_version': pilot._last_report_version,
            'reported_versions': sorted(pilot._reported_versions),
            'help_attempted': pilot._help_attempted,
            'trace_count': len(pilot.trace), 'controller_cpu_ns': pilot.cpu_ns}


COLD_FLAGS = {'ready': True, 'last_t': None, 'last_report_version': None,
              'reported_versions': [], 'help_attempted': False,
              'trace_count': 0, 'controller_cpu_ns': 0}


def load_cold(pilot, payload):
    """Explicit diagnostic loader, never an operational controller state restore.

    No policy act, GuidanceSearch, help or report decision runs during preload.
    Copies protect the immutable caller's payload from mutable controller APIs.
    """
    pilot.reset(deepcopy(payload['briefing']))
    for bundle in payload['prefix_bundles']:
        pilot.estimator.update(deepcopy(bundle))
    if flags(pilot) != COLD_FLAGS:
        raise ValueError('preload changed cold policy flags')


def _one(spec, payload, cell, repetition, factory, cpu_clock, wall_clock):
    pilot, before, after, before_flags, after_flags, trace = (None,) * 6
    error, stage, action = None, 'setup', None
    c0, w0 = H._clock(cpu_clock), H._clock(wall_clock)
    try:
        pilot = factory(SearchSettings(**cell['search']), AdapterSettings(**cell['adapter']),
                        cell['coordinate_allowance_nm'])
        load_cold(pilot, payload)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    setup_cpu = H._delta(c0, H._clock(cpu_clock))
    setup_wall = H._delta(w0, H._clock(wall_clock))
    cpu = wall = None
    if error is None:
        stage = 'before-audit'
        try:
            before, before_flags = H.state(pilot.estimator), flags(pilot)
            H.validate_history(before, spec)
            if before_flags != COLD_FLAGS:
                raise ValueError('non-cold policy before final act')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    if error is None:
        stage = 'final'
        c0, w0 = H._clock(cpu_clock), H._clock(wall_clock)
        try:
            action = pilot.act(deepcopy(payload['final_bundle']))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        cpu = H._delta(c0, H._clock(cpu_clock))
        wall = H._delta(w0, H._clock(wall_clock))
    if pilot is not None:
        try:
            after, after_flags, trace = H.state(pilot.estimator), flags(pilot), pilot.trace
            if error is None:
                stage = 'after-audit'
                H.validate_history(after, spec, final=True)
                if (after_flags['last_t'] != spec['final_second'] or len(trace) != 1 or
                        trace[0]['t'] != spec['final_second'] or trace[0]['actions'] != action):
                    raise ValueError('one final act and matching action trace required')
        except Exception as exc:
            if error is None:
                error, stage = f'{type(exc).__name__}: {exc}', 'after-audit'
    return {'family': 'late-cold', 'cell': cell['id'], 'workload': spec['id'],
            'repetition': repetition, 'input_sha256': spec['input_sha256'],
            'error': error, 'error_stage': stage if error else None,
            'setup_cpu_ns': setup_cpu, 'setup_wall_ns': setup_wall,
            'cpu_ns': cpu, 'wall_ns': wall, 'before': before, 'after': after,
            'before_flags': before_flags, 'after_flags': after_flags,
            'final_action': action, 'trace': trace, 'scope': SCOPE,
            'accounting': 'setup and final disjoint; audits/export outside; nested counters not added'}


def profile(manifest_path, payload_root, factory=AdaptiveController,
            cpu_clock=time.process_time_ns, wall_clock=time.perf_counter_ns):
    """After review: fixed 2,304 intervals, no generation or flight.

    Loading and validating all frozen bytes happens before any measurement.
    Failure rows remain in schedule; no replacement/reordering or early fit test.
    """
    manifest, payloads, package = H.read_payloads(manifest_path, payload_root)
    rows = [_one(spec, payload, cell, rep, factory, cpu_clock, wall_clock)
            for rep in range(Q.REPETITIONS) for cell in Q.policy_grid()
            for spec, payload in zip(manifest['sequences'], payloads)]
    return {'stage': 'controlled-late-act-diagnostic-only', 'flight_authorized': False,
            'package_sha256': package, 'scope': SCOPE, 'rows': rows}


def projection(control_rows, late_rows, short_inputs, history_manifest):
    """Same-campaign 576 +2,304 maxima; finite illustrative projection only.

    This API checks input/key/scoped-row consistency. The caller must establish
    same execution machine/source/invocation provenance; rows alone cannot.
    """
    baseline = S.projection(control_rows, short_inputs)
    if Q.digest(history_manifest) != H.MANIFEST_DIGEST:
        raise ValueError('accepted history manifest required')
    expected = {(c['id'], s['id'], r): s['input_sha256']
                for r in range(Q.REPETITIONS) for c in Q.policy_grid()
                for s in history_manifest['sequences']}
    keys = [(r['cell'], r['workload'], r['repetition']) for r in late_rows]
    if len(keys) != 2304 or len(set(keys)) != 2304 or set(keys) != set(expected):
        raise ValueError('complete unique 2304 late-act intervals required')
    for key, row in zip(keys, late_rows):
        if row['family'] != 'late-cold' or row['input_sha256'] != expected[key] or row['scope'] != SCOPE:
            raise ValueError('late-act input or scope differs')
    known = lambda v: type(v) is int and v >= 0
    if (baseline['projected_cpu_h'] is None or any(
            row['error'] or row['before_flags'] != COLD_FLAGS or
            any(not known(row[k]) for k in ('cpu_ns', 'wall_ns', 'setup_cpu_ns', 'setup_wall_ns'))
            for row in late_rows)):
        return {'projected_cpu_h': None, 'disposition': 'incomplete-or-unknown; no flight clearance'}
    maxima = {c['id']: max(baseline['maxima_cpu_ns'][c['id']],
                          max(r['cpu_ns'] for r in late_rows if r['cell'] == c['id']))
              for c in Q.policy_grid()}
    hours = sum(maxima.values()) * len(Q.SENSITIVITY_SEEDS) * Q.HORIZON_STEPS / 1e9 / 3600
    return {'projected_cpu_h': hours, 'maxima_cpu_ns': maxima, 'budget_cpu_h': Q.BUDGET_CPU_H,
            'disposition': 'stop-and-revise' if hours > Q.BUDGET_CPU_H else 'next-review-only; no flight clearance',
            'actual_late_setup_cpu_ns': sum(r['setup_cpu_ns'] for r in late_rows),
            'actual_late_final_cpu_ns': sum(r['cpu_ns'] for r in late_rows),
            'basis': 'same-campaign per-cell maximum over 576+2304 acts times20 times3600; not a global bound',
            'non_additive': 'never add estimator-only or nested telemetry to inclusive act costs'}
