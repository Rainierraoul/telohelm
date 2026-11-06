"""D26 timing harness for cross-review before execution. No world or reserved sampler.

Importing this module does not profile. Explicit campaign() repeats the unchanged
144 first-act API and then 432 fixed final-act intervals. Setup is separate actual
cost; nested telemetry is never added. The expanded maximum is illustrative.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
from . import qualification as Q
from .adapter import AdapterSettings
from .guidance import SearchSettings

MANIFEST_DIGEST = '1baee96ba879cc63805c616b0dbc0ddcdb07e6539dbbcafe75478468cfa6eee6'


def read_inputs(path: Path) -> dict:
    body = json.loads(Path(path).read_text())
    saved = body.pop('body_sha256')
    if saved != MANIFEST_DIGEST or Q.digest(body) != saved:
        raise ValueError('the cross-reviewed timing manifest is required')
    sequences = body['sequences']
    if len(sequences) != 6 or len({s['id'] for s in sequences}) != 6:
        raise ValueError('six unique sequences required')
    for seq in sequences:
        payload = {k: v for k, v in seq.items() if k != 'input_sha256'}
        if Q.digest(payload) != seq['input_sha256']:
            raise ValueError('sequence digest differs')
        if seq['timed_bundle_index'] != len(seq['bundles']) - 1:
            raise ValueError('only the final act is timed')
    return body


def _delta(start, end):
    if type(start) is not int or type(end) is not int or end < start:
        return None
    return end - start


def profile_final(inputs, factory=Q.AdaptiveController,
                  cpu_clock=time.process_time_ns, wall_clock=time.perf_counter_ns):
    """Fresh instance for each (repeat, cell, sequence). No state across instances.

    factory/reset/all prefix acts are one separate interval. The final act includes
    its input copy/update/decision. Trace inspection is outside both intervals.
    Prefix help and public belief state are retained as actually observed, never
    inferred from a sequence's name. Partial/setup/final/audit failures are explicit.
    """
    rows = []
    for repetition in range(Q.REPETITIONS):
        for cell in Q.policy_grid():
            for seq in inputs['sequences']:
                actions, pilot, error, stage = [], None, None, 'setup'
                begin_cpu, begin_wall = cpu_clock(), wall_clock()
                try:
                    pilot = factory(SearchSettings(**cell['search']), AdapterSettings(**cell['adapter']),
                                    cell['coordinate_allowance_nm'])
                    pilot.reset(deepcopy(seq['briefing']))
                    for bundle in seq['bundles'][:-1]:
                        actions.append(pilot.act(deepcopy(bundle)))
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                setup_cpu, setup_wall = _delta(begin_cpu, cpu_clock()), _delta(begin_wall, wall_clock())
                final_cpu = final_wall = None
                final_action = None
                if error is None:
                    stage = 'final'
                    begin_cpu, begin_wall = cpu_clock(), wall_clock()
                    try:
                        final_action = pilot.act(deepcopy(seq['bundles'][-1]))
                    except Exception as exc:
                        error = f'{type(exc).__name__}: {exc}'
                    final_cpu, final_wall = _delta(begin_cpu, cpu_clock()), _delta(begin_wall, wall_clock())
                trace = None
                try:
                    if pilot is not None:
                        trace = pilot.trace
                except Exception as exc:
                    if error is None:
                        stage, error = 'audit', f'{type(exc).__name__}: {exc}'
                rows.append({'family': 'stateful', 'cell': cell['id'], 'workload': seq['id'],
                             'repetition': repetition, 'input_sha256': seq['input_sha256'],
                             'error': error, 'error_stage': stage if error else None,
                             'setup_cpu_ns': setup_cpu, 'setup_wall_ns': setup_wall,
                             'cpu_ns': final_cpu, 'wall_ns': final_wall,
                             'prefix_actions': deepcopy(actions), 'final_action': deepcopy(final_action),
                             'trace': deepcopy(trace),
                             'scope': 'setup separate; final act inclusive; trace export outside intervals'})
    return rows


def projection(rows, inputs):
    """Complete unique 576 intervals only; errors/unknown setup remain unavailable."""
    expected = {(family, c['id'], w['id'], r):
                (Q.digest(w) if family == 'first' else w['input_sha256'])
                for family, workloads in [('first', Q.public_workloads()), ('stateful', inputs['sequences'])]
                for c in Q.policy_grid() for w in workloads for r in range(Q.REPETITIONS)}
    keys = [(x['family'], x['cell'], x['workload'], x['repetition']) for x in rows]
    if len(keys) != len(expected) or set(keys) != set(expected) or len(set(keys)) != len(keys):
        raise ValueError('complete unique 576 campaign intervals required')
    for key, row in zip(keys, rows):
        if row['input_sha256'] != expected[key]:
            raise ValueError('campaign input hash differs')
    clocks = lambda x: type(x) is int and x >= 0
    if any(x['error'] or not clocks(x['cpu_ns']) or not clocks(x['wall_ns']) or
           (x['family'] == 'stateful' and
            (not clocks(x['setup_cpu_ns']) or not clocks(x['setup_wall_ns']))) for x in rows):
        return {'projected_cpu_h': None, 'disposition': 'incomplete-or-unknown; no flight clearance'}
    maxima = {c['id']: max(x['cpu_ns'] for x in rows if x['cell'] == c['id']) for c in Q.policy_grid()}
    hours = sum(maxima.values()) * len(Q.SENSITIVITY_SEEDS) * Q.HORIZON_STEPS / 1e9 / 3600
    return {'projected_cpu_h': hours, 'budget_cpu_h': Q.BUDGET_CPU_H, 'maxima_cpu_ns': maxima,
            'disposition': 'stop-and-revise' if hours > Q.BUDGET_CPU_H else 'next-review-only; no flight clearance',
            'basis': 'max of 144 inclusive first acts and 432 final acts per cell times20 times3600; not global bound',
            'actual_timed_cpu_ns': sum(x['cpu_ns'] for x in rows),
            'actual_stateful_setup_cpu_ns': sum(x['setup_cpu_ns'] for x in rows if x['family'] == 'stateful'),
            'actual_timed_wall_ns': sum(x['wall_ns'] for x in rows),
            'actual_stateful_setup_wall_ns': sum(x['setup_wall_ns'] for x in rows if x['family'] == 'stateful'),
            'accounting': 'nonoverlapping measured intervals; export/bootstrap outside; no nested counter addition'}


def campaign(path: Path, factory=Q.AdaptiveController,
             cpu_clock=time.process_time_ns, wall_clock=time.perf_counter_ns):
    """Explicit measured-feasibility API, only after exact implementation cross-review.

    The caller must pin source/dependencies/machine/load and preserve every row.
    No timing execution or protocol/flight authorization occurs merely on import.
    """
    inputs = read_inputs(path)
    first = [dict(r, family='first') for r in Q.profile_search(factory, cpu_clock, wall_clock)]
    rows = first + profile_final(inputs, factory, cpu_clock, wall_clock)
    return {'stage': 'measured-feasibility-only', 'flight_authorized': False,
            'manifest_body_sha256': MANIFEST_DIGEST, 'rows': rows, 'projection': projection(rows, inputs)}
