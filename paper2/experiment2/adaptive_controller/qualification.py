"""Prospective controller qualification preparation, not a flight runner or clearance.

The default command prints a draft recipe/fingerprint. It does not generate reserved
cases, profile the controller, fly a world, certify solvability or call a model.
Profile execution is an explicit API for use only after this package's cross-review.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import time

from .adapter import AdaptiveController; from .adapter import AdapterSettings
from .guidance import SearchSettings

SENSITIVITY_SEEDS = tuple(range(4540, 4560))
VERIFICATION_SEEDS = tuple(range(4560, 4600))
BUDGET_CPU_H = 72.  # TURN139: prospective sensitivity allocation accepted in TURN138/D25.
REPETITIONS = 3
HORIZON_STEPS = 3600


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def policy_grid():
    """Exact 24 prospectively fixed full-policy cells from DEVELOPMENT-PLAN.md."""
    return [{'id': f'h{h}-l{l}-r{r:g}-s{s}',
             'search': asdict(SearchSettings(horizon_s=h, heading_spacing_deg=l)),
             'adapter': asdict(AdapterSettings(report_guard_nm=r, stale_after_s=s)),
             'coordinate_allowance_nm': 1e-6}
            for h, l, r, s in product((150, 300, 600), (10, 20), (.15, .20), (10, 30))]


def constructed_recipes():
    """Named coverage requirements, not yet materialized or labelled worlds.

    Explicit numeric geometry and validation/flight script must be committed and
    cross-reviewed before these can become executable episode inputs.
    """
    recipes = [{'id': 'still-air', 'wind_kt_from': [0, 0]}]
    recipes += [{'id': f'wind-{s}-{d}', 'wind_kt_from': [s, d]}
                for s, d in product((20, 40, 60), (0, 90, 180, 270))]
    recipes += [{'id': f'orbit-{d:g}', 'fractional_radius_change': d} for d in (-.2, 0., .2)]
    recipes += [{'id': f'delay-{d}', 'delay_s': d} for d in (0, 5, 30, 120)]
    recipes += [
        {'id': 'finite-dropout', 'dropout_s': [20, 80]},
        {'id': 'permanent-dropout', 'dropout_s': [20, None]},
        {'id': 'delay-and-dropout', 'delay_s': 30, 'dropout_s': [20, None]},
        {'id': 'old-report-after-help', 'delay_s': 30, 'requires': 'new help measurement then older delivery'},
        {'id': 'tanker-change-while-turning', 'goal_tanker': 'SHELL 2', 'requires': 'change during a turn'},
        {'id': 'distance-change-while-turning', 'requires': 'new version/distance during a turn'},
        {'id': 'goal-change-with-help-pending', 'requires': 'goal change between accepted help and reply'},
        {'id': 'report-boundary', 'requires': 'inside/on/outside guard and public tolerance; inbound/outbound'},
        {'id': 'contact-boundary', 'requires': 'crossing and grazing contact; retained numerical alternatives'},
        {'id': 'deadline-boundary', 'requires': 'contact before/on/after deadline; goal event at same boundary'},
        {'id': 'different-command-and-actual', 'requires': 'actual heading differs from command'},
    ]
    return recipes


def public_workloads():
    """Fixed invented public observations for prospective search timing only.

    No hidden world, future observation, outcome or solvability label is supplied.
    These do not bound the cost of every belief state reached during a flight.
    """
    briefing = {'own_callsign': 'RECEIVER', 'control_callsign': 'CONTROL',
                'goal': {'version': 1, 'tanker': 'A', 'report_at_nm': 20., 'deadline': None},
                'tankers': [{'callsign': 'A', 'centre': [65., 47.], 'radius': 8.,
                             'speed_nm_s': 280/3600, 'direction': 'clockwise', 'position_at_0': [65., 55.]}]}
    result = []
    for name, second, measured in [('fresh', 0, True), ('stale', 180, False)]:
        bundle = {'t': second, 'own': {'x': 0., 'y': 0., 'heading_deg': 0., 'commanded_heading_deg': 90.,
                                      'tas_nm_s': .125, 'track_deg': 0., 'ground_speed_nm_s': .125},
                  'goal': None, 'refusals': [], 'reports': [{'callsign': 'A', 'x': 65., 'y': 55.,
                    'measured_at': 0, 'received_at': second}] if measured else []}
        result.append({'id': name, 'briefing': deepcopy(briefing), 'bundle': bundle})
    return result


def draft_package():
    """Fingerprint the draft recipe, code and environment; do not call it a freeze."""
    here = Path(__file__).resolve().parent
    study2 = here.parent
    from telohelm.taskpacks.tanker.world import episode, physics
    paths = {str(p.relative_to(study2)): p for p in sorted(here.glob('*.py'))}
    paths.update({'world/episode.py': Path(episode.__file__), 'world/physics.py': Path(physics.__file__)})
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()}
    body = {'stage': 'draft-preparation-only', 'flight_authorized': False,
            'python': platform.python_version(), 'source_sha256': hashes,
            'sensitivity_seeds': list(SENSITIVITY_SEEDS), 'verification_seeds': list(VERIFICATION_SEEDS),
            'grid': policy_grid(), 'constructed_recipes': constructed_recipes(),
            'workloads': public_workloads(), 'profile_repetitions': REPETITIONS,
            'budget_cpu_h': BUDGET_CPU_H,
            'open': ['protocol incorporates seed reservation before generation',
                     'exact sampler/draw order and materialized geometry/input identities',
                     'frozen validation/flight script and independent labels',
                     'cross-review before timing execution; measured feasibility before flights']}
    return body | {'draft_sha256': digest(body)}


def profile_search(factory=AdaptiveController, cpu_clock=time.process_time_ns, wall_clock=time.perf_counter_ns):
    """After cross-review only: three interleaved repetitions of 24 cells × 2 public inputs.

    Measures factory/reset/act inclusively once. Failure/unknown remains explicit.
    Nested search/estimator CPU counters are diagnostics, never added.
    """
    rows = []
    for repetition in range(REPETITIONS):
        for cell in policy_grid():
            for workload in public_workloads():
                start_cpu, start_wall = cpu_clock(), wall_clock()
                error = None
                try:
                    pilot = factory(SearchSettings(**cell['search']), AdapterSettings(**cell['adapter']),
                                    cell['coordinate_allowance_nm'])
                    pilot.reset(deepcopy(workload['briefing']))
                    pilot.act(deepcopy(workload['bundle']))
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                end_cpu, end_wall = cpu_clock(), wall_clock()
                rows.append({'cell': cell['id'], 'workload': workload['id'], 'repetition': repetition,
                             'input_sha256': digest(workload), 'error': error,
                             'cpu_ns': None if start_cpu is None or end_cpu is None else end_cpu-start_cpu,
                             'wall_ns': None if start_wall is None or end_wall is None else end_wall-start_wall})
    return rows


def feasibility_projection(rows):
    """Illustrative max-observed-call projection, never a global execution-time bound."""
    expected = {(c['id'], w['id'], r) for c in policy_grid() for w in public_workloads() for r in range(REPETITIONS)}
    keys = [(x['cell'], x['workload'], x['repetition']) for x in rows]
    if len(keys) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError('complete unique profile cells/workloads/repetitions required')
    if any(x['error'] or type(x['cpu_ns']) is not int or x['cpu_ns'] < 0 for x in rows):
        return {'projected_cpu_h': None, 'disposition': 'incomplete-or-unknown; no flight clearance'}
    per_cell = {c['id']: max(x['cpu_ns'] for x in rows if x['cell'] == c['id']) for c in policy_grid()}
    # Reset charged at every second overestimates that component for these inputs;
    # other belief states can still cost more, so this is not an upper bound on flights.
    hours = sum(per_cell.values()) * len(SENSITIVITY_SEEDS) * HORIZON_STEPS / 1e9 / 3600
    return {'projected_cpu_h': hours, 'budget_cpu_h': BUDGET_CPU_H,
            'disposition': 'stop-and-revise' if hours > BUDGET_CPU_H else 'next-review-only; no flight clearance',
            'basis': 'maximum observed inclusive call per cell, 20 scenarios × 3600 steps; not a universal bound'}


if __name__ == '__main__':
    print(json.dumps(draft_package(), indent=2))
