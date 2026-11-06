"""Old/new equivalence on 60 committed invented public sequences. No worlds/profile/results.

Reference and inputs were pushed at 0be6683 before source modification. All binary
floating values, candidate trajectories/metrics/order and public behavior must match;
only timing telemetry and propagation-work counts may differ.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import paper2.experiment2.adaptive_controller.guidance as current
from paper2.experiment2.adaptive_controller.adapter import AdaptiveController; from paper2.experiment2.adaptive_controller.adapter import AdapterSettings
from paper2.experiment2.adaptive_controller.qualification import digest

FIXTURES = Path(__file__).parent / 'fixtures'
SOURCE = FIXTURES / 'controller_guidance_turn134.py'
REFERENCE_SHA256 = '0ff8f8fb212399cdd7a9c19d1cd146fe8f485df95279d90d2c398a307e46a1e9'
EQUIVALENCE_STATS = []
INPUTS = json.loads((FIXTURES / 'controller_reuse_inputs.json').read_text())
SPEC = importlib.util.spec_from_file_location('paper2.experiment2.adaptive_controller._frozen_turn134', SOURCE)
REFERENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REFERENCE
SPEC.loader.exec_module(REFERENCE)


def bits(value):
    if type(value) is float:
        return {'binary_float': value.hex()}
    if isinstance(value, dict):
        return {k: bits(v) for k, v in value.items() if k != 'cpu_ns'}
    if isinstance(value, (list, tuple)):
        return [bits(v) for v in value]
    return value


def test_pushed_input_and_reference_pins():
    body = {k: v for k, v in INPUTS.items() if k != 'input_sha256'}
    assert digest(body) == '303aee3fbaa85ca5e096ac130fdb4ebc5fbf610b1735cb8497cc936e1f69a181'
    assert len(INPUTS['cases']) == 60
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == REFERENCE_SHA256


def run_case(module, case, monkeypatch):
    pilot = AdaptiveController(current.SearchSettings(**case['search']), AdapterSettings(**case['adapter']))
    pilot.reset(deepcopy(case['briefing']))
    pilot.search = module.GuidanceSearch(module.SearchSettings(**case['search']))
    metrics, outputs, propagation = [], [], [0]
    original_propagate, original_metrics = module.propagate, module.prediction_metrics
    last_track = [None, None]
    def propagate(*args, **kwargs):
        propagation[0] += 1
        return original_propagate(*args, **kwargs)
    def measure(track, target, times, report_distance, wind, already):
        # Hash exact coordinates, headings, commands and TAS per candidate. Reuse
        # this audit hash within a candidate's hypotheses, not in controller code.
        if track is not last_track[0]:
            last_track[:] = [track, digest(bits([asdict(v) for v in track]))]
        value = original_metrics(track, target, times, report_distance, wind, already)
        metrics.append({'track': last_track[1], 'target': digest(bits(target)),
                        'times': times, 'report_distance': bits(report_distance),
                        'wind': bits(wind), 'already': already, 'metrics': bits(value),
                        'turn_magnitude': bits(abs(module.heading_error(track[1].commanded_heading_deg,
                                                                       track[0].heading_deg)))})
        return value
    with monkeypatch.context() as m:
        m.setattr(module, 'propagate', propagate)
        m.setattr(module, 'prediction_metrics', measure)
        for b in case['bundles']:
            before = digest(b)
            start = len(metrics)
            try:
                action = pilot.act(deepcopy(b))
                trace = pilot.trace[-1]
                candidate_ranks = []
                plan = trace.get('plan')
                if plan:
                    n = plan['forecast_hypotheses']
                    # Every candidate's complete lexicographic rank, in original
                    # order, from its per-hypothesis metrics and first turn.
                    for i in range(plan['candidates']):
                        group = metrics[start+i*n:start+(i+1)*n]
                        distances = [float.fromhex(x['metrics'][0]['binary_float']) for x in group]
                        nearest = [float.fromhex(x['metrics'][1]['binary_float']) for x in group]
                        reports = [x['metrics'][2] for x in group]
                        rank = (not all(reports), not all(x <= .25 for x in distances),
                                max(distances), max(nearest), group[0]['turn_magnitude'], i)
                        candidate_ranks.append(bits(rank))
                outputs.append({'action': bits(action), 'trace': bits(trace), 'ranks': candidate_ranks})
            except Exception as exc:
                outputs.append({'error': [type(exc).__name__, str(exc)], 'trace': bits(pilot.trace)})
            assert digest(b) == before
    return {'outputs': outputs, 'metrics': metrics, 'propagations': propagation[0],
            'searches': pilot.search.searches}


@pytest.mark.parametrize('case', INPUTS['cases'], ids=lambda x: x['id'])
def test_exact_candidate_and_public_behavior_equivalence(case, monkeypatch):
    old = run_case(REFERENCE, case, monkeypatch)
    new = run_case(current, case, monkeypatch)
    expected_errors = [1, 2, 3] if case['id'] == 'errors-and-state-preservation' else []
    assert [i for i, x in enumerate(old['outputs']) if 'error' in x] == expected_errors
    assert [i for i, x in enumerate(new['outputs']) if 'error' in x] == expected_errors
    assert new['outputs'] == old['outputs']
    assert new['metrics'] == old['metrics']
    assert new['searches'] == old['searches']
    assert new['propagations'] <= old['propagations']
    if old['propagations']:
        assert new['propagations'] < old['propagations']

    EQUIVALENCE_STATS.append({'case': case['id'], 'outputs_sha256': digest(new['outputs']),
                              'candidate_metrics_sha256': digest(new['metrics']),
                              'old_propagations': old['propagations'], 'new_propagations': new['propagations'],
                              'searches': new['searches'], 'expected_error_indices': expected_errors})
