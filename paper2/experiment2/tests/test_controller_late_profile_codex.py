"""Invented short public histories/stub searches and clocks; no timing campaign."""
from copy import deepcopy
from itertools import count
import pytest

import paper2.experiment2.adaptive_controller.late_profile as L; import paper2.experiment2.adaptive_controller.history_inputs as I
import paper2.experiment2.adaptive_controller.history_profile as H; import paper2.experiment2.adaptive_controller.qualification as Q; import paper2.experiment2.adaptive_controller.stateful_profile as S
from paper2.experiment2.adaptive_controller.adapter import AdaptiveController
from .test_controller_adapter_codex import fixed_plan


def short_input():
    definition = {'id': 'invented-short', 'final_second': 12, 'targets': 2,
                  'cadence_s': 1, 'final_reports': 'fresh'}
    payload = I.materialize(definition)
    spec = definition | {'prefix_bundle_count': 12, 'retained_per_target_after_final': 13,
                         'input_sha256': Q.digest(payload)}
    return spec, payload


def factory(*args):
    pilot = AdaptiveController(*args)
    original_reset = pilot.reset
    def reset(briefing):
        original_reset(briefing)
        pilot.search.choose = lambda *a, **k: fixed_plan()
    pilot.reset = reset
    return pilot


def one(factory_=factory, cpu=None):
    spec, data = short_input()
    saved = deepcopy(data)
    result = L._one(spec, data, Q.policy_grid()[0], 0, factory_,
                    cpu or count(0, 10).__next__, count(0, 20).__next__)
    assert data == saved
    return result


def test_preload_retains_estimator_history_without_prior_policy_decisions():
    row = one()
    assert row['error'] is None
    assert row['before_flags'] == L.COLD_FLAGS
    assert row['before']['t'] == 11 and row['after']['t'] == 12
    assert all(v['retained'] == 12 for v in row['before']['targets'].values())
    assert all(v['retained'] == 13 for v in row['after']['targets'].values())
    assert len(row['trace']) == 1 and row['trace'][0]['t'] == 12
    assert row['trace'][0]['actions'] == row['final_action']
    assert row['setup_cpu_ns'] == row['cpu_ns'] == 10
    assert row['setup_wall_ns'] == row['wall_ns'] == 20


def test_loader_fresh_reset_clears_old_policy_flags_and_does_not_call_act():
    pilot = AdaptiveController()
    pilot._help_attempted = True
    pilot._reported_versions = {99}
    pilot._last_t = 99
    pilot.act = lambda b: pytest.fail('preload must not execute policy')
    _, payload = short_input()
    L.load_cold(pilot, payload)
    assert L.flags(pilot) == L.COLD_FLAGS


def test_setup_final_audit_and_clock_failures_are_retained():
    def broken(*args):
        raise RuntimeError('invented constructor failure')
    row = one(broken)
    assert row['error_stage'] == 'setup' and row['cpu_ns'] is None
    def bad_act(*args):
        obj = factory(*args)
        obj.act = lambda b: (_ for _ in ()).throw(ValueError('invented final failure'))
        return obj
    assert one(bad_act)['error_stage'] == 'final'
    row = one(cpu=lambda: None)
    assert row['cpu_ns'] is None and row['setup_cpu_ns'] is None
    assert one(cpu=count(100, -1).__next__)['cpu_ns'] is None
    def corrupt_trace(*args):
        obj = factory(*args)
        act = obj.act
        def changed(b):
            result = act(b)
            obj._trace[0]['t'] += 1
            return result
        obj.act = changed
        return obj
    assert one(corrupt_trace)['error_stage'] == 'after-audit'


def test_exact_schedule_uses_all_frozen_bytes_once_before_measurement(monkeypatch):
    # Actual accepted package reading, no estimator or policy executed.
    from pathlib import Path
    root = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/controller'
    seen = []
    def fake(spec, payload, cell, rep, *args):
        assert Q.digest(payload) == spec['input_sha256']
        seen.append((rep, cell['id'], spec['id']))
        return {'family': 'late-cold'}
    monkeypatch.setattr(L, '_one', fake)
    result = L.profile(root / 'long-history-inputs.json', root / 'history-payloads-turn151')
    assert len(result['rows']) == len(seen) == len(set(seen)) == 2304
    assert seen == [(r, c['id'], s['id']) for r in range(3) for c in Q.policy_grid()
                    for s in H.read_manifest(root / 'long-history-inputs.json')['sequences']]
    assert result['flight_authorized'] is False


def projection_rows():
    from pathlib import Path
    root = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/controller'
    short = S.read_inputs(root / 'stateful-timing-inputs.json')
    history = H.read_manifest(root / 'long-history-inputs.json')
    controls = []
    for fam, workloads in [('first', Q.public_workloads()), ('stateful', short['sequences'])]:
        for rep in range(3):
            for cell in Q.policy_grid():
                for w in workloads:
                    controls.append({'family': fam, 'cell': cell['id'], 'workload': w['id'],
                                     'repetition': rep, 'input_sha256': Q.digest(w) if fam == 'first' else w['input_sha256'],
                                     'error': None, 'cpu_ns': 100, 'wall_ns': 200,
                                     'setup_cpu_ns': 50, 'setup_wall_ns': 60})
    late = [{'family': 'late-cold', 'cell': c['id'], 'workload': s['id'], 'repetition': r,
             'input_sha256': s['input_sha256'], 'scope': L.SCOPE, 'error': None,
             'before_flags': deepcopy(L.COLD_FLAGS), 'cpu_ns': 10, 'wall_ns': 20,
             'setup_cpu_ns': 1, 'setup_wall_ns': 2}
            for r in range(3) for c in Q.policy_grid() for s in history['sequences']]
    return controls, late, short, history


def test_combined_projection_cannot_lower_same_campaign_maxima():
    controls, late, short, history = projection_rows()
    result = L.projection(controls, late, short, history)
    assert result['maxima_cpu_ns'] == {c['id']: 100 for c in Q.policy_grid()}
    late[0]['cpu_ns'] = 1000
    assert L.projection(controls, late, short, history)['maxima_cpu_ns'][late[0]['cell']] == 1000
    assert 'never add' in result['non_additive']


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'hash', 'unknown', 'error', 'warm', 'short-unknown'])
def test_incomplete_or_unavailable_evidence_never_gives_numeric_fit(change):
    controls, late, short, history = projection_rows()
    if change == 'missing': late.pop()
    if change == 'duplicate': late[-1] = deepcopy(late[0])
    if change == 'hash': late[0]['input_sha256'] = '0' * 64
    if change == 'unknown': late[0]['setup_wall_ns'] = None
    if change == 'error': late[0]['error'] = 'invented failure'
    if change == 'warm': late[0]['before_flags']['help_attempted'] = True
    if change == 'short-unknown': controls[0]['cpu_ns'] = None
    if change in ('missing', 'duplicate', 'hash'):
        with pytest.raises(ValueError): L.projection(controls, late, short, history)
    else:
        assert L.projection(controls, late, short, history)['projected_cpu_h'] is None
