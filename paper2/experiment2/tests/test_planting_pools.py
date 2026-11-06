"""Invented pool controls: no held-back source is acquired by these tests."""
from copy import deepcopy
import json

import pytest

import paper2.experiment2.outcomecases as O; import paper2.experiment2.planting_pools as P


def case(role, decision, label, ordinal, value=0, stream=4401):
    source = O.partition()[stream][role if role in ('eligibility', 'final') else 'replay'][0]
    if role == 'fresh_wrong':
        source = O.partition()[stream]['replay'][1]
    row = {'decision': decision, 'disposition': label,
           'moment': {'source': source, 'phase': 1, 'index': 0, 'episode': f'invented-{source}-{ordinal}',
                      't': ordinal, 'ctx': {'goal_version': 1, 'x': ordinal},
                      'goal': {'version': 1, 'tanker': 'SHELL 1', 'report_at_nm': 10., 'deadline': None}},
           'output': {'heading_deg': value} if decision == 'heading' else {'report_trigger_nm': value + 1},
           'evidence': {'construction': 'invented', 'source': source, 'phase': 1, 'index': 0, 'second': ordinal}}
    return O._case(stream, role, row, ordinal, rewrite=1 if role == 'fresh_wrong' else None)


def pool_file(tmp_path, role, values=None):
    spec = P.specification(4401, role, eligibility_per_decision=2, final_total=4)
    rows = []
    for i, decision in enumerate(('heading', 'report')):
        labels = ('right', 'wrong') if role == 'eligibility' else ('wrong', 'wrong')
        for j, label in enumerate(labels):
            value = (0, 5)[j] if values is None else values[j]
            rows.append(case(role, decision, label, 1000 + 100 * i + j + (10000 if role == 'final' else 0), value))
    path = tmp_path / (role + '.json')
    P.seal(path, {'status': 'complete', 'specification': spec, 'cases': rows})
    return path


def test_partition_roles_reserved_sources_and_sizes():
    e, f = P.specification(4401, 'eligibility'), P.specification(4401, 'final')
    assert e['sources'] == [4410] and f['sources'] == [4411]
    assert e['quotas']['heading'] == {'right': 50, 'wrong': 50}
    assert sum(q['wrong'] for q in f['quotas'].values()) == 200
    for stream in (4301, 4433, 4400, 4499, True):
        with pytest.raises(ValueError):
            P.specification(stream, 'final')
    for size in (0, 1, 2.0, True):
        with pytest.raises(ValueError):
            P.specification(4401, 'final', final_total=size)


def test_sealed_pool_refuses_tampering_wrong_source_role_and_missing_cases(tmp_path):
    path = pool_file(tmp_path, 'eligibility')
    record = P.load(path, 4401, 'eligibility')
    assert len(record['cases']) == 4
    value = json.loads(path.read_text())
    value['record']['cases'][0]['output']['heading_deg'] = 42
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='digest'):
        P.load(path, 4401, 'eligibility')
    for mutate in (lambda r: r['cases'][0]['origin'].update(seed=4433),
                   lambda r: r['cases'][0].update(split='final'), lambda r: r['cases'].pop()):
        changed = deepcopy(record)
        mutate(changed)
        bad = tmp_path / ('bad-' + str(len(list(tmp_path.iterdir()))))
        P.seal(bad, changed)
        with pytest.raises(ValueError):
            P.load(bad, 4401, 'eligibility')


def test_role_disjointness_is_about_context_output_not_source_metadata():
    e, f = case('eligibility', 'heading', 'wrong', 1, 5), case('final', 'heading', 'wrong', 1, 5)
    with pytest.raises(ValueError, match='overlap'):
        P.disjoint([e], [f])
    f['context']['x'] = 2
    P.disjoint([e], [f])


def fake_rows(source):
    rows = []
    for i, decision in enumerate(('heading', 'report')):
        for j, label in enumerate(('right', 'wrong', 'wrong')):
            c = case('eligibility', decision, label, 100 * i + j, 0 if label == 'right' else j + 4)
            c['origin']['seed'] = source
            m = {'source': source, 'phase': 1, 'index': 0, 'episode': c['origin']['episode'],
                 't': c['origin']['second'], 'ctx': c['context'], 'goal': c['goal']}
            rows.append({'moment': m, 'decision': decision, 'output': c['output'], 'disposition': label,
                         'evidence': {**c['label']['evidence'], 'source': source}})
    return rows


def test_builder_fixes_sources_before_drawing_verifies_selection_and_never_refills(tmp_path, monkeypatch):
    calls = []
    def moments(source, per_phase, phases):
        assert (tmp_path / 'ok' / 'plan.json').exists()
        calls.append(source)
        return fake_rows(source)
    monkeypatch.setattr(O, 'moments', moments)
    monkeypatch.setattr(O, 'label_all', lambda rows, workers: O._keep(rows))
    verified = []
    monkeypatch.setattr(O, 'verify', lambda c: verified.append(c) is None)
    record = P.build(4401, 'eligibility', tmp_path / 'ok', per_phase=1, phases=(1,), eligibility_per_decision=2)
    assert calls == [4410] and len(verified) == len(record['cases']) == 4
    assert P.load(tmp_path / 'ok' / 'pool.json', 4401, 'eligibility') == record
    with pytest.raises(FileExistsError):
        P.build(4401, 'eligibility', tmp_path / 'ok')
    assert calls == [4410]


def test_insufficient_yield_and_bad_verification_leave_failure_records(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(O, 'moments', lambda source, *_: calls.append(source) or fake_rows(source))
    monkeypatch.setattr(O, 'label_all', lambda rows, workers: O._keep(rows))
    monkeypatch.setattr(O, 'verify', lambda c: True)
    with pytest.raises(ValueError, match='insufficient'):
        P.build(4401, 'eligibility', tmp_path / 'short')
    assert calls == [4410]
    assert P.unseal(tmp_path / 'short' / 'failure.json')['status'] == 'failed'
    monkeypatch.setattr(O, 'verify', lambda c: False)
    with pytest.raises(ValueError, match='reconstruction'):
        P.build(4401, 'final', tmp_path / 'bad', final_total=2)
    assert P.unseal(tmp_path / 'bad' / 'failure.json')['status'] == 'failed'
