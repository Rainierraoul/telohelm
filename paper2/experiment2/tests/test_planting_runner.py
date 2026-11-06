"""Selection/final separation and exact case-count controls using authored toy checks."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper2.experiment2.planting_runner as R; import paper2.experiment2.planting_pools as P
from .test_planting_pools import case; from .test_planting_pools import pool_file
import telohelm.runtime.sandbox as S

BASE = "def check(ctx, out):\n    return 'accept' if out.heading_deg <= 1.0 else 'reject'\n"
PLACEMENT = {'stream': 4401, 'phase': 1, 'episode_index': 2, 'decision': 'heading', 'ordinal': 1, 'counter': 1}


def toy(source, entry, args, **kw):
    # Authored bounded toy source only, no model or world. Native coverage is separate below.
    ns = {}
    exec(compile(source, '<authored-fixture>', 'exec'), ns)
    try:
        value = ns[entry](*(SimpleNamespace(**x) for x in args))
        return SimpleNamespace(status='ok', result=value)
    except Exception:
        return SimpleNamespace(status='error', result=None)


def replays():
    return ([case('stored_right', 'heading', 'right', i, 0) for i in range(100)] +
            [case('stored_wrong', 'heading', 'wrong', 200 + i, 5) for i in range(40)] +
            [case('fresh_wrong', 'heading', 'wrong', 300 + i, 5) for i in range(100)])


def selected(tmp_path, run=toy, counter=1, eligibility_values=None):
    eligibility = pool_file(tmp_path, 'eligibility', eligibility_values)
    out = tmp_path / 'selection'
    record = R.select(out, {**PLACEMENT, 'counter': counter}, BASE, R.source_sha(BASE), eligibility, replays(),
                      rewrite=1, run=run)
    return out, record


def test_first_eligible_mutation_is_chosen_without_final_access_and_replay_does_not_reselect(tmp_path):
    out, r = selected(tmp_path)
    assert r['planted'] and r['selected_operation'] == 1 and r['caught']
    assert r['mutant_replay']['wrong_accepted'] == 140
    assert len(r['attempts']) == 1      # a caught first choice cannot select another using replay information
    assert not list(tmp_path.glob('final*'))
    assert P.unseal(out / 'selection.json') == r
    with pytest.raises(FileExistsError):
        R.select(out, PLACEMENT, BASE, R.source_sha(BASE), tmp_path / 'eligibility.json', replays(), rewrite=1, run=toy)


def test_counter_wrap_and_all_inapplicable_or_nonexpanding_reasons_survive(tmp_path):
    _, r = selected(tmp_path, counter=3)
    assert [a['operation'] for a in r['attempts']] == [3, 4, 5, 6, 7]
    assert [a['status'] for a in r['attempts']] == ['no_expansion', 'inapplicable', 'inapplicable', 'no_expansion', 'eligible']
    assert r['selected_operation'] == 7


def test_errors_and_invalid_verdicts_do_not_count_as_acceptance_or_eligible_expansion(tmp_path):
    def bad_first(source, *args, **kwargs):
        if '10.0' in source:
            return SimpleNamespace(status='ok', result='anything')
        return toy(source, *args, **kwargs)
    _, r = selected(tmp_path, run=bad_first)
    assert r['attempts'][0]['status'] == 'execution_error' and r['selected_operation'] == 7
    assert {v['status'] for v in r['attempts'][0]['eligibility_verdicts']} == {'invalid_verdict'}


def test_unqualified_baseline_and_no_expansion_skip_final_without_even_opening_it(tmp_path):
    def wrong_baseline(source, entry, args, **kw):
        return SimpleNamespace(status='ok', result='accept')
    out, r = selected(tmp_path, run=wrong_baseline)
    assert r['status'] == 'baseline_unqualified' and not r['planted']
    assert R.final(out / 'selection.json', Path('/does-not-exist'), tmp_path / 'final') == {
        'status': 'not_evaluated', 'reason': 'no completed planted mutation'}
    other = tmp_path / 'other'
    other.mkdir()
    out, r = selected(other, eligibility_values=(0, 0))
    assert r['status'] == 'no_eligible_mutation' and len(r['attempts']) == 10
    assert R.final(out / 'selection.json', Path('/does-not-exist'), other / 'final')['status'] == 'not_evaluated'


def test_final_counts_baseline_errors_and_mutation_added_errors_separately(tmp_path):
    out, _ = selected(tmp_path)
    final = pool_file(tmp_path, 'final', values=(0, 5))
    result = R.final(out / 'selection.json', final, tmp_path / 'measurement', run=toy)
    assert result['status'] == 'complete' and result['cases'] == result['paired_valid_cases'] == 2
    assert result['baseline_wrong_accepted'] == result['monitored_wrong_accepted'] == 1
    assert result['mutant_wrong_accepted'] == result['unmonitored_wrong_accepted'] == 2
    assert result['added_wrong_accepted'] == 1 and result['removed_wrong_accepted'] == 0
    assert 'static' in result['interpretation']


def test_execution_failure_in_final_is_not_reported_as_a_correct_rejection(tmp_path):
    out, _ = selected(tmp_path)
    f = pool_file(tmp_path, 'final')
    def failing(*args, **kw):
        return SimpleNamespace(status='timeout', result=None)
    result = R.final(out / 'selection.json', f, tmp_path / 'measurement', run=failing)
    assert result['status'] == 'execution_incomplete' and result['paired_valid_cases'] == 0
    assert all(v['status'] == 'timeout' for v in result['mutant_verdicts'])


def test_pool_overlap_stale_fresh_ordinal_and_wrong_frozen_source_are_refused(tmp_path):
    e = pool_file(tmp_path, 'eligibility')
    with pytest.raises(ValueError, match='frozen'):
        R.select(tmp_path / 'bad', PLACEMENT, BASE, '0' * 64, e, replays(), rewrite=1, run=toy)
    with pytest.raises(ValueError, match='ordinal'):
        R.select(tmp_path / 'bad', PLACEMENT, BASE, R.source_sha(BASE), e, replays(), rewrite=2, run=toy)
    rows = replays()
    pool = P.unseal(e)
    pool['cases'][0]['context'] = rows[0]['context']
    overlap = tmp_path / 'overlap'
    P.seal(overlap, pool)
    with pytest.raises(ValueError, match='overlap'):
        R.select(tmp_path / 'bad', PLACEMENT, BASE, R.source_sha(BASE), overlap, rows, rewrite=1, run=toy)
    assert not (tmp_path / 'bad').exists()


def test_final_overlap_or_relabelled_role_is_refused_before_execution(tmp_path):
    out, _ = selected(tmp_path)
    f = pool_file(tmp_path, 'final')
    record = P.unseal(f)
    selection = P.unseal(out / 'selection.json')
    record['cases'][0]['context'] = selection['eligibility_cases'][0]['context']
    record['cases'][0]['output'] = selection['eligibility_cases'][0]['output']
    overlap = tmp_path / 'overlap'
    P.seal(overlap, record)
    with pytest.raises(ValueError, match='overlap'):
        R.final(out / 'selection.json', overlap, tmp_path / 'measurement', run=toy)
    assert not (tmp_path / 'measurement').exists()


def test_crash_leaves_bound_inputs_and_progress_and_cannot_be_retried(tmp_path):
    n = 0
    def cut(*args, **kwargs):
        nonlocal n
        n += 1
        if n == 242:     # replay 240 calls, eligibility call 2
            raise RuntimeError('injected crash')
        return toy(*args, **kwargs)
    with pytest.raises(RuntimeError, match='crash'):
        selected(tmp_path, run=cut)
    root = tmp_path / 'selection'
    assert (root / 'inputs.json').exists() and (root / 'progress.jsonl').exists()
    assert P.unseal(root / 'failure.json')['status'] == 'failed'
    assert not (root / 'selection.json').exists()
    with pytest.raises(FileExistsError):
        R.select(root, PLACEMENT, BASE, R.source_sha(BASE), tmp_path / 'eligibility.json', replays(), rewrite=1, run=toy)


def test_placement_schedule_is_shared_deterministic_and_balanced_by_decision():
    rows = R.placement_plan(4401, 6001)
    assert rows == R.placement_plan(4401, 6001) and len(rows) == 10
    assert [r['counter'] for r in rows] == list(range(1, 11))
    for phase in range(1, 6):
        p = [r for r in rows if r['phase'] == phase]
        assert len({r['episode_index'] for r in p}) == 2 and {r['decision'] for r in p} == {'heading', 'report'}
    with pytest.raises(ValueError):
        R.placement_plan(4301, 6001)


@pytest.mark.skipif(not S.available(), reason='native containment unavailable')
def test_native_sandbox_verdicts_agree_with_hand_calculated_toy():
    rows = [case('eligibility', 'heading', 'right', 1, 0), case('eligibility', 'heading', 'wrong', 2, 5)]
    values = R.evaluate(BASE, rows)
    assert [v['status'] for v in values] == ['ok', 'ok']
    assert [v['accepted'] for v in values] == [True, False]


def test_cli_preserves_full_case_provenance_from_a_real_store(tmp_path, monkeypatch):
    import json
    import sys
    import telohelm.qualification.store as H
    eligibility = pool_file(tmp_path, 'eligibility')
    store = H.CaseStore(tmp_path / 'store', verifier=lambda c: True, verifier_name='invented fixture', development=True)
    for c in replays() + [case('eligibility', 'heading', 'wrong', 9000, 5),
                          case('final', 'heading', 'wrong', 9001, 5)]:
        store.admit(c)
    placement = tmp_path / 'placement.json'
    placement.write_text(json.dumps(PLACEMENT))
    source = tmp_path / 'check.py'
    source.write_text(BASE)
    calls = []
    def select(out, p, c0, expected_sha, e, cases, *, rewrite):
        # The old CLI feeds only payloads, so this real intake fails before the repair.
        wrong, right = R._replay_cases(cases, 4401, 'heading', rewrite)
        assert len(wrong) == 140 and len(right) == 100
        assert all(c['split'] not in ('eligibility', 'final') for c in cases)
        assert expected_sha == R.FROZEN_C0['heading'] and e == eligibility and c0 == BASE
        calls.append(cases)
        return {'status': 'rehearsed', 'planted': False}
    monkeypatch.setattr(R, 'select', select)
    monkeypatch.setattr(sys, 'argv', ['planting_runner', 'select', '--placement', str(placement), '--eligibility',
                                    str(eligibility), '--replay', str(tmp_path / 'store'), '--check', str(source),
                                    '--out', str(tmp_path / 'run'), '--rewrite', '1'])
    R.main()
    assert len(calls) == 1


def test_input_provenance_cannot_change_during_selection(tmp_path):
    eligibility = pool_file(tmp_path, 'eligibility')
    rows, placement, n = replays(), dict(PLACEMENT), 0
    def concurrent_edit(source, *args, **kwargs):
        nonlocal n
        n += 1
        if n == 1:
            rows[0]['output']['heading_deg'] = 999
            placement['counter'] = 10
        return toy(source, *args, **kwargs)
    r = R.select(tmp_path / 'run', placement, BASE, R.source_sha(BASE), eligibility, rows, rewrite=1, run=concurrent_edit)
    assert r['placement']['counter'] == 1 and r['replay_cases'][0]['output']['heading_deg'] == 0


def test_a_truncated_wrong_set_cannot_masquerade_as_a_complete_replay(tmp_path):
    e = pool_file(tmp_path, 'eligibility')
    rows = [c for c in replays() if c['split'] != 'stored_wrong' or c['origin']['ordinal'] == 200]
    with pytest.raises(ValueError, match='40 stored wrong'):
        R.select(tmp_path / 'bad', PLACEMENT, BASE, R.source_sha(BASE), e, rows, rewrite=1, run=toy)
    assert not (tmp_path / 'bad').exists()
