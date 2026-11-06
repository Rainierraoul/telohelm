"""Hand-authored original-state traces only; no reference or stream generation."""
from dataclasses import asdict, replace, FrozenInstanceError
import hashlib
import json
import pytest
import paper2.experiment2.reference_suffix_codex as S; import paper2.experiment2.original_binding as B; import paper2.experiment2.original_replay as R
from .test_original_binding_codex import fixture
from .test_original_replay_codex import TEXT
import telohelm.taskpacks.tanker.world.episode as E


def inputs(prior=False, change=False):
    capture, issue, kw = fixture()
    if prior or change:
        saved = json.loads(capture.body)
        ep = R._restore(saved)
        if prior:
            ep._report(TEXT, 1, False)  # invented pre-existing report subphase
        if change:
            goal = E.Goal(2, 'SHELL 1', 2., 18)
            ep.s = replace(ep.s, goals=ep.s.goals + ((14, goal),))
            ep.pending_goals.append((14, goal))
        capture = R.capture(ep, R.Identity(**saved['identity']), context=saved['context'],
                            input_bundle=saved['input_bundle'], issue=saved['issue'],
                            actor_pending=saved['actor_pending'])
        kw['expected_capture_sha256'] = capture.sha256
    binding = B.bind(capture, issue, **kw)
    rows = [{'t': t, 'actions': {'goal_version': 1, 'heading_deg': 90, 'report': TEXT}
             if t == 10 and not prior else {}, 'accepted': True} for t in range(10, 18)]
    source = {'kind': 'fixed invented unit controls', 'no_reference_pilot_executed': True}
    return binding, rows, source


def make(prior=False):
    b, rows, source = inputs(prior)
    ref = S.retain_reference(b, rows, 18, source, expected_binding_sha256=b.sha256)
    return b, ref


def build(b, ref, candidate=None, **kw):
    return S.build(b, ref, {'goal_version': 1, 'heading_deg': 90, 'report': TEXT}
                   if candidate is None else candidate, expected_binding_sha256=b.sha256,
                   expected_reference_sha256=ref.sha256, **kw)


def replay(b, ref, p):
    return S.replay(b, ref, p, expected_binding_sha256=b.sha256,
                    expected_reference_sha256=ref.sha256, expected_plan_sha256=p.sha256)


def test_candidate_occupies_one_original_interval_and_suffix_is_unchanged():
    b, ref = make(); p = build(b, ref)
    raw = json.loads(p.body)
    assert raw['candidate_second'] == 10 and raw['continuation_seconds'] == list(range(11, 18))
    assert p.continuation == [r['actions'] for r in json.loads(ref.body)['rows'][1:]]
    out = replay(b, ref, p)
    assert out['upper_elapsed_s'] == 8 and out['admissible'] is False
    assert [r['second'] for r in out['evidence']['steps']] == list(range(10, 18))
    assert sum(r['prefix'] for r in out['evidence']['steps']) == 1
    assert out['evidence']['outcome']['reports_under_goal'] == 1
    assert raw['binding_plus_plan_bytes'] == len(b.body) + len(R.encode([p.candidate, p.continuation]))


def test_missing_report_cannot_be_repaired_by_suffix_or_reclassified_impossible():
    b, ref = make(); out = replay(b, ref, build(b, ref, {}))
    assert out['upper_elapsed_s'] is None and out['disposition'] == 'unknown'
    assert out['evidence']['outcome']['reports_under_goal'] == 0


def test_retained_original_report_is_preserved_without_repeating_it():
    b, ref = make(prior=True); out = replay(b, ref, build(b, ref, {}))
    assert out['upper_elapsed_s'] == 8
    assert out['evidence']['terminal']['record']['reports'] == json.loads(b.body)['original']['state']['record']['reports']
    assert out['evidence']['outcome']['reports_under_goal'] == 1


def test_bytes_and_returned_actions_are_detached_from_caller_mutation():
    b, rows, source = inputs()
    ref = S.retain_reference(b, rows, 18, source, expected_binding_sha256=b.sha256)
    rows[-1]['actions']['help'] = {}; source['changed'] = True
    assert 'changed' not in json.loads(ref.body)['reference_source']
    p = build(b, ref); action = p.candidate; action['heading_deg'] = 0
    continuation = p.continuation; continuation[0]['heading_deg'] = 0
    assert p.candidate['heading_deg'] == 90 and p.continuation[0] == {}
    with pytest.raises(FrozenInstanceError): p.body = b'changed'


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'boolean-time', 'help', 'stale-goal', 'refused', 'extra'])
def test_malformed_reference_or_changed_goal_is_refused(kind):
    b, rows, source = inputs()
    if kind == 'missing': rows.pop()
    if kind == 'duplicate': rows[2]['t'] = 11
    if kind == 'boolean-time': rows[0]['t'] = True
    if kind == 'help': rows[2]['actions'] = {'goal_version': 1, 'help': {'reason': 'missing_or_stale_evidence', 'text': 'x'}}
    if kind == 'stale-goal': rows[2]['actions'] = {'goal_version': 2, 'heading_deg': 90}
    if kind == 'refused': rows[2]['accepted'] = False
    if kind == 'extra': rows.append({'t': 18, 'actions': {}, 'accepted': True})
    with pytest.raises(ValueError): S.retain_reference(b, rows, 18, source, expected_binding_sha256=b.sha256)


@pytest.mark.parametrize('end', [True, 10, 19, 3601])
def test_reference_ending_is_absolute_and_within_original_limits(end):
    b, rows, source = inputs()
    with pytest.raises(ValueError): S.retain_reference(b, rows, end, source, expected_binding_sha256=b.sha256)


def test_goal_change_in_interval_is_not_silently_truncated_or_rewritten():
    b, rows, source = inputs(change=True)
    with pytest.raises(ValueError, match='goal changes'):
        S.retain_reference(b, rows, 18, source, expected_binding_sha256=b.sha256)


def test_independent_binding_and_reference_anchors_cannot_be_replaced():
    b, ref = make()
    with pytest.raises(ValueError): S.build(b, ref, {}, expected_binding_sha256='0' * 64, expected_reference_sha256=ref.sha256)
    raw = json.loads(ref.body); raw['public_history_sha256'] = '0' * 64
    body = R.encode(raw); other = S.Reference(body, hashlib.sha256(body).hexdigest())
    with pytest.raises(ValueError, match='identity conflict'): build(b, other)
    with pytest.raises(ValueError): S.build(b, other, {}, expected_binding_sha256=b.sha256, expected_reference_sha256=ref.sha256)
    with pytest.raises(ValueError): build(replace(b, admissible=True), ref)


def test_actual_payload_and_steps_keep_original_replay_caps():
    b, ref = make()
    with pytest.raises(ValueError, match='byte limit'): build(b, ref, limits=R.Limits(input_bytes=1024))
    with pytest.raises(ValueError, match='step cap'): build(b, ref, limits=R.Limits(max_steps=7))
    with pytest.raises(ValueError): build(b, ref, {'goal_version': 2, 'heading_deg': 90})
    with pytest.raises(ValueError): build(b, ref, {'goal_version': 1, 'help': {}})


def test_changed_suffix_cannot_be_repaired_or_used_with_recomputed_plan_hash():
    b, ref = make(); p = build(b, ref)
    raw = json.loads(p.body); raw['continuation'][0] = {'goal_version': 1, 'report': TEXT}
    body = R.encode(raw); changed = S.Plan(body, hashlib.sha256(body).hexdigest())
    with pytest.raises(ValueError, match='immutable original suffix'): replay(b, ref, changed)
