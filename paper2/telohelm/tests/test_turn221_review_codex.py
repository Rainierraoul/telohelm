"""Independent retention controls; invented records and stub flights, no inference/world."""
from types import SimpleNamespace

import pytest

import telohelm.learning as G
from telohelm.runtime.library import Library
from .test_candidates import Proposer; from .test_candidates import reply
from .test_regression_trial import coordinator; from .test_regression_trial import history


@pytest.mark.parametrize('calls,helped,complete,solvable,eligible', [
    (0, False, True, True, True), (1, False, True, True, False),
    (None, False, True, True, False), (0, True, True, True, False),
    (0, False, False, True, False),
    # Since 27 October the reference label is recorded only; an unknown one no longer excludes an episode
    # (paper2/notes/PRE-FREEZE-DESIGN-CHANGES-CLAUDE.md, item 1).
    (0, False, True, None, True),
])
def test_own_history_requires_completed_autonomous_solvable_success(tmp_path, calls, helped, complete, solvable, eligible):
    co = G.Coordinator(Library(tmp_path / 'lib'), tmp_path, None, lambda _: solvable, None,
                       regression=8, attempt_every=100)
    outcome = {'goal_met': True, 'joined': True, 'report_correct': True, 'settled': True,
               'help_requests': 1 if helped else 0}
    if calls is not None:
        outcome['model_calls'] = calls
    co.after_episode(SimpleNamespace(episode_id='old'), outcome, complete)
    assert bool(co.regression_set([])) is eligible


def test_a_flown_summary_with_a_stale_evidence_request_is_neither_eligible_nor_retained(tmp_path):
    """Codex's TURN295 witness: a summary counts help at its top level and nests the accepted requests."""
    co = G.Coordinator(Library(tmp_path / 'lib'), tmp_path, None, lambda _: True, None, regression=8, attempt_every=100)
    stale = {'goal_met': True, 'joined': True, 'report_correct': True, 'settled': True, 'model_calls': 0,
             'help_requests': 1, 'outcome': {'help_requests': [{'t': 3, 'reason': 'missing_or_stale_evidence'}]}}
    method = {**stale, 'outcome': {'help_requests': [{'t': 3, 'reason': 'missing_tool'}]}}
    co.after_episode(SimpleNamespace(episode_id='stale'), stale, True)
    co.after_episode(SimpleNamespace(episode_id='method'), method, True)
    assert [c.eligible for c in co.completed] == [False, True]
    assert co.regression_set([]) == []          # both asked for help, so neither met the goal by itself


def test_retention_selection_has_no_duplicate_or_recent_case_and_keeps_boundaries(tmp_path):
    co = coordinator(tmp_path, 8, history(60))
    for length in range(1, 50):
        co.completed = co.completed[:length] if length == 1 else [
            G.Completed(i, f'e{i:02d}', SimpleNamespace(), True, None, True, alone=True)
            for i in range(1, length + 1)]
        recent = co.completed[-8:]
        for bound in range(1, 9):
            co.regression = bound
            picked = co.regression_set(recent)
            ids = [x.episode for x in picked]
            assert len(ids) <= bound and len(ids) == len(set(ids))
            assert not set(ids) & {x.episode for x in recent}
            if picked:
                assert picked[-1] == co.completed[-9]
            if len(picked) > 1:
                assert picked[0] == co.completed[0]


def test_failed_retention_keeps_sampling_same_history_and_adopts_next_candidate(tmp_path):
    co = coordinator(tmp_path, 4, history(20))
    co.candidates = 3
    co.propose = Proposer([reply(10), reply(20)])
    flown = []

    def fly(shadow, completed, root):
        candidate = shadow.adopted('heading', 'procedure')[0]
        tag = 10 if '10}' in shadow.entry(candidate)['source'] else 20
        flown.append((tag, completed.episode))
        return {'episode': completed.episode, 'goal_met': not (tag == 10 and completed.episode == 'e01'),
                'help': 0, 'component': {'passed': True}}

    co.fly = fly
    event = co.attempt(co.completed[-1])
    assert event['passed'] and event['candidate_call'] == 2 and len(co.propose.sent) == 2
    assert [e for tag, e in flown if tag == 10] == [e for tag, e in flown if tag == 20]
    assert len(co.library.adopted('heading', 'procedure')) == 1
    assert '20}' in co.library.entry(co.library.adopted('heading', 'procedure')[0])['source']
    start = next(e for e in co.events if e['kind'] == 'attempt_start')
    assert len(start['regression_set']) == 4


def test_help_cannot_rescue_whole_goal_retention(tmp_path):
    co = coordinator(tmp_path, 8, history(20))
    co.fly = lambda shadow, c, root: {'episode': c.episode, 'goal_met': True,
                                     'help': int(c.episode == 'e01'), 'component': {'passed': True}}
    result = co.attempt(co.completed[-1])
    assert result['passed'] is False and not co.library.adopted('heading', 'procedure')


def test_unsettled_nominal_success_is_not_demonstrated_competence(tmp_path):
    co = G.Coordinator(Library(tmp_path / 'lib'), tmp_path, None, lambda _: True, None,
                       regression=8, attempt_every=100)
    co.after_episode(SimpleNamespace(episode_id='uncertain'),
                     {'goal_met': True, 'joined': True, 'report_correct': True, 'model_calls': 0,
                      'help_requests': 0, 'settled': False}, True)
    assert not co.regression_set([])


@pytest.mark.parametrize('outcome', [
    {'goal_met': True, 'settled': False, 'end_reason': 'join'},
    {'goal_met': True, 'settled': True, 'end_reason': None},
])
def test_real_retention_result_requires_settled_natural_ending(tmp_path, outcome):
    co = coordinator(tmp_path, 8, history(20))

    def fly(shadow, c, root):
        saved = outcome if c.episode == 'e01' else {'goal_met': True, 'settled': True, 'end_reason': 'join'}
        return {'episode': c.episode, 'goal_met': True, 'help': 0, 'component': {'passed': True},
                'outcome': saved}

    co.fly = fly
    result = co.attempt(co.completed[-1])
    assert result['passed'] is False and not co.library.adopted('heading', 'procedure')
