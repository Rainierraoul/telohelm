"""TURN139 exact-source CPU controls: invented candidates, no flight/model/reserved seed."""
from dataclasses import asdict
from decimal import Decimal, localcontext
import hashlib
import json
import math

import pytest
import paper2.experiment2.streams as G; import telohelm.taskpacks.tanker.world.physics as P; import telohelm.taskpacks.tanker.world.episode as E
import telohelm.runtime.models as M
from .test_turn137_review_codex import gateway
from paper2.telohelm.tests.test_turn65_review_codex import MSG


class Choices:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []
    def integers(self, low, high):
        v = next(self.values)
        assert low <= v < high
        self.calls.append((low, high))
        return v


def receiver():
    return P.ReceiverState(0., 0., 0., 0., 0., .125, .01)


def tanker(name, cx, radius=1., delta=0.):
    return E.Tanker(name, P.Orbit(cx, 0., radius, .075, 0., delta))


def test_exact_integral_join_window_and_conservative_shrinking_radius():
    # tau = (11.5 - (1 + .25) - .25) / .125 = 80 exactly.
    # Last strictly safe integer = 79; latest change = 69, exclusive RNG upper 70.
    rng = Choices([69, 1, 4])
    one, two = tanker('SHELL 1', 11.5, delta=-.25), tanker('SHELL 2', 60., radius=2.)
    t, goal = G._goal_change(rng, receiver(), (0., 0.), one, two, E.Goal(7, 'SHELL 1', 10.))
    assert t == 69 and rng.calls == [(60, 70), (0, 2), (0, 5)]
    assert goal == E.Goal(8, 'SHELL 2', 30.) and goal.deadline is None
    assert one.orbit.r0 + abs(one.orbit.delta) > one.orbit.r0


def test_inclusive_distance_eligibility_boundary_excludes_original_distance():
    # At second 60, L = 31.5 - .125*60 - 6 = 18; D <= 15 exactly.
    rng = Choices([60, 0, 0])
    t, goal = G._goal_change(rng, receiver(), (0., 0.), tanker('SHELL 1', 31.5, 6.),
                             tanker('SHELL 2', 60.), E.Goal(1, 'SHELL 1', 10.))
    assert t == 60 and goal == E.Goal(2, 'SHELL 1', 15.)
    assert rng.calls == [(60, 192), (0, 2), (0, 1)]


def test_empty_window_and_eligible_set_keep_why_without_extra_draws():
    no_window = Choices([])
    with pytest.raises(G.Unusable) as w:
        G._goal_change(no_window, receiver(), (0., 0.), tanker('SHELL 1', 10.),
                       tanker('SHELL 2', 60.), E.Goal(1, 'SHELL 1', 10.))
    assert w.value.reason == 'no goal-change window'
    assert w.value.details == {'t_max': 59, 'tau': 70.}
    assert no_window.calls == []
    rng = Choices([69, 0])
    with pytest.raises(G.Unusable) as w:
        G._goal_change(rng, receiver(), (0., 0.), tanker('SHELL 1', 11.25),
                       tanker('SHELL 2', 60.), E.Goal(1, 'SHELL 1', 10.))
    assert w.value.reason == 'no eligible new goal'
    assert w.value.details['eligible_nm'] == [] and w.value.details['t_g'] == 69
    assert w.value.details['bound_nm'] == -1.375 and rng.calls == [(60, 70), (0, 2)]


def test_unusable_and_reference_rejections_retained_before_success(monkeypatch):
    s = G.draw(4401, 1, 0)
    calls = []
    def draw(stream, phase, index, attempt):
        calls.append(('draw', attempt))
        if attempt == 0:
            raise G.Unusable('no eligible new goal', {'eligible_nm': [], 't_g': 61})
        return s
    def oracle(scenario, ask_help):
        assert scenario is s and ask_help is False
        calls.append(('oracle', len(calls)))
        met = len(calls) >= 5
        return {'goal_met': met, 'settled': True, 'possible_goal_met': [met], 'flags': [], 'alternatives': []}
    monkeypatch.setattr(G, 'draw', draw)
    monkeypatch.setattr(G.phase5, 'fly', oracle)
    out = G.solvable_episode(4401, 4, 0)
    assert [a['attempt'] for a in out.attempts] == [0, 1, 2]
    assert [a['solvable'] for a in out.attempts] == [False, False, True]
    assert [a['reason'] for a in out.attempts] == ['no eligible new goal', 'not solvable by the reference pilot', None]
    assert out.attempts[0]['details'] == {'eligible_nm': [], 't_g': 61}
    assert [a['seed'] for a in out.attempts] == [G.episode_seed(4401, 4, 0, i) for i in range(3)]
    assert len([c for c in calls if c[0] == 'oracle']) == 2


def test_unusable_cap_is_finite_and_does_not_invoke_oracle(monkeypatch):
    seen = []
    def draw(stream, phase, index, attempt):
        seen.append(attempt)
        raise G.Unusable('no goal-change window', {'t_max': 59})
    monkeypatch.setattr(G, 'draw', draw)
    monkeypatch.setattr(G, 'ATTEMPT_CAP', 3)
    monkeypatch.setattr(G.phase5, 'fly', lambda *a, **k: pytest.fail('unusable candidate must not be flown'))
    with pytest.raises(G.GenerationError, match='3 attempts'):
        G.solvable_episode(4401, 4, 0)
    assert seen == [0, 1, 2]
    # This establishes finite termination only; the exception currently lacks the
    # attempt records. A collector must retain them before the scored freeze.


def test_phase4_candidate_bounds_with_independent_high_precision_envelope():
    rows = []
    for stream in (4401, 4402, 4403, 4404):
        for index in range(20):
            seed = int.from_bytes(hashlib.sha256(f'telohelm-stream|{stream}|4|{index}|0'.encode()).digest()[:8], 'big')
            assert seed == G.episode_seed(stream, 4, index, 0)
            try:
                s = G.draw(stream, 4, index)
            except G.Unusable as u:
                assert u.reason in ('no goal-change window', 'no eligible new goal')
                rows.append((stream, index, u.reason))
                continue
            (_, old), (t, new) = s.goals
            assert new.version == old.version + 1 and new.deadline is None
            assert s == G.draw(stream, 4, index)
            def D(v): return Decimal.from_float(float(v))
            with localcontext() as ctx:
                ctx.prec = 60
                speed = D(s.receiver.tas) + (D(s.wind[0])**2 + D(s.wind[1])**2).sqrt()
                def lower(k, second):
                    o = k.orbit
                    dist = ((D(o.cx)-D(s.receiver.x))**2 + (D(o.cy)-D(s.receiver.y))**2).sqrt()
                    return dist - speed*second - D(o.r0)-abs(D(o.delta))
                assert lower(s.tankers[0], t+10) > D(E.JOIN_TOLERANCE)
                named = next(k for k in s.tankers if k.callsign == new.tanker)
                assert D(new.report_at_nm) <= lower(named, t)-3
            if new.tanker == old.tanker:
                assert new.report_at_nm != old.report_at_nm
            rows.append((stream, index, 'candidate', t, new.tanker, new.report_at_nm))
    assert len(rows) == 80 and any(r[2] != 'candidate' for r in rows)


def test_legacy_refusal_preserves_original_journal_and_never_reinfers(tmp_path):
    provider = M.ScriptedProvider(lambda _: M.ModelResponse('answered', 'invented', 11, 7))
    original = tmp_path/'original'
    first, _ = gateway(tmp_path, 'first', 'build', original, provider)
    first.submit('reasoning_model', MSG, 256, 1, 0, 0, 0.)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    records = [json.loads(line) for line in original.read_text().splitlines()]
    for rec in records: rec.pop('purpose', None)
    legacy = tmp_path/'legacy'
    legacy.write_text(''.join(json.dumps(r)+'\n' for r in records))
    before = legacy.read_bytes()
    for purpose in ('running', 'build'):
        second, ledger = gateway(tmp_path, purpose, purpose, legacy, provider)
        with pytest.raises(M.GatewayError, match='original purpose is unknown'):
            second.submit('reasoning_model', MSG, 256, 1, 0, 0, 0.)
        assert ledger.calls == [] and provider.calls == 1 and legacy.read_bytes() == before
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest


def test_exhausted_cap_retains_attempt_records(monkeypatch):
    monkeypatch.setattr(G, 'ATTEMPT_CAP', 2)
    def draw(stream, phase, index, attempt):
        raise G.Unusable('no eligible new goal', {'eligible_nm': [], 't_g': 61})
    monkeypatch.setattr(G, 'draw', draw)
    with pytest.raises(G.GenerationError) as caught:
        G.solvable_episode(4401, 4, 0)
    records = getattr(caught.value, 'attempts', None)
    assert records is not None and [r['attempt'] for r in records] == [0, 1]
    assert [r['seed'] for r in records] == [G.episode_seed(4401, 4, 0, i) for i in range(2)]
    assert all(r['reason'] == 'no eligible new goal' and r['details']['eligible_nm'] == [] for r in records)


def test_nominal_success_with_unsettled_reference_outcome_is_not_certified(monkeypatch):
    # Fixed invented oracle evidence, no actual pilot flight. Keep a possible
    # failure as unknown: it cannot be silently labelled a solvable candidate.
    uncertain = {'goal_met': True, 'settled': False, 'possible_goal_met': [False, True],
                 'flags': [{'kind': 'invented numerical boundary'}], 'alternatives': []}
    settled = {'goal_met': True, 'settled': True, 'possible_goal_met': [True],
               'flags': [], 'alternatives': []}
    replies = iter([uncertain, settled])
    monkeypatch.setattr(G.phase5, 'fly', lambda *a, **k: next(replies))
    out = G.solvable_episode(4401, 4, 0)
    # ADAPTED by Claude, TURN 140: attempt 1 of s4401-p4-e00 is a recorded Unusable draw (no eligible new goal),
    # which is replaced without flying, so the assertions apply to the attempts that flew. Original assertions:
    # len(out.attempts) == 2; out.attempts[1]['solvable'] is True.
    flown = [a for a in out.attempts if 'outcome' in a]
    assert len(flown) == 2 and flown[0]['solvable'] is not True
    assert flown[0]['reason'] and flown[1]['solvable'] is True
    assert flown[0]['outcome'] == uncertain
    assert [a['reason'] for a in out.attempts if 'outcome' not in a] == ['no eligible new goal']
