"""Stream generation, phases 1 and 2 (MVP). Development seeds only."""
from __future__ import annotations

import math

import pytest

import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.phase5 as phase5
import paper2.experiment2.streams as G


def test_episodes_are_reproducible_and_named_by_stream_phase_and_index():
    a, b = G.draw(4401, 1, 3), G.draw(4401, 1, 3)
    assert a == b and a.episode_id == "s4401-p1-e03"
    assert G.draw(4401, 1, 4) != a and G.draw(4402, 1, 3) != a and G.draw(4401, 1, 3, attempt=1) != a


def test_phase_1_is_still_air_and_phase_2_has_one_of_the_12_winds():
    assert all(G.draw(4401, 1, i).wind == (0.0, 0.0) for i in range(20))
    allowed = {tuple(round(v, 12) for v in phase5.wind_vector(s, d)) for s, d in G.WINDS}
    winds = {tuple(round(v, 12) for v in G.draw(4401, 2, i).wind) for i in range(40)}
    assert winds <= allowed and len(winds) > 3 and (0.0, 0.0) not in winds


def test_draws_respect_the_ranges():
    for i in range(30):
        s = G.draw(4403, 2, i)
        r = s.receiver
        assert -5 <= r.x <= 5 and -5 <= r.y <= 5 and math.isclose(r.heading, math.radians(90))
        t1, t2 = s.tankers
        assert 50 <= t1.orbit.cx <= 80 and 35 <= t1.orbit.cy <= 60 and 6 <= t1.orbit.r0 <= 10
        assert 44 <= abs(t2.orbit.cy - t1.orbit.cy) <= 56 and t1.callsign == "SHELL 1" and t2.callsign == "SHELL 2"
        assert s.dropouts == () and s.delay_s == 0 and s.goals[0][1].tanker == "SHELL 1"
        assert s.goals[0][1].report_at_nm in G.DISTANCES


def test_generated_episodes_are_reference_pilot_solvable_and_attempts_are_kept():
    g = G.solvable_episode(4401, 1, 0)
    assert phase5.fly(g.scenario, ask_help=False)["goal_met"] and g.attempts[-1]["solvable"] is True
    assert all(a["reason"] == "not solvable by the reference pilot" for a in g.attempts[:-1])


def test_unsolvable_draws_are_replaced_and_the_cap_stops_generation(monkeypatch):
    calls = {"n": 0}
    def fly(s, ask_help):
        calls["n"] += 1
        return {"goal_met": calls["n"] >= 3, "settled": True}
    monkeypatch.setattr(G.phase5, "fly", fly)
    g = G.solvable_episode(4401, 1, 0)
    assert [a["solvable"] for a in g.attempts] == [False, False, True]
    monkeypatch.setattr(G.phase5, "fly", lambda s, ask_help: {"goal_met": False, "settled": True})
    monkeypatch.setattr(G, "ATTEMPT_CAP", 4)
    with pytest.raises(G.GenerationError):
        G.solvable_episode(4401, 1, 1)


def test_phases_not_yet_generated_are_refused():
    with pytest.raises(G.GenerationError):
        G.draw(4401, 5, 0)


def _lower(s, tanker, t):
    import math as m
    o, r = tanker.orbit, s.receiver
    return m.hypot(o.cx - r.x, o.cy - r.y) - (r.tas + m.hypot(*s.wind)) * t - (o.r0 + abs(o.delta))


def test_phase_4_adds_one_goal_change_inside_its_window_and_eligible_against_every_path():
    seen = 0
    for i in range(20):
        try:
            s = G.draw(4405, 4, i)
        except G.Unusable as u:
            assert u.reason in ("no goal-change window", "no eligible new goal") and "t_max" in u.details
            continue
        seen += 1
        (t0, g1), (tg, g2) = s.goals
        assert t0 == 0 and g2.version == g1.version + 1 and g2.deadline is None and g1.deadline is None
        shell1, shell2 = s.tankers
        tau = (_lower(s, shell1, 0.0) - E.JOIN_TOLERANCE) / (s.receiver.tas + math.hypot(*s.wind))
        assert 60 <= tg <= math.ceil(tau) - 1 - 10
        assert _lower(s, shell1, tg) > E.JOIN_TOLERANCE           # no path can have joined before the change
        new = shell1 if g2.tanker == "SHELL 1" else shell2
        assert g2.report_at_nm in G.DISTANCES and g2.report_at_nm <= _lower(s, new, tg) - 3
        if g2.tanker == "SHELL 1":
            assert g2.report_at_nm != g1.report_at_nm
        assert G.draw(4405, 4, i) == s                             # reproducible
    assert seen >= 10


def test_unusable_phase_4_draws_are_kept_with_their_reason_and_replaced():
    reasons = [a["reason"] for i in range(20) for a in G.solvable_episode(4405, 4, i).attempts]
    assert reasons.count(None) == 20
    assert all(r in (None, "no goal-change window", "no eligible new goal", "not solvable by the reference pilot")
               for r in reasons)


@pytest.mark.parametrize("index", [0, 5, 10, 15])
def test_each_phase_4_block_yields_a_reference_pilot_solvable_episode_with_a_goal_change(index):
    g = G.solvable_episode(4401, 4, index)
    assert len(g.scenario.goals) == 2 and phase5.fly(g.scenario, ask_help=False)["goal_met"]


def test_phase_3_blocks_carry_their_disturbances_with_wind():
    allowed = {tuple(round(v, 12) for v in phase5.wind_vector(s, d)) for s, d in G.WINDS}
    for i in range(20):
        s = G.draw(4401, 3, i)
        block = G.PHASE3_BLOCKS[i // 5]
        assert tuple(round(v, 12) for v in s.wind) in allowed
        shell1, shell2 = s.tankers
        assert shell2.orbit.delta == 0.0
        orbit = block in ("orbit", "all")
        assert (shell1.orbit.delta != 0.0) == orbit
        assert abs(shell1.orbit.delta) <= 0.2 * shell1.orbit.r0
        if block in ("dropout", "all"):
            ((who, start, end),) = s.dropouts
            assert who == "SHELL 1" and 60 <= start <= 600 and end - start in (5, 30)
        else:
            assert s.dropouts == ()
        assert (s.delay_s in (5, 30, 120)) == (block in ("delay", "all"))
        assert s.delay_s == 0 or block in ("delay", "all")


def test_phase_3_draws_are_reproducible_and_change_with_the_attempt():
    a, b = G.draw(4401, 3, 7), G.draw(4401, 3, 7)
    assert a == b and G.draw(4401, 3, 7, attempt=1) != a and a.episode_id == "s4401-p3-e07"


@pytest.mark.parametrize("index", [0, 5, 10, 15])
def test_each_phase_3_block_yields_a_reference_pilot_solvable_episode(index):
    g = G.solvable_episode(4401, 3, index)
    assert g.attempts[-1]["solvable"] is True and phase5.fly(g.scenario, ask_help=False)["goal_met"]


def test_only_a_settled_reference_success_certifies_and_every_attempt_keeps_its_outcome(monkeypatch):
    """GEN139.UNSETTLED with real flights' shape: an unsettled nominal success is replaced, its whole outcome kept."""
    outcomes = iter([{"goal_met": True, "settled": False, "flags": [{"kind": "invented"}]},
                     {"goal_met": False, "settled": True},
                     {"goal_met": True, "settled": True}])
    monkeypatch.setattr(G.phase5, "fly", lambda s, ask_help: next(outcomes))
    g = G.solvable_episode(4401, 1, 0)
    assert [a["reason"] for a in g.attempts] == ["reference outcome not settled", "not solvable by the reference pilot",
                                                None]
    assert g.attempts[0]["outcome"]["flags"] == [{"kind": "invented"}] and g.attempts[2]["solvable"] is True


def test_the_cap_error_keeps_every_attempt(monkeypatch):
    monkeypatch.setattr(G, "ATTEMPT_CAP", 3)
    monkeypatch.setattr(G.phase5, "fly", lambda s, ask_help: {"goal_met": False, "settled": True})
    with pytest.raises(G.GenerationError) as e:
        G.solvable_episode(4401, 1, 0)
    assert [a["attempt"] for a in e.value.attempts] == [0, 1, 2]
    assert [a["seed"] for a in e.value.attempts] == [G.episode_seed(4401, 1, 0, i) for i in range(3)]



def test_the_learning_label_is_unknown_when_the_reference_outcome_is_unsettled(monkeypatch):
    import telohelm.learning as L
    for outcome, label in (({"goal_met": True, "settled": True}, True), ({"goal_met": False, "settled": True}, False),
                           ({"goal_met": True, "settled": False}, None), ({"goal_met": False}, None)):
        monkeypatch.setattr(G.phase5, "fly", lambda s, ask_help, o=outcome: o)
        assert L.reference_solvable(G.draw(4401, 1, 0)) is label


def test_a_stream_run_keeps_its_generation_records(tmp_path):
    import paper2.experiment2.comparators as K
    K.run_pilot_stream(K.conditions()["calculator"], 4401, (1,), 2, tmp_path / "c", max_steps=3)
    import json as _json
    recs = [_json.loads(l) for l in (tmp_path / "c" / "generation.jsonl").read_text().splitlines()]
    assert [r["episode"] for r in recs] == ["s4401-p1-e00", "s4401-p1-e01"]
    assert all(r["attempts"][-1]["solvable"] and "outcome" in r["attempts"][-1] for r in recs)
