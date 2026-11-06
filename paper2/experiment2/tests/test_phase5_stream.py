"""Phase 5 in the stream generator (study2/world/streams.py, help_episode): help cases of DISTURBANCES.md section 11."""
import gzip
import hashlib
import json
from dataclasses import replace

import pytest

import paper2.experiment2.run_stream as RS
import telohelm.taskpacks.tanker.world.phase5 as phase5; import paper2.experiment2.streams as streams


def test_the_two_worlds_differ_only_in_the_sign_of_the_radius_change():
    d, flown = streams.draw_help(4401, 0)
    a, b = d.world_a, d.world_b
    assert flown in ("a", "b") and a.episode_id == b.episode_id == "s4401-p5-e00"
    assert (a.receiver, a.wind, a.goals, a.dropouts, a.delay_s) == (b.receiver, b.wind, b.goals, b.dropouts, b.delay_s)
    assert a.tankers[1] == b.tankers[1]
    oa, ob = a.tankers[0].orbit, b.tankers[0].orbit
    assert replace(oa, delta=0.0) == replace(ob, delta=0.0)
    assert oa.delta == pytest.approx(0.2 * oa.r0) and ob.delta == pytest.approx(-0.2 * ob.r0)
    assert a.dropouts == (("SHELL 1", d.blackout, None),) and 20 <= d.blackout <= 59


def test_a_draw_is_reproducible_from_its_episode_seed():
    d1, f1 = streams.draw_help(4401, 3, 2)
    d2, f2 = streams.draw_help(4401, 3, 2)
    assert (d1, f1) == (d2, f2) and d1.seed == streams.episode_seed(4401, 5, 3, 2)
    assert streams.draw_help(4401, 3, 1)[0] != d1


@pytest.fixture(scope="module")
def accepted():
    return streams.episode(4401, 5, 0)


def test_the_accepted_episode_is_the_flown_world_with_the_certified_deadline(accepted):
    *rejected, last = accepted.attempts
    assert last["help_certified"] and last["reason"] is None
    assert all(not r["help_certified"] and r["reason"] for r in rejected)
    d, flown = streams.draw_help(4401, 0, last["attempt"])
    world = d.world_a if flown == "a" else d.world_b
    goal = accepted.scenario.goals[0][1]
    assert goal.deadline == accepted.witness["deadline"] == last["assessment"]["deadline"]
    assert accepted.scenario == replace(world, goals=((0, goal),))


def test_the_reference_pilot_meets_the_goal_with_help_by_the_deadline(accepted):
    o = phase5.fly(accepted.scenario, ask_help=True)
    assert o["goal_met"] is True and o["settled"] is True


def test_attempt_records_keep_trace_hashes_and_the_witness_keeps_the_traces(accepted):
    flight = accepted.attempts[-1]["assessment"]["witness"]["a"]
    full = accepted.witness["witness"]["a"]["trace"]
    assert flight["trace"]["steps"] == len(full["steps"])
    assert flight["trace"]["sha256"] == hashlib.sha256(json.dumps(full, sort_keys=True, default=str).encode()).hexdigest()


def test_generation_stops_at_the_cap_with_every_attempt(monkeypatch):
    monkeypatch.setattr(streams, "ATTEMPT_CAP", 3)
    monkeypatch.setattr(phase5, "assess_draw", lambda d: {"label": "unknown", "reason": "no deadline window"})
    with pytest.raises(streams.GenerationError) as e:
        streams.episode(4401, 5, 0)
    assert [a["attempt"] for a in e.value.attempts] == [0, 1, 2]
    assert streams.cap_record(4401, 5, 0, e.value)["accepted"] is False


def test_other_phases_still_come_from_the_solvable_generator():
    assert streams.episode(4401, 1, 0).scenario == streams.solvable_episode(4401, 1, 0).scenario


def test_the_stream_runner_keeps_each_witness_beside_the_run(tmp_path):
    out = tmp_path / "run"
    RS.run_stream(1, 1, out, stream_seed=4401, phases=(5,), empty_library=True, max_steps=2)
    (row,) = [json.loads(l) for l in (out / "generation.jsonl").read_text().splitlines()]
    data = (out / row["witness"]["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["witness"]["sha256"]
    witness = json.loads(gzip.decompress(data))
    assert witness["label"] == "help certified" and witness["deadline"] == row["attempts"][-1]["assessment"]["deadline"]
