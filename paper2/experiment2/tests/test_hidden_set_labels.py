"""Tests for the bound-based heading labels (note B8, decision D10), on a made-up still-air scenario."""
from __future__ import annotations

import math

from paper2.experiment2.hidden_set.labels import DELTA_S; from paper2.experiment2.hidden_set.labels import label_heading
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P
from telohelm.taskpacks.tanker.world.pilot import ReferencePilot


def scenario(deadline=None):
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200))),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    return E.Scenario(episode_id="t", receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                      goals=((0, E.Goal(1, "SHELL 1", 20.0, deadline=deadline)),))


def at_second(sc, t):
    ep = E.Episode(sc)
    start = ep.reset()
    pilot = ReferencePilot(start["briefing"], ask_help=False)
    bundle = start["bundle"]
    for _ in range(t):
        bundle = ep.step(pilot.act(bundle))
    return ep, pilot, bundle


def consistent(result):
    """Every label follows from the stored bounds by the rule, and no infinite bound gives 'right'."""
    us, ls = result["upper_star"], result["lower_star"]
    for c, r in result["labels"].items():
        if r["label"] == "right":
            assert math.isfinite(r["upper"]) and math.isfinite(ls) and r["upper"] - ls < DELTA_S / 2
        if r["label"] == "wrong":
            assert r["lower"] is not None and r["lower"] - us > DELTA_S


def test_labels_follow_the_rule_and_leave_the_episode_untouched():
    ep, pilot, bundle = at_second(scenario(), 100)
    before = (ep.t, ep.receiver, len(ep.record.actions))
    own = bundle["own"]["heading_deg"]
    result = label_heading(ep, pilot, bundle, [own, (own + 180) % 360, (own + 30) % 360])
    assert (ep.t, ep.receiver, len(ep.record.actions)) == before
    consistent(result)
    assert math.isfinite(result["reference"]["upper"])          # the reference pilot meets the whole goal
    assert result["labels"][own]["label"] != "wrong"
    for r in result["labels"].values():
        assert r["lower"] <= r["upper"]                           # the relaxation never exceeds a witness


def test_an_impossible_deadline_makes_every_heading_wrong_for_the_deadline():
    ep, pilot, bundle = at_second(scenario(deadline=150), 100)
    result = label_heading(ep, pilot, bundle, [0.0, 90.0])
    assert {r["label"] for r in result["labels"].values()} == {"wrong (deadline)"}
    assert result["lower_star"] == math.inf


def test_one_second_of_a_reversed_heading_is_labelled_right():
    """The consequence of one-second action semantics (B8.16.1): the aircraft turns at about 1.4° a second,
    so one second of any command barely moves it, and the witness loses less than a second. Wrong headings
    in the hidden set therefore have to come from sequence cases."""
    ep, pilot, bundle = at_second(scenario(), 100)
    own = bundle["own"]["heading_deg"]
    result = label_heading(ep, pilot, bundle, [(own + 180) % 360])
    r = result["labels"][(own + 180) % 360]
    assert r["label"] == "right" and r["upper"] - result["upper_star"] < 1.0


def test_goal_met_inside_the_prefix_uses_elapsed_time_from_t():
    """B8.20.1: a join inside the one-second prefix gives U_c = t_end − t and L_c = max(0, t_end − 1 − t)."""
    ep, pilot, bundle = at_second(scenario(), 0)
    while True:                                        # fly the reference to find its join second J
        nxt = ep.step(pilot.act(bundle))
        if nxt is None:
            break
        bundle = nxt
    join = ep.record.joined_at
    ep, pilot, bundle = at_second(scenario(), join - 1)
    result = label_heading(ep, pilot, bundle, [bundle["own"]["heading_deg"]])
    r = result["labels"][bundle["own"]["heading_deg"]]
    assert r["prefix"] == "met" and r["upper"] == 1.0 and r["lower"] == 0.0


def test_a_witness_that_crosses_a_goal_change_is_no_witness():
    """R22.5: a physical goal change to SHELL 2 at second 200, labelled at 100. Every witness continuation
    meets the new goal, so none may bound the original goal's completion; all labels are unknown."""
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200))),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    sc = E.Scenario(episode_id="t", receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                    goals=((0, E.Goal(1, "SHELL 1", 20.0)), (200, E.Goal(2, "SHELL 2", 20.0))))
    ep, pilot, bundle = at_second(sc, 100)
    own = bundle["own"]["heading_deg"]
    result = label_heading(ep, pilot, bundle, [own, (own + 90) % 360])
    assert result["reference"]["witness"] == "goal changed during the witness"
    assert {r["label"] for r in result["labels"].values()} == {"unknown"}
    assert math.isinf(result["upper_star"])
