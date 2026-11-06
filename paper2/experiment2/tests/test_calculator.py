"""Tests for paper 1's calculator as an Experiment 2 condition, on made-up scenarios. Nothing here is a result."""
from __future__ import annotations

import math

from paper2.experiment2.baselines.calculator import Paper1Calculator; from paper2.experiment2.baselines.calculator import condition_outcome
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P


def scenario(**kw):
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200))),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    base = dict(episode_id="test", receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                goals=((0, E.Goal(1, "SHELL 1", 20.0)),))
    base.update(kw)
    return E.Scenario(**base)


def fly(sc, steps=1800):
    ep = E.Episode(sc)
    start = ep.reset()
    calc = Paper1Calculator(start["briefing"])
    bundle = start["bundle"]
    for _ in range(steps):
        bundle = ep.step(calc.act(bundle))
        if bundle is None:
            break
    return ep, calc


def test_still_air_join_and_report_through_the_trigger():
    ep, calc = fly(scenario())
    out = ep.outcome()
    assert calc.declined is None
    assert out["end_reason"] == "join" and out["goal_met"]
    assert [r["via_trigger"] for r in ep.record.reports] == [True]


def test_declines_in_wind_and_then_holds():
    ep, calc = fly(scenario(wind=(P.kt(20.0), 0.0)), steps=50)
    assert calc.declined == {"t": 0, "reason": "wind"}
    assert all(not a["actions"] for a in ep.record.actions)


def test_declines_when_the_report_goes_stale():
    ep, calc = fly(scenario(dropouts=(("SHELL 1", 30, None),)), steps=100)
    assert calc.declined == {"t": 40, "reason": "stale or missing report"}    # last report measured at 29


def test_declines_when_the_orbit_departs_from_the_briefing():
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200), delta=1.6)),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    ep, calc = fly(scenario(tankers=tankers), steps=120)
    assert calc.declined is not None and calc.declined["reason"] == "orbit differs from the briefing"
    assert calc.declined["t"] == 61                                           # first report after the change starts


def test_a_new_goal_gets_its_own_trigger():
    sc = scenario(goals=((0, E.Goal(1, "SHELL 1", 20.0)), (60, E.Goal(2, "SHELL 1", 15.0))))
    ep, calc = fly(sc)
    assert calc.declined is None and ep.outcome()["goal_met"]
    assert [r["goal_version"] for r in ep.record.reports] == [2]


def test_a_long_delay_declines_at_second_11_not_at_the_start():
    """B9.20.1: the first report is delivered fresh at time 0 even with a 30 s delay."""
    ep, calc = fly(scenario(delay_s=30), steps=40)
    assert calc.declined == {"t": 11, "reason": "stale or missing report"}


def test_a_declined_episode_is_never_completed_by_the_condition():
    out = {"goal_met": True, "settled": True}               # as if the held heading later met the goal
    calc = Paper1Calculator.__new__(Paper1Calculator)
    calc.declined = {"t": 11, "reason": "stale or missing report"}
    r = condition_outcome(out, calc)
    assert r["physical_goal_met"] and not r["completed_by_condition"]
