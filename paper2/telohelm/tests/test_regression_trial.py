"""A regression check in the trial (study2/learning.py, Coordinator.regression_set; REGRESSION-TRIAL-PLAN.md)."""
import json
from types import SimpleNamespace

import pytest

import telohelm.learning as G
from telohelm.runtime.library import Library
from .test_candidates import Proposer; from .test_candidates import reply


def coordinator(tmp_path, regression, completed, fails=()):
    """A coordinator whose finished episodes are given as (episode, alone) pairs; a candidate's trial flight fails on
    the episodes named in fails, and every flight is recorded."""
    co = G.Coordinator(Library(tmp_path / "lib"), tmp_path, Proposer([reply(20)]), lambda _: True, lambda *_: None,
                       candidates=1, learn_decisions=("heading",), regression=regression)
    co.flown = []

    def fly(shadow, c, root):
        co.flown.append(c.episode)
        return {"episode": c.episode, "goal_met": c.episode not in fails, "help": 0}
    co.fly = fly
    for i, (name, alone) in enumerate(completed, start=1):
        co.completed.append(G.Completed(i, name, SimpleNamespace(), True, None, True, alone=alone))
    co.queue = [{"episode": completed[-1][0], "failed": "join"}]
    return co


def history(n, alone=True):
    return [(f"e{i:02d}", alone) for i in range(1, n + 1)]


def test_off_by_default_the_trial_is_unchanged(tmp_path):
    co = coordinator(tmp_path, 0, history(12))
    co.attempt(co.completed[-1])
    assert co.flown == [f"e{i:02d}" for i in range(5, 13)]


def test_the_regression_set_is_earlier_episodes_met_alone_spread_and_keeping_the_newest(tmp_path):
    co = coordinator(tmp_path, 3, history(20))
    recent = co.completed[-8:]
    picked = [c.episode for c in co.regression_set(recent)]
    assert picked == ["e01", "e07", "e12"] and not set(picked) & {c.episode for c in recent}
    mixed = coordinator(tmp_path / "m", 8, [(f"e{i:02d}", i % 2 == 0) for i in range(1, 21)])
    assert all(int(c.episode[1:]) % 2 == 0 for c in mixed.regression_set(mixed.completed[-8:]))   # only alone ones
    assert coordinator(tmp_path / "c", 8, history(6, alone=False)).regression_set([]) == []      # cold start


def test_a_candidate_that_breaks_an_earlier_episode_is_not_adopted(tmp_path):
    co = coordinator(tmp_path, 8, history(20), fails={"e03"})
    event = co.attempt(co.completed[-1])
    assert not event["passed"] and "e03" in co.flown
    assert not co.library.adopted("heading", "procedure")
    regression = [r for r in event["detail"]["trial"] if r.get("regression")]
    assert regression and any(r["episode"] == "e03" and not r["goal_met"] for r in regression)


def test_the_regression_set_is_flown_only_after_the_recent_episodes_pass(tmp_path):
    co = coordinator(tmp_path, 8, history(20), fails={"e20"})
    co.attempt(co.completed[-1])
    assert co.flown == [f"e{i:02d}" for i in range(13, 21)]


def test_the_setting_is_validated(tmp_path):
    with pytest.raises(ValueError):
        G.Coordinator(Library(tmp_path / "lib"), tmp_path, None, lambda _: True, lambda *_: None, regression=-1)


def test_a_regression_episode_must_meet_the_whole_goal_not_only_the_candidates_part(tmp_path):
    co = coordinator(tmp_path, 8, history(20))

    def fly(shadow, c, root):
        co.flown.append(c.episode)
        # the candidate's own part passes everywhere, but the whole goal is lost on e03, a regression episode
        return {"episode": c.episode, "goal_met": c.episode != "e03", "help": 0, "component": {"passed": True}}
    co.fly = fly
    event = co.attempt(co.completed[-1])
    assert not event["passed"] and "e03" in co.flown
