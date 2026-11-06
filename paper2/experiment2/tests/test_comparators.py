"""The fixed-policy comparator runner (MVP step 5), against the checks in COMPARATOR-RUNNER-CONTRACT-CODEX.md.
Invented scenarios and pilots, and short real generated episodes for the calculator and the controller. CPU only;
nothing here is a result."""
from __future__ import annotations

import json
import math

import pytest

import paper2.experiment2.comparators as K
from telohelm.runtime.trusted_log import verify
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P
import paper2.experiment2.streams as G


def scenario(deadline=12, wind=(0.0, 0.0), goals=None, episode_id="invented"):
    orbit = P.Orbit(60.0, 40.0, 8.0, P.kt(280), 0.5)
    receiver = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450), P.turn_rate(450))
    goals = goals or ((0, E.Goal(1, "SHELL 1", 20.0, deadline)),)
    return E.Scenario(episode_id=episode_id, receiver=receiver, wind=wind, tankers=(E.Tanker("SHELL 1", orbit),),
                      goals=goals)


class Invented:
    """A pilot that records what it is given and replies from a script: t -> action (default: hold)."""
    made = []

    def __init__(self, script=None, raise_at=None):
        self.script, self.raise_at = script or {}, raise_at
        self.resets, self.seen = 0, []
        Invented.made.append(self)

    def reset(self, briefing):
        self.resets += 1
        self.briefing = briefing

    def act(self, bundle):
        self.seen.append(bundle["t"])
        bundle["own"]["x"] = 999.0                         # a pilot that writes on its input must not reach the world
        if bundle["t"] == self.raise_at:
            raise ValueError("invented failure")
        return self.script.get(bundle["t"], {})


def condition(**kw):
    return K.Condition("invented", "invented", lambda: Invented(**kw))


def summary(path):
    return json.loads((path / "summary.json").read_text())


def log(path):
    return [json.loads(l)["payload"] for l in (path / "episode.log").read_text().splitlines()]


def test_each_episode_gets_a_fresh_pilot_reset_once_with_public_copies(tmp_path):
    Invented.made = []
    a = K.run_pilot_episode(scenario(), condition(), tmp_path / "a", "s")
    b = K.run_pilot_episode(scenario(episode_id="other"), condition(), tmp_path / "b", "s")
    assert len(Invented.made) == 2 and all(p.resets == 1 for p in Invented.made)
    assert a["ended_at"] == b["ended_at"] == 12              # the pilot's writes on its copies changed nothing
    ev = log(tmp_path / "a")
    bundles = [e for e in ev if e["kind"] == "bundle"]
    assert all(e["bundle"]["own"]["x"] != 999.0 for e in bundles)
    assert all(e["sha256"] == K._sha(e["bundle"]) for e in bundles)
    assert verify(tmp_path / "a" / "episode.log", tmp_path / "a" / "episode.heads")["ok"]
    assert "scenario_sha256" in ev[1] and set(Invented.made[0].briefing) == {"own_callsign", "control_callsign",
                                                                              "tankers", "goal"}


def test_cadence_starts_at_zero_rises_by_one_and_never_acts_at_the_end(tmp_path):
    Invented.made = []
    r = K.run_pilot_episode(scenario(deadline=12), condition(), tmp_path / "e", "s")
    seen = Invented.made[0].seen
    assert seen == list(range(12)) and r["action_steps"] == 12 and r["end_reason"] == "deadline" and r["complete"]


def test_goal_changes_reach_the_pilot_and_an_old_tag_is_refused_by_the_world_not_repaired(tmp_path):
    goals = ((0, E.Goal(1, "SHELL 1", 20.0, None)), (5, E.Goal(2, "SHELL 1", 25.0, 10)))
    script = {t: {"goal_version": 1, "heading_deg": 45.0} for t in range(10)}
    K.run_pilot_episode(scenario(goals=goals), condition(script=script), tmp_path / "g", "s")
    ev = log(tmp_path / "g")
    goal_bundles = [e["t"] for e in ev if e["kind"] == "bundle" and e["bundle"]["goal"] is not None]
    assert goal_bundles == [0, 5]
    steps = {e["t"]: e for e in ev if e["kind"] == "world_step"}
    assert all(steps[t]["accepted"] for t in range(5))
    assert all(not steps[t]["accepted"] and steps[t]["refusal"] == "goal version out of date" for t in range(5, 10))
    decisions = {e["t"]: e for e in ev if e["kind"] == "decision"}
    assert decisions[7]["goal_tag"] == 1 and decisions[7]["current_goal_version"] == 2


def test_none_and_malformed_actions_go_to_the_world_and_are_refused(tmp_path):
    script = {1: None, 2: {"goal_version": 1, "heading_deg": 400.0}, 3: {"goal_version": 1, "heading_deg": 10.0}}
    r = K.run_pilot_episode(scenario(), condition(script=script), tmp_path / "m", "s")
    steps = {e["t"]: e for e in log(tmp_path / "m") if e["kind"] == "world_step"}
    assert steps[1]["refusal"] == "actions must be an object" and steps[2]["refusal"] == "heading out of range"
    assert steps[3]["accepted"]
    shares = r["method_seconds"]
    assert shares["shares"] == {"invented": 1, "held": 11} and sum(shares["shares"].values()) == r["action_steps"]
    assert shares["held_by_issuer"] == {"initial": 3, "invented": 8}


def test_a_step_cap_is_censored_and_the_natural_end_is_complete(tmp_path):
    cut = K.run_pilot_episode(scenario(), condition(), tmp_path / "c", "s", max_steps=3)
    full = K.run_pilot_episode(scenario(), condition(), tmp_path / "f", "s")
    assert (cut["complete"], cut["censored_by_step_limit"], cut["action_steps"]) == (False, True, 3)
    assert (full["complete"], full["censored_by_step_limit"]) == (True, False)


def test_a_policy_failure_keeps_the_partial_records_and_stops_the_run(tmp_path):
    with pytest.raises(K.PolicyFailed):
        K.run_pilot_episode(scenario(), condition(raise_at=4), tmp_path / "x", "s")
    s = summary(tmp_path / "x")
    assert s["status"] == "policy_failed" and s["outcome"] is None and s["action_steps"] == 4
    assert log(tmp_path / "x")[-1]["kind"] == "policy_failed"
    rows = [json.loads(l) for l in (tmp_path / "x" / "work.jsonl").read_text().splitlines()]
    assert [r["scope"] for r in rows].count("policy_act") == 5          # the failed call is charged too


def test_policy_time_is_inclusive_and_a_pilots_own_counter_is_only_a_diagnostic(tmp_path):
    class Counting(Invented):
        cpu_ns = 123
    r = K.run_pilot_episode(scenario(), K.Condition("invented", "invented", Counting), tmp_path / "t", "s")
    rows = [json.loads(l) for l in (tmp_path / "t" / "work.jsonl").read_text().splitlines()]
    assert r["costs"]["policy_cpu_ns"] == sum(x["cpu_ns"] for x in rows if x["scope"] in ("policy_reset", "policy_act"))
    assert r["costs"]["model_calls"] == 0 and r["diagnostics"]["policy_reported_cpu_ns"] == 123
    assert {x["scope"] for x in rows} == {"policy_reset", "policy_act", "world_step"}


def test_the_calculator_declines_in_wind_and_the_episode_still_counts_as_complete(tmp_path):
    r = K.run_pilot_episode(scenario(wind=(0.01, 0.0)), K.conditions()["calculator"], tmp_path / "w", "s")
    assert r["complete"] and r["condition_outcome"]["declined"]["reason"] == "wind"
    assert r["condition_outcome"]["completed_by_condition"] is False


@pytest.mark.parametrize("name", ["calculator", "adaptive_controller"])
def test_both_fixed_conditions_fly_a_generated_episode_through_the_runner(tmp_path, name):
    sc = G.solvable_episode(4401, 1, 0).scenario
    r = K.run_pilot_episode(sc, K.conditions()[name], tmp_path / name, "s", max_steps=20)
    assert r["status"] == "ok" and r["action_steps"] == 20 and r["censored_by_step_limit"]
    assert r["method_seconds"]["shares"][name] >= 1          # it flew, through its own source



def test_ctrl_wrap_witness_the_controllers_propagation_can_produce_360():
    """The smallest form of MVP126.CTRL.WRAP, for Codex: a tiny negative angle wraps to exactly 360.0."""
    assert math.degrees(-1e-16) % 360 == 360.0


# Arms the trigger until the aircraft is within 1 NM of the report distance, then leaves it. Setting it every second
# would re-arm it every second, and a trigger armed in a second cannot fire until the next, so it would never fire.
TRIGGER = ('def decide(ctx):\n'
           '    if ctx.observed_range_nm is not None and ctx.observed_range_nm > ctx.report_at_nm + 1:\n'
           '        return {"report_trigger_nm": ctx.report_at_nm}\n'
           '    return {}\n')


def two_goal_scenario():
    orbit = P.Orbit(22.0, 0.0, 8.0, P.kt(280), 0.5)
    receiver = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450), P.turn_rate(450))
    goals = ((0, E.Goal(1, "SHELL 1", 20.0, None)), (60, E.Goal(2, "SHELL 1", 15.0, 200)))
    return E.Scenario(episode_id="invented-parity", receiver=receiver, wind=(0.0, 0.0),
                      tankers=(E.Tanker("SHELL 1", orbit),), goals=goals)


def _plain(x):
    return json.loads(json.dumps(x, default=str))


def test_replaying_telohelms_actions_through_the_fixed_path_gives_the_same_inputs_and_outcome(tmp_path):
    """Codex's TURN 127 parity: the two paths apply the same world rules. Telohelm flies with the stocked heading
    procedure and a report-trigger procedure through a goal change to a deadline or join; the fixed path then replays
    its exact actions. Every public input and the whole outcome must agree."""
    from copy import deepcopy
    import paper2.experiment2.mvp_seed as seed
    import telohelm.runtime.broker as B; import telohelm.runtime.episode_loop as L; import telohelm.runtime.prediction as PT
    from telohelm.runtime.ledger import Ledger
    from telohelm.runtime.library import Library
    from telohelm.runtime.trusted_log import TrustedLog
    sc = two_goal_scenario()
    lib = Library(tmp_path / "library")
    for kind, decision, source in (("procedure", "heading", seed.AIM), ("check", "heading", seed.HEADING_CHECK),
                                   ("procedure", "report", TRIGGER), ("check", "report", seed.REPORT_CHECK)):
        lib.set_status(lib.add({"kind": kind, "decision": decision, "conditions": [], "source": source,
                                "origin": {"calls": []}}), "adopted", "invented parity fixture")
    ledger = Ledger()
    br = B.Broker(sc.episode_id, TrustedLog(tmp_path / "b.log", tmp_path / "b.heads", "s"), ledger, PT.predict)
    telo = L.run_episode(sc, lib, L.NoModels(), TrustedLog(tmp_path / "t.log", tmp_path / "t.heads", "s"), ledger,
                         broker=br)
    inputs = B.load_inputs(tmp_path / "b.log", tmp_path / "b.heads", sc.episode_id)
    actions = telo["actions"]
    assert any("report_trigger_nm" in a for a in actions) and any(r["via_trigger"] for r in telo["outcome"]["reports"])

    class Replay:
        def reset(self, briefing):
            pass

        def act(self, bundle):
            return deepcopy(actions[bundle["t"]])

    fixed = K.run_pilot_episode(sc, K.Condition("replay", "replay", Replay), tmp_path / "fixed", "s")
    ev = log(tmp_path / "fixed")
    assert _plain([e["briefing"] for e in ev if e["kind"] == "briefing"][0]) == _plain(inputs["briefing"])
    assert _plain([e["bundle"] for e in ev if e["kind"] == "bundle"]) == _plain([b["bundle"] for b in inputs["bundles"]])
    assert [e["t"] for e in ev if e["kind"] == "bundle"] == [b["t"] for b in inputs["bundles"]]
    assert _plain(fixed["outcome"]) == _plain(telo["outcome"])
    assert fixed["outcome"]["goal"]["version"] == 2 and fixed["end_reason"] in ("join", "deadline")


def test_an_old_target_help_reply_and_an_older_report_after_a_newer_one_reach_the_pilot_in_world_order(tmp_path):
    """The help reply about SHELL 1 arrives after the goal has moved to SHELL 2 and does not change the goal; with a
    5-second report delay, a regular report measured before the help reply is delivered after it."""
    t1 = E.Tanker("SHELL 1", P.Orbit(60.0, 40.0, 8.0, P.kt(280), 0.5))
    t2 = E.Tanker("SHELL 2", P.Orbit(60.0, -10.0, 8.0, P.kt(280), 1.5))
    receiver = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450), P.turn_rate(450))
    goals = ((0, E.Goal(1, "SHELL 1", 20.0, None)), (10, E.Goal(2, "SHELL 2", 20.0, 40)))
    sc = E.Scenario(episode_id="invented-help", receiver=receiver, wind=(0.0, 0.0), tankers=(t1, t2), goals=goals,
                    delay_s=5)
    Invented.made = []
    help_at_1 = {1: {"goal_version": 1, "help": {"reason": "missing_tool", "text": "invented"}}}
    K.run_pilot_episode(sc, condition(script=help_at_1), tmp_path / "h", "s")
    ev = log(tmp_path / "h")
    bundles = {e["t"]: e["bundle"] for e in ev if e["kind"] == "bundle"}
    reply = [r for r in bundles[31]["reports"] if r.get("help_reply")]
    assert [(r["callsign"], r["measured_at"]) for r in reply] == [("SHELL 1", 31)]
    assert bundles[31]["goal"] is None
    assert [e for e in ev if e["kind"] == "decision" and e["t"] == 31][0]["current_goal_version"] == 2
    later = [r for r in bundles[32]["reports"] if r["callsign"] == "SHELL 1"]
    assert later and all(r["measured_at"] < 31 and not r.get("help_reply") for r in later)


def test_held_seconds_keep_the_original_decision_and_goal_tag_across_a_goal_change(tmp_path):
    goals = ((0, E.Goal(1, "SHELL 1", 20.0, None)), (5, E.Goal(2, "SHELL 1", 25.0, 10)))
    r = K.run_pilot_episode(scenario(goals=goals), condition(script={2: {"goal_version": 1, "heading_deg": 45.0}}),
                            tmp_path / "o", "s")
    steps = {e["t"]: e for e in log(tmp_path / "o") if e["kind"] == "world_step"}
    assert steps[2]["heading"] == {"issued": "invented/invented/2"}
    assert steps[8]["heading"] == {"held": {"decision_id": "invented/invented/2", "goal_version": 1},
                                   "issuer": "invented"}
    ms = r["method_seconds"]
    assert ms["seconds"] == 10 and ms["fractions"] == {"invented": 0.1, "held": 0.9}


def test_a_declined_calculator_episode_that_later_meets_the_goal_physically_stays_in_the_denominator(tmp_path):
    """Reports stop for good at 5 s, so the calculator declines as stale at 15 s (the last report is measured at 4 s); the held heading and the trigger it
    set at 0 s still join and report. Physically met, not completed by the condition, and still a complete episode."""
    tanker = E.Tanker("SHELL 1", P.Orbit(12.0, 0.0, 1.0, 0.001, 0.0))
    receiver = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450), P.turn_rate(450))
    sc = E.Scenario(episode_id="invented-decline", receiver=receiver, wind=(0.0, 0.0), tankers=(tanker,),
                    goals=((0, E.Goal(1, "SHELL 1", 5.0, None)),), dropouts=(("SHELL 1", 5, None),))
    out = tmp_path / "stream"
    r = K.run_pilot_episode(sc, K.conditions()["calculator"], out / sc.episode_id, "s")
    c = r["condition_outcome"]
    assert c["declined"]["reason"] == "stale or missing report" and c["declined"]["t"] == 15
    assert r["complete"] and c["physical_goal_met"] is True and c["completed_by_condition"] is False
