"""The opt-in disagreement watch and own-report tracking (runtime/disagreement.py, runtime/own_reports.py; Codex's
TURN229 rulings; paper2/notes/HANDOVER-SIGNALS.md).

The replica of the report trigger must know every report the world makes; the gap must be zero on the briefed orbit
and open an incident after an orbit change; while an incident is open the decisions that read the evidence are
withheld from their procedures before selection, climb to a model told the gap, have a trigger their procedure left
armed withdrawn, and pass attribution; withholding lifts when no model can be asked; with both options off nothing
changes.
"""
from __future__ import annotations

import json
import math

import pytest

import telohelm.learning as G
import paper2.experiment2.run_stream as RS
from telohelm.taskpacks.tanker.execution import TANKER_EVIDENCE; from telohelm.taskpacks.tanker.execution import briefed_position; from telohelm.taskpacks.tanker.execution import briefing_gap
import telohelm.runtime.models as M
from telohelm.runtime.context import ContextBuilder
from telohelm.runtime.disagreement import COOLDOWN_S; from telohelm.runtime.disagreement import Watch; from telohelm.runtime.disagreement import models_exhausted
from telohelm.runtime.library import Library
from telohelm.runtime.own_reports import OwnTrigger
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P
import paper2.experiment2.streams as streams

AIM = '''import math
def decide(ctx):
    return {"heading_deg": math.degrees(math.atan2(ctx.report_x - ctx.x, ctx.report_y - ctx.y)) % 360}
'''
# Arms the trigger every 100 s, never knowing whether it has fired; a trigger cannot fire in its arming second, so one
# set every second would never fire.
TRIGGER = "def decide(ctx):\n    return {'report_trigger_nm': ctx.report_at_nm + 0.25} if ctx.t % 100 == 0 else {}\n"
ACCEPT = "def check(ctx, out):\n    return 'accept'\n"


def scenario(delta=0.0, report_at=10.0, name="watch-test"):
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200), delta)),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    return E.Scenario(episode_id=name, receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                      goals=((0, E.Goal(1, "SHELL 1", report_at)),))


def stocked(tmp_path, name, heading=AIM, report=TRIGGER) -> Library:
    lib = Library(tmp_path / f"library-{name}")
    for kind, decision, source in (("procedure", "heading", heading), ("check", "heading", ACCEPT),
                                   ("procedure", "report", report), ("check", "report", ACCEPT)):
        sha = lib.add({"kind": kind, "decision": decision, "conditions": [], "source": source,
                       "origin": {"calls": []}})
        lib.set_status(sha, "adopted", "test")
    return lib


def stand_in(seen, report_trigger=False):
    """Small and reasoning models alike: head for the newest report; never report, or, with report_trigger, set a
    trigger when asked without a gap; every context is kept."""
    def answer(request):
        body = json.loads(request.messages[-1][1])
        ctx, decision = body["context"], body["decision"]
        seen.append((decision, ctx))
        if decision == "report" and report_trigger and "briefing_disagreement" not in ctx:
            return M.ModelResponse("answered", json.dumps({"report_trigger_nm": ctx["report_at_nm"] + 0.25}), 50, 5)
        if decision == "heading" and ctx.get("report_x") is not None:
            h = math.degrees(math.atan2(ctx["report_x"] - ctx["x"], ctx["report_y"] - ctx["y"])) % 360
            return M.ModelResponse("answered", json.dumps({"heading_deg": round(h, 3)}), 50, 5)
        return M.ModelResponse("answered", "{}", 50, 5)
    return answer


def fly(tmp_path, monkeypatch, sc, name, steps=900, heading=AIM, report=TRIGGER, report_trigger=False, **flight):
    seen = []
    monkeypatch.setattr(RS.seed, "stand_in_model", stand_in(seen, report_trigger))
    out = tmp_path / name
    out.mkdir()
    done = RS.run_episode(sc, stocked(tmp_path, name, heading, report), out, "dev-test", steps,
                          {"small_model": 2.0, "reasoning_model": 60.0}, None,
                          {"allowance": None, "reasoning": None, "small_interval_s": 30, **flight})
    log = [json.loads(l)["payload"] for l in (out / sc.episode_id / "episode.log").read_text().splitlines()]
    return done["record"], log, seen


# -- the replica of the report trigger --------------------------------------------------------------------------------

def policy(ctx, t):
    """Arms the trigger every second, as step 4's report procedures did, with a direct report or a clearing now and
    then, and heads for the newest report."""
    a = {"goal_version": ctx["goal_version"]}
    if ctx["report_x"] is not None:
        a["heading_deg"] = math.degrees(math.atan2(ctx["report_x"] - ctx["x"], ctx["report_y"] - ctx["y"])) % 360
    if t % 97 == 41:
        a["report"] = "Wedgetail, aircraft 2, %g NM from %s" % (ctx["report_at_nm"], ctx["goal_tanker"])
    elif t % 53 == 7:
        a["report_trigger_nm"] = None
    else:
        a["report_trigger_nm"] = ctx["report_at_nm"] + 0.25
    if t % 211 == 5:
        a["heading_deg"] = 400.0               # refused: the world refuses the whole action
    return a


@pytest.mark.parametrize("phase,index", [(1, 0), (2, 3), (3, 0), (3, 7), (3, 12), (3, 16), (4, 2), (4, 9), (4, 18)])
def test_the_replica_knows_every_report_the_world_makes(phase, index):
    ep = E.Episode(streams.episode(4401, phase, index).scenario)
    start = ep.reset()
    builder, trig, bundle = ContextBuilder(start["briefing"]), OwnTrigger(), start["bundle"]
    while bundle is not None:
        ctx = builder.update(bundle)
        trig.at(ctx["goal_version"])
        action = policy(ctx, ctx["t"])
        bundle = ep.step(action)
        trig.after(ctx["t"], ctx, action, ep.record.actions[-1]["accepted"], "procedure")
    world = [(r["t"], r["goal_version"], r["via_trigger"]) for r in ep.outcome()["reports"]]
    mine = [(e["t"], e["goal_version"], e["via_trigger"]) for e in trig.events if e["kind"] == "own_report"]
    assert mine == world and len(world) >= 1


# -- the gap -----------------------------------------------------------------------------------------------------------

def gaps(sc, seconds):
    ep = E.Episode(sc)
    start = ep.reset()
    builder, bundle, out = ContextBuilder(start["briefing"]), start["bundle"], []
    for _ in range(seconds):
        if bundle is None:
            break
        ctx = builder.update(bundle)
        out.append(briefing_gap(start["briefing"], ctx))
        bundle = ep.step({})
    return start["briefing"], out


def test_the_gap_is_zero_on_the_briefed_orbit_and_grows_after_an_orbit_change():
    briefing, still = gaps(scenario(), 400)
    assert max(g for g in still if g is not None) < 1e-9
    tanker = briefing["tankers"][0]
    assert briefed_position(tanker, 0.0) == pytest.approx(tuple(tanker["position_at_0"]), abs=1e-12)
    _, changed = gaps(scenario(delta=3.0), 400)
    assert max(changed[:60]) < 1e-9 and max(changed) > TANKER_EVIDENCE.tolerance
    for phase, index in ((1, 0), (3, 5), (3, 10)):             # still air, dropout, late radar: the orbit as briefed
        _, g = gaps(streams.episode(4401, phase, index).scenario, 600)
        assert max(x for x in g if x is not None) < 1e-9


# -- the watch on its own ----------------------------------------------------------------------------------------------

class View:
    tolerance, unit, affects, withdraw, meaning = 1.0, "NM", frozenset({"heading", "report"}), {}, "a gap"

    def __init__(self):
        self.next = None

    def gap(self, briefing, ctx):
        return self.next


def test_an_incident_opens_above_the_tolerance_and_closes_after_the_cooldown_or_a_goal_change():
    view = View()
    w = Watch(view, {})
    view.next = 0.5
    assert w.update({"t": 0, "goal_version": 1}) == [] and not w.withholds("heading", 1)
    view.next = 2.0
    assert [e["kind"] for e in w.update({"t": 1, "goal_version": 1})] == ["incident_opened"]
    assert w.withholds("heading", 1) and w.withholds("report", 1) and not w.withholds("help", 1)
    assert not w.withholds("heading", 2)
    view.next = 0.2
    for t in range(2, 2 + COOLDOWN_S):
        assert w.update({"t": t, "goal_version": 1}) == []
    assert [e["why"] for e in w.update({"t": 2 + COOLDOWN_S, "goal_version": 1})] == ["agreed"]
    assert not w.withholds("heading", 1)
    view.next = 3.0
    w.update({"t": 100, "goal_version": 1})
    view.next = 0.1
    w.update({"t": 101, "goal_version": 1})
    view.next = 4.0                              # disagreeing again before the cooldown restarts it
    w.update({"t": 102, "goal_version": 1})
    assert w.open["largest"] == 4.0 and w._agreeing_since is None
    kinds = [e["kind"] for e in w.update({"t": 103, "goal_version": 2})]
    assert kinds == ["incident_closed", "incident_opened"] and w.open["goal_version"] == 2
    assert [i["why_closed"] for i in w.summary()["incidents"]] == ["agreed", "goal changed", None]


def test_withholding_lifts_when_no_model_can_be_asked():
    class Ladder:
        goal_version, no_reasoning = 1, set()

        def role_for(self, decision, reason):
            return "reasoning_model"
    ladder = Ladder()
    assert not models_exhausted(ladder, "heading")
    ladder.no_reasoning.add(("heading", 1))
    assert models_exhausted(ladder, "heading") and not models_exhausted(ladder, "report")
    view = View()
    view.next = 5.0
    w = Watch(view, {})
    w.update({"t": 0, "goal_version": 1})
    w.lift("heading", 1, 0, "no model can be asked")
    assert not w.withholds("heading", 1) and w.withholds("report", 1)
    assert w.summary()["incidents"][0]["lifted"] == [{"decision": "heading", "t": 0, "why": "no model can be asked"}]


# -- in flight ---------------------------------------------------------------------------------------------------------

def test_after_an_orbit_change_the_procedures_are_withheld_and_a_model_told_the_gap_flies(tmp_path, monkeypatch):
    rec, log, seen = fly(tmp_path, monkeypatch, scenario(delta=3.0), "on", watch=True, own_reports=True)
    incidents = rec["disagreement"]["incidents"]
    assert incidents and incidents[0]["opened_at"] > 60
    opened = incidents[0]["opened_at"]
    decisions = [e for e in log if e.get("kind") == "decision"]
    before = [d for d in decisions if d["t"] < opened]
    lifted = {x["decision"]: x["t"] for x in incidents[0]["lifted"]}
    during = [d for d in decisions if opened <= d["t"] < lifted.get(d["decision"], 10 ** 9)]
    assert all(d["source"] == "procedure" for d in before if d["decision"] == "heading")
    assert during and all(d.get("withheld") == 1 and d["source"] != "procedure" for d in during)
    assert any(d["source"] == "small_model" for d in during)
    # The small model's allowance runs out and the reasoning model is refused: withholding lifts, recorded, and the
    # procedures fly again rather than leave the decision with no method.
    for decision, t in lifted.items():
        assert any(d["decision"] == decision and d["t"] >= t and d["source"] == "procedure" for d in decisions)
    told = [ctx for decision, ctx in seen if "briefing_disagreement" in ctx]
    assert told and told[0]["briefing_disagreement"]["since_t"] == opened
    assert told[0]["briefing_disagreement"]["gap_nm"] > TANKER_EVIDENCE.tolerance
    pending = [e for e in log if e.get("kind") == "standing_order_withdrawal_pending"]
    withdrawn = [e for e in log if e.get("kind") == "standing_order_withdrawn"]
    reports = rec["outcome"]["reports"]
    if not reports or reports[0]["t"] >= opened:
        # The procedure's trigger was still armed when the incident opened: its withdrawal is pending from that second,
        # tried in it and accepted by the world; the model asked in that second is told only that it is pending.
        assert len(pending) == 1 and pending[0]["t"] == opened and pending[0]["armed"]["by"] == "procedure"
        assert len(withdrawn) == 1 and withdrawn[0]["t"] == opened
        first = [c["briefing_disagreement"]["standing_orders"] for c in told if c["t"] == opened]
        later = [c["briefing_disagreement"]["standing_orders"] for c in told if c["t"] > opened]
        assert all(o == {"report": {"state": "armed, withdrawal pending"}} for o in first)
        assert later and all(o == {"report": {"state": "withdrawn", "at": opened}} for o in later)
        # No triggered report while the report decision was withheld; after the lift the procedure may set one again.
        assert not any(r["via_trigger"] and opened < r["t"] < lifted.get("report", 10 ** 9) for r in reports)
    assert rec["own_reports"] == [{"kind": "own_report", "t": r["t"], "goal_version": r["goal_version"],
                                   "via_trigger": r["via_trigger"]} for r in reports]


def test_with_both_options_off_the_episode_is_flown_as_before(tmp_path, monkeypatch):
    rec, log, seen = fly(tmp_path, monkeypatch, scenario(delta=3.0), "off")
    assert "disagreement" not in rec and "own_reports" not in rec
    assert not any("withheld" in e for e in log) and not any("briefing_disagreement" in ctx for _, ctx in seen)
    assert not any(e.get("kind") in ("incident_opened", "standing_order_withdrawn", "own_report") for e in log)


def test_own_reports_stops_a_rearmed_trigger_reporting_twice(tmp_path, monkeypatch):
    off, _, _ = fly(tmp_path, monkeypatch, scenario(report_at=20.0, name="twice"), "twice", steps=400)
    on, _, _ = fly(tmp_path, monkeypatch, scenario(report_at=20.0, name="once"), "once", steps=400, own_reports=True)
    assert len(off["outcome"]["reports"]) >= 2
    assert len(on["outcome"]["reports"]) == 1 and on["outcome"]["reports"][0]["via_trigger"]
    assert "disagreement" not in on


def test_the_watch_needs_the_ladder(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="needs the flight ladder"):
        fly(tmp_path, monkeypatch, scenario(), "noladder", steps=5, watch=True, no_ladder=True)


def test_the_learner_is_told_the_gap_only_when_there_was_one():
    assert G.briefing_disagreement({}) == {}
    assert G.briefing_disagreement({"disagreement": {"tolerance": 1.0, "unit": "NM", "incidents": []}}) == {}
    got = G.briefing_disagreement({"disagreement": {"tolerance": 1.0, "unit": "NM", "incidents": [
        {"opened_at": 143, "largest": 5.704}, {"opened_at": 900, "largest": 7.2}]}})
    assert got == {"briefing_disagreement": {"first_at_s": 143, "largest_gap_nm": 7.2, "tolerance_nm": 1.0}}


# -- the repairs after Codex's TURN231 review ----------------------------------------------------------------------------

def test_lifts_belong_to_one_incident_and_a_second_without_a_gap_restarts_the_agreement():
    view = View()
    w = Watch(view, {})
    view.next = 5.0
    w.update({"t": 0, "goal_version": 1})
    w.lift("heading", 1, 0, "no model can be asked")
    view.next = 0.1
    for t in range(1, 2 + COOLDOWN_S):                                # agreeing from second 1, closed at 61
        w.update({"t": t, "goal_version": 1})
    assert w.open is None
    view.next = 5.0
    w.update({"t": 100, "goal_version": 1})
    assert w.open["incident"] == 2 and w.withholds("heading", 1)      # the new incident starts with no lifts
    view.next = 0.1
    for t in range(101, 131):
        w.update({"t": t, "goal_version": 1})
    view.next = None                                                   # nothing to measure for one second
    w.update({"t": 131, "goal_version": 1})
    view.next = 0.1
    for t in range(132, 132 + COOLDOWN_S):                            # agreeing again only from second 132
        assert w.update({"t": t, "goal_version": 1}) == []
    assert [e["why"] for e in w.update({"t": 132 + COOLDOWN_S, "goal_version": 1})] == ["agreed"]


def first_incident(tmp_path, monkeypatch, sc, name):
    rec, _, _ = fly(tmp_path, monkeypatch, sc, name, steps=200, watch=True)
    return rec["disagreement"]["incidents"][0]["opened_at"]


def test_a_refused_withdrawal_is_tried_again_until_the_world_accepts_it(tmp_path, monkeypatch):
    real = E.Episode._action_error
    refused = []

    def refuse_first_clearing(self, actions):
        if isinstance(actions, dict) and "report_trigger_nm" in actions and actions["report_trigger_nm"] is None \
                and not refused:
            refused.append(self.t)
            return "refused for the test"
        return real(self, actions)
    monkeypatch.setattr(E.Episode, "_action_error", refuse_first_clearing)
    rec, log, seen = fly(tmp_path, monkeypatch, scenario(delta=3.0), "refused", steps=200, watch=True)
    opened = rec["disagreement"]["incidents"][0]["opened_at"]
    kinds = [(e["kind"], e["t"]) for e in log if e.get("kind", "").startswith("standing_order")]
    assert kinds == [("standing_order_withdrawal_pending", opened), ("standing_order_withdrawal_attempted", opened),
                     ("standing_order_withdrawal_refused", opened), ("standing_order_withdrawal_attempted", opened + 1),
                     ("standing_order_withdrawn", opened + 1)]
    assert refused == [opened]
    told = {c["t"]: c["briefing_disagreement"]["standing_orders"] for _, c in seen if "briefing_disagreement" in c}
    assert all(o == {"report": {"state": "armed, withdrawal pending"}} for t, o in told.items() if t <= opened + 1)


def climbing_at(t, source):
    """A procedure that gives no output at second t, so its decision climbs to a model then."""
    body = source.split("def decide(ctx):\n", 1)
    return body[0] + "def decide(ctx):\n    if ctx.t == %d:\n        raise ValueError('no answer')\n" % t + body[1]


def test_replies_asked_before_the_incident_do_not_act_and_fresh_requests_follow(tmp_path, monkeypatch):
    opened = first_incident(tmp_path, monkeypatch, scenario(delta=3.0), "probe")
    rec, log, seen = fly(tmp_path, monkeypatch, scenario(delta=3.0), "early", steps=260, watch=True,
                         heading=climbing_at(opened - 1, AIM), report=climbing_at(opened - 1, TRIGGER),
                         report_trigger=True)
    decisions = [e for e in log if e.get("kind") == "decision"]
    asked = [(d, c) for d, c in seen if c["t"] == opened - 1]
    assert sorted(d for d, _ in asked) == ["heading", "report"] and not any("briefing_disagreement" in c for _, c in asked)
    aside = [d for d in decisions if (d.get("reply_checks") or {}).get("refused") == "asked_before_incident"]
    assert sorted(d["decision"] for d in aside) == ["heading", "report"]
    assert all(d["source"] == "none" and d["part"] is None and d["withheld"] == 1 for d in aside)
    assert all(d["reply_checks"]["asked_at"] == opened - 1 for d in aside)
    assert not any(e["kind"] == "escalated" for e in rec["ladder"])        # setting aside is no sign of failure
    fresh = [(d, c) for d, c in seen if c["t"] > opened - 1]
    assert {d for d, c in fresh if "briefing_disagreement" in c} == {"heading", "report"}
    charged = [r for r in (tmp_path / "early" / "watch-test" / "ledger.jsonl").read_text().splitlines()]
    assert len(charged) >= 4                                               # the set-aside calls stay charged


def test_a_goal_change_ends_the_incident_and_old_replies_stay_out_of_date(tmp_path, monkeypatch):
    opened = first_incident(tmp_path, monkeypatch, scenario(delta=3.0), "probe2")
    sc = scenario(delta=3.0, name="goal-change")
    sc = E.Scenario(episode_id=sc.episode_id, receiver=sc.receiver, wind=sc.wind, tankers=sc.tankers,
                    goals=((0, E.Goal(1, "SHELL 1", 10.0)), (opened + 1, E.Goal(2, "SHELL 2", 10.0))))
    rec, log, seen = fly(tmp_path, monkeypatch, sc, "goal", steps=260, watch=True,
                         heading=climbing_at(opened - 1, AIM), report=climbing_at(opened - 1, TRIGGER))
    incidents = rec["disagreement"]["incidents"]
    assert incidents[0]["opened_at"] == opened and incidents[0]["why_closed"] == "goal changed"
    assert all(i["goal_version"] == 1 for i in incidents)                 # SHELL 2 keeps its briefed orbit
    replies = [e for e in log if e.get("kind") == "model_reply" and e["t"] > opened]
    assert replies and all(r.get("refusal") == "out_of_date" for r in replies if r.get("goal_version") == 1)
    after = [e for e in log if e.get("kind") == "decision" and e["goal_version"] == 2]
    assert after and all("withheld" not in d for d in after)
    assert all(d["source"] == "procedure" for d in after if d["decision"] == "heading")


def test_with_the_reasoning_call_spent_on_a_set_aside_reply_the_procedure_flies_again(tmp_path, monkeypatch):
    opened = first_incident(tmp_path, monkeypatch, scenario(delta=3.0), "probe3")
    monkeypatch.setattr(M, "CALL_LIMITS", {**M.CALL_LIMITS, "small_model": 0, "reasoning_model": 1})
    rec, log, seen = fly(tmp_path, monkeypatch, scenario(delta=3.0), "spent", steps=260, watch=True,
                         heading=climbing_at(opened - 1, AIM))
    aside = [e for e in log if e.get("kind") == "decision"
             and (e.get("reply_checks") or {}).get("refused") == "asked_before_incident"]
    assert [d["decision"] for d in aside] == ["heading"] and aside[0]["reply_checks"]["source"] == "reasoning_model"
    lifts = rec["disagreement"]["incidents"][0]["lifted"]
    assert [x["decision"] for x in lifts if x["decision"] == "heading"] == ["heading"]
    lift_t = next(x["t"] for x in lifts if x["decision"] == "heading")
    assert lift_t > aside[0]["t"]
    later = [e for e in log if e.get("kind") == "decision" and e["decision"] == "heading" and e["t"] >= lift_t]
    assert later and all(d["source"] == "procedure" for d in later)
    rows = [json.loads(r) for r in (tmp_path / "spent" / "watch-test" / "ledger.jsonl").read_text().splitlines()]
    assert sum(r.get("kind") == "reasoning_model" for r in rows) == 1      # charged once, not refunded
