"""Fixed analytic report fixtures and event transitions; no acquisition/draw."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
import math
import pytest
from paper2.experiment2.monitor_labels import Original
from paper2.experiment2.report_evidence import ReportGoal; from paper2.experiment2.report_evidence import Trigger; from paper2.experiment2.report_evidence import advance_trigger; from paper2.experiment2.report_evidence import direct_report
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P

O = Original("fixed-report", 4, 180, *(["a" * 64] * 5), decision="report")
G = ReportGoal(4, "SHELL 1", 25)
TEXT = "Wedgetail, aircraft 2, 25 NM from SHELL 1"


@pytest.mark.parametrize("text,rng,rate,count,expected", [
    (TEXT, 25, -.1, 0, "correct_component"),
    (TEXT.replace("25 NM", "99 NM"), 25, -.1, 0, "incorrect_component"),
    (TEXT.replace("SHELL 1", "SHELL 2"), 25, -.1, 0, "incorrect_component"),
    (TEXT, 26, -.1, 0, "incorrect_component"),
    (TEXT, 25, .1, 0, "incorrect_component"),
    (TEXT, 25, -.1, 1, "incorrect_component"),
    (TEXT, 25, -.1, 2, "incorrect_component"),
    (TEXT, 25.25, -.1, 0, "unknown"),
    (TEXT, 25, 0, 0, "unknown"),
    (TEXT, float("nan"), -.1, 0, "unknown"),
    (TEXT, 25, float("inf"), 0, "unknown"),
])
def test_report_obligations(text, rng, rate, count, expected):
    p = direct_report(O, G, text, rng, rate, prior_count=count)
    assert p.verdict == expected and p.admissible is False
    assert json.loads(p.evidence_json)["prior_count"] == count


@pytest.mark.parametrize("text,rng,rate", [
    (TEXT, 25, -.1), (TEXT, 26, -.1), (TEXT, 25, .1), (TEXT, 25.25, -.1),
    ("prefix Wedgetail aircraft 2 SHELL 1 extra 025.005 suffix", 25, -.1),
    (TEXT.replace("25 NM", "99 NM"), 25, -.1),
])
def test_independent_component_matches_fixed_world_report_oracle(text, rng, rate):
    # Stationary tanker at (0,0); receiver at (rng,0) with ground x velocity rate.
    orbit = P.Orbit(0, -1, 1, 0, 0)
    heading = math.pi / 2 if rate >= 0 else -math.pi / 2
    r = P.ReceiverState(180, rng, 0, heading, heading, abs(rate), 0)
    sc = E.Scenario("fixed-report", r, (0, 0), (E.Tanker("SHELL 1", orbit),),
                    ((0, E.Goal(4, "SHELL 1", 25)),))
    ep = E.Episode(sc); ep.t = 180
    ep._report(text, 4, False)
    rec = ep.record.reports[0]
    p = direct_report(O, G, text, rng, rate)
    facts = json.loads(p.evidence_json)["predicates"]
    assert facts["content_ok"] == rec["content_ok"]
    assert facts["inbound"] == rec["inbound"]
    assert (p.verdict == "unknown") == bool(ep.record.flags)
    if not ep.record.flags:
        assert (p.verdict == "correct_component") == rec["correct"]


@pytest.mark.parametrize("o,g,text,count", [
    (replace(O, goal_version=5), G, TEXT, 0), (replace(O, input_sha256="x"), G, TEXT, 0),
    (replace(O, second=True), G, TEXT, 0), (replace(O, decision="heading"), G, TEXT, 0),
    (O, G, " ", 0), (O, G, "x" * 201, 0), (O, G, TEXT, True),
])
def test_invalid_original_or_public_field_refuses(o, g, text, count):
    assert direct_report(o, g, text, 25, -.1, count).verdict == "invalid"


def test_unknown_retained_flags_and_identity_and_no_goal_success_claim():
    p = direct_report(O, G, TEXT, 25, -.1, numerical_flags=("prefix join unresolved",))
    assert p.verdict == "unknown" and "prefix join unresolved" in p.reason
    q = direct_report(replace(O, second=181), G, TEXT, 25, -.1)
    assert q.fingerprint != direct_report(O, G, TEXT, 25, -.1).fingerprint
    assert "whole-goal witness still required" in q.reason and not q.admissible
    with pytest.raises(FrozenInstanceError): q.original.second = 181


def test_trigger_retention_firing_and_order_text_with_different_threshold():
    a = advance_trigger(G, 180, None, 30, {"goal_version": 4, "report_trigger_nm": 28})
    assert a.text is None and a.state.armed_at == 180
    b = advance_trigger(G, 181, a.state, None, {})
    assert b.state == a.state and b.text is None
    c = advance_trigger(G, 182, b.state, 28, {})
    assert c.via_trigger and c.text == TEXT and c.state is None
    assert advance_trigger(G, 183, c.state, 27, {}).text is None


def test_rearm_null_direct_and_old_goal_preserve_event_order():
    a = Trigger(25, 4, 179, 26)
    b = advance_trigger(G, 180, a, 24, {"goal_version": 4, "report_trigger_nm": 25})
    assert b.text is None and b.state.armed_at == 180
    assert advance_trigger(G, 181, b.state, 24, {}).text is None  # no strict decrease
    assert advance_trigger(G, 181, a, 24, {"goal_version": 4, "report_trigger_nm": None}).state is None
    d = advance_trigger(G, 180, a, 24, {"goal_version": 4, "report_trigger_nm": 25, "report": TEXT})
    assert d.text == TEXT and not d.via_trigger and d.state is None
    old = advance_trigger(replace(G, version=5), 180, a, 24, {})
    assert old.state == a and old.text is None
    refused = advance_trigger(G, 180, a, 24, {"goal_version": 3, "report": TEXT})
    assert refused.state is None and refused.text == TEXT and refused.via_trigger


def test_trigger_matches_world_with_delivered_measurement_not_truth():
    from paper2.telohelm.tests.test_episode_loop import scenario
    ep = E.Episode(scenario()); ep.t = 180
    ep._delivered = {ep.goal.tanker: {"x": 0, "y": 24, "measured_at": 170}}
    ep.trigger = E.Trigger(25, ep.goal.version, 26, 179)
    goal = ReportGoal(ep.goal.version, ep.goal.tanker, ep.goal.report_at_nm, ep.s.control_callsign)
    step = advance_trigger(goal, 180, Trigger(25, ep.goal.version, 179, 26), ep._observed_range(), {})
    ep._check_trigger()
    assert step.text == ep.record.reports[0]["text"] and step.via_trigger and ep.trigger is None
    assert ep.record.reports[0]["correct"] is False  # observed24 is not current true range


@pytest.mark.parametrize("cmd", [{"goal_version": 4, "report": "", "report_trigger_nm": 25},
    {"goal_version": 4, "report_trigger_nm": float("nan")}])
def test_refused_component_keeps_ordinary_existing_trigger_check(cmd):
    a = Trigger(25, 4, 179, 26)
    p = advance_trigger(G, 180, a, 24, cmd)
    assert p.state is None and p.text == TEXT and p.via_trigger and not p.admissible


def test_world_refused_old_action_still_checks_retained_trigger():
    from paper2.telohelm.tests.test_episode_loop import scenario
    ep = E.Episode(scenario()); ep.t = 180
    ep._delivered = {ep.goal.tanker: {"x": 0, "y": 24, "measured_at": 170}}
    ep.trigger = E.Trigger(25, ep.goal.version, 26, 179)
    ep._apply_actions({"goal_version": 99, "report": "refused text"})
    ep._check_trigger()
    assert len(ep.record.refusals) == 1 and len(ep.record.reports) == 1
    assert ep.record.reports[0]["via_trigger"] and ep.record.reports[0]["text"] != "refused text"


def test_full_action_must_be_admitted_and_projected_explicitly():
    with pytest.raises(ValueError):
        advance_trigger(G, 180, None, 25, {"goal_version": 4, "heading_deg": 360, "report": TEXT})
    with pytest.raises(ValueError): advance_trigger(G, 180, None, True, {})


def test_component_cannot_be_promoted_even_by_trusting_verifier(tmp_path):
    from .test_case_store import case
    from telohelm.qualification.store import CaseStore; from telohelm.qualification.store import StoreError
    c = case(decision="report")
    c["label"]["evidence"] = asdict(direct_report(O, G, TEXT, 25, -.1))
    s = CaseStore(tmp_path / "store", verifier=lambda c: True, verifier_name="invented")
    with pytest.raises(StoreError, match="unverified proposal"):
        s.admit(c)
    assert s.cases() == []
