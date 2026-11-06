"""Hand-constructed original-clock and whole-goal controls; no generator draws."""
from dataclasses import replace
import hashlib
import json
import math

import pytest
import paper2.experiment2.original_replay as R
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P

TEXT = "Wedgetail, aircraft 2, 1 NM from SHELL 1"


def world(*, second=10, deadline=18, delay=0, delta=0, changes=(), report=None):
    receiver = P.ReceiverState(0., 0., 0., math.pi/2, math.pi/2, .1, .02)
    tanker = E.Tanker("SHELL 1", P.Orbit(2., -1., 1., 0. if delta == 0 else .05, 0., delta=delta))
    ep = E.Episode(E.Scenario("invented-original", receiver, (0., 0.), (tanker,),
                   ((0, E.Goal(1, "SHELL 1", 1., deadline)), *changes), delay_s=delay))
    initial = ep.reset()
    bundle = initial["bundle"]
    for _ in range(second):
        bundle = ep.step({})
        assert bundle is not None
    if report is not None:
        ep._report(report, 1, False)
    return ep, bundle


def original(ep, bundle, decision="heading"):
    identity = R.Identity(ep.s.episode_id, "invented-unit", 4420, "diagnostic", decision,
                          "action", ep.t, len(ep.record.actions), ep.goal.version, 0, 0, None)
    return R.capture(ep, identity, context={"t": ep.t, "goal_version": ep.goal.version},
                     input_bundle=bundle, issue={"request_id": "fixed", "issued_second": ep.t,
                                               "issued_step": len(ep.record.actions)}, actor_pending={})


def run(c, candidate=None, continuation=None, **kw):
    return R.replay(c, {"goal_version": 1, "heading_deg": 90, "report": TEXT} if candidate is None else candidate,
                    [{}] * 10 if continuation is None else continuation, expected_capture_sha256=c.sha256, **kw)


def body(e):
    return json.loads(e.body)


def test_shifted_clock_report_then_join_at_inclusive_deadline():
    ep, bundle = world()
    c = original(ep, bundle)
    e = run(c, continuation=[{}]*7)
    b = body(e)
    assert e.disposition == "nominal_upper" and e.upper_elapsed_s == 8
    assert b["terminal"]["t"] == 18 and b["outcome"]["reports_under_goal"] == 1
    assert len(b["steps"]) == 8 and b["steps"][0]["second"] == 10
    assert b["steps"][-1]["join"]["decision"] == "join"
    assert not e.admissible and e.qualification.startswith("conditional")
    assert ep.t == 10 and not ep.record.reports  # original untouched


def test_independent_world_trace_matches_each_surviving_bundle_and_ending():
    ep, bundle = world(delay=2)
    c = original(ep, bundle)
    e = run(c, continuation=[{}]*7)
    for row in body(e)["steps"]:
        got = ep.step(row["action"])
        assert got == row.get("bundle")
        assert json.loads(R.encode(R._trace_state(ep))) == row["after"]
    assert ep.outcome() == body(e)["outcome"]


@pytest.mark.parametrize("prior, candidate", [(None, {}), ("bad report", {}), (TEXT, {"goal_version":1,"report":TEXT})])
def test_contact_without_exactly_one_correct_report_never_has_upper(prior, candidate):
    ep, bundle = world(report=prior)
    e = run(original(ep,bundle), candidate=candidate, continuation=[{}]*7)
    assert e.upper_elapsed_s is None and e.disposition == "unknown"


def test_prior_correct_report_is_retained_without_repeating_it():
    ep,bundle=world(report=TEXT)
    e=run(original(ep,bundle),candidate={},continuation=[{}]*7)
    assert e.upper_elapsed_s == 8 and body(e)["outcome"]["reports_under_goal"] == 1


def test_join_before_report_cannot_be_repaired_after_ending():
    ep,bundle=world(second=17)
    e=run(original(ep,bundle),candidate={},continuation=[{"goal_version":1,"report":TEXT}])
    assert e.upper_elapsed_s is None and len(body(e)["steps"]) == 1
    assert body(e)["terminal"]["t"] == 18


def test_deadline_precedes_a_later_join():
    ep,bundle=world(deadline=17)
    e=run(original(ep,bundle),continuation=[{}]*7)
    assert e.upper_elapsed_s is None and body(e)["outcome"]["end_reason"] == "deadline"


def test_plan_cap_and_exhaustion_are_unknown_or_invalid_never_infinite():
    ep,bundle=world()
    c=original(ep,bundle)
    short=run(c,continuation=[{}])
    assert short.disposition == "unknown" and short.upper_elapsed_s is None
    excessive=run(c,limits=R.Limits(max_steps=2))
    assert excessive.disposition == "invalid" and not body(excessive)["steps"]


def test_combined_query_cap_retains_attempted_step_and_partial_work():
    ep,bundle=world()
    e=run(original(ep,bundle),limits=R.Limits(total_distance_queries=2))
    b=body(e)
    assert e.disposition == "unknown" and "work cap" in e.reason
    assert len(b["steps"]) == 1 and len(b["distance_queries"]) == 2
    assert b["steps"][0]["accepted"] is True and b["terminal"]["t"] == 10
    assert len(b["terminal"]["record"]["reports"]) == 1


def test_invalid_full_action_is_atomic_but_retained_trigger_still_checks():
    ep,bundle=world(second=9)
    ep._apply_actions({"goal_version":1,"report_trigger_nm":1.1})
    bundle=ep.step({})  # armed_at9; current second10 is the first decreasing threshold
    assert not ep.record.reports
    c=original(ep,bundle)
    e=run(c,candidate={"goal_version":1,"heading_deg":360,"report_trigger_nm":None})
    b=body(e)
    assert e.disposition == "invalid" and e.upper_elapsed_s is None
    assert b["steps"][0]["accepted"] is False
    assert b["terminal"]["trigger"] is None
    assert len(b["terminal"]["record"]["reports"]) == 1
    assert b["terminal"]["record"]["reports"][0]["via_trigger"]


def test_trigger_uses_ordered_text_and_fires_once_on_later_second():
    ep,bundle=world(second=9)
    e=run(original(ep,bundle,decision="report"),candidate={"goal_version":1,"report_trigger_nm":1.05},
          continuation=[{}]*8)
    b=body(e)
    assert e.upper_elapsed_s == 9
    report=b["terminal"]["record"]["reports"][0]
    assert report["t"] == 10 and report["text"] == TEXT and report["via_trigger"]
    assert b["steps"][0]["after"]["trigger"]["armed_at"] == 9


def test_rearming_prevents_firing_and_null_or_direct_report_clears():
    ep,bundle=world(second=9)
    c=original(ep,bundle,decision="report")
    arm={"goal_version":1,"report_trigger_nm":1.05}
    e=run(c,candidate=arm,continuation=[arm]*8)
    assert e.upper_elapsed_s is None and not body(e)["terminal"]["record"]["reports"]
    e=run(c,candidate=arm,continuation=[{"goal_version":1,"report_trigger_nm":None}]+[{}]*7)
    assert not body(e)["terminal"]["record"]["reports"]
    e=run(c,candidate=arm,continuation=[{"goal_version":1,"report":TEXT,"report_trigger_nm":1.05}]+[{}]*7)
    assert e.upper_elapsed_s == 9 and len(body(e)["terminal"]["record"]["reports"]) == 1
    assert body(e)["steps"][1]["after"]["trigger"] is None


def test_old_goal_trigger_is_not_retagged():
    ep,bundle=world()
    ep.trigger=E.Trigger(2.,0,2.,5)
    e=run(original(ep,bundle),candidate={},continuation=[{}]*7)
    assert e.upper_elapsed_s is None and not body(e)["terminal"]["record"]["reports"]
    assert body(e)["terminal"]["trigger"]["goal_version"] == 0


def test_goal_change_in_plan_is_unknown_and_original_schedule_retained():
    ep,bundle=world(changes=((12,E.Goal(2,"SHELL 1",2.,30)),))
    e=run(original(ep,bundle))
    assert "goal changes" in e.reason and not body(e)["steps"]
    assert body(e)["original"]["scenario"]["goals"][1][0] == 12


def test_change_after_terminal_plan_does_not_erase_earlier_completion():
    ep,bundle=world(changes=((25,E.Goal(2,"SHELL 1",2.,30)),))
    e=run(original(ep,bundle),continuation=[{}]*7)
    assert e.upper_elapsed_s == 8


def test_invalid_changing_radius_preserved_and_explicitly_unknown():
    ep,bundle=world(second=0,delta=.01,deadline=30)
    # This otherwise valid fixture is made unsupported by a negative ending radius.
    ep.s=replace(ep.s,tankers=(replace(ep.s.tankers[0],orbit=replace(ep.s.tankers[0].orbit,delta=-2.)),))
    ep.tankers={k.callsign:k for k in ep.s.tankers}
    e=run(original(ep,bundle))
    assert "changing motion" in e.reason and not body(e)["steps"]
    assert body(e)["original"]["scenario"]["tankers"][0]["orbit"]["delta"] == -2.


def test_sensitive_prior_report_or_join_cannot_be_erased():
    ep,bundle=world()
    ep.record.flags.append({"t":9,"kind":"invented unresolved original"})
    e=run(original(ep,bundle))
    assert e.upper_elapsed_s is None and not body(e)["steps"]
    assert body(e)["terminal"]["record"]["flags"]


def test_report_boundary_remains_unknown_even_if_nominally_correct():
    ep,bundle=world(second=0,deadline=30)
    # Fixed original 1.25 NM from stationary tanker, inbound .1 NM/s.
    ep.receiver=replace(ep.receiver,x=.75)
    e=run(original(ep,bundle),continuation=[{}]*10)
    assert e.disposition == "unknown" and "unsettled" in e.reason
    assert body(e)["terminal"]["record"]["flags"][0]["kind"] == "report range near tolerance"


def test_delivered_measurement_queue_help_and_actor_pending_survive_capture():
    ep,bundle=world(delay=3)
    ep._apply_actions({"goal_version":1,"help":{"reason":"missing_tool","text":"fixed"}})
    c=original(ep,bundle)
    saved=json.loads(c.body)
    restored=R._restore(saved)
    assert json.loads(R.encode(R._state(restored))) == saved["state"]
    assert len(saved["state"]["queue"]) == 3 and saved["state"]["delivered"]["SHELL 1"]["measured_at"] == 7
    assert saved["state"]["pending_help"][0][0] == 40


def test_canonical_capture_is_detached_and_digest_is_not_authority():
    ep,bundle=world()
    c=original(ep,bundle)
    ep.queue.clear(); bundle["own"]["x"]=999
    assert json.loads(c.body)["input_bundle"]["own"]["x"] != 999
    changed=json.loads(c.body);changed["state"]["receiver"]["x"]=100
    data=R.encode(changed);forged=R.Capture(data,hashlib.sha256(data).hexdigest())
    e=R.replay(forged,{},[],expected_capture_sha256=c.sha256)
    assert e.disposition == "invalid" and "independently retained" in e.reason


def test_source_or_settings_conflict_refuses_before_any_propagation(monkeypatch):
    ep,bundle=world();c=original(ep,bundle)
    import telohelm.taskpacks.tanker.dynamics as D
    monkeypatch.setattr(D,"TIE_BAND",1e-8)  # the band moved with the public dynamics
    e=run(c)
    assert e.disposition == "invalid" and "settings conflict" in e.reason
    assert not body(e)["steps"]


def test_fault_retains_completed_prefix_and_error_queries(monkeypatch):
    ep,bundle=world();c=original(ep,bundle)
    true=P.join_assessment
    def fault(*a,**k):
        if a[1] == 11.: raise RuntimeError("invented second interval failure")
        return true(*a,**k)
    monkeypatch.setattr(P,"join_assessment",fault)
    e=run(c)
    assert e.disposition == "unknown" and "second interval failure" in e.reason
    assert len(body(e)["steps"]) == 2 and body(e)["steps"][0]["after"]["t"] == 11


@pytest.mark.parametrize("field,value", [("second",True),("action_step",100),("goal_version",2),("episode","other")])
def test_conflicting_original_identity_refused(field,value):
    ep,bundle=world()
    i=R.Identity(ep.s.episode_id,"unit",4420,"diagnostic","heading","action",ep.t,len(ep.record.actions),1,0,0,None)
    with pytest.raises(ValueError):
        R.capture(ep,replace(i,**{field:value}),context={"t":ep.t,"goal_version":1},
                  input_bundle=bundle,issue={"fixed":True},actor_pending={})


def test_closed_original_episode_cannot_have_new_decision():
    ep,bundle=world(second=17)
    ep.step({})
    with pytest.raises(ValueError): original(ep,{"t":ep.t})


def test_nonfinite_original_is_refused_without_replacing_input():
    ep,bundle=world();ep.receiver=replace(ep.receiver,x=float("nan"))
    with pytest.raises(ValueError): original(ep,bundle)


def test_bool_or_oversized_work_limits_refused():
    ep,bundle=world();c=original(ep,bundle)
    with pytest.raises(ValueError):run(c,limits=R.Limits(max_steps=True))
    with pytest.raises(ValueError):run(c,limits=R.Limits(total_distance_queries=32769))


def test_reconstruction_checks_conclusion_plan_and_original_not_caller_boolean():
    ep,bundle=world();c=original(ep,bundle);e=run(c)
    assert R.reproduce(c,e,expected_capture_sha256=c.sha256)
    assert not R.reproduce(c,replace(e,upper_elapsed_s=1),expected_capture_sha256=c.sha256)
    assert not R.reproduce(c,replace(e,admissible=True),expected_capture_sha256=c.sha256)
    changed=body(e);changed["candidate"]["report"]="wrong"
    data=R.encode(changed)
    forged=replace(e,body=data,sha256=hashlib.sha256(data).hexdigest())
    assert not R.reproduce(c,forged,expected_capture_sha256=c.sha256)
    assert not R.reproduce(c,e,expected_capture_sha256="0"*64)


def test_nonfinite_query_retained_as_failure_not_erased_by_serialisation(monkeypatch):
    ep,bundle=world();c=original(ep,bundle)
    monkeypatch.setattr(E.Episode,"_true_range",lambda *a:float("inf"))
    e=run(c)
    assert e.disposition == "unknown" and "nonfinite distance" in e.reason
    assert body(e)["distance_queries"][0]["range_nm"] == {"nonfinite_float":"inf"}
    assert len(body(e)["steps"]) == 1


def test_join_work_cap_unknown_preserves_interval_enclosure(monkeypatch):
    ep,bundle=world();c=original(ep,bundle)
    monkeypatch.setattr(P,"join_assessment",lambda *a,**k:("unresolved",P.Approach(.3,.2,10.5,3,"unresolved")))
    e=run(c)
    assert e.upper_elapsed_s is None and "unsettled" in e.reason
    assert body(e)["steps"][0]["join"]["enclosure"]["evaluations"] == 3
    assert len(body(e)["outcome"]["alternatives"]) == 1


def test_pending_help_after_join_is_retained_but_never_delivered():
    ep,bundle=world()
    ep._apply_actions({"goal_version":1,"help":{"reason":"missing_tool","text":"fixed"}})
    c=original(ep,bundle);e=run(c,continuation=[{}]*7)
    assert e.upper_elapsed_s == 8
    b=body(e)
    assert b["terminal"]["pending_help"][0][0] == 40
    assert not any(r.get("help_reply") for r in b["terminal"]["delivered"].values())


def test_full_invocation_and_retained_report_clock_are_detached():
    ep,bundle=world();c=original(ep,bundle,decision="report")
    candidate={"goal_version":1,"report":TEXT};plan=[{}]*7
    e=run(c,candidate=candidate,continuation=plan)
    candidate["report"]="changed";plan[0]["heading_deg"]=180
    b=body(e)
    assert b["candidate"]["report"] == TEXT and b["continuation"][0] == {}
    assert b["terminal"]["record"]["reports"][0]["t"] == 10
    assert b["original"]["identity"]["second"] == 10 and e.upper_elapsed_s == 8


@pytest.mark.parametrize("data", [b"not JSON", b"[]"])
def test_malformed_capture_refused_even_if_its_digest_was_supplied(data):
    c=R.Capture(data,hashlib.sha256(data).hexdigest())
    e=run(c)
    assert e.disposition == "invalid" and e.upper_elapsed_s is None


@pytest.mark.parametrize("where,field", [("state","pending_help"),("record","reports"),("root","issue")])
def test_missing_original_field_is_not_silently_defaulted(where,field):
    ep,bundle=world();saved=json.loads(original(ep,bundle).body)
    target=saved if where == "root" else saved["state"]["record"] if where == "record" else saved["state"]
    del target[field]
    data=R.encode(saved);c=R.Capture(data,hashlib.sha256(data).hexdigest())
    e=run(c)
    assert e.disposition == "invalid" and "original" in e.reason


def changing_world():
    # A tiny, finite change during this invented complete episode. Speed and
    # radial rate satisfy the reviewed domain; no generator or reference pilot.
    receiver=P.ReceiverState(0.,0.,0.,math.pi/2,math.pi/2,.1,.02)
    tanker=E.Tanker("SHELL 1",P.Orbit(2.,-1.,1.,.002,0.,delta=.005,start=2.,duration=10.))
    ep=E.Episode(E.Scenario("invented-changing",receiver,(0.,0.),(tanker,),((0,E.Goal(1,"SHELL 1",1.,30)),)))
    bundle=ep.reset()["bundle"]
    for _ in range(10):bundle=ep.step({})
    return ep,bundle


def test_changing_replay_finishes_with_retained_quadrature_and_original_clock():
    ep,bundle=changing_world();c=original(ep,bundle)
    e=run(c,continuation=[{}]*12)
    assert e.disposition=="nominal_upper" and e.upper_elapsed_s==8
    b=body(e)
    assert b["integrand_evaluations"]>0
    assert any(q["kind"]=="velocity" and q["integrand_evaluations"]>0 for q in b["tanker_queries"])
    assert any(q["kind"]=="position" and q["second"]==11 for q in b["tanker_queries"])
    assert all(q["state"]=="ok" for q in b["tanker_queries"])
    assert not e.admissible
    for row in b["steps"]:
        ep.step(row["action"])
        assert ep.t==row["after"]["t"]
        assert ep.receiver==P.ReceiverState(**row["after"]["receiver"])
    assert ep.outcome()["goal_met"] and ep.record.ended_at==18
    expected=ep.record.reports[0];actual=b["terminal"]["record"]["reports"][0]
    assert actual["true_range"]==pytest.approx(expected["true_range"],abs=1e-12)
    assert actual["range_derivative_nm_s"]==pytest.approx(expected["range_derivative_nm_s"],abs=1e-12)


@pytest.mark.parametrize("limits,expected",[(R.Limits(quadrature_per_query=1),"quadrature integrand"),
                                          (R.Limits(total_integrand_evaluations=2),"quadrature integrand"),
                                          (R.Limits(total_tanker_queries=1),"tanker query")])
def test_changing_work_caps_retain_partial_query_work(limits,expected):
    ep,bundle=changing_world();e=run(original(ep,bundle),limits=limits)
    assert e.disposition=="unknown" and "work cap" in e.reason and expected in e.reason
    b=body(e)
    assert len(b["steps"])==1 and b["tanker_queries"]
    assert b["integrand_evaluations"]<=limits.total_integrand_evaluations
    assert len(b["tanker_queries"])<=limits.total_tanker_queries


def test_trace_threshold_stops_after_retaining_previous_step():
    ep,bundle=world();e=run(original(ep,bundle),limits=R.Limits(trace_stop_bytes=6000))
    assert e.disposition=="unknown" and "trace stop byte" in e.reason
    assert body(e)["steps"] and body(e)["steps"][0]["after"]["t"]==11


def test_input_budget_refuses_before_propagation():
    ep,bundle=world();e=run(original(ep,bundle),limits=R.Limits(input_bytes=1024))
    assert e.disposition=="invalid" and "input byte" in e.reason and not body(e)["steps"]


def test_changing_backend_nonconvergence_retains_failed_query(monkeypatch):
    ep,bundle=changing_world();c=original(ep,bundle)
    import scipy.integrate
    monkeypatch.setattr(scipy.integrate,"quad",lambda *a,**k:(0.,0.,{},"invented nonconvergence"))
    e=run(c)
    assert e.disposition=="unknown" and "did not converge" in e.reason
    assert body(e)["tanker_queries"][0]["state"]=="error"
    assert body(e)["tanker_queries"][0]["point"] is None


def test_adapter_domain_error_is_retained_before_propagation():
    ep,bundle=world()
    # A constant orbit ignores change duration in the production formula; the
    # stricter motion adapter refuses the unsupported zero duration explicitly.
    ep.s=replace(ep.s,tankers=(replace(ep.s.tankers[0],orbit=replace(ep.s.tankers[0].orbit,duration=0.)),))
    ep.tankers={k.callsign:k for k in ep.s.tankers}
    e=run(original(ep,bundle))
    assert e.disposition=="unknown" and "motion adapter binding" in e.reason
    assert not body(e)["steps"]


def test_compact_trace_reconstructs_all_action_report_and_flag_history():
    ep,bundle=changing_world();e=run(original(ep,bundle),continuation=[{}]*12);b=body(e)
    record=b["original"]["state"]["record"]
    alternatives=b["original"]["state"]["join_alternatives"]
    for row in b["steps"]:
        assert R.digest(record)==row["before"]["record_sha256"]
        for key,delta in row["record_delta"].items():record[key].extend(delta)
        record.update(row["after"]["record_status"])
        alternatives.extend(row["join_alternatives_delta"])
        assert R.digest(record)==row["after"]["record_sha256"]
        assert R.digest(alternatives)==row["after"]["join_alternatives_sha256"]
    assert record==b["terminal"]["record"]


def test_retained_original_body_is_not_mutated_by_replay_record_aliases():
    ep,bundle=world(delay=2);c=original(ep,bundle);e=run(c)
    assert R.encode(body(e)["original"])==c.body
    assert R.digest(body(e)["original"])==c.sha256
    saved=json.loads(c.body);before=R.encode(saved);restored=R._restore(saved)
    restored.record.actions.append({"invented":True});restored._delivered.clear()
    assert R.encode(saved)==before
