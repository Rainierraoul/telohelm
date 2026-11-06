"""Typed original public-input/issue controls, fixed invented worlds only."""
from dataclasses import asdict,replace
import hashlib
import json
import pytest
import paper2.experiment2.original_binding as B; import paper2.experiment2.original_replay as R
from telohelm.runtime.context import ContextBuilder
from telohelm.runtime.delivery import Pending
from .test_original_replay_codex import world; from .test_original_replay_codex import TEXT


def fixture(role="procedure",pending=False,zero_latency=False):
    ep,_=world(second=0)
    start=ep.reset();briefing=start["briefing"];prefix=[start["bundle"]]
    cb=ContextBuilder(briefing);contexts=[cb.update(prefix[0])]
    for _ in range(10):
        prefix.append(ep.step({}));contexts.append(cb.update(prefix[-1]))
    model=role in ("small_model","reasoning_model")
    issued=10 if not model or zero_latency else 8
    source={"role":role,"actual_payload":"fixed invented source"}
    issue=B.Issue("req1" if model else None,role,"heading",1,issued,issued,10,
                  "post-issuance" if model and zero_latency else "reply" if model else "decisions",
                  R.digest(contexts[issued]),R.digest(source))
    call=Pending(1,"heading",role,"req1",1,issued,issued,10,issue.context_sha256,"answered",
                 '{"heading_deg":90}',False) if model else None
    ps=(Pending(2,"report","small_model","future",1,9,9,11,R.digest(contexts[9]),"answered","{}",False),) if pending else ()
    identity=R.Identity(ep.s.episode_id,"invented-unit",4420,"diagnostic","heading","action",10,10,1,0,0,None)
    c=R.capture(ep,identity,context=contexts[-1],input_bundle=prefix[-1],issue=asdict(issue),
                actor_pending={"calls":[asdict(p) for p in ps],
                               "public_history_sha256":R.digest({"briefing":briefing,"bundles":prefix})})
    kwargs={"briefing":briefing,"bundles":prefix,"issued_context":contexts[issued],"source_body":source,
            "issued_call":call,"pending":ps,"expected_capture_sha256":c.sha256}
    return c,issue,kwargs


@pytest.mark.parametrize("role",["procedure","small_model","reasoning_model","none"])
def test_all_source_roles_bind_and_replay_same_original_clock(role):
    c,i,kw=fixture(role);b=B.bind(c,i,**kw)
    out=B.replay_bound(b,{"goal_version":1,"heading_deg":90,"report":TEXT},[{}]*7,expected_binding_sha256=b.sha256)
    assert out["evidence"]["upper_elapsed_s"]==8 and not out["admissible"]
    assert out["binding"]["issue"]["issued_second"]==(8 if "model" in role else 10)
    assert out["binding"]["original"]["identity"]["second"]==10


def test_zero_latency_has_explicit_post_issuance_subphase():
    c,i,kw=fixture("small_model",zero_latency=True);b=B.bind(c,i,**kw)
    assert json.loads(b.body)["issue"]["subphase"]=="post-issuance"
    with pytest.raises(ValueError):B.bind(c,replace(i,subphase="reply"),**kw)


@pytest.mark.parametrize("field,value",[("goal_version",2),("decision","report"),("issued_second",11),
                                      ("issued_step",11),("available_second",9),("issued_second",True),
                                      ("context_sha256","0"*64),("source_sha256","0"*64),("request_id","other")])
def test_conflicting_issue_never_retags_the_original(field,value):
    c,i,kw=fixture("small_model")
    with pytest.raises(ValueError):B.bind(c,replace(i,**{field:value}),**kw)


def test_input_context_source_and_capture_mutations_are_refused():
    for changed in ("bundles","issued_context","source_body","expected_capture_sha256"):
        c,i,kw=fixture("small_model")
        if changed=="bundles":kw[changed][3]["own"]["x"]=999
        elif changed=="issued_context":kw[changed]={**kw[changed],"x":999}
        elif changed=="source_body":kw[changed]={"different":True}
        else:kw[changed]="0"*64
        with pytest.raises(ValueError):B.bind(c,i,**kw)


def test_missing_or_noncontiguous_public_history_refused():
    c,i,kw=fixture()
    with pytest.raises(ValueError):B.bind(c,i,**{**kw,"bundles":kw["bundles"][1:]})
    kw["bundles"][3]["t"]=4
    with pytest.raises(ValueError):B.bind(c,i,**kw)


def test_pending_keeps_issuance_context_and_late_availability():
    c,i,kw=fixture("small_model",pending=True);b=B.bind(c,i,**kw)
    saved=json.loads(b.body)
    assert saved["pending"]["calls"][0]["available_second"]==11
    bad=replace(kw["pending"][0],available_second=10)
    with pytest.raises(ValueError):B.bind(c,i,**{**kw,"pending":(bad,)})
    with pytest.raises(ValueError):B.bind(c,i,**{**kw,"pending":(kw["pending"][0],kw["pending"][0])})


def test_model_role_or_status_conflict_refused():
    c,i,kw=fixture("small_model")
    for p in (None,replace(kw["issued_call"],role="reasoning_model"),replace(kw["issued_call"],status="timeout")):
        with pytest.raises(ValueError):B.bind(c,i,**{**kw,"issued_call":p})


def test_binding_is_detached_and_forged_hash_cannot_replace_original_anchor():
    c,i,kw=fixture();b=B.bind(c,i,**kw)
    kw["bundles"][0]["own"]["x"]=900
    assert json.loads(b.body)["bundles"][0]["own"]["x"]!=900
    saved=json.loads(b.body);saved["issue"]["issued_second"]=9
    data=R.encode(saved);forged=B.Binding(data,hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError):B.replay_bound(forged,{},[],expected_binding_sha256=b.sha256)


def test_changed_context_builder_settings_refused(monkeypatch):
    c,i,kw=fixture();b=B.bind(c,i,**kw)
    monkeypatch.setattr(B.C,"HISTORY",59)
    with pytest.raises(ValueError):B.replay_bound(b,{},[],expected_binding_sha256=b.sha256)


def test_typed_payload_budget_counts_history_not_just_world_snapshot():
    c,i,kw=fixture();b=B.bind(c,i,**kw)
    with pytest.raises(ValueError):
        B.replay_bound(b,{},[],expected_binding_sha256=b.sha256,limits=R.Limits(input_bytes=1024))


def test_pending_conflicting_issue_step_or_context_refused():
    c,i,kw=fixture("small_model",pending=True)
    for p in (replace(kw["pending"][0],issued_second=11),replace(kw["pending"][0],context_sha256="0"*64)):
        with pytest.raises(ValueError):B.bind(c,i,**{**kw,"pending":(p,)})


def complete_fixture(changing=False):
    from .test_original_replay_codex import changing_world
    import telohelm.taskpacks.tanker.world.episode as E
    template=changing_world()[0] if changing else world(second=0)[0]
    ep=E.Episode(template.s);start=ep.reset();cb=ContextBuilder(start["briefing"])
    context=cb.update(start["bundle"]);source={"role":"procedure","source":"fixed hand-authored finite unit controls"}
    issue=B.Issue(None,"procedure","heading",1,0,0,0,"decisions",R.digest(context),R.digest(source))
    prefix=[start["bundle"]]
    identity=R.Identity(ep.s.episode_id,"invented-unit",4420,"diagnostic","heading","action",0,0,1,0,0,None)
    c=R.capture(ep,identity,context=context,input_bundle=start["bundle"],issue=asdict(issue),
                actor_pending={"calls":[],"public_history_sha256":R.digest({"briefing":start["briefing"],"bundles":prefix})})
    b=B.bind(c,issue,briefing=start["briefing"],bundles=prefix,issued_context=context,source_body=source,
             expected_capture_sha256=c.sha256)
    actions=[{} for _ in range(18)];actions[0]={"goal_version":1,"heading_deg":90};actions[10]={"goal_version":1,"report":TEXT}
    return ep,b,actions


@pytest.mark.parametrize("changing",[False,True])
def test_complete_typed_fixture_from_reset_matches_world_ending(changing):
    ep,b,actions=complete_fixture(changing)
    out=B.replay_bound(b,actions[0],actions[1:],expected_binding_sha256=b.sha256)
    assert out["disposition"]=="nominal_upper" and out["upper_elapsed_s"]==18
    for action in actions:
        if ep.done:break
        ep.step(action)
    assert ep.record.ended_at==18 and ep.outcome()["goal_met"]
    assert out["evidence"]["integrand_evaluations"]>0 if changing else out["evidence"]["integrand_evaluations"]==0


def test_caller_cannot_upgrade_binding_qualification_or_admissibility():
    c,i,kw=fixture();b=B.bind(c,i,**kw)
    for forged in (replace(b,qualification="certified"),replace(b,admissible=True)):
        with pytest.raises(ValueError):B.replay_bound(forged,{},[],expected_binding_sha256=b.sha256)
