"""TURN53 invented-input repair checks; no world trajectory or model execution."""
import hashlib
import json
import sys
import pytest
import telohelm.runtime.broker as B
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify


def msg():
    return {"capability": "predict", "args": {"goal_version": 0,
        "start": {"x_nm": 0, "y_nm": 1, "heading_deg": 50, "tas_nm_s": .1},
        "headings_deg": [50, 51], "wind_nm_s": [.01, 0],
        "orbit": {"cx_nm": 20, "cy_nm": 30, "radius_nm": 5, "speed_nm_s": .08, "angle_rad": 1}}}


def valid(a):
    return {"positions_nm": [[1, 2] for _ in a["headings_deg"]],
            "closest_approach_nm": 1, "closest_at_s": 0}


def read(path):
    return [json.loads(l)["payload"] for l in path.read_text().splitlines()[1:]]


@pytest.fixture
def make(tmp_path):
    count = 0
    def f(tool=valid, replay=False, clock=None):
        nonlocal count
        count += 1
        path=tmp_path/str(count); head=tmp_path/(str(count)+".heads")
        ticks=iter(range(1000)); ledger=Ledger()
        b=B.Broker("invented53",TrustedLog(path,head,"turn53"),ledger,tool,replay=replay,
                   clock=clock or (lambda: float(next(ticks))))
        return b,ledger,path,head
    return f


@pytest.mark.parametrize("where", ["heading", "wind", "orbit", "speed", "position"])
def test_large_integer_refused_at_each_numeric_location(make,where):
    obj=msg(); a=obj["args"]; huge=10**400
    if where=="heading": a["headings_deg"][0]=huge
    elif where=="wind": a["wind_nm_s"][0]=huge
    elif where=="orbit": a["orbit"]["cx_nm"]=huge
    elif where=="speed": a["start"]["tas_nm_s"]=huge
    else: a["start"]["x_nm"]=huge
    b,l,_,_=make()
    with b.open_call("heading") as c:
        assert json.loads(c.request(json.dumps(obj)))=={"ok":False,"refusal":"malformed"}
    assert b.predictions_used==0 and l.calls==[]


@pytest.mark.parametrize("field,bad", [
    ("positions_nm", [[True,0],[0,0]]), ("positions_nm",[[0,float("nan")],[0,0]]),
    ("positions_nm",[[0,0]]), ("positions_nm",[[0,0],[0,0],[0,0]]),
    ("closest_approach_nm",-1), ("closest_approach_nm",float("inf")),
    ("closest_at_s",-1), ("closest_at_s",3), ("closest_at_s",True),
    ("positions_nm",[[10**400,0],[0,0]])])
def test_bad_result_fields_fail_closed_count_once(make,field,bad):
    result=valid(msg()["args"]); result[field]=bad
    b,l,path,head=make(lambda a: result)
    with b.open_call("heading") as c:
        assert json.loads(c.request(json.dumps(msg())))=={"ok":False,"refusal":"tool_failed"}
    assert b.predictions_used==1 and len(l.calls)==1 and l.calls[0].status=="failed"
    assert verify(path,head)["ok"]


@pytest.mark.parametrize("exc",[KeyboardInterrupt,SystemExit])
def test_start_is_durable_before_tool_and_interruption_is_charged(make,exc):
    outputs=[]
    for replay in (False,True):
        b=l=path=head=None
        def stopped(a):
            start=read(path)[-1]
            assert start["kind"]=="prediction_start" and start["request"]==msg()
            assert verify(path,head)["ok"]
            raise exc("invented53")
        b,l,path,head=make(stopped,replay)
        with pytest.raises(exc):
            with b.open_call("heading") as c:
                c.request(json.dumps(msg()))
        assert c.closed and not b._open
        assert len(l.calls)==1 and l.calls[0].status=="interrupted" and l.calls[0].wall_s==1
        assert l.calls[0].scope==("replay" if replay else "running")
        ev=read(path); start=next(e for e in ev if e["kind"]=="prediction_start")
        end=next(e for e in ev if e["kind"]=="prediction_interrupted")
        assert start["request_id"]==end["request_id"]==l.calls[0].request_id
        outputs.append(c.request(json.dumps(msg())))
        assert verify(path,head)["ok"]
    assert outputs[0]==outputs[1]


def test_regression_atomic_and_episode_end_closes_all_calls_once(make):
    b,l,path,head=make(); b.at(3,5,2)
    with pytest.raises(B.BrokerError): b.at(4,6,1)
    assert (b.t,b.action_step,b.goal_version)==(3,5,2)
    c1=b.open_call("one"); c2=b.open_call("two")
    b.end_episode()
    for c in (c1,c2):
        assert c.closed and json.loads(c.request(json.dumps(msg())))["refusal"]=="call_closed"
        c.__exit__(None,None,None)
    assert len([e for e in read(path) if e["kind"]=="call_close"])==2
    assert not b._open and not l.calls and verify(path,head)["ok"]


def test_finite_horizon_gives_large_but_finite_reply_payload(make):
    # Illustrates transport planning. A shape-valid maximum-horizon reply can exceed 2MB.
    n=B.MAX_PREDICTION_SECONDS; big=int(sys.float_info.max)
    obj=msg(); obj["args"]["headings_deg"]=[0]*n
    def largest(a):
        return {"positions_nm":[[-big,-big] for _ in range(n)],
                "closest_approach_nm":big,"closest_at_s":n}
    b,l,path,head=make(largest)
    with b.open_call("heading") as c:
        raw=c.request(json.dumps(obj))
    assert json.loads(raw)["ok"]
    assert 2_000_000<len(raw.encode())<2_250_000
    assert len(l.calls)==1 and verify(path,head)["ok"]


@pytest.mark.parametrize("exc",[KeyboardInterrupt,SystemExit])
# Fixed in TURN 54 (Claude): BR53.RET elapsed-clock interruption after tool return drops ledger record
def test_interruption_while_measuring_finished_prediction_retains_ledger(make,exc):
    ticks=0
    def clock():
        nonlocal ticks
        ticks+=1
        if ticks==2: raise exc("elapsed clock interrupted")
        return 1.0
    b,l,path,head=make(clock=clock)
    with pytest.raises(exc):
        with b.open_call("heading") as c:
            c.request(json.dumps(msg()))
    assert c.closed and b.predictions_used==1
    assert len(l.calls)==1 and l.calls[0].status=="interrupted" and l.calls[0].wall_s is None
    assert any(e["kind"]=="prediction_interrupted" for e in read(path))
