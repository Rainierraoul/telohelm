"""TURN55 phase/exception/unknown-time controls with invented stubs; no world/model work."""
import json
import pytest
from telohelm.runtime.broker import Broker
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify


def request():
    return json.dumps({"capability":"predict","args":{"goal_version":0,
        "start":{"x_nm":0,"y_nm":0,"heading_deg":0,"tas_nm_s":.1},
        "headings_deg":[0],"wind_nm_s":[0,0],
        "orbit":{"cx_nm":10,"cy_nm":10,"radius_nm":3,"speed_nm_s":.1,"angle_rad":0}}})


def result():
    return {"positions_nm":[[0,0]],"closest_approach_nm":1,"closest_at_s":1}


def records(path):
    return [json.loads(l)["payload"] for l in path.read_text().splitlines()[1:]]


def make(tmp_path,tool,clock,replay=False):
    path=tmp_path/"log"; head=tmp_path/"heads"; ledger=Ledger()
    b=Broker("invented55",TrustedLog(path,head,"turn55"),ledger,tool,clock=clock,replay=replay)
    return b,ledger,path,head


@pytest.mark.parametrize("phase",["start clock","tool","end clock"])
@pytest.mark.parametrize("exc",[KeyboardInterrupt,SystemExit])
@pytest.mark.parametrize("replay",[False,True])
def test_one_interrupted_record_exact_exception_no_retry(tmp_path,phase,exc,replay):
    original=exc("invented original"); ticks=0; invoked=[]
    def clock():
        nonlocal ticks
        ticks+=1
        if (phase=="start clock" and ticks==1) or (phase=="end clock" and ticks==2): raise original
        return float(ticks)
    def tool(args):
        invoked.append(args)
        assert records(path)[-1]["kind"]=="prediction_start" and verify(path,head)["ok"]
        if phase=="tool": raise original
        return result()
    b,l,path,head=make(tmp_path,tool,clock,replay)
    with pytest.raises(exc) as caught:
        with b.open_call("heading") as c:
            c.request(request())
    assert caught.value is original
    assert len(invoked)==(0 if phase=="start clock" else 1)
    assert b.predictions_used==1 and c.closed and len(l.calls)==1
    row=l.calls[0]
    assert row.status=="interrupted" and row.scope==("replay" if replay else "running")
    assert row.wall_s==(1 if phase=="tool" else None)
    assert row.started_s==(None if phase=="start clock" else 1)
    ev=records(path); terminal=[e for e in ev if e["kind"] in ("prediction_end","prediction_interrupted")]
    assert len(terminal)==1 and terminal[0]["kind"]=="prediction_interrupted"
    assert terminal[0]["phase"]==phase and terminal[0]["request_id"]==row.request_id
    assert json.loads(c.request(request()))=={"ok":False,"refusal":"call_closed"}
    assert len(l.calls)==1 and verify(path,head)["ok"]


@pytest.mark.parametrize("phase",["start clock","end clock"])
@pytest.mark.parametrize("replay",[False,True])
def test_ordinary_clock_fault_retains_answer_unknown_time(tmp_path,phase,replay):
    ticks=0; invoked=[]
    def clock():
        nonlocal ticks
        ticks+=1
        if ticks==(1 if phase=="start clock" else 2): raise RuntimeError("invented clock fault")
        return float(ticks)
    def tool(a): invoked.append(a); return result()
    b,l,path,head=make(tmp_path,tool,clock,replay)
    with b.open_call("heading") as c:
        reply=json.loads(c.request(request()))
    assert reply["ok"] and len(invoked)==1 and b.predictions_used==1
    assert len(l.calls)==1 and l.calls[0].status=="answered" and l.calls[0].wall_s is None
    assert l.calls[0].started_s==(None if phase=="start clock" else 1)
    terminal=[e for e in records(path) if e["kind"] in ("prediction_end","prediction_interrupted")]
    assert len(terminal)==1 and terminal[0]["kind"]=="prediction_end" and terminal[0]["status"]=="answered"
    assert verify(path,head)["ok"]


@pytest.mark.parametrize("exc",[KeyboardInterrupt,SystemExit])
def test_original_tool_interrupt_survives_ordinary_clock_failure(tmp_path,exc):
    original=exc("original"); ticks=0; invoked=[]
    def clock():
        nonlocal ticks
        ticks+=1
        if ticks==2: raise RuntimeError("secondary timing failure")
        return 1.0
    def tool(a): invoked.append(a); raise original
    b,l,path,head=make(tmp_path,tool,clock)
    with pytest.raises(exc) as caught:
        with b.open_call("heading") as c: c.request(request())
    assert caught.value is original and len(invoked)==1
    assert len(l.calls)==1 and l.calls[0].status=="interrupted" and l.calls[0].wall_s is None
    assert len([e for e in records(path) if e["kind"]=="prediction_interrupted"])==1
    assert verify(path,head)["ok"]


@pytest.mark.parametrize("broken",[False,True])
def test_completed_answer_or_failure_has_one_identified_terminal(tmp_path,broken):
    ticks=iter([10.,12.]); invoked=[]
    def tool(a):
        invoked.append(a)
        if broken: raise ValueError("invented private tool error")
        return result()
    b,l,path,head=make(tmp_path,tool,lambda:next(ticks))
    with b.open_call("heading") as c: reply=c.request(request())
    assert len(invoked)==1 and len(l.calls)==1 and l.calls[0].wall_s==2
    status="failed" if broken else "answered"
    assert l.calls[0].status==status
    terminal=[e for e in records(path) if e["kind"] in ("prediction_end","prediction_interrupted")]
    assert len(terminal)==1 and terminal[0]["request_id"]==l.calls[0].request_id and terminal[0]["status"]==status
    assert "private" not in reply and verify(path,head)["ok"]
