"""Independent invented CPU controls for frozen TURN121. No real provider/server requests."""
import io,json,math
from pathlib import Path
import pytest
import telohelm.learning as G; import telohelm.local_model as LM; import telohelm.token_count as TC
import telohelm.runtime.prediction as T; import telohelm.runtime.models as M
import telohelm.taskpacks.tanker.world.physics as P
from .test_prediction import args
from paper2.experiment2.tests.test_learning import coordinator; from paper2.experiment2.tests.test_learning import finish; from paper2.experiment2.tests.test_learning import passing


def gateway(tmp_path,monkeypatch,answer):
    seen=M.ScriptedProvider(answer)
    monkeypatch.setattr(LM,'provider',lambda *a,**kw:seen)
    return G.GatewayProposer('http://127.0.0.1:1','invented',tmp_path,'invented'),seen


def test_actual_token_boundary_allows_4096_refuses_4097_without_charge(tmp_path,monkeypatch):
    p,seen=gateway(tmp_path,monkeypatch,lambda req:M.ModelResponse('answered','{}',tokens_in=4096,tokens_out=2))
    kw=LM.SETTINGS['reasoning_model']['chat_template_kwargs']
    # Real pinned tokenizer/template, no surrogate byte/word count. This vocabulary gives one token per word.
    overhead=TC.count([('user','word')],kw)-1
    exact=[('user','word '*(4096-overhead))];larger=[('user','word '*(4097-overhead))]
    assert TC.count(exact,kw)==4096 and TC.count(larger,kw)==4097
    assert p(exact,{'attempt':1,'call':1,'episode':'invented'})=='{}'
    assert p(larger,{'attempt':1,'call':2,'episode':'invented'})==''
    assert seen.calls==len(p.ledger.calls)==1
    rows=[json.loads(l) for l in (tmp_path/'learning-attempt-1/preflight.jsonl').read_text().splitlines()]
    assert [(r['input_tokens'],r['sent']) for r in rows]==[(4096,True),(4097,False)]


def test_large_format_repair_is_counted_and_not_sent_or_charged(tmp_path,monkeypatch):
    p,seen=gateway(tmp_path/'gate',monkeypatch,lambda req:M.ModelResponse('answered','not json '*5000,tokens_in=700,tokens_out=5000))
    co,_=coordinator(tmp_path/'co',p,fly=passing)
    finish(co,5)
    assert co.events[-1]['kind']=='attempt_failed' and co.events[-1]['calls']==3
    rows=[json.loads(l) for l in (tmp_path/'gate/learning-attempt-1/preflight.jsonl').read_text().splitlines()]
    assert [r['sent'] for r in rows]==[True,False,False]
    assert all(r['input_tokens']>4096 for r in rows[1:])
    assert seen.calls==len(p.ledger.calls)==1 and p.ledger.calls[0].tokens_out==5000


def test_server_token_count_mismatch_is_fatal(monkeypatch):
    import urllib.request
    class InventedOpener:
        def open(self,*a,**kw):return io.BytesIO(b'{"count":4097}')
    monkeypatch.setattr(urllib.request,'build_opener',lambda *a,**kw:InventedOpener())
    script=(next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1/cluster/mvp-serve.sbatch').read_text()
    start=script.index('import json, urllib.request');end=script.index("' >>",start)
    code=script[start:end].replace("from study2", "from telohelm")
    with pytest.raises(SystemExit):
        exec(compile(code,'<exact-job-token-count-probe>','exec'),{})


def test_prediction_positions_against_independent_small_step_rk4():
    # Known-command windows short enough that commanded heading is never reached during its turn.
    tas=450/3600; omega=9.80665*math.tan(math.pi/6)/(450*1852/3600)
    headings=[150]*10+[30]*10; wx,wy=.002,-.001
    result=T.predict(args(headings,heading=90,tas=tas,wind=(wx,wy)))
    x=y=0.;h=math.pi/2;dt=.01
    for second,target in enumerate(headings):
        target=target*math.pi/180
        for _ in range(100):
            delta=(target-h+math.pi)%(2*math.pi)-math.pi
            assert abs(delta)>omega*dt
            dh=math.copysign(omega,delta)
            vx=lambda angle:tas*math.sin(angle)+wx
            vy=lambda angle:tas*math.cos(angle)+wy
            x+=dt*(vx(h)+4*vx(h+dh*dt/2)+vx(h+dh*dt))/6
            y+=dt*(vy(h)+4*vy(h+dh*dt/2)+vy(h+dh*dt))/6;h+=dh*dt
        assert result['positions_nm'][second]==pytest.approx([x,y],abs=1e-9)


def test_prediction_search_limit_hit_is_measured_not_assumed(monkeypatch):
    # Constant true distance around a stationary receiver forces the Lipschitz branch-and-bound to its work cap.
    import telohelm.taskpacks.tanker.dynamics as PD   # the prediction tool's search now lives with the public dynamics
    original=PD.closest_approach;seen=[];evaluations=[]
    def measured(distance,*a,**kw):
        count=0
        def d(t):
            nonlocal count
            count+=1;return distance(t)
        found=original(d,*a,**kw);seen.append(found);evaluations.append(count);return found
    monkeypatch.setattr(PD,'closest_approach',measured)
    r=T.predict(args([0]*3600,heading=0,tas=0,cx=0,cy=0,radius=10,speed=300/3600))
    assert seen[0].status=='unresolved' and evaluations[0]<=20000 and evaluations[0]==seen[0].evaluations
    assert seen[0].upper-seen[0].lower > T.GAP_NM
    assert r['closest_approach_nm']==pytest.approx(10,abs=1e-9)
    assert 0<=r['closest_at_s']<=3600
    assert len(r['positions_nm'])==3600
