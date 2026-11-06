"""TURN137 CPU/invented controls only: no model/provider network, world flight or reserved draw."""
import json,math
from pathlib import Path
import pytest
import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog
from paper2.telohelm.tests.test_turn65_review_codex import MSG
import paper2.experiment2.streams as G


def gateway(tmp_path,name,purpose,journal,provider,replay=False):
    d=tmp_path/name;d.mkdir();ledger=Ledger()
    return M.ModelGateway('e','condition','stream',provider,TrustedLog(d/'log',d/'heads','invented'),ledger,
                          journal,purpose=purpose,replay=replay),ledger


@pytest.mark.parametrize('purpose',['running','build'])
def test_saved_known_purpose_retained_on_normal_fresh_ledger_recovery(tmp_path,purpose):
    provider=M.ScriptedProvider(lambda _:M.ModelResponse('answered','invented',11,7));j=tmp_path/'journal'
    first,_=gateway(tmp_path,'first',purpose,j,provider)
    first.submit('reasoning_model',MSG,256,1,0,0,0.)
    second,l= gateway(tmp_path,'second',purpose,j,provider)
    assert second.submit('reasoning_model',MSG,256,1,0,0,0.)['recovered']
    assert provider.calls==1 and len(l.calls)==1 and l.calls[0].scope==purpose
    assert l.calls[0].tokens_in==11


def test_pre_purpose_record_does_not_guess_an_original_scope(tmp_path):
    provider=M.ScriptedProvider(lambda _:M.ModelResponse('answered','invented',11,7));j=tmp_path/'original'
    first,_=gateway(tmp_path,'first','build',j,provider)
    first.submit('reasoning_model',MSG,256,1,0,0,0.)
    # An invented record with the precise pre-purpose response schema. Preserve
    # the original evidence; this copy tests omission/legacy handling, not an attack claim.
    records=[json.loads(line) for line in j.read_text().splitlines()]
    for rec in records: rec.pop('purpose',None)
    legacy=tmp_path/'legacy';legacy.write_text(''.join(json.dumps(r)+'\n' for r in records))
    second,l=gateway(tmp_path,'second','running',legacy,provider)
    try:
        out=second.submit('reasoning_model',MSG,256,1,0,0,0.)
    except M.GatewayError:
        assert l.calls==[] and provider.calls==1
        return
    assert provider.calls==1
    # Either refuse admission or preserve verified resources under an explicitly
    # unavailable original-purpose disposition. Never give them a guessed flight tag.
    assert (out.get('status')=='refused' and not l.calls) or (
        out.get('purpose_unknown') is True and all(c.scope not in ('running','build') for c in l.calls))


def test_phase3_schedule_blocks_and_disturbance_ranges_without_flying():
    expected=('orbit','dropout','delay','all')
    seen={'orbit':set(),'dropout':set(),'delay':set()}
    for stream in (4401,4402,4403,4404):
        for index in range(20):
            s=G.draw(stream,3,index);block=expected[index//5];one,two=s.tankers
            assert s.receiver.t==0 and two.orbit.delta==0.
            assert one.orbit.r0*.8 <= one.orbit.radius(180) <= one.orbit.r0*1.2
            if block in ('orbit','all'):
                seen['orbit'].add(math.copysign(1,one.orbit.delta))
                assert abs(one.orbit.delta) <= one.orbit.r0*.2
            else: assert one.orbit.delta==0.
            if block in ('dropout','all'):
                name,start,end=s.dropouts[0]
                assert name=='SHELL 1' and 60<=start<=600 and end-start in (5,30)
                seen['dropout'].add(end-start)
            else: assert s.dropouts==()
            if block in ('delay','all'):
                assert s.delay_s in (5,30,120);seen['delay'].add(s.delay_s)
            else: assert s.delay_s==0
            assert math.isclose(math.hypot(*s.wind)*3600,20,abs_tol=1e-10) or math.isclose(
                math.hypot(*s.wind)*3600,40,abs_tol=1e-10) or math.isclose(math.hypot(*s.wind)*3600,60,abs_tol=1e-10)
            assert s==G.draw(stream,3,index)
    assert seen=={'orbit':{-1.,1.},'dropout':{5,30},'delay':{5,30,120}}


def test_phase4_analytic_range_envelope_and_strict_integer_time():
    # Worst-case envelope is a mathematical relaxation of turn-constrained motion,
    # not a flown path. Choose arbitrary points inside both bounded disks.
    centre=(50.,35.);start=(5.,5.);radius_upper=12.;speed=.15
    distance=math.dist(centre,start)
    last=math.ceil((distance-radius_upper-.25)/speed)-1
    tmax=last-10
    assert tmax==268 and distance-speed*(tmax+10)-radius_upper>.25
    assert math.ceil((10.-1.-.25)/.125)-1==69 # exact equality at t70 is not strictly safe
    for t in (60,tmax):
        lower=distance-speed*t-radius_upper
        for theta in (0,.7,1.8,3.14,5.1):
            receiver=(start[0]+speed*t*math.cos(theta),start[1]+speed*t*math.sin(theta))
            for phi in (0,.8,2.2,4.9):
                tanker=(centre[0]+radius_upper*math.cos(phi),centre[1]+radius_upper*math.sin(phi))
                assert math.dist(receiver,tanker)>=lower-1e-12
        allowed=[d for d in (10,15,20,25,30) if d<=lower-3]
        assert all(d+3<=lower for d in allowed)
    assert distance-10==pytest.approx(44.08326913195984)
    # The envelope proves current outside-distance eligibility; it says nothing
    # about a policy's future success, finite-horizon solvability or report timing.
