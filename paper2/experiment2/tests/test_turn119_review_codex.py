"""Independent frozen TURN119 controls; invented CPU cases, no real model requests."""

import telohelm.learning_monitor as GM  # replay qualification moved to the monitor
import json
from pathlib import Path
import pytest
import telohelm.learning as G; import telohelm.local_model as LM; import paper2.experiment2.run_stream as RS
import telohelm.runtime.replay as RP; import telohelm.runtime.sandbox as S
from telohelm.runtime.models import ModelResponse; from telohelm.runtime.models import ScriptedProvider; from telohelm.runtime.models import CALL_LIMITS
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P
from .test_learning import coordinator; from .test_learning import Scripted; from .test_learning import finish; from .test_learning import reply; from .test_learning import passing; from .test_learning import BY_FLAG


def test_settings_equal_study1_and_repaired_output_cap():
    config=json.loads((next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1/study1/config.cluster.json').read_text())
    assert LM.SETTINGS == {'small_model':config['model_settings']['T1'], 'reasoning_model':config['model_settings']['T2']}
    assert G.PROTOCOL_OUTPUT_TOKENS == 16384
    assert CALL_LIMITS == {'reasoning_model':4,'small_model':30}


def test_turn116_first_object_and_unknown_call_count():
    assert LM.first_object('prefix {"heading_deg":45} another {} trailing }') == {'heading_deg':45}
    assert RS._unknown_calls([ModelResponse('answered','{}',tokens_in=1,tokens_out=None)]) == 1


def test_proposer_repair_limit_and_costs(tmp_path,monkeypatch):
    provider=ScriptedProvider(lambda request:ModelResponse('cut_off',None,tokens_in=10,tokens_out=16384))
    monkeypatch.setattr(LM,'provider',lambda *a,**kw:provider)
    proposer=G.GatewayProposer('http://127.0.0.1:1','invented',tmp_path/'gateway','invented')
    co,_=coordinator(tmp_path/'co',proposer,fly=passing)
    finish(co,5)
    assert co.events[-1]['kind']=='attempt_failed' and co.events[-1]['calls']==3
    assert provider.calls==len(proposer.ledger.calls)==3
    assert sum(c.tokens_out for c in proposer.ledger.calls)==3*16384
    assert sum(c.tokens_in for c in proposer.ledger.calls)==30


def test_over_input_budget_never_reaches_provider(tmp_path,monkeypatch):
    provider=ScriptedProvider(lambda request:ModelResponse('answered','{}',tokens_in=4097,tokens_out=2))
    monkeypatch.setattr(LM,'provider',lambda *a,**kw:provider)
    proposer=G.GatewayProposer('http://127.0.0.1:1','invented',tmp_path,'invented')
    proposer([('user','word '*5000)],{'attempt':1,'call':1,'episode':'invented'})
    assert provider.calls==0


def test_world_accepts_report_with_point_two_nm_range_error():
    orbit=P.Orbit(0,0,8,P.kt(280),0)
    receiver=P.ReceiverState(0,20.2,8,3*P.math.pi/2,3*P.math.pi/2,P.kt(450),P.turn_rate(450))
    sc=E.Scenario(episode_id='invented-report',receiver=receiver,wind=(0,0),tankers=(E.Tanker('SHELL 1',orbit),),goals=((0,E.Goal(1,'SHELL 1',20)),))
    world=E.Episode(sc)
    world._report('Wedgetail, aircraft 2, 20 NM from SHELL 1',1,False)
    assert world.record.reports[-1]['correct'] and E.REPORT_TOLERANCE==0.25


@pytest.mark.parametrize('prompt',[G.PROPOSER_RULES,LM.ASK['report']])
def test_prompt_report_rule_matches_world(prompt):
    assert f'{E.REPORT_TOLERANCE} NM' in prompt


def test_helped_successes_do_not_meet_floor(tmp_path):
    co,_=coordinator(tmp_path,Scripted(),fly=passing)
    # Populate completed observations through the real after-episode path; no invented model/world inference.
    import telohelm.taskpacks.tanker.world.phase5 as phase5
    for i in range(1,22):
        co.after_episode(phase5.draw(i).world_a,{'goal_met':True,'joined':True,'report_correct':True,'help_requests':[{'reason':'missing_tool'}]},True)
    assert co.breaches==1


def test_replay_refuses_missing_five_right_cases(tmp_path,monkeypatch):
    def invented_run(source,entry,args,**kw):
        return S.SandboxResult('ok','accept' if args[1]['heading_deg']<200 else 'reject')
    def replay(*a,**kw): return RP.replay_test(*a,**kw,run=invented_run)
    monkeypatch.setattr(GM,'replay_test',replay)
    corpus=G.Corpus(wrong=[{'context':{'t':0},'output':{'heading_deg':300}}],right=[{'context':{'t':0},'output':{'heading_deg':0}} for _ in range(95)])
    co,_=coordinator(tmp_path,Scripted(),fly=passing,corpus=corpus)
    assert co.check_replay(BY_FLAG,None)[0] is False


def test_job_preserves_failed_stream_exit_code(tmp_path):
    import subprocess
    script=(next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1/cluster/mvp-serve.sbatch').read_text()
    footer=script[script.index('"$PY" -m study2.run_stream'):]
    # Exact footer, with a failing local executable substituted for the runner; no Slurm/server contact.
    prefix=f'PY=/usr/bin/false\nRUN_ARGS="--stream 4401"\nLOGS={str(tmp_path)!r}\n'
    result=subprocess.run(['/bin/bash','-c',prefix+footer],capture_output=True,text=True)
    assert 'exit 1' in result.stdout
    assert result.returncode == 1
