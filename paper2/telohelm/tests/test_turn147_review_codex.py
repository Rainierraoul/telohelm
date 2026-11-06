"""Component attribution controls with invented full outcomes/decision records only."""
from copy import deepcopy
import pytest
from telohelm.learning import component_trial
from paper2.experiment2.tests.test_learning import _o; from paper2.experiment2.tests.test_learning import _d; from paper2.experiment2.tests.test_learning import _rep


def test_current_goal_accepted_candidate_controls_pass():
    heading=_d(0,'heading','cand',{'heading_deg':45.});heading['goal_version']=1
    assert component_trial('heading','cand',_o(),[heading])['passed'] is True
    report=_d(9,'report','cand',{'report':'x'});report['goal_version']=1
    o=_o(reports=[_rep(9)],report_correct=True,report_correct_options=[True])
    assert component_trial('report','cand',o,[report])['passed'] is True


def test_old_goal_candidate_use_does_not_verify_current_goal_heading_component():
    old=_d(0,'heading','cand',{'heading_deg':45.});old['goal_version']=1
    current=_d(12,'heading','other',{'heading_deg':90.});current['goal_version']=2
    o=_o(goal={'version':2})
    assert component_trial('heading','cand',o,[old,current])['passed'] is False


def test_direct_report_ownership_must_match_original_decision_goal():
    d=_d(9,'report','cand',{'report':'x'});d['goal_version']=1
    o=_o(goal={'version':2},reports=[_rep(9,version=2)],report_correct=True,report_correct_options=[True])
    assert component_trial('report','cand',o,[d])['passed'] is False


def test_trigger_ownership_requires_same_original_goal_and_active_trigger():
    d=_d(5,'report','cand',{'report_trigger_nm':20.});d['goal_version']=1
    o=_o(goal={'version':2},reports=[_rep(30,via_trigger=True,version=2)],
         report_correct=True,report_correct_options=[True])
    assert component_trial('report','cand',o,[d])['passed'] is False


def test_cleared_trigger_cannot_supply_ownership_even_with_nominal_correct_report():
    d=_d(5,'report','cand',{'report_trigger_nm':None});d['goal_version']=1
    o=_o(reports=[_rep(30,via_trigger=True)],report_correct=True,report_correct_options=[True])
    assert component_trial('report','cand',o,[d])['passed'] is False


def test_shadow_result_retains_original_candidate_decision_evidence(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import telohelm.learning as G
    from telohelm.runtime.library import Library
    record=_d(0,'heading','cand',{'heading_deg':45.})
    record.update(goal_version=1,context_sha256='invented-public-input-digest',request_id=None,
                  discarded_for_version=None,source={'kind':'procedure'})
    o=_o(goal_met=False)
    def run(scenario,library,models,log,ledger,**kwargs):
        log.append(deepcopy(record))
        return {'outcome':deepcopy(o),'actions':[{'goal_version':1,'heading_deg':45.}]}
    monkeypatch.setattr(G.L,'run_episode',run)
    lib=Library(tmp_path/'library')
    co=G.Coordinator(lib,tmp_path,None,lambda s:True,lambda a:{})
    co._candidate=('cand','heading')
    root=tmp_path/'shadow';root.mkdir()
    c=G.Completed(1,'e',SimpleNamespace(episode_id='e'),False,'both',True)
    result=co._fly(lib,c,root)
    assert result['component']['passed'] is True and result['outcome']==o
    assert result.get('decision_evidence')==[record]
