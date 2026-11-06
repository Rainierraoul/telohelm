"""Opt-in post-cold-start floor: historical failures remain reported; current job/defaults untouched."""
from types import SimpleNamespace
import telohelm.learning as G; import paper2.experiment2.run_stream as RS
from telohelm.runtime.library import Library


def make(tmp_path, opt=False):
    return G.Coordinator(Library(tmp_path/'library'),tmp_path,lambda *_:'not JSON',
        lambda s:None if s.episode_id.startswith('unknown') else True,lambda *_:None,
        attempt_every=1000,floor_after_cold_start=opt)


def finish(c, i, ok, unknown=False, helped=False, stale=False, summary_shape=False):
    requests = [{'reason':'missing_or_stale_evidence'}] if stale else [{}] if helped else []
    record = {'goal_met':ok,'joined':ok,'report_correct':ok,'help_requests':requests}
    if summary_shape:          # a flown episode's summary: a count on top, the accepted requests nested (TURN295)
        record = {**record, 'help_requests':len(requests), 'outcome':dict(record)}
    c.after_episode(SimpleNamespace(episode_id=('unknown' if unknown else 'e')+str(i)), record, True)


def breaches(c):
    return [e for e in c.events if e['kind']=='floor_breach']


def test_legacy_floor_still_assesses_the_old_window_at_twenty_one(tmp_path):
    c=make(tmp_path)
    for i in range(1,22):finish(c,i,i>10)
    assert len(breaches(c))==1 and breaches(c)[0]['rate']==0.55


def test_opt_in_does_not_rewrite_for_cold_failures_after_sustained_success(tmp_path):
    c=make(tmp_path,True)
    for i in range(1,41):finish(c,i,i>10)
    assert not breaches(c)
    assert len(c.completed)==40 and sum(not x.goal_met for x in c.completed)==10
    assert sum(e['kind']=='completed' for e in c.events)==40


def test_full_post_cold_window_breaches_rearms_and_breaches_again(tmp_path):
    c=make(tmp_path,True)
    for i in range(1,40):finish(c,i,i!=21)
    assert not breaches(c)            # only19 post-cold observations
    finish(c,40,False)
    assert len(breaches(c))==1 and breaches(c)[0]['rate']==0.9
    finish(c,41,True)                 # first miss left window;19/20 is healthy
    assert c.armed and any(e['kind']=='floor_rearmed' for e in c.events)
    finish(c,42,False)
    assert len(breaches(c))==2 and breaches(c)[1]['rate']==0.9


def test_misses_still_queue_reassessment_before_floor_warmup(tmp_path):
    c=make(tmp_path,True)
    for i in range(1,22):finish(c,i,False)
    assert len(c.queue)==21 and c.queue[-1]=={'episode':'e21','failed':'both'}
    assert not breaches(c)


import pytest


@pytest.mark.parametrize('summary_shape', [False, True])
def test_a_stale_evidence_episode_never_fills_warmup_or_performance_window(tmp_path, summary_shape):
    # Until 27 October the episodes left out were those of unknown reference solvability; since then they are those
    # with an accepted stale-evidence help request (PRE-FREEZE-DESIGN-CHANGES-CLAUDE.md, item 1).
    c=make(tmp_path,True)
    for i in range(1,21):finish(c,i,False,stale=True,summary_shape=summary_shape)
    for i in range(21,60):finish(c,i,i<58)
    assert not breaches(c)            #39 eligible episodes, regardless of stream ordinal
    finish(c,60,False)
    assert len(breaches(c))==1
    assert len(c.completed)==60


def test_help_is_not_unassisted_success_after_warmup(tmp_path):
    c=make(tmp_path,True)
    for i in range(1,41):finish(c,i,True,helped=i in (39,40))
    assert len(breaches(c))==1 and breaches(c)[0]['rate']==0.9


def test_cli_defaults_legacy_and_can_request_post_cold_floor(tmp_path,monkeypatch):
    seen={}
    monkeypatch.setattr(RS,'run_stream',lambda *a,**k:seen.update(k) or {'summary':{
        'completed':0,'episodes':0,'goal_met_of_completed':0,'joined_of_completed':0,'model_calls':0}})
    RS.main(['--out',str(tmp_path/'a')])
    assert seen['floor_after_cold_start'] is False
    RS.main(['--out',str(tmp_path/'b'),'--floor-after-cold-start'])
    assert seen['floor_after_cold_start'] is True


def test_runner_binds_flag_into_coordinator(tmp_path,monkeypatch):
    seen={}
    class Built(Exception):pass
    def spy(*a,**k):seen.update(k);raise Built
    monkeypatch.setattr(G,'Coordinator',spy)
    import pytest
    with pytest.raises(Built):
        RS.run_stream(1,1,tmp_path/'out',learn={'base_url':'http://127.0.0.1:1','model':'m'},
                      floor_after_cold_start=True)
    assert seen['floor_after_cold_start'] is True
