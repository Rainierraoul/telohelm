"""Independent immutable-case/split witnesses on invented trusted-writer inputs."""

import telohelm.learning_monitor as GM  # replay qualification moved to the monitor
import copy
import pytest
import telohelm.learning as G
import telohelm.qualification.store as H
from paper2.experiment2.tests.test_case_store import case; from paper2.experiment2.tests.test_case_store import trusting; from paper2.experiment2.tests.test_case_store import stocked; from paper2.experiment2.tests.test_case_store import coordinator_with


def test_seed_domains_have_fixed_distinct_inputs_without_a_draw():
    keys = [(s,split,role,i) for s in H.SCORED_STREAMS for split in H.SPLITS
            for role in H.DECISIONS for i in range(3)]
    seeds = [H.draw_seed(s,split,role,"geometry",i,0,rewrite=1 if split=="fresh_wrong" else None)
             for s,split,role,i in keys]
    assert len(set(seeds)) == len(keys)
    # Finite examples do not prove collision freedom of truncated SHA-256.


@pytest.mark.parametrize("path",["caller","reader","replay"])
def test_admitted_originals_do_not_change_with_caller_or_reader_views(tmp_path,path):
    s = trusting(tmp_path)
    c = case(); original = copy.deepcopy(c)
    s.admit(c)
    if path=="caller":
        c["context"]["goal_version"] = 99
    elif path=="reader":
        s.cases()[0]["output"]["heading_deg"] = 180
    else:
        wrong, right, missing = s.replay_cases(4301,"heading",None)
        right[0]["context"]["goal_version"] = 33
    assert s.cases()[0] == original
    assert H.CaseStore(s.root).cases()[0] == original


@pytest.mark.parametrize("split,stream", [("final",4301),("eligibility",4301),("confirmed_wrong",4302)])
def test_live_confirmation_refuses_unrelated_split_or_stream_before_work(tmp_path, monkeypatch, split, stream):
    store = stocked(tmp_path)
    co, lib = coordinator_with(tmp_path, store)
    before = store.log_path.read_bytes()
    monkeypatch.setattr(GM,"replay_test",lambda *a,**kw: pytest.fail("unrelated confirmation ran replay"))
    with pytest.raises(H.StoreError):
        co.confirm_wrong(case(split,"heading",kind="geometry",ordinal=777,stream=stream))
    assert store.log_path.read_bytes() == before
