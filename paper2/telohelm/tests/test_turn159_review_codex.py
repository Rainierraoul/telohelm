"""Independent repair controls with invented cases; no scientific acquisition."""
import copy
import pytest
import telohelm.qualification.store as H
from paper2.experiment2.tests.test_case_store import case; from paper2.experiment2.tests.test_case_store import trusting; from paper2.experiment2.tests.test_case_store import coordinator_with


def test_verifier_cannot_change_the_snapshot_it_authorises(tmp_path):
    original = case()
    def verifier(c):
        c['context']['goal_version'] = 888
        c['output']['heading_deg'] = 270
        return True
    s = H.CaseStore(tmp_path/'cases', verifier=verifier, verifier_name='invented')
    cid = s.admit(original)
    assert s.cases() == [original]
    assert s.records[0]['case_id'] == cid == H.case_id(original)
    assert H.CaseStore(s.root).cases() == [original]


def test_undo_views_are_detached_from_disk_and_original(tmp_path):
    s = trusting(tmp_path)
    original = case('confirmed_wrong',kind='stale',ordinal=5)
    s.admit(original)
    before = copy.deepcopy(s.cases())
    wrong, right = s.undo_cases(4301,'heading')
    wrong[0]['context']['goal_version'] = 100
    assert s.cases() == before == H.CaseStore(s.root).cases()


@pytest.mark.parametrize('invalid', [float('nan'), object()])
def test_unserialisable_confirmation_is_refused_before_reconciliation(tmp_path,monkeypatch,invalid):
    s = trusting(tmp_path)
    co, lib = coordinator_with(tmp_path,s)
    touched = []
    monkeypatch.setattr(co,'reconcile_pairs',lambda: touched.append('reconcile'))
    c = case('confirmed_wrong',kind='stale',ordinal=5)
    c['context']['unserialisable'] = invalid
    with pytest.raises(H.StoreError):
        co.confirm_wrong(c)
    assert touched == []
    assert s.records == []
