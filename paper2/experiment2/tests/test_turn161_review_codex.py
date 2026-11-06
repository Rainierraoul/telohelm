"""Invented boundary records only; no generated flights or model calls."""
import copy
import pytest
import telohelm.qualification.store as H
import paper2.experiment2.devcases as D
from .test_case_store import case


def dev_case():
    c = case(stream=4401)
    c['label']['rule'] = D.RULE
    return c


def test_normal_marker_blocks_scored_restart(tmp_path):
    s = H.CaseStore(tmp_path, verifier=lambda c: True, development=True)
    s.admit(dev_case())
    with pytest.raises(H.StoreError, match='development store'):
        H.CaseStore(tmp_path)
    assert H.CaseStore(tmp_path, development=True).cases() == [dev_case()]


def test_torn_initial_marker_cannot_open_as_scored_after_recovery(tmp_path):
    (tmp_path/'cases.log').write_bytes(b'12 partial')
    s = H.CaseStore(tmp_path, verifier=lambda c: True, development=True)
    assert s.recover() == len(b'12 partial')
    s.admit(dev_case())
    with pytest.raises(H.StoreError, match='development store'):
        H.CaseStore(tmp_path)


def report_case(construction='early_report'):
    c = case(split='fresh_wrong', decision='report', kind='identity', stream=4401, rewrite=1)
    c['goal'] = {'version':1, 'tanker':'SHELL 1', 'report_at_nm':25}
    c['context'] = {'t':12, 'goal_version':1, 'observed_range_nm':35}
    c['output'] = {'report':D.report_text('CONTROL', 25, 'SHELL 1')}
    c['label'].update(rule=D.RULE, evidence={'construction':construction, 'report_at_nm':25,
                                          'observed_range_nm':35, 'control_callsign':'CONTROL'})
    return c


def test_early_report_construction_positive():
    assert D.verify(report_case()) is True


@pytest.mark.parametrize('text', ['garbage', 'CONTROL, aircraft 2, 99 NM from SHELL 1'])
def test_early_report_verifier_refuses_a_different_construction(text):
    c = report_case(); c['output']['report'] = text
    assert H.case_errors(c, development=True) == []
    assert D.verify(c) is False


def test_reference_report_requires_exact_output():
    c = report_case('reference_report')
    c['split'] = 'stored_right'; c['kind'] = None; c['label']['disposition'] = 'right'
    c['label']['evidence']['reference_report'] = c['output']['report']
    assert D.verify(c) is True
    changed = copy.deepcopy(c); changed['output']['report'] = 'garbage'
    assert D.verify(changed) is False
