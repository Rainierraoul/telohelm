"""Saved evidence checks with fake shadow runner; no world/model outcomes executed."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
import telohelm.learning as G
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import verify
from paper2.experiment2.tests.test_learning import _o; from paper2.experiment2.tests.test_learning import _d; from paper2.experiment2.tests.test_learning import _rep


def setup(tmp_path, monkeypatch):
    lib = Library(tmp_path / 'library')
    source = 'def decide(ctx):\n    return {"heading_deg": 45.0}\n'
    sha = lib.add({'kind': 'procedure', 'decision': 'heading', 'conditions': [], 'source': source,
                   'origin': {'invented': True}})
    co = G.Coordinator(lib, tmp_path, None, lambda s: True, lambda a: {})
    co._candidate = (sha, 'heading')
    records = []
    def fake(scenario, library, models, log, ledger, **kwargs):
        record = _d(0, 'heading', sha, {'heading_deg': 45.0 + len(records)})
        record.update(context_sha256=f'invented-input-{len(records)}', source={'kind': 'procedure'},
                      request_id=None, discarded_for_version=None)
        records.append(record)
        log.append(record)
        return {'outcome': _o(goal_met=False), 'actions': [deepcopy(record['part'])]}
    monkeypatch.setattr(G.L, 'run_episode', fake)
    c = G.Completed(1, 'e', SimpleNamespace(episode_id='e', invented=True), False, 'both', True)
    return co, lib, c, source, records


def fly(co, lib, c, root):
    root.mkdir()
    return co._fly(lib, c, root)


def test_original_decisions_source_and_copied_log_heads_survive_scratch_cleanup(tmp_path, monkeypatch):
    co, lib, c, source, records = setup(tmp_path, monkeypatch)
    result = fly(co, lib, c, tmp_path / 'shadow')
    G.shutil.rmtree(tmp_path / 'shadow')
    saved = Path(result['evidence_dir'])
    assert result['decision_evidence'] == records
    assert result['candidate_source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    assert verify(saved / 'log', saved / 'heads')['ok'] is True
    assert verify(saved / 'broker.log', saved / 'broker.heads')['ok'] is True


def test_repeated_trial_never_overwrites_previous_protected_evidence(tmp_path, monkeypatch):
    co, lib, c, source, records = setup(tmp_path, monkeypatch)
    first = fly(co, lib, c, tmp_path / 'shadow1')
    saved = Path(first['evidence_dir']) / 'log'
    original = saved.read_bytes()
    second = fly(co, lib, c, tmp_path / 'shadow2')
    assert first['decision_evidence'] != second['decision_evidence']
    assert saved.read_bytes() == original


def test_missing_goal_tags_and_direct_supersession_never_verify_a_trigger():
    missing = _d(0, 'heading', 'cand', {'heading_deg': 45.0}, goal_version=None)
    assert G.component_trial('heading', 'cand', _o(), [missing])['passed'] is False
    setter = _d(5, 'report', 'cand', {'report_trigger_nm': 20.0})
    direct = _d(7, 'report', 'other', {'report': 'x'})
    o = _o(reports=[_rep(30, via_trigger=True)], report_correct=True, report_correct_options=[True])
    # Inconsistent invented ending, not claimed reachable in the correct world.
    assert G.component_trial('report', 'cand', o, [setter, direct])['passed'] is False
