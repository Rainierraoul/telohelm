"""Reproduction controls from invented public bundles and fake shadow runner only."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import telohelm.learning as G
import paper2.experiment2.adaptive_controller.qualification as Q
from telohelm.runtime.context import ContextBuilder
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import verify
from .test_learning import _o; from .test_learning import _d


@dataclass
class InventedScenario:
    episode_id: str
    marker: int


def context_sha(ctx):
    # Exact canonical rule in episode_loop.run_episode's local context_sha,
    # not a public models.py API. This binds the reconstruction control.
    return hashlib.sha256(json.dumps(ctx, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def test_retained_public_bundles_reconstruct_context_digests_and_library(tmp_path, monkeypatch):
    lib = Library(tmp_path / 'library')
    source = 'def decide(ctx):\n    return {"heading_deg": 45.0}\n'
    sha = lib.add({'kind': 'procedure', 'decision': 'heading', 'conditions': [], 'source': source,
                   'origin': {'invented': True}})
    lib.set_status(sha, 'adopted', 'invented')
    co = G.Coordinator(lib, tmp_path, None, lambda s: True, lambda a: {})
    co._candidate = (sha, 'heading')
    workload = Q.public_workloads()[0]
    briefing = deepcopy(workload['briefing'])
    bundles = [deepcopy(workload['bundle']) for _ in range(2)]
    bundles[1]['t'] = 1
    bundles[1]['reports'] = []
    records = []

    def fake(scenario, library, models, log, ledger, broker, **kwargs):
        builder = ContextBuilder(deepcopy(briefing))
        broker.checkpoint('briefing', briefing=deepcopy(briefing))
        for bundle in bundles:
            broker.at(t=bundle['t'], action_step=bundle['t'], goal_version=1)
            broker.checkpoint('bundle', bundle=deepcopy(bundle), bundle_sha256=hashlib.sha256(
                json.dumps(bundle, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                           allow_nan=False).encode()).hexdigest())
            ctx = builder.update(deepcopy(bundle))
            record = _d(bundle['t'], 'heading', sha, {'heading_deg': 45.0})
            record.update(context_sha256=context_sha(ctx), request_id=None, source='procedure',
                          discarded_for_version=None)
            records.append(deepcopy(record))
            log.append(record)
        return {'outcome': _o(goal_met=False), 'actions': [r['part'] for r in records]}

    monkeypatch.setattr(G.L, 'run_episode', fake)
    scenario = InventedScenario('e', 7)
    c = G.Completed(1, 'e', scenario, False, 'both', True)
    root = tmp_path / 'shadow'
    root.mkdir()
    result = co._fly(lib, c, root)
    G.shutil.rmtree(root)
    saved = Path(result['evidence_dir'])
    assert result['decision_evidence'] == records
    assert json.loads((saved / 'scenario.json').read_text()) == {'episode_id': 'e', 'marker': 7}
    assert verify(saved / 'log', saved / 'heads')['ok']
    assert verify(saved / 'broker.log', saved / 'broker.heads')['ok']
    settings = json.loads((saved / 'settings.json').read_text())
    assert settings['candidate'] == sha and settings['decision'] == 'heading'
    assert settings['completed_ordinal'] == 1 and settings['trial_ordinal'] == 'trial-0000'
    assert (saved / 'ledger.jsonl').read_bytes() == b''  # fake runner performs no charged work
    copied = Library(saved / 'shadow-library')
    assert copied.entry(sha)['source'] == source and copied.adopted('heading', 'procedure') == [sha]
    events = [json.loads(l)['payload'] for l in (saved / 'broker.log').read_text().splitlines()]
    start = next(e for e in events if e['kind'] == 'briefing')
    reconstructed = ContextBuilder(start['briefing'])
    digests = [context_sha(reconstructed.update(e['bundle'])) for e in events if e['kind'] == 'bundle']
    assert digests == [r['context_sha256'] for r in result['decision_evidence']]
    assert all(not f.stat().st_mode & 0o222 for f in saved.rglob('*') if f.is_file())
    pins = {str(f.relative_to(saved)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in saved.rglob('*') if f.is_file()}
    assert len(pins) >= 9


def test_numbering_never_reuses_existing_files_or_sparse_directories(tmp_path):
    parent = tmp_path / 'candidate-episode'
    parent.mkdir()
    (parent / 'trial-0000').write_text('invented existing non-directory')
    (parent / 'trial-0001').mkdir()
    (parent / 'trial-0010').mkdir()
    made = G._new_evidence_dir(parent)
    assert made.name == 'trial-0002'
    assert (parent / 'trial-0000').read_text() == 'invented existing non-directory'
    assert G._new_evidence_dir(parent) != made


def test_original_estimator_separates_mutable_aliases_even_without_new_reports():
    """Short estimator-only baseline for a future transaction/equivalence proposal."""
    from paper2.experiment2.adaptive_controller.estimation import EpisodeEstimator
    workload = Q.public_workloads()[0]
    est = EpisodeEstimator()
    est.reset(deepcopy(workload['briefing']))
    est.update(deepcopy(workload['bundle']))
    old_targets = est.targets
    old = old_targets['A']
    old_observation = old.observations[0]
    bundle = deepcopy(workload['bundle'])
    bundle.update(t=1, reports=[])
    est.update(bundle)
    new = est.targets['A']
    assert est.targets is not old_targets and new is not old
    assert new.observations is not old.observations
    assert new.observations[0] == old_observation and new.observations[0] is not old_observation
    old.observations.clear()
    assert len(new.observations) == 1  # in-place reuse would expose this alias
