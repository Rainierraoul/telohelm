"""Public input construction and bounded estimator validation; no GuidanceSearch/time study."""
from copy import deepcopy
import json
from pathlib import Path
import paper2.experiment2.adaptive_controller.history_inputs as H; import paper2.experiment2.adaptive_controller.qualification as Q
from paper2.experiment2.adaptive_controller.estimation import EpisodeEstimator


def test_precise_history_manifest_is_reproducible_and_hashes_numeric_inputs():
    body=H.manifest()
    root=next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
    assert body==json.loads((next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/controller/long-history-inputs.json').read_text())
    assert body['count']==32 and body['measurement_authorized'] is False
    before=deepcopy(body)
    for spec in body['sequences']:
        semantic={k:spec[k] for k in ('id','final_second','targets','cadence_s','final_reports')}
        payload=H.materialize(semantic)
        assert H.materialize(spec)==payload
        assert Q.digest(payload)==spec['input_sha256']
        assert len(payload['prefix_bundles'])==spec['prefix_bundle_count']
        assert sum(len(b['reports']) for b in payload['prefix_bundles'])==spec['prefix_report_count']
        assert payload['prefix_bundles'][-1]['t']<payload['final_bundle']['t']
    assert body==before


def test_small_equivalent_geometry_history_accepts_units_and_retains_all_reports():
    for n in (1,2):
        for cadence in (1,12):
            for mode in ('none','fresh'):
                spec={'id':'bounded-unit','final_second':60,'targets':n,'cadence_s':cadence,'final_reports':mode}
                payload=H.materialize(spec); est=EpisodeEstimator();est.reset(payload['briefing'])
                for b in payload['prefix_bundles']:est.update(b)
                est.update(payload['final_bundle'])
                expected=len(payload['prefix_bundles'])+(mode=='fresh')
                assert all(len(t.observations)==expected for t in est.targets.values())
                latest=60 if mode=='fresh' else payload['prefix_bundles'][-1]['t']
                assert all(t.latest.measured_at==latest for t in est.targets.values())
                assert est.goal.version==1 and est.goal.tanker=='A'
                assert all(b['refusals']==[] and b['goal'] is None for b in payload['prefix_bundles'])
    # The actual180..3599 histories were constructed/hashed, not replayed/timed here.
