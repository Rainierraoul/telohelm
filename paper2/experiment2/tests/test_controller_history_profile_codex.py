"""Frozen payload exports plus toy estimators/invented clocks only, never timing."""
from copy import deepcopy
from itertools import count
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import paper2.experiment2.adaptive_controller.history_profile as P; import paper2.experiment2.adaptive_controller.history_inputs as H; import paper2.experiment2.adaptive_controller.qualification as Q

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
MANIFEST = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/controller/long-history-inputs.json'


class Toy:
    instances = []

    def __init__(self, allowance):
        assert allowance == 1e-6
        self.instances.append(self)
        self.t = None

    def reset(self, briefing):
        self.goal = NS(**briefing['goal'])
        self.targets = {b['callsign']: NS(observations={0: 1}, latest=NS(measured_at=0, received_at=0),
                                        change_low=-1.6, change_high=1.6) for b in briefing['tankers']}
        briefing.clear()  # verify copies protect frozen data

    def update(self, bundle):
        self.t = bundle['t']
        for r in bundle['reports']:
            b = self.targets[r['callsign']]
            b.observations[r['measured_at']] = 1
            b.latest = NS(measured_at=r['measured_at'], received_at=r['received_at'])
        bundle.clear()


def inputs():
    manifest = P.read_manifest(MANIFEST)
    return manifest, [H.materialize(s) for s in manifest['sequences']]


def profile(factory=Toy, cpu=None):
    manifest, payloads = inputs()
    original = deepcopy(payloads)
    Toy.instances = []
    rows = P.profile_final(manifest, payloads, factory, cpu or count(0, 10).__next__,
                           count(0, 20).__next__, lambda: {'value': None, 'unit': None})
    assert payloads == original
    return manifest, rows


def test_full_payload_freeze_and_tamper_refusal(tmp_path):
    package = P.freeze_payloads(MANIFEST, tmp_path / 'payloads')
    manifest, payloads, sha = P.read_payloads(MANIFEST, tmp_path / 'payloads')
    assert len(payloads) == 32 and sha == package['body_sha256']
    with pytest.raises(FileExistsError):
        P.freeze_payloads(MANIFEST, tmp_path / 'payloads')
    path = tmp_path / 'payloads' / package['entries'][0]['file']
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='bytes differ'):
        P.read_payloads(MANIFEST, tmp_path / 'payloads')


def test_96_fresh_toy_instances_scopes_counts_and_immutable_inputs():
    manifest, rows = profile()
    assert len(Toy.instances) == len({id(t) for t in Toy.instances}) == 96
    assert [(r['repetition'], r['input']) for r in rows] == [
        (rep, s['id']) for rep in range(3) for s in manifest['sequences']]
    for row in rows:
        assert row['setup_cpu_ns'] == row['cpu_ns'] == 10
        assert row['setup_wall_ns'] == row['wall_ns'] == 20
        spec = next(s for s in manifest['sequences'] if s['id'] == row['input'])
        assert row['before']['t'] < spec['final_second'] == row['after']['t']
        assert all(b['retained'] == spec['retained_per_target_after_final']
                   for b in row['after']['targets'].values())
    result = P.summarize(rows, manifest)
    assert result['complete_known_timing'] is True and result['final_cpu_ns'] == 960
    assert 'no feasibility' in result['disposition'] and 'never add' in result['non_additive']


def test_unknown_clocks_failures_and_invalid_campaigns_stay_explicit():
    manifest, rows = profile(cpu=lambda: None)
    assert P.summarize(rows, manifest)['final_cpu_ns'] is None
    class Broken(Toy):
        def reset(self, briefing):
            raise RuntimeError('invented setup failure')
    manifest, failed = profile(factory=Broken)
    assert all(r['error_stage'] == 'setup' and r['cpu_ns'] is None for r in failed)
    assert P.summarize(failed, manifest)['complete_known_timing'] is False
    with pytest.raises(ValueError, match='unique'):
        P.summarize(rows[:-1] + [rows[0]], manifest)
    changed = deepcopy(rows)
    changed[0]['input_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash differs'):
        P.summarize(changed, manifest)
    m, data = inputs()
    data[0]['final_bundle']['t'] += 1
    with pytest.raises(ValueError, match='payload differs'):
        P.profile_final(m, data, factory=Toy)


def test_clock_raising_or_regressing_is_unknown():
    assert P._delta(True, 2) is None and P._delta(2, 1) is None
    def broken():
        raise RuntimeError('invented missing clock')
    assert P._clock(broken) is None
