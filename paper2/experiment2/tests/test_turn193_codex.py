from pathlib import Path
"""Cheap source/sampler diagnostics; no model, server, world or signals."""
import subprocess

import pytest

import paper2.experiment2.outcomecases as O
import paper2.experiment2.streams as streams
from .test_option_c import HEADING


@pytest.mark.parametrize('phase', [3, 4])
def test_disturbed_phase_attempts_cycle_all_four_blocks(monkeypatch, phase):
    seen = []
    def unavailable(source, p, index):
        seen.append((source, p, index))
        raise streams.GenerationError('invented unavailable draw')
    monkeypatch.setattr(streams, 'solvable_episode', unavailable)
    assert O.moments(4400, 5, (phase,)) == []
    assert [i for _, _, i in seen] == [0, 5, 10, 15, 1]


@pytest.mark.parametrize('name', ['poc-step3-stream.sbatch', 'poc-step4.sbatch', 'poc-step5.sbatch'])
def test_jobs_use_owned_ports_for_readiness_parity_clients_and_watch(name):
    path = (next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / "dcarr-runtime-0.1/cluster") / name
    script = path.read_text()
    assert subprocess.run(['bash', '-n', str(path)], capture_output=True).returncode == 0
    assert script.index('source "$CODE/cluster/poc-serve.sh"') < script.index('poc_serve || exit 1')
    assert script.index('poc_serve || exit 1') < script.index('T2_PORT')
    assert 'http://127.0.0.1:8000' not in script and 'http://127.0.0.1:8001' not in script
    assert '--base-url "http://127.0.0.1:$T1_PORT"' in script
    assert '--learn-url "http://127.0.0.1:$T2_PORT"' in script
    assert 'os.environ["T2_PORT"]' in script
    assert script.index('poc_watch "$learn_pid"') < script.index('wait "$learn_pid"')
