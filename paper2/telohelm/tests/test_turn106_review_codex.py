"""TURN106 cold-load review with harmless local sentinels and fixture logs only.

Unreviewed temporary code writes one sentinel file; no model/server/cluster use.
"""
import json
import os
import py_compile
import subprocess
import sys

import pytest

import telohelm.runtime.exchange as X; import telohelm.runtime.models as M
from .test_bootstrap import REPO; from .test_bootstrap import SCRIPT; from .test_bootstrap import cold_copy; from .test_bootstrap import frozen_manifest
from .test_exchange_join import adapter
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit  # noqa: F401
from .test_turn100_review_codex import logged; from .test_turn100_review_codex import rewritten


def cold_witness(tmp_path, kind):
    name = 'review106'
    root, pkg = cold_copy(tmp_path, name)
    frozen_manifest(tmp_path, pkg)
    sentinel = tmp_path / 'unreviewed-code-ran'
    effect = '\nfrom pathlib import Path as _P\n_P(' + repr(str(sentinel)) + ').write_text("invented sentinel")\n'
    if kind == 'source':
        with (pkg / 'models.py').open('a') as f:
            f.write(effect)
    elif kind == 'bytecode':
        # Source hashes still match the independently frozen manifest. The
        # unreviewed unchecked-hash cache is executable even with bytecode
        # writing disabled. No source/cache pin is changed after cataloguing.
        source = pkg / 'models.py'; original = source.read_bytes()
        source.write_bytes(original + effect.encode())
        py_compile.compile(str(source), doraise=True,
                           invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
        source.write_bytes(original)
    elif kind == 'parent':
        (root / name / '__init__.py').write_text(effect)
    registry = tmp_path / 'registry'; registry.touch(); (tmp_path / 'run').mkdir()
    args = {'runtime': str(pkg), 'package': name + '.runtime', 'manifest': str(tmp_path / 'manifest.json'),
            'expected': str(tmp_path / 'expected'), 'registry': str(registry), 'dir': str(tmp_path / 'run'),
            'supply': True}
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(REPO), str(root)]), 'PYTHONDONTWRITEBYTECODE': '1'}
    done = subprocess.run([sys.executable, '-c', SCRIPT, json.dumps(args)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    return out, sentinel


# Fixed in TURN 107 (Claude): BOOT106.PRELOAD: unreviewed runtime code executes before identity refusal
@pytest.mark.parametrize('kind', ['source', 'bytecode'])
def test_unreviewed_runtime_code_is_refused_before_it_executes(tmp_path, kind):
    out, sentinel = cold_witness(tmp_path, kind)
    assert out.get('bootstrap_error') or out['problems']  # actual refusal, not model success
    assert not sentinel.exists(), out


# Fixed in TURN 107 (Claude): BOOT106.PARENT: application parent initializer executes outside catalog and is admitted
def test_application_parent_initializer_requires_an_independent_pin(tmp_path):
    out, sentinel = cold_witness(tmp_path, 'parent')
    assert not sentinel.exists(), out
    assert out.get('bootstrap_error') or out['problems']


@pytest.mark.parametrize('latency', [0, 0.0])
def test_valid_same_typed_zero_latency_remains_a_match(gateway, tmp_path, latency):
    wire, store, journal, original = logged(gateway, tmp_path)
    gw, _, _, later = gateway(provider=adapter(wire, X.ExchangeStore(store), reportable=True), journal=journal)
    assert submit(gw)['recovered'] and wire.calls == 1
    def change(event):
        if event['kind'] == 'model_start':
            event['request']['modelled_latency_s'] = latency
            event['available_second'] = event['request']['sim_second']
            event['available_step'] = event['request']['action_step']
    o = rewritten(original, tmp_path / 'o-log', tmp_path / 'o-heads', change)
    l = rewritten(later, tmp_path / 'l-log', tmp_path / 'l-heads', change)
    assert M.compare_attribution(o, l) == []
