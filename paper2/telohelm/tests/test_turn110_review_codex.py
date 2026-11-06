"""TURN110 independent helper-loader and actual-layout recovery controls.

All temporary modules/cache effects and transport replies are invented; no server.
"""
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys

import pytest

from .test_coldlaunch import SCRIPT; from .test_coldlaunch import install; from .test_coldlaunch import launch


# Fixed in TURN 111 (Claude): BOOT110.HELPER.CACHE: source-pinned helper executes unreviewed cached code
@pytest.mark.parametrize('helper', ['bootstrap.py', 'codecapture.py'])
def test_unreviewed_helper_bytecode_is_not_executed(tmp_path, helper):
    inst, helpers, pins = install(tmp_path)
    sentinel = tmp_path / 'unreviewed-helper-cache-ran'
    source = helpers / helper; original = source.read_bytes()
    source.write_bytes(original + ('\nfrom pathlib import Path as _P\n_P(' + repr(str(sentinel))
                                   + ').write_text("fixture")\n').encode())
    py_compile.compile(str(source), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    source.write_bytes(original)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins[helper]
    out = launch(tmp_path, inst, helpers, pins)
    assert not sentinel.exists(), out


# Fixed in TURN 111 (Claude): BOOT110.HELPER.LOADED: a new valid helper pin silently returns the old module
def test_new_helper_identity_cannot_silently_reuse_a_previous_module(tmp_path):
    inst, helpers, _ = install(tmp_path)
    other = tmp_path / 'helpers2'; shutil.copytree(helpers, other)
    for d, marker in [(helpers, 'first'), (other, 'second')]:
        with (d / 'bootstrap.py').open('a') as f:
            f.write('\nREVIEW110_MARKER = ' + repr(marker) + '\n')
    program = '''import importlib.util,json,sys
from pathlib import Path
d1,d2=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location("fixture_launcher",d1/"coldlaunch.py")
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
import hashlib
def pins(d):return {n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in L.HELPERS}
first=L.load(d1,pins(d1))
assert first.REVIEW110_MARKER=="first"
try:
 second=L.load(d2,pins(d2))
 out={"refused":False,"second_marker":second.REVIEW110_MARKER}
except L.LaunchError:
 out={"refused":True}
print(json.dumps(out))
'''
    env = {**os.environ, 'PYTHONPATH': str(inst), 'PYTHONDONTWRITEBYTECODE': '1'}
    done = subprocess.run([sys.executable, '-c', program, str(helpers), str(other)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out['refused'] or out['second_marker'] == 'second', out


@pytest.mark.parametrize('path_used', ['store', 'journal'])
def test_actual_layout_recovery_preserves_verified_duration_and_cost_once(tmp_path, path_used):
    inst, helpers, pins = install(tmp_path)
    script = SCRIPT + '''
print(json.dumps({"text":out.get("text"),"wire_calls":Wire.calls,"recovered":out.get("recovered"),
                  "reportable":out.get("reportable"),"cost":[[c.tokens_in,c.tokens_out,c.wall_s] for c in calls],
                  "log_ok":boot.modules["trusted_log"].verify(run/"log",run/"heads")["ok"]}))
'''
    def run(name, journal):
        args = {'helpers': str(helpers), 'pins': pins, 'runtime': str(inst / 'telohelm' / 'runtime'),
                'manifest': str(tmp_path / 'manifest.json'), 'expected': str(tmp_path / 'expected'),
                'registry': str(tmp_path / 'registry'), 'run': str(tmp_path / name),
                'store': str(tmp_path / 'store'), 'journal': str(tmp_path / journal), 'supply': True}
        env = {**os.environ, 'PYTHONPATH': str(inst), 'PYTHONDONTWRITEBYTECODE': '1'}
        done = subprocess.run([sys.executable, '-c', script, json.dumps(args)], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=30)
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])
    original = run('run1', 'journal')
    later = run('run2', 'journal' if path_used == 'journal' else 'new-journal')
    assert original['wire_calls'] == 1 and original['cost'][0][:2] == [11, 4]
    assert original['cost'][0][2] is not None
    assert later['wire_calls'] == 0 and later['cost'] == original['cost']
    assert later['text'] == 'invented answer' and later['reportable'] is True
    assert original['log_ok'] and later['log_ok']
    if path_used == 'journal':
        assert later['recovered'] is True
