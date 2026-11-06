"""TURN112 independent source-only loading and initialization cuts.

Fresh interpreters, harmless temporary helpers only; no application/model use.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from .test_coldlaunch import install


PREFIX = '''import importlib.util,hashlib,json,sys
from pathlib import Path
good,bad=map(Path,sys.argv[1:3])
spec=importlib.util.spec_from_file_location("fixture_launcher",good/"coldlaunch.py")
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
def pins(d):return {n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in L.HELPERS}
'''


def execute(tmp_path, inst, good, bad, program):
    env = {**os.environ, 'PYTHONPATH': str(inst), 'PYTHONDONTWRITEBYTECODE': '1'}
    done = subprocess.run([sys.executable, '-c', PREFIX + program, str(good), str(bad)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('helper', ['bootstrap.py', 'codecapture.py'])
@pytest.mark.parametrize('exception', ['RuntimeError("fixture")', 'SystemExit(2)'])
def test_each_initialization_stage_cleans_up_before_good_retry(tmp_path, helper, exception):
    inst, good, _ = install(tmp_path)
    bad = tmp_path / 'bad'; shutil.copytree(good, bad)
    with (bad / helper).open('a') as f:
        f.write('\nraise ' + exception + '\n')
    program = '''try:
 L.load(bad,pins(bad))
except L.LaunchError:
 failed=True
else:
 failed=False
left=sorted(n for n in ("telohelm_bootstrap","telohelm_codecapture") if n in sys.modules)
B=L.load(good,pins(good))
print(json.dumps({"failed":failed,"left":left,"same":L.load(good,pins(good)) is B,
                  "bound":B.C is sys.modules["telohelm_codecapture"],
                  "no_cache":not (good/"__pycache__").exists()}))
'''
    assert execute(tmp_path, inst, good, bad, program) == {
        'failed': True, 'left': [], 'same': True, 'bound': True, 'no_cache': True}


def test_checked_source_buffer_is_what_executes_at_a_post_read_file_cut(tmp_path):
    inst, good, _ = install(tmp_path)
    program = '''wanted=pins(good)
sentinel=good/"changed-file-ran"
real=L._exec_source
def cut(name,path,source):
 if name=="telohelm_codecapture":
  p=good/"bootstrap.py"
  p.write_bytes(p.read_bytes()+("\\nfrom pathlib import Path as _P\\n_P("+repr(str(sentinel))+").write_text('fixture')\\n").encode())
 return real(name,path,source)
L._exec_source=cut
B=L.load(good,wanted)
print(json.dumps({"changed_source":pins(good)!=wanted,"sentinel":sentinel.exists(),
                  "recorded_original":B.__telohelm_identity__["digests"]==wanted,
                  "bound":B.C is sys.modules["telohelm_codecapture"]}))
'''
    assert execute(tmp_path, inst, good, good, program) == {
        'changed_source': True, 'sentinel': False, 'recorded_original': True, 'bound': True}
