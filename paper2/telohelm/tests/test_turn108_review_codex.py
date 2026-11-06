"""TURN108 independent positive boundaries and explicit deployment-scope probe.

Temporary source/cache pins, invented observer/Wire, fresh interpreters only.
"""
import hashlib
import json
import py_compile
import subprocess
import sys

from .test_bootstrap import REPO; from .test_bootstrap import RUNTIME; from .test_bootstrap import SCRIPT; from .test_bootstrap import _args; from .test_bootstrap import _cold; from .test_bootstrap import cold_copy; from .test_bootstrap import frozen_manifest
import telohelm.runtime.manifest as F


def execute(tmp_path, root, pkg, name, script):
    args, env = _args(tmp_path, root, pkg, name)
    done = subprocess.run([sys.executable, '-c', script, json.dumps(args)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_independently_pinned_correct_runtime_cache_is_accepted(tmp_path):
    root, pkg = cold_copy(tmp_path, 'review108cache')
    m = frozen_manifest(tmp_path, pkg)
    py_compile.compile(str(pkg / 'models.py'), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
    m['runtime']['bytecode_sha256'] = {str(p.relative_to(pkg)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in pkg.rglob('*.pyc')}
    (tmp_path / 'manifest.json').write_text(json.dumps(m)); (tmp_path / 'expected').write_text(F.fingerprint(m))
    out = _cold(tmp_path, root, pkg, 'review108cache')
    assert out['problems'] == [] and out['status'] == 'answered' and out['calls'] == 1
    assert len(out['pyc']) == 1 and out['pyc'][0].startswith('__pycache__/models.')


def test_preflight_itself_does_not_execute_even_authorized_initializer(tmp_path):
    root, pkg = cold_copy(tmp_path, 'review108preflight')
    sentinel = tmp_path / 'authorized-init-executed'
    (pkg / '__init__.py').write_text('from pathlib import Path\nPath(' + repr(str(sentinel)) + ').write_text("fixture")\n')
    frozen_manifest(tmp_path, pkg)
    script = SCRIPT[:SCRIPT.index('try:\n    boot = B.cold_start')]
    script += '''pre = B.preflight(Path(args["runtime"]), args["package"], manifest, Path(args["expected"]), "fixture", "fixture")
print(json.dumps({"phase": pre["catalog"]["phase"], "loaded": args["package"] in sys.modules}))
'''
    out = execute(tmp_path, root, pkg, 'review108preflight', script)
    assert out == {'phase': 'pre-import', 'loaded': False} and not sentinel.exists()


def test_factory_recovery_retains_original_cost_once_in_shared_ledger(tmp_path):
    root, pkg = cold_copy(tmp_path, 'review108recovery'); frozen_manifest(tmp_path, pkg)
    script = SCRIPT[:SCRIPT.index('out = gw.submit')]
    script += '''out = gw.submit("small_model", [("user", "invented request")], 256, 1, 10, 10, 2.4)
original_duration = gw._ledger.calls[0].wall_s
second, problems2 = B.build_gateway(boot, "episode-a", "condition-a", "stream-a", Wire(), Path(args["dir"]) / "x",
                                    Path(args["dir"]) / "log2", Path(args["dir"]) / "heads2",
                                    Path(args["dir"]) / "journal", Path(args["registry"]), ledger=gw._ledger)
recovered = second.submit("small_model", [("user", "invented request")], 256, 1, 10, 10, 2.4)
checks = [boot.modules["trusted_log"].verify(Path(args["dir"]) / l, Path(args["dir"]) / h)["ok"]
          for l,h in [("log","heads"),("log2","heads2")]]
print(json.dumps({"problems": problems + problems2, "recovered": recovered["recovered"], "text": recovered["text"],
                  "calls": Wire.calls, "tokens_in": sum(c.tokens_in or 0 for c in gw._ledger.calls),
                  "tokens_out": sum(c.tokens_out or 0 for c in gw._ledger.calls),
                  "duration_kept": gw._ledger.calls[0].wall_s == original_duration,
                  "duration_known": original_duration is not None, "checks": checks,
                  "registry_rows": len(Path(args["registry"]).read_text().splitlines())}))
'''
    out = execute(tmp_path, root, pkg, 'review108recovery', script)
    assert out == {'problems': [], 'recovered': True, 'text': 'invented answer', 'calls': 1,
                   'tokens_in': 11, 'tokens_out': 4, 'duration_kept': True, 'duration_known': True,
                   'checks': [True, True], 'registry_rows': 2}


def test_current_repository_namespace_is_a_documented_scope_refusal(tmp_path):
    # Importing telohelm.bootstrap necessarily imports its telohelm ancestor. The
    # general cold factory therefore cannot cold-load telohelm.runtime itself.
    # This is a scope probe, not evidence that a deployment entry point works.
    m = frozen_manifest(tmp_path, RUNTIME)
    m['runtime']['code_sha256'] = {str(p.relative_to(RUNTIME)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in RUNTIME.rglob('*.py')}
    m['runtime']['bytecode_sha256'] = {str(p.relative_to(RUNTIME)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in RUNTIME.rglob('*.pyc')}
    (tmp_path / 'manifest.json').write_text(json.dumps(m)); (tmp_path / 'expected').write_text(F.fingerprint(m))
    args, env = _args(tmp_path, REPO, RUNTIME, 'telohelm')
    done = subprocess.run([sys.executable, '-c', SCRIPT, json.dumps(args)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert 'bootstrap_error' in out and 'ancestors are already imported (telohelm)' in out['bootstrap_error']
