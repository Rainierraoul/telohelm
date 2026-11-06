"""The cold launcher on the actual telohelm.runtime layout (TURN 109). Each test builds an independently frozen temporary
installation: a copy of the real telohelm package initializer and the whole runtime package (without caches), and a
separate helper directory holding coldlaunch.py, bootstrap.py and codecapture.py. Fresh interpreters run with only the
installation on the search path and bytecode writing off. Observations other than code, Python and dependencies are
invented fixtures; the transport is an invented in-process stand-in. No server or cluster is contacted."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import telohelm.runtime.manifest as F
from .test_manifest import invented

STUDY2 = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm'

SCRIPT = '\nimport importlib.util, json, sys\nfrom pathlib import Path\nargs = json.loads(sys.argv[1])\nspec = importlib.util.spec_from_file_location("telohelm_coldlaunch", Path(args["helpers"]) / "coldlaunch.py")\nlaunch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)\ntry:\n    B = launch.load(Path(args["helpers"]), args["pins"])\nexcept launch.LaunchError as e:\n    print(json.dumps({"launch_error": str(e), "telohelm_loaded": "telohelm" in sys.modules})); sys.exit(0)\nmanifest = json.loads(Path(args["manifest"]).read_text())\n\nclass Wire:\n    origin, max_response_bytes = manifest["transport"]["origin"], manifest["transport"]["max_response_bytes"]\n    calls = 0\n    def exchange(self, url, body, timeout_s):\n        Wire.calls += 1\n        reply = json.dumps({"id": "cmpl-invented", "model": "invented-small",\n                            "usage": {"prompt_tokens": 11, "completion_tokens": 4},\n                            "choices": [{"message": {"role": "assistant", "content": "invented answer"},\n                                         "finish_reason": "stop"}]}).encode()\n        raw = b"HTTP/1.0 200 OK\\r\\nContent-Length: " + str(len(reply)).encode() + b"\\r\\n\\r\\n" + reply\n        return {"url": url, "outcome": "received", "request_wire": b"POST /v1/chat/completions HTTP/1.0\\r\\n\\r\\n"\n                + json.dumps(body, sort_keys=True).encode(), "reply_wire": raw, "reply_complete": True, "elapsed_s": 0.01}\n\ndef invented(S, name):\n    want = {"prompts": manifest["prompts"], "weights": manifest["weights"]["files"],\n            "launch": manifest["server"]["launch_args"],\n            "model_list": sorted({n for v in manifest["names"]["served"].values() for n in v}),\n            "effective_defaults": {k: manifest["generation"][k] for k in ("max_tokens", "stop", "seed")},\n            "environment": {k: manifest["environment"][k] for k in ("hardware", "precision", "concurrency", "batching")}}\n    return S.observed(want[name], "invented fixture, not an observation", "invented", "invented")\n\ndef observe(S, config, evidence):\n    names = ["prompts", "weights", "launch", "model_list", "effective_defaults", "environment"]\n    return B.cpu_observer(supplied={n: invented(S, n) for n in names} if args["supply"] else {})(S, config, evidence)\n\ntelohelm_before = "telohelm" in sys.modules\ntry:\n    boot = B.cold_start(Path(args["runtime"]), "telohelm.runtime", manifest, Path(args["expected"]),\n                        Path(args["registry"]), observe, "invented capture time", "invented frozen installation")\nexcept B.BootstrapError as e:\n    print(json.dumps({"bootstrap_error": str(e), "telohelm_before": telohelm_before})); sys.exit(0)\nrun = Path(args["run"]); run.mkdir(exist_ok=True)\nfrom telohelm.runtime.ledger import Ledger       # the runtime is loaded and admitted by now\ngw, problems = B.build_gateway(boot, "episode-a", "condition-a", "stream-a", Wire(), Path(args["store"]), run / "log",\n                               run / "heads", Path(args["journal"]), Path(args["registry"]), ledger=Ledger())\nout = gw.submit("small_model", [("user", "invented request")], 256, 1, 10, 10, 2.4)\ncalls = gw.ledger.calls\nprint(json.dumps({"problems": problems, "status": out["status"], "text": out.get("text"),\n                  "reportable": out.get("reportable"), "recovered": out.get("recovered"), "wire_calls": Wire.calls,\n                  "tokens": [[c.tokens_in, c.tokens_out] for c in calls], "telohelm_before": telohelm_before,\n                  "telohelm_file": sys.modules["telohelm"].__file__}))\n'


def install(tmp_path):
    """An independently frozen temporary installation and a separate helper directory."""
    inst = tmp_path / "inst"
    (inst / "telohelm").mkdir(parents=True)
    shutil.copyfile(next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/__init__.py', inst / "telohelm" / "__init__.py")
    shutil.copytree(next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/runtime', inst / "telohelm" / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    for f in ("coldlaunch.py", "bootstrap.py", "codecapture.py"):
        shutil.copyfile(STUDY2 / f, helpers / f)
    runtime = inst / "telohelm" / "runtime"
    m = invented()
    m["runtime"]["code_sha256"] = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(runtime.rglob("*.py"))}
    m["runtime"]["ancestor_sha256"] = {"telohelm": hashlib.sha256((inst / "telohelm" / "__init__.py").read_bytes()).hexdigest()}
    m["runtime"]["python"] = platform.python_version()
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    (tmp_path / "expected").write_text(F.fingerprint(m))
    (tmp_path / "registry").touch()
    pins = {f: hashlib.sha256((helpers / f).read_bytes()).hexdigest() for f in ("bootstrap.py", "codecapture.py")}
    return inst, helpers, pins


def launch(tmp_path, inst, helpers, pins, run="run1", supply=True, journal="journal"):
    args = {"helpers": str(helpers), "pins": pins, "runtime": str(inst / "telohelm" / "runtime"),
            "manifest": str(tmp_path / "manifest.json"), "expected": str(tmp_path / "expected"),
            "registry": str(tmp_path / "registry"), "run": str(tmp_path / run), "store": str(tmp_path / "store"),
            "journal": str(tmp_path / journal), "supply": supply}
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(inst), PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(args)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_actual_runtime_layout_cold_loads_and_answers(tmp_path):
    inst, helpers, pins = install(tmp_path)
    out = launch(tmp_path, inst, helpers, pins)
    assert out["telohelm_before"] is False and out["problems"] == []
    assert out["telohelm_file"] == str((inst / "telohelm" / "__init__.py").resolve())
    assert out["status"] == "answered" and out["text"] == "invented answer" and out["reportable"] is True
    assert out["wire_calls"] == 1 and out["tokens"] == [[11, 4]]


def test_only_cpu_observations_refuse_before_any_call(tmp_path):
    inst, helpers, pins = install(tmp_path)
    out = launch(tmp_path, inst, helpers, pins, supply=False)
    assert "weights: unknown" in out["problems"] and out["status"] == "refused" and out["wire_calls"] == 0


def test_a_changed_telohelm_initializer_is_refused_before_it_runs(tmp_path):
    inst, helpers, pins = install(tmp_path)
    sentinel = tmp_path / "unreviewed-code-ran"
    with (inst / "telohelm" / "__init__.py").open("a") as f:
        f.write(f"\nfrom pathlib import Path as _P\n_P({str(sentinel)!r}).write_text('ran')\n")
    out = launch(tmp_path, inst, helpers, pins)
    assert "the ancestor telohelm is not the pinned file" in out["bootstrap_error"] and not sentinel.exists()


def test_a_changed_runtime_module_is_refused_before_it_runs(tmp_path):
    inst, helpers, pins = install(tmp_path)
    sentinel = tmp_path / "unreviewed-code-ran"
    with (inst / "telohelm" / "runtime" / "models.py").open("a") as f:
        f.write(f"\nfrom pathlib import Path as _P\n_P({str(sentinel)!r}).write_text('ran')\n")
    out = launch(tmp_path, inst, helpers, pins)
    assert "refused before import" in out["bootstrap_error"] and not sentinel.exists()


def test_a_helper_that_is_not_the_pinned_one_is_not_executed(tmp_path):
    inst, helpers, pins = install(tmp_path)
    sentinel = tmp_path / "unreviewed-helper-ran"
    with (helpers / "bootstrap.py").open("a") as f:
        f.write(f"\nfrom pathlib import Path as _P\n_P({str(sentinel)!r}).write_text('ran')\n")
    out = launch(tmp_path, inst, helpers, pins)
    assert out["launch_error"] == "bootstrap.py is not the pinned helper" and not sentinel.exists()
    assert out["telohelm_loaded"] is False


def test_a_second_run_recovers_the_first_run_once_through_the_registry(tmp_path):
    inst, helpers, pins = install(tmp_path)
    first = launch(tmp_path, inst, helpers, pins, run="run1")
    assert first["status"] == "answered" and first["wire_calls"] == 1
    second = launch(tmp_path, inst, helpers, pins, run="run2")            # a new process, same journal and store
    assert second["problems"] == [] and second["recovered"] is True and second["wire_calls"] == 0
    assert second["text"] == "invented answer" and second["reportable"] is True and second["tokens"] == [[11, 4]]
    assert len((tmp_path / "registry").read_text().splitlines()) == 2


# TURN 111 (Codex's BOOT110): helpers run from their checked source bytes, one generation per process.

GENERATIONS = r'''
import hashlib, importlib.util, json, shutil, sys
from pathlib import Path
d1, d2, bad = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("fixture_launcher", d1 / "coldlaunch.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
pins = lambda d: {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in L.HELPERS}
out = {}
try:
    L.load(bad, pins(bad))
except L.LaunchError as e:
    out["failed_init"] = str(e)
out["left_behind"] = sorted(n for n in ("telohelm_bootstrap", "telohelm_codecapture") if n in sys.modules)
first = L.load(d1, pins(d1))
out["same_again"] = L.load(d1, pins(d1)) is first
out["bound"] = first.C is sys.modules["telohelm_codecapture"] and first.C.__file__ == str((d1 / "codecapture.py").resolve())
out["no_cache_file"] = not (d1 / "__pycache__").exists()
try:
    L.load(d2, pins(d2)); out["second"] = "loaded"
except L.LaunchError as e:
    out["second"] = str(e)
print(json.dumps(out))
'''


def test_one_helper_generation_per_process_and_no_partial_reuse(tmp_path):
    inst, helpers, _ = install(tmp_path)
    other = tmp_path / "helpers2"; shutil.copytree(helpers, other)
    with (other / "bootstrap.py").open("a") as f:
        f.write("\nSECOND_GENERATION = True\n")
    bad = tmp_path / "helpers-bad"; shutil.copytree(helpers, bad)
    with (bad / "bootstrap.py").open("a") as f:
        f.write("\nraise RuntimeError('invented failure during initialisation')\n")
    env = {**os.environ, "PYTHONPATH": str(inst), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", GENERATIONS, str(helpers), str(other), str(bad)], cwd=str(tmp_path),
                          env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["failed_init"] == "bootstrap.py failed to initialise: RuntimeError" and out["left_behind"] == []
    assert out["same_again"] is True and out["bound"] is True and out["no_cache_file"] is True
    assert out["second"] == "a different generation of the helpers is already loaded in this process"
