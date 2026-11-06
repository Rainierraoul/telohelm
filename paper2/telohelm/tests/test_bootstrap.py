"""The cold bootstrap (TURN 105), run in fresh interpreters on a temporary copy of the runtime package with bytecode
writing off. The code, Python and dependency items are observed; the others are invented fixtures or unknown. No server
or cluster is contacted: the transport is an invented in-process stand-in."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import telohelm.bootstrap as B
import telohelm.runtime.manifest as F
from .test_manifest import invented

REPO = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1'
RUNTIME = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/runtime'
NEEDED = ("__init__.py", "manifest.py", "startup.py", "models.py", "exchange.py", "transport.py", "trusted_log.py",
          "ledger.py")

SCRIPT = '\nimport json, sys\nfrom pathlib import Path\nimport telohelm.bootstrap as B\n\nargs = json.loads(sys.argv[1])\nmanifest = json.loads(Path(args["manifest"]).read_text())\n\nclass Wire:\n    origin, max_response_bytes = manifest["transport"]["origin"], manifest["transport"]["max_response_bytes"]\n    calls = 0\n    def exchange(self, url, body, timeout_s):\n        Wire.calls += 1\n        reply = json.dumps({"id": "cmpl-invented", "model": "invented-small",\n                            "usage": {"prompt_tokens": 11, "completion_tokens": 4},\n                            "choices": [{"message": {"role": "assistant", "content": "invented answer"},\n                                         "finish_reason": "stop"}]}).encode()\n        raw = b"HTTP/1.0 200 OK\\r\\nContent-Length: " + str(len(reply)).encode() + b"\\r\\n\\r\\n" + reply\n        return {"url": url, "outcome": "received", "request_wire": b"POST /v1/chat/completions HTTP/1.0\\r\\n\\r\\n"\n                + json.dumps(body, sort_keys=True).encode(), "reply_wire": raw, "reply_complete": True, "elapsed_s": 0.01}\n\ndef entry(S, name):\n    want = {"prompts": manifest["prompts"], "weights": manifest["weights"]["files"],\n            "launch": manifest["server"]["launch_args"],\n            "model_list": sorted({n for v in manifest["names"]["served"].values() for n in v}),\n            "effective_defaults": {k: manifest["generation"][k] for k in ("max_tokens", "stop", "seed")},\n            "environment": {k: manifest["environment"][k] for k in ("hardware", "precision", "concurrency", "batching")}}\n    return S.observed(want[name], "invented fixture, not an observation", "invented", "invented")\n\nsupplied_names = ["prompts", "weights", "launch", "model_list", "effective_defaults", "environment"]\n\ndef observe(S, config, evidence):\n    extra = {n: entry(S, n) for n in supplied_names} if args["supply"] else {}\n    return B.cpu_observer(supplied=extra)(S, config, evidence)\n\ntry:\n    boot = B.cold_start(Path(args["runtime"]), args["package"], manifest, Path(args["expected"]), Path(args["registry"]),\n                        observe, "invented capture time", "invented read-only copy")\nexcept B.BootstrapError as e:\n    print(json.dumps({"bootstrap_error": str(e)})); sys.exit(0)\ngw, problems = B.build_gateway(boot, "episode-a", "condition-a", "stream-a", Wire(), Path(args["dir"]) / "x",\n                               Path(args["dir"]) / "log", Path(args["dir"]) / "heads", Path(args["dir"]) / "journal",\n                               Path(args["registry"]))\nout = gw.submit("small_model", [("user", "invented request")], 256, 1, 10, 10, 2.4)\nprint(json.dumps({"problems": problems, "status": out["status"], "text": out.get("text"),\n                  "reportable": out.get("reportable"), "calls": Wire.calls, "phase": boot.catalog["phase"],\n                  "pyc": sorted(boot.catalog["bytecode"])}))\n'


def cold_copy(tmp_path, name):
    """A temporary package holding a copy of the runtime modules the gateway needs."""
    root = tmp_path / "coldroot"
    pkg = root / name / "runtime"
    pkg.mkdir(parents=True)
    (root / name / "__init__.py").write_text("")
    for f in NEEDED:
        shutil.copyfile(RUNTIME / f, pkg / f)
    return root, pkg


def frozen_manifest(tmp_path, pkg):
    """The manifest, with the code pins computed here, independently of the bootstrap, from the copied files."""
    m = invented()
    m["runtime"]["code_sha256"] = {f: hashlib.sha256((pkg / f).read_bytes()).hexdigest() for f in NEEDED}
    m["runtime"]["python"] = platform.python_version()
    # TURN 107 (Codex's BOOT106.PARENT): the runtime package's ancestors are pinned too.
    m["runtime"]["ancestor_sha256"] = {pkg.parent.name: hashlib.sha256((pkg.parent / "__init__.py").read_bytes()).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    (tmp_path / "expected").write_text(F.fingerprint(m))
    return m


def run(tmp_path, supply=True, name="coldstudy"):
    root, pkg = cold_copy(tmp_path, name)
    frozen_manifest(tmp_path, pkg)
    registry = tmp_path / "registry"
    registry.touch()
    (tmp_path / "run").mkdir()
    args = {"runtime": str(pkg), "package": f"{name}.runtime", "manifest": str(tmp_path / "manifest.json"),
            "expected": str(tmp_path / "expected"), "registry": str(registry), "dir": str(tmp_path / "run"),
            "supply": supply}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(root)]), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(args)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1]), registry


def test_a_cold_start_in_a_fresh_interpreter_admits_and_answers(tmp_path):
    out, registry = run(tmp_path)
    assert out["problems"] == [] and out["phase"] == "pre-import" and out["pyc"] == []
    assert out["status"] == "answered" and out["text"] == "invented answer" and out["reportable"] is True
    assert out["calls"] == 1 and len(registry.read_text().splitlines()) == 1        # this run's root published


def test_a_cold_start_with_only_cpu_observations_refuses_before_any_call(tmp_path):
    out, _ = run(tmp_path, supply=False)
    assert {"prompts: unknown", "weights: unknown", "launch: unknown", "model_list: unknown",
            "effective_defaults: unknown", "environment: unknown"} <= set(out["problems"])
    assert out["status"] == "refused" and out["calls"] == 0 and out["text"] is None


def test_a_changed_runtime_file_after_freezing_refuses(tmp_path):
    root, pkg = cold_copy(tmp_path, "coldstudy2")
    frozen_manifest(tmp_path, pkg)
    with (pkg / "ledger.py").open("a") as f:
        f.write("\n# an invented change after the pins were frozen\n")
    registry = tmp_path / "registry"; registry.touch(); (tmp_path / "run").mkdir()
    args = {"runtime": str(pkg), "package": "coldstudy2.runtime", "manifest": str(tmp_path / "manifest.json"),
            "expected": str(tmp_path / "expected"), "registry": str(registry), "dir": str(tmp_path / "run"),
            "supply": True}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(root)]), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(args)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)
    out = json.loads(done.stdout.strip().splitlines()[-1])
    # TURN 107: refused by the preflight, before any runtime code is imported (Codex's BOOT106.PRELOAD).
    assert "refused before import" in out["bootstrap_error"]


def test_a_warm_interpreter_cannot_cold_start(tmp_path):
    import telohelm.runtime  # noqa: F401  (already imported by this test process)
    with pytest.raises(B.BootstrapError):
        B.cold_start(RUNTIME, 'telohelm.runtime', invented(), tmp_path / "expected", tmp_path / "registry",
                     B.cpu_observer(), "t", "p")


def test_a_missing_registry_fails_closed(tmp_path):
    root, pkg = cold_copy(tmp_path, "coldstudy3")
    frozen_manifest(tmp_path, pkg)
    (tmp_path / "run").mkdir()
    args = {"runtime": str(pkg), "package": "coldstudy3.runtime", "manifest": str(tmp_path / "manifest.json"),
            "expected": str(tmp_path / "expected"), "registry": str(tmp_path / "absent"), "dir": str(tmp_path / "run"),
            "supply": True}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(root)]), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(args)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)
    assert "no protected roots registry" in json.loads(done.stdout.strip().splitlines()[-1])["bootstrap_error"]



# TURN 107 (Codex's CB106 contract): the preflight runs before any application code.

def _args(tmp_path, root, pkg, name):
    registry = tmp_path / "registry"; registry.touch(); (tmp_path / "run").mkdir(exist_ok=True)
    return ({"runtime": str(pkg), "package": f"{name}.runtime", "manifest": str(tmp_path / "manifest.json"),
             "expected": str(tmp_path / "expected"), "registry": str(registry), "dir": str(tmp_path / "run"),
             "supply": True},
            {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(root)]), "PYTHONDONTWRITEBYTECODE": "1"})


def _cold(tmp_path, root, pkg, name):
    args, env = _args(tmp_path, root, pkg, name)
    done = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(args)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("change,expect", [
    (lambda m: m["runtime"].pop("ancestor_sha256"), "must pin exactly the runtime package's ancestors"),
    (lambda m: m["runtime"].update(ancestor_sha256={"coldstudy7": "0" * 64}), "is not the pinned file"),
    (lambda m: m["runtime"].update(bytecode_sha256={"x.pyc": "0" * 64}), "not exactly the pinned bytecode"),
])
def test_a_manifest_that_does_not_authorise_the_files_refuses_before_import(tmp_path, change, expect):
    root, pkg = cold_copy(tmp_path, "coldstudy7")
    m = frozen_manifest(tmp_path, pkg)
    change(m)
    (tmp_path / "manifest.json").write_text(json.dumps(m)); (tmp_path / "expected").write_text(F.fingerprint(m))
    assert expect in _cold(tmp_path, root, pkg, "coldstudy7")["bootstrap_error"]


def test_a_manifest_that_is_not_the_frozen_one_refuses_before_import(tmp_path):
    root, pkg = cold_copy(tmp_path, "coldstudy8")
    frozen_manifest(tmp_path, pkg)
    (tmp_path / "expected").write_text("0" * 64)
    assert "not the frozen one" in _cold(tmp_path, root, pkg, "coldstudy8")["bootstrap_error"]


def test_cached_bytecode_beside_an_ancestor_refuses_before_import(tmp_path):
    import py_compile
    root, pkg = cold_copy(tmp_path, "coldstudy9")
    frozen_manifest(tmp_path, pkg)
    py_compile.compile(str(pkg.parent / "__init__.py"), doraise=True)
    assert "has cached bytecode beside it" in _cold(tmp_path, root, pkg, "coldstudy9")["bootstrap_error"]


def test_a_runtime_that_resolves_elsewhere_refuses_before_import(tmp_path):
    root, pkg = cold_copy(tmp_path, "coldstudy10")
    frozen_manifest(tmp_path, pkg)
    other = tmp_path / "elsewhere"; shutil.copytree(pkg, other)
    args, env = _args(tmp_path, root, pkg, "coldstudy10")
    args["runtime"] = str(other)                                        # catalogue one directory, import another
    done = subprocess.run([sys.executable, "-c", SCRIPT, json.dumps(args)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=120)
    assert "not the catalogued" in json.loads(done.stdout.strip().splitlines()[-1])["bootstrap_error"]
