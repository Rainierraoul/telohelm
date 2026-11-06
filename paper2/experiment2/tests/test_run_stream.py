"""The MVP stream runner (MVP step 1): it runs, and its outcome, ledger, attribution and summary agree with the logs.
Short episodes with the stand-in model and prediction tool; nothing here is a result."""
from __future__ import annotations

import json

import pytest

import paper2.experiment2.run_stream as R
import telohelm.runtime.attribution as A
from telohelm.runtime.trusted_log import verify


def test_a_short_stream_runs_and_its_records_agree(tmp_path):
    done = R.run_stream(2, 1, tmp_path / "out", max_steps=30)
    s, eps = done["summary"], done["episodes"]
    assert s["episodes"] == 2 and [e["episode"] for e in eps] == ["dev-1", "dev-2"]
    for e in eps:
        d = tmp_path / "out" / e["episode"]
        for name in ("broker", "gateway", "episode"):
            assert verify(d / f"{name}.log", d / f"{name}.heads")["ok"]
        # The attribution in the summary is the derivation from the protected broker log.
        again = A.derive(d / "broker.log", d / "broker.heads", e["episode"])
        assert e["share_of_seconds"] == again["shares"] and e["seconds"] == again["K"] == e["action_steps"] == 30
        assert abs(sum(v for v in e["share_of_seconds"].values()) - 1.0) < 1e-9
        assert json.loads((d / "summary.json").read_text())["episode"] == e["episode"]
    assert s["model_calls"] == sum(e["model_calls"] for e in eps)
    assert s["tokens_in"] == 200 * s["model_calls"] and s["tokens_out"] == 12 * s["model_calls"]
    assert s["share_of_seconds"]["incomplete"]["K"] == 60                # stopped by the step limit, not an ending
    assert json.loads((tmp_path / "out" / "stream-summary.json").read_text())["episodes"] == 2


def test_the_library_is_shared_across_the_stream(tmp_path):
    R.run_stream(2, 5, tmp_path / "out", max_steps=5)
    from telohelm.runtime.library import Library
    lib = Library(tmp_path / "out" / "library")
    assert len(lib.adopted("heading", "procedure")) == 1 and len(lib.adopted("report", "check")) == 1



def test_the_job_gates_containment_before_any_model_and_fails_on_this_mac():
    """Codex's TURN 119: a pytest exit of 0 would also allow every test to be skipped. The job runs contained code
    first, before serve.sh, and stops when the Linux route is unavailable, as on this Mac."""
    import re
    import subprocess
    import sys
    from pathlib import Path
    script = (next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1/cluster/mvp-serve.sbatch').read_text()
    assert script.index("S.run(") < script.index('bash "$CODE/cluster/serve.sh"')
    assert "grep -q -e skipped" in script
    probe = re.search(r"\"\$PY\" -c '(.*?)' >>", script, re.S).group(1)
    r = subprocess.run([sys.executable, "-c", probe], cwd=next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1', capture_output=True)
    if sys.platform != "linux":
        assert r.returncode != 0


def _token_probe():
    from pathlib import Path
    script = (next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1/cluster/mvp-serve.sbatch').read_text()
    start = script.index("import json, urllib.request")
    return compile((script[start:script.index("' >>", start)]).replace("from study2", "from telohelm"), "<job-token-count-probe>", "exec")


def test_the_jobs_token_check_passes_when_the_server_agrees_and_fails_when_it_cannot_ask(monkeypatch, capsys):
    """Codex's MVP121.TOKEN.PARITY: the job compares a fresh and a repair request with the server's /tokenize. An
    invented server that counts as the offline counter does lets it through; one that cannot be reached stops it."""
    import io
    import json
    import urllib.request
    import telohelm.token_count as T

    class Agrees:
        def open(self, req, timeout=None):
            body = json.loads(req.data)
            m = [(x["role"], x["content"]) for x in body["messages"]]
            return io.BytesIO(json.dumps({"count": T.count(m, body["chat_template_kwargs"])}).encode())

    class Unreachable:
        def open(self, req, timeout=None):
            raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **kw: Agrees())
    exec(_token_probe(), {})
    out = capsys.readouterr().out
    assert "fresh request" in out and "repair request" in out
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **kw: Unreachable())
    with pytest.raises(OSError):
        exec(_token_probe(), {})
