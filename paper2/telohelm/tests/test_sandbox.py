"""Containment tests for model-written code (architecture section 6). Each forbidden access is tried
inside the sandbox and must be refused. Where it makes sense, the same probe is first run without the
sandbox, to show that it would have succeeded, so a refusal is not an artefact of a broken probe."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import telohelm.runtime.sandbox as S

pytestmark = pytest.mark.skipif(not S.available(), reason="macOS sandbox route only")

REPO = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
PROTECTED = [
    next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/results/experiment1/frozen/judges.json',
    next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/taskpacks/tanker/world/episode.py',
    next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'coordination/TURN.json',
]


def probe(body: str) -> str:
    """Procedure source whose decide() reports whether the body succeeded or was refused."""
    return ("def decide(arg):\n    try:\n"
            + "".join(f"        {line}\n" for line in body.splitlines())
            + "        return {'done': True}\n"
            "    except Exception as e:\n        return {'denied': type(e).__name__}\n")


def unsandboxed(code: str, payload=None, **kw) -> dict:
    """The same runner without the sandbox profile, for control runs."""
    msg = json.dumps({"code": code, "entry": "decide", "input": payload, "wrap": False})
    out = subprocess.run([sys.executable, "-I", "-S", "-c", S.RUNNER], input=msg.encode(), capture_output=True, **kw)
    return json.loads(out.stdout)["result"]


def contained(code: str, payload=None) -> dict:
    r = S.run(code, "decide", payload)
    assert r.status == "ok", r
    return r.result


@pytest.fixture
def hidden(tmp_path):
    f = tmp_path / "monitor-private" / "labels.json"
    f.parent.mkdir()
    f.write_text('{"label": "wrong"}')
    return f


def test_a_procedure_runs_and_returns_its_action():
    code = "import math\ndef decide(b):\n    return {'heading_deg': math.degrees(math.atan2(b['dx'], b['dy'])) % 360}\n"
    r = S.run(code, "decide", {"dx": 1.0, "dy": 0.0})
    assert r.status == "ok" and r.result == {"heading_deg": 90.0}
    assert r.wall_seconds > 0


@pytest.mark.parametrize("op", ["open(P).read()", "import os; os.stat(P)"])
def test_reading_a_private_file_is_refused(hidden, op):
    body = f"P = {str(hidden)!r}\n" + op.replace("; ", "\n")
    assert unsandboxed(probe(body)) == {"done": True}
    assert contained(probe(body)) == {"denied": "PermissionError"}


def test_protected_repository_files_are_refused():
    for f in PROTECTED:
        assert f.exists()
        assert contained(probe(f"open({str(f)!r}).read()")) == {"denied": "PermissionError"}, f


def test_listing_the_scenario_directory_is_refused(hidden):
    body = f"import os\nos.listdir({str(hidden.parent)!r})"
    assert unsandboxed(probe(body)) == {"done": True}
    assert contained(probe(body)) == {"denied": "PermissionError"}
    assert contained(probe("import os\nos.listdir('/Users')")) == {"denied": "PermissionError"}


def test_writing_a_file_is_refused(tmp_path):
    target = tmp_path / "log.jsonl"
    assert contained(probe(f"open({str(target)!r}, 'a').write('forged')")) == {"denied": "PermissionError"}
    assert not target.exists()


def test_network_is_refused():
    body = "import socket\ns = socket.socket()\ns.settimeout(1)\ns.connect(('127.0.0.1', 9))"
    assert contained(probe(body)) == {"denied": "PermissionError"}


@pytest.mark.parametrize("body", ["import subprocess\nsubprocess.run(['/bin/echo', 'x'])", "import os\nos.fork()"])
def test_starting_processes_is_refused(body):
    assert contained(probe(body)) == {"denied": "PermissionError"}


def test_inherited_descriptors_are_closed(hidden):
    fd = os.open(hidden, os.O_RDONLY)
    os.set_inheritable(fd, True)
    try:
        body = f"import os\nos.read({fd}, 100)"
        assert unsandboxed(probe(body), close_fds=False) == {"done": True}
        assert contained(probe(body)) == {"denied": "OSError"}
    finally:
        os.close(fd)


def test_environment_is_empty(monkeypatch):
    monkeypatch.setenv("TELOHELM_TEST_SECRET", "label=wrong")
    code = "import os\ndef decide(_):\n    return sorted(os.environ)\n"
    assert "TELOHELM_TEST_SECRET" in unsandboxed(code)
    assert contained(code) == [] or "TELOHELM_TEST_SECRET" not in contained(code)


def test_processor_time_is_limited():
    r = S.run("def decide(_):\n    while True:\n        pass\n", "decide", None)
    assert r.status == "timeout"


def test_errors_and_bad_replies_are_reported_not_raised():
    assert S.run("def decide(_):\n    raise ValueError('no')\n", "decide", None).status == "error"
    assert S.run("def decide(_):\n    return {1, 2}\n", "decide", None).status == "error"       # not JSON
    big = S.run("def decide(_):\n    return 'x' * 300000\n", "decide", None)
    assert big.status == "bad_reply"
    assert S.run("def wrong_name(_):\n    return 1\n", "decide", None).status == "error"


# --- gaps found by the threat list of paper2/notes/GAP-EXPERIMENTS-PLAN.md (G4) ------------------------

PROCESS_LIST = """def decide(_):
    try:
        import ctypes, ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        mib = (ctypes.c_int * 3)(1, 14, 0)             # kern.proc.all
        size = ctypes.c_size_t(0)
        rc = libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0)
        return {'done': True} if rc == 0 and size.value > 0 else {'denied': 'sysctl'}
    except Exception as e:
        return {'denied': type(e).__name__}
"""
PARENT_ARGS = """def decide(_):
    try:
        import ctypes, ctypes.util, os
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        mib = (ctypes.c_int * 3)(1, 49, os.getppid()) # kern.procargs2
        size = ctypes.c_size_t(65536)
        buf = ctypes.create_string_buffer(65536)
        rc = libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0)
        return {'done': True} if rc == 0 else {'denied': 'sysctl'}
    except Exception as e:
        return {'denied': type(e).__name__}
"""


@pytest.mark.parametrize("code", [PROCESS_LIST, PARENT_ARGS])
def test_other_processes_cannot_be_inspected(code):
    """With sysctl reads allowed, both probes succeeded inside the sandbox (found 10 October). Now ctypes
    itself cannot load there, and the probes are refused."""
    assert unsandboxed(code) == {"done": True}
    assert "denied" in contained(code)


def test_memory_above_the_limit_is_stopped():
    r = S.run("def decide(_):\n    x = bytearray(2 * 1024**3)\n    x[::4096] = b'1' * len(x[::4096])\n    return len(x)\n",
              "decide", None)
    assert r.status == "memory"
    small = S.run("def decide(_):\n    return len(bytearray(50 * 1024**2))\n", "decide", None)
    assert small.status == "ok"


def test_signals_to_other_processes_are_refused():
    assert contained(probe("import os\nos.kill(os.getppid(), 0)")) == {"denied": "PermissionError"}


def test_memory_reading_agrees_with_ps():
    child = subprocess.Popen([sys.executable, "-c",
                              "x = bytearray(200 * 1024**2); x[::4096] = b'1' * len(x[::4096]); import time; time.sleep(2)"])
    try:
        import time
        time.sleep(0.8)
        ps = int(subprocess.run(["ps", "-o", "rss=", "-p", str(child.pid)], capture_output=True, text=True).stdout) / 1024
        assert S.resident_mb(child.pid) == pytest.approx(ps, rel=0.05)
    finally:
        child.wait()


def test_a_blocked_input_write_stays_inside_the_deadline(monkeypatch):
    """R24.IO: the child sleeps 0.2 s before reading, and the message is padded past the pipe buffer, so a
    blocking write would hold the parent. With a 0.04 s limit the call must end near 0.04 s."""
    import time
    monkeypatch.setattr(S, "RUNNER", "import time\ntime.sleep(0.2)\n" + S.RUNNER)
    start = time.perf_counter()
    r = S.run("def decide(ctx):\n    return 1\n", "decide", {"padding": "x" * 131072}, wall_seconds=0.04)
    elapsed = time.perf_counter() - start
    assert r.status == "timeout" and elapsed < 0.15, (r.status, elapsed)


def test_normal_calls_still_work_after_the_io_change():
    r = S.run("def decide(ctx):\n    return len(ctx['padding'])\n", "decide", {"padding": "x" * 131072})
    assert r.status == "ok" and r.result == 131072 and r.cpu_seconds is not None
