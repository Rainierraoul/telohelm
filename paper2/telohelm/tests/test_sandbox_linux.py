"""Containment tests for the Linux route (TURN 117), the same checks as tests/test_sandbox.py for macOS. Each forbidden
access is tried inside the sandbox and must fail; where it makes sense the same probe first runs without the sandbox,
to show that it would have succeeded. On Linux a file outside the sandbox's root does not exist there, so a refusal
may be FileNotFoundError as well as PermissionError; any refusal counts, and success never does."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import telohelm.runtime.sandbox as S

pytestmark = pytest.mark.skipif(sys.platform != "linux" or not S.available(), reason="Linux sandbox route only")

REPO = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())


def probe(body: str) -> str:
    return ("def decide(arg):\n    try:\n"
            + "".join(f"        {line}\n" for line in body.splitlines())
            + "        return {'done': True}\n"
            "    except BaseException as e:\n        return {'denied': type(e).__name__}\n")


def unsandboxed(code: str, payload=None, **kw) -> dict:
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
    assert r.status == "ok" and r.result == {"heading_deg": 90.0} and r.cpu_seconds is not None


@pytest.mark.parametrize("op", ["open(P).read()", "import os; os.stat(P)"])
def test_reading_a_private_file_is_refused(hidden, op):
    body = f"P = {str(hidden)!r}\n" + op.replace("; ", "\n")
    assert unsandboxed(probe(body)) == {"done": True}
    assert "denied" in contained(probe(body))


def test_repository_and_home_files_cannot_be_read():
    for f in [next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'coordination/TURN.json', next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/taskpacks/tanker/world/episode.py',
              Path.home() / ".bashrc", Path("/etc/passwd")]:
        if f.exists():
            assert "denied" in contained(probe(f"open({str(f)!r}).read()")), f


def test_the_visible_filesystem_is_only_the_runtime():
    code = "import os\ndef decide(_):\n    return sorted(os.listdir('/'))\n"
    install_top = Path(os.path.realpath(sys.executable)).parts[1]
    assert set(contained(code)) <= {"usr", "dev", "lib", "lib64", install_top}
    assert "denied" in contained(probe("import os\nos.listdir('/proc')"))


def test_writing_anywhere_is_refused(tmp_path):
    target = tmp_path / "log.jsonl"
    assert "denied" in contained(probe(f"open({str(target)!r}, 'a').write('forged')"))
    assert not target.exists()
    for p in ["/x", "/dev/forged", "/usr/lib/forged"]:
        assert "denied" in contained(probe(f"open({p!r}, 'w').write('x')")), p
    py_root = str(Path(os.path.realpath(sys.executable)).parents[1])
    assert "denied" in contained(probe(f"open({py_root + '/forged'!r}, 'w').write('x')"))


def test_network_is_refused():
    for target in ["('127.0.0.1', 9)", "('8.8.8.8', 53)"]:
        body = f"import socket\ns = socket.socket()\ns.settimeout(1)\ns.connect({target})"
        assert "denied" in contained(probe(body)), target


@pytest.mark.parametrize("body", ["import subprocess\nsubprocess.run(['/bin/echo', 'x'])", "import os\nos.fork()",
                                  "import os\nos.execv('/usr/bin/prlimit', ['prlimit'])"])
def test_starting_processes_is_refused(body):
    assert "denied" in contained(probe(body))


def test_privileges_cannot_be_regained():
    body = ("import ctypes\nlibc = ctypes.CDLL(None)\n"
            "assert libc.mount(b'none', b'/usr', b'tmpfs', 0, None) == 0, 'mount refused'")
    assert "denied" in contained(probe(body))
    code = ("def decide(_):\n    import os\n    try:\n        os.setuid(0)\n        os.chroot('/')\n"
            "        return {'done': True}\n    except BaseException as e:\n        return {'denied': type(e).__name__}\n")
    assert "denied" in contained(code)


def test_inherited_descriptors_are_closed(hidden):
    fd = os.open(hidden, os.O_RDONLY)
    os.set_inheritable(fd, True)
    try:
        body = f"import os\nos.read({fd}, 100)"
        assert unsandboxed(probe(body), close_fds=False) == {"done": True}
        assert "denied" in contained(probe(body))
    finally:
        os.close(fd)


def test_environment_is_empty(monkeypatch):
    monkeypatch.setenv("TELOHELM_TEST_SECRET", "label=wrong")
    code = "import os\ndef decide(_):\n    return sorted(os.environ)\n"
    assert "TELOHELM_TEST_SECRET" in unsandboxed(code)
    assert "TELOHELM_TEST_SECRET" not in contained(code)


def test_other_processes_cannot_be_seen_or_signalled():
    assert "denied" in contained(probe("import os\nos.kill(1, 0)\nos.kill(os.getppid(), 0)\nassert os.getppid() > 1"))
    assert "denied" in contained(probe("open('/proc/1/cmdline').read()"))


def test_processor_time_is_limited():
    assert S.run("def decide(_):\n    while True:\n        pass\n", "decide", None).status == "timeout"


def test_wall_time_is_limited():
    r = S.run("def decide(_):\n    import time\n    time.sleep(30)\n", "decide", None, wall_seconds=1.0)
    assert r.status == "timeout" and r.wall_seconds < 3


def test_memory_above_the_limit_is_stopped():
    big = S.run("def decide(_):\n    x = bytearray(2 * 1024**3)\n    return len(x)\n", "decide", None)
    assert big.status == "memory"
    assert S.run("def decide(_):\n    return len(bytearray(50 * 1024**2))\n", "decide", None).status == "ok"


def test_errors_and_bad_replies_are_reported_not_raised():
    assert S.run("def decide(_):\n    raise ValueError('no')\n", "decide", None).status == "error"
    assert S.run("def decide(_):\n    return {1, 2}\n", "decide", None).status == "error"
    assert S.run("def decide(_):\n    return 'x' * 300000\n", "decide", None).status == "bad_reply"


def test_nothing_is_left_behind():
    before = set(os.listdir("/tmp"))
    S.run("def decide(_):\n    return 1\n", "decide", None)
    assert set(os.listdir("/tmp")) == before
