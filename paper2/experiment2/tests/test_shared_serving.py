"""Shared model servers (study2/runtime/slots.py) and concurrent comparison arms (study2/comparison.py run --concurrent)."""
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import paper2.experiment2.comparison as C; import telohelm.local_model as local_model
import telohelm.runtime.slots as slots

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())


class Slow:
    """A stand-in transport that takes `hold` seconds and records when each call ran."""

    def __init__(self, log: Path, hold: float = 0.3):
        self.log, self.hold = log, hold

    def __call__(self, url, body, timeout_s):
        start = time.time()
        time.sleep(self.hold)
        with self.log.open("a") as f:
            f.write(json.dumps({"pid": os.getpid(), "start": start, "end": time.time(), "timeout_s": timeout_s}) + "\n")
        return 200, {"ok": True}


def most_at_once(log: Path) -> int:
    """The most calls in flight at the same moment (an end and a start at the same time do not overlap)."""
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    events = sorted([(r["start"], 1) for r in rows] + [(r["end"], -1) for r in rows], key=lambda e: (e[0], e[1]))
    level = most = 0
    for _, step in events:
        level += step
        most = max(most, level)
    return most


def test_one_slot_lets_one_request_through_at_a_time_and_logs_each_wait(tmp_path):
    t = slots.SlotTransport(Slow(tmp_path / "calls.jsonl"), tmp_path / "slots", 1)
    threads = [threading.Thread(target=t, args=("http://x/v1", {"i": i}, 600.0)) for i in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert most_at_once(tmp_path / "calls.jsonl") == 1
    calls = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert {c["timeout_s"] for c in calls} == {600.0}               # waiting never shortens a request's own limit
    waits = sorted(r["wait_s"] for l in (tmp_path / "slots" / "waits.jsonl").read_text().splitlines()
                   if (r := json.loads(l))["event"] == "acquired")
    assert len(waits) == 3 and waits[0] < 0.2 and waits[-1] >= 0.5


def _worker(directory, log, n):
    t = slots.SlotTransport(Slow(Path(log)), Path(directory), n)
    for i in range(2):
        t("http://x/v1", {"pid": os.getpid(), "i": i}, 60.0)


def test_the_limit_holds_across_processes(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(str(tmp_path / "slots"), str(tmp_path / "calls.jsonl"), 2))
             for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert most_at_once(tmp_path / "calls.jsonl") == 2
    assert sum(json.loads(l)["event"] == "acquired" for l in
               (tmp_path / "slots" / "waits.jsonl").read_text().splitlines()) == 8


def test_the_provider_is_wrapped_only_for_a_configured_server(tmp_path, monkeypatch):
    url = "http://127.0.0.1:20001"
    monkeypatch.delenv(slots.ENV, raising=False)
    assert not isinstance(local_model.provider(url, "m")._transport, slots.SlotTransport)
    monkeypatch.setenv(slots.ENV, json.dumps({url: {"dir": str(tmp_path / "s"), "n": 1}}))
    wrapped = local_model.provider(url, "m", role="reasoning_model")._transport
    assert isinstance(wrapped, slots.SlotTransport) and wrapped.n == 1 and wrapped.origin == url
    assert not isinstance(local_model.provider("http://127.0.0.1:20002", "m")._transport, slots.SlotTransport)
    monkeypatch.setenv(slots.ENV, json.dumps({url: {"dir": str(tmp_path / "s"), "n": 0}}))
    with pytest.raises(ValueError):
        local_model.provider(url, "m")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("cmp") / "schedule"
    C.prepare(root, 4401, 6401, phases=(1,))
    return root


def test_two_arms_fly_at_once_from_separate_processes(prepared):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, "-m", 'paper2.experiment2.comparison', "run", "--root", str(prepared), "--max-steps", "3", "--concurrent"]
    procs = [subprocess.Popen(cmd + ["--arm", arm], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             for arm in ("calculator", "adaptive_controller")]
    outs = [p.communicate(timeout=600) for p in procs]
    assert [p.returncode for p in procs] == [0, 0], [o[1][-2000:] for o in outs]
    for arm in ("calculator", "adaptive_controller"):
        state = json.loads((prepared / f"{arm}-state.json").read_text())
        assert state["status"] == "returned" and len(state["positions"]) == 20
        assert state["worker_pid"] in {p.pid for p in procs}
        assert state["execution_started_epoch_s"] <= state["execution_finished_epoch_s"] <= time.time()
        assert state["execution_wall_s"] >= 0
    index = json.loads((prepared / "cost-index.json").read_text())
    assert index["arms"]["calculator"]["status_file"] and index["arms"]["adaptive_controller"]["status_file"]
    assert not list(prepared.glob("*.tmp"))
    # The serial mode still refuses a second dispatch of an arm already claimed.
    with pytest.raises(FileExistsError):
        C.run_arm(prepared, "calculator", max_steps=3)
