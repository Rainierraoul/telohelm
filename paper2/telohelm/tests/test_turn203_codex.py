"""Independent shared-serving controls; dummy transports and CPU processes only."""
import fcntl
import json
import multiprocessing
from pathlib import Path
import time

import pytest

import telohelm.runtime.slots as slots


class Inner:
    origin = "http://127.0.0.1:32123"
    max_response_bytes = 1000

    def __call__(self, url, body, timeout_s):
        if body.get("fail"):
            raise TimeoutError("dummy timeout")
        return 200, {"timeout": timeout_s}

    def exchange(self, url, body, timeout_s):
        return {"outcome": "received", "request_wire": b"request", "reply_wire": b"reply", "timeout": timeout_s}


def rows(root):
    p = root / "waits.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def test_raw_exchange_path_keeps_bytes_and_uses_the_same_slot(tmp_path):
    t = slots.SlotTransport(Inner(), tmp_path, 1)
    result = t.exchange(Inner.origin + "/v1/chat/completions", {"messages": []}, 7.)
    assert result["request_wire"] == b"request" and result["reply_wire"] == b"reply"
    assert result["timeout"] == 7.


def test_two_wrappers_cannot_disagree_about_one_directorys_capacity(tmp_path):
    slots.SlotTransport(Inner(), tmp_path, 1)
    with pytest.raises(ValueError, match="configuration"):
        slots.SlotTransport(Inner(), tmp_path, 2)


def _queued(directory, ready):
    t = slots.SlotTransport(Inner(), Path(directory), 1)
    ready.set()
    t(Inner.origin + "/v1/chat/completions", {}, 3.)


def test_killed_waiter_keeps_its_queue_receipt_before_any_slot_is_available(tmp_path):
    slots.SlotTransport(Inner(), tmp_path, 1)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    with (tmp_path / "slot-0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proc = ctx.Process(target=_queued, args=(str(tmp_path), ready))
        proc.start()
        try:
            assert ready.wait(5)
            end = time.monotonic() + 1
            while not rows(tmp_path) and time.monotonic() < end:
                time.sleep(.01)
            assert [r.get("event") for r in rows(tmp_path)] == ["queued"]
        finally:
            proc.terminate()
            proc.join(5)
            fcntl.flock(lock, fcntl.LOCK_UN)


def test_exception_releases_lease_and_records_separate_wait_and_service(tmp_path, monkeypatch):
    monkeypatch.setenv("TELOHELM_ARM", "full")
    monkeypatch.setenv("TELOHELM_STREAM", "4401")
    t = slots.SlotTransport(Inner(), tmp_path, 1)
    with pytest.raises(TimeoutError):
        t(Inner.origin + "/v1", {"fail": True}, 2.)
    assert t(Inner.origin + "/v1", {}, 9.) == (200, {"timeout": 9.})
    events = rows(tmp_path)
    assert [r["event"] for r in events] == ["queued", "acquired", "released"] * 2
    assert len({r["wait_id"] for r in events}) == 2
    assert all(r["arm"] == "full" and r["stream"] == "4401" for r in events)
    assert events[2]["outcome"] == "TimeoutError"
    assert events[1]["wait_s"] >= 0 and events[2]["held_s"] >= 0
    assert events[4]["monotonic_s"] >= events[2]["monotonic_s"]


def test_two_servers_cannot_silently_share_a_slot_directory(tmp_path):
    slots.SlotTransport(Inner(), tmp_path, 1)
    other = Inner()
    other.origin = "http://127.0.0.1:32124"
    with pytest.raises(ValueError, match="configuration"):
        slots.SlotTransport(other, tmp_path, 1)

