"""Tests for the procedure library and its crash recovery (architecture section 7)."""
from __future__ import annotations

import json

import pytest

import telohelm.runtime.library as L


def entry(n: int, kind: str = "procedure") -> dict:
    return {"kind": kind, "decision": "heading", "conditions": [],
            "source": f"def decide(ctx):\n    return {{'heading_deg': {n}}}\n", "origin": {"calls": [f"r{n}"]}}


def test_entries_are_content_addressed_and_immutable(tmp_path):
    lib = L.Library(tmp_path)
    a = lib.add(entry(1))
    assert lib.add(entry(1)) == a                                  # same content, same fingerprint
    assert lib.entry(a)["source"].endswith("{'heading_deg': 1}\n")
    (tmp_path / "content" / f"{a}.json").write_text("{}")
    with pytest.raises(L.LibraryError):
        lib.entry(a)


def test_statuses_are_projections_and_can_be_restored(tmp_path):
    lib = L.Library(tmp_path)
    a, b = lib.add(entry(1)), lib.add(entry(2))
    lib.set_status(a, "adopted", "passed replay")
    seq_before = lib.projection()["seq"]
    lib.set_status(a, "suspended", "replay fault")
    lib.set_status(b, "adopted", "repair")
    assert lib.adopted("heading", "procedure") == [b]
    earlier = lib.projection(upto_seq=seq_before)
    assert earlier["entries"][a]["status"] == "adopted" and earlier["entries"][b]["status"] == "candidate"
    assert json.loads((tmp_path / "index.json").read_text()) == json.loads(L.canonical(lib.projection()))
    with pytest.raises(L.LibraryError):
        lib.set_status(a, "deleted", "")


def test_log_survives_reopening(tmp_path):
    lib = L.Library(tmp_path)
    a = lib.add(entry(1))
    lib.set_status(a, "adopted", "ok")
    again = L.Library(tmp_path)
    assert again.projection() == lib.projection()


def test_crash_after_content_leaves_unreferenced_content_and_no_event(tmp_path):
    L.Library(tmp_path).add(entry(1))
    with pytest.raises(L.Crash):
        L.Library(tmp_path, crash_at="after_content").add(entry(2))
    lib = L.Library(tmp_path)
    r = lib.recover()
    assert r["removed_bytes"] == 0 and len(r["unreferenced"]) == 1
    assert len(lib.projection()["entries"]) == 1


def test_crash_mid_event_removes_and_records_the_torn_record(tmp_path):
    L.Library(tmp_path).add(entry(1))
    with pytest.raises(L.Crash):
        L.Library(tmp_path, crash_at="mid_event").add(entry(2))
    lib = L.Library(tmp_path)
    assert len(lib.events) == 1
    with pytest.raises(L.LibraryError, match="torn"):
        lib.add(entry(3))                                           # no writing past a torn record
    r = lib.recover()
    assert r["removed_bytes"] > 0 and lib.events[-1]["kind"] == "recovery"
    assert len(lib.projection()["entries"]) == 1
    again = L.Library(tmp_path)
    assert again.events == lib.events                             # the log is whole again after recovery


def test_crash_after_event_rebuilds_a_stale_index(tmp_path):
    lib0 = L.Library(tmp_path)
    a = lib0.add(entry(1))
    with pytest.raises(L.Crash):
        L.Library(tmp_path, crash_at="after_event").set_status(a, "adopted", "ok")
    stale = json.loads((tmp_path / "index.json").read_text())
    assert stale["entries"][a]["status"] == "candidate"            # the index missed the completed event
    lib = L.Library(tmp_path)
    lib.recover()
    assert json.loads((tmp_path / "index.json").read_text())["entries"][a]["status"] == "adopted"


def test_a_corrupted_record_stops_the_log_there(tmp_path):
    lib = L.Library(tmp_path)
    lib.add(entry(1))
    lib.add(entry(2))
    data = (tmp_path / "events.log").read_bytes().replace(b'"heading"', b'"HEADING"', 1)
    (tmp_path / "events.log").write_bytes(data)
    damaged = L.Library(tmp_path)
    assert len(damaged.events) == 0                                 # first record no longer matches its fingerprint
    with pytest.raises(L.LibraryError, match="damaged"):
        damaged.recover()                                           # later records are never cut away
    assert (tmp_path / "events.log").read_bytes() == data
    with pytest.raises(L.LibraryError, match="damaged"):
        damaged.add(entry(3))


def test_the_library_applies_the_reply_checks_itself(tmp_path):
    lib = L.Library(tmp_path)
    bad = dict(entry(1), source="import os\ndef decide(ctx):\n    return {}\n")
    with pytest.raises(L.LibraryError, match="refused"):
        lib.add(bad)
    with pytest.raises(L.LibraryError, match="refused"):
        lib.add(dict(entry(1), kind=[]))


# -- groups of status changes (Codex's PAIR153.VISIBLE and UNDO.CUT) ------------------------------------------

def two_adopted(tmp_path):
    lib = L.Library(tmp_path)
    old, new = lib.add(entry(1)), lib.add(entry(2))
    lib.set_status(old, "adopted", "start")
    return lib, old, new


def test_a_group_is_invisible_until_closed_and_then_whole(tmp_path):
    lib, old, new = two_adopted(tmp_path)
    plan = [{"entry": new, "status": "adopted", "reason": "r"}, {"entry": old, "status": "suspended", "reason": "r"}]
    g = lib.open_group(plan, "swap")
    lib.set_status(new, "adopted", "r", group=g)
    assert lib.adopted("heading", "procedure") == [old]              # half a group is never seen
    lib.set_status(old, "suspended", "r", group=g)
    assert lib.adopted("heading", "procedure") == [old]
    lib.close_group(g)
    assert lib.adopted("heading", "procedure") == [new]
    assert lib.projection(upto_seq=g)["entries"][new]["status"] == "candidate"


def test_an_interrupted_group_is_finished_once_from_its_plan_after_a_restart(tmp_path):
    lib, old, new = two_adopted(tmp_path)
    plan = [{"entry": new, "status": "adopted", "reason": "r"}, {"entry": old, "status": "suspended", "reason": "r"}]
    g = lib.open_group(plan, "swap")
    lib.set_status(new, "adopted", "r", group=g)                     # the process stops here
    again = L.Library(tmp_path)
    again.recover()
    assert again.adopted("heading", "procedure") == [old] and list(again.open_groups()) == [g]
    with pytest.raises(L.LibraryError):
        again.open_group(plan, "a second group while one is open")
    assert again.complete_open_groups() == [g]
    assert again.complete_open_groups() == []                        # a repeat restart finds nothing to do
    third = L.Library(tmp_path)
    assert third.adopted("heading", "procedure") == [new] and not third.open_groups()


def test_a_torn_group_record_leaves_the_earlier_state(tmp_path):
    lib, old, new = two_adopted(tmp_path)
    lib.crash_at = "mid_event"
    with pytest.raises(L.Crash):
        lib.open_group([{"entry": new, "status": "adopted", "reason": "r"}], "swap")
    again = L.Library(tmp_path)
    assert again.recover()["removed_bytes"] > 0
    assert again.adopted("heading", "procedure") == [old] and not again.open_groups()


def test_a_status_change_cannot_name_a_closed_or_unknown_group(tmp_path):
    lib, old, new = two_adopted(tmp_path)
    g = lib.open_group([{"entry": new, "status": "adopted", "reason": "r"}], "one")
    lib.close_group(g)
    with pytest.raises(L.LibraryError):
        lib.set_status(new, "adopted", "late", group=g)
    with pytest.raises(L.LibraryError):
        lib.open_group([{"entry": "no-such-entry", "status": "adopted", "reason": "r"}], "bad plan")
