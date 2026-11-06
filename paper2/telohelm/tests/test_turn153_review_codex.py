"""Exact TURN152 paired-adoption controls; invented cases, no model/world flight.

Strict markers preserve unmet D27 requirements. Remove only after repair.
"""

import telohelm.learning_monitor as GM  # replay qualification moved to the monitor
import json
from types import SimpleNamespace

import pytest

import telohelm.learning as G
from telohelm.runtime.library import Library
from telohelm.runtime.procedures import reply_errors
from telohelm.runtime.chooser import choose


def pair():
    return {"kind": "pair", "decision": "heading",
            "procedure": {"conditions": [], "source": 'def decide(ctx):\n    return {"heading_deg": 77.0}\n'},
            "check": {"conditions": [], "source": "def check(ctx, out):\n    return 'accept'\n"},
            "reason": {"code": "missing_tool", "text": "invented CPU control"}}


def setup(tmp_path, monkeypatch):
    lib = Library(tmp_path / "library")
    old = {}
    for kind, source in (("procedure", 'def decide(ctx):\n    return {"heading_deg": 12.0}\n'),
                         ("check", "def check(ctx, out):\n    return 'reject'\n")):
        old[kind] = lib.add({"kind": kind, "decision": "heading", "conditions": [],
                            "source": source, "origin": {"invented": True}})
        lib.set_status(old[kind], "adopted", "invented starting library")
    seen = []
    def fly(shadow, c, root):
        seen.append({k: shadow.adopted("heading", k) for k in ("procedure", "check")})
        return {"goal_met": True, "help": 0, "episode": c.episode}
    co = G.Coordinator(lib, tmp_path, None, lambda s: True, lambda a: {}, fly=fly)
    monkeypatch.setattr(co, "check_replay", lambda *a: (True, {"invented_replay_pass": True}))
    boundary = G.Completed(5, "invented-e", None, False, "both", True)
    return co, lib, old, boundary, seen


def adopt(co, boundary):
    return co._attempt_pair(1, 1, pair(), boundary, [boundary], [{"failed": "both"}])


def restart(tmp_path):
    lib = Library(tmp_path / "library")
    lib.recover()
    return G.Coordinator(lib, tmp_path, None, lambda s: True, lambda a: {}), lib


def states(lib):
    return {s: x["status"] for s, x in lib.projection()["entries"].items()}


def test_pair_schema_rejects_cross_decision_and_malformed_halves():
    p = pair()
    assert reply_errors(p) == []
    p["procedure"]["decision"] = "report"
    assert reply_errors(p)
    p = pair()
    p["check"] = None
    assert reply_errors(p)
    p = pair()
    p["decision"] = []
    assert reply_errors(p)


def test_same_process_pair_and_linked_undo_positive(tmp_path, monkeypatch):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    e = adopt(co, b)
    assert e["passed"] and seen == [{"procedure": [e["candidate"]], "check": [e["paired_check"]]}]
    assert states(lib)[old["procedure"]] == states(lib)[old["check"]] == "suspended"
    monkeypatch.setattr(GM, "replay_test", lambda *a, **k: {"wrong_accepted": [0]})
    co.confirm_wrong({"context": {"t": 0}, "output": {"heading_deg": 77}})
    assert states(lib)[e["candidate"]] == "suspended"
    assert states(lib)[e["paired_check"]] == "retired"
    assert states(lib)[old["procedure"]] == states(lib)[old["check"]] == "adopted"


def test_manual_pending_recovery_positive(tmp_path, monkeypatch):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    real = lib.set_status
    def cut(sha, status, why, **kw):
        if "pair from attempt" in why and lib.entry(sha)["kind"] == "procedure":
            raise RuntimeError("invented process stop")
        return real(sha, status, why, **kw)
    monkeypatch.setattr(lib, "set_status", cut)
    with pytest.raises(RuntimeError):
        adopt(co, b)
    recovered, lib2 = restart(tmp_path)
    assert len(recovered.reconcile_pairs()) == 1
    assert recovered.reconcile_pairs() == []
    row = json.loads((tmp_path / "pairs.jsonl").read_text().splitlines()[0])
    assert lib2.adopted("heading", "procedure") == [row["procedure"]]
    assert lib2.adopted("heading", "check") == [row["check"]]
    assert recovered.pair_of[row["check"]] == row["procedure"]


def test_completed_pair_restores_undo_dependencies_on_restart(tmp_path, monkeypatch):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    e = adopt(co, b)
    recovered, lib2 = restart(tmp_path)
    recovered.reconcile_pairs()
    monkeypatch.setattr(GM, "replay_test", lambda *a, **k: {"wrong_accepted": [0]})
    recovered.confirm_wrong({"context": {"t": 0}, "output": {"heading_deg": 77}})
    assert states(lib2)[e["paired_check"]] == "retired", "completed original check must remain monitored after restart"
    assert states(lib2)[e["candidate"]] == "suspended"
    assert states(lib2)[old["procedure"]] == states(lib2)[old["check"]] == "adopted"


def test_interrupted_pair_is_hidden_or_reconciled_before_chooser_use(tmp_path, monkeypatch):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    real = lib.set_status
    def cut(sha, status, why, **kw):
        if "pair from attempt" in why and lib.entry(sha)["kind"] == "procedure":
            raise RuntimeError("invented process stop")
        return real(sha, status, why, **kw)
    monkeypatch.setattr(lib, "set_status", cut)
    with pytest.raises(RuntimeError):
        adopt(co, b)
    recovered, lib2 = restart(tmp_path)
    # An ordinary reader can use the live library before the optional manual reconcile call.
    # No sandbox executed: the stand-in reports each inspected method's result.
    def run(source, entry, payload, **kw):
        value = {"heading_deg": 12.0} if entry == "decide" else "accept"
        return SimpleNamespace(status="ok", result=value, error=None, wall_seconds=0, cpu_seconds=0)
    result = choose("heading", {"report_x": 1, "report_y": 2}, lib2, run=run)
    assert result.get("use") != old["procedure"] or result["checks"] == [old["check"]], (
        "old procedure exposed with candidate check before joint publication")


def test_interrupted_linked_undo_finishes_before_pair_is_used_again(tmp_path, monkeypatch):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    e = adopt(co, b)
    monkeypatch.setattr(GM, "replay_test", lambda *a, **k: {"wrong_accepted": [0]})
    real = lib.set_status
    def cut(sha, status, why, **kw):
        if sha == e["candidate"] and status == "suspended":
            raise RuntimeError("invented undo stop")
        return real(sha, status, why, **kw)
    monkeypatch.setattr(lib, "set_status", cut)
    with pytest.raises(RuntimeError):
        co.confirm_wrong({"context": {"t": 0}, "output": {"heading_deg": 77}})
    recovered, lib2 = restart(tmp_path)
    recovered.reconcile_pairs()
    assert states(lib2)[e["paired_check"]] == "retired"
    assert states(lib2)[e["candidate"]] == "suspended", "original dependent procedure must not survive an interrupted undo"
    assert states(lib2)[old["procedure"]] == states(lib2)[old["check"]] == "adopted"
