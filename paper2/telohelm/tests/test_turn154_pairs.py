"""TURN 154 repairs of Codex's PAIR153 controls: dependencies read from the library, adoption and undo as one
library group, and an interrupted group finished once before anything new. Invented entries; no flights."""
from __future__ import annotations

import telohelm.learning_monitor as GM  # replay qualification moved to the monitor

import pytest

import telohelm.learning as G
from .test_turn153_review_codex import adopt; from .test_turn153_review_codex import restart; from .test_turn153_review_codex import setup; from .test_turn153_review_codex import states


def test_dependencies_survive_two_restarts_and_undo_still_works(tmp_path, monkeypatch):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    e = adopt(co, b)
    first, _ = restart(tmp_path)
    assert first.reconcile_pairs() == []
    second, lib3 = restart(tmp_path)
    assert second.reconcile_pairs() == []
    assert second.pair_of == {e["paired_check"]: e["candidate"]}
    assert second.replaced[e["paired_check"]] == old["check"]
    assert second.replaced[e["candidate"]] == old["procedure"]
    monkeypatch.setattr(GM, "replay_test", lambda *a, **k: {"wrong_accepted": [0]})
    second.confirm_wrong({"context": {"t": 0}, "output": {"heading_deg": 77}})
    assert states(lib3)[e["paired_check"]] == "retired" and states(lib3)[e["candidate"]] == "suspended"
    assert states(lib3)[old["procedure"]] == states(lib3)[old["check"]] == "adopted"
    assert second.pair_of == {}


def test_a_single_rewrite_is_adopted_as_one_group(tmp_path, monkeypatch):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    real = lib.set_status
    def cut(sha, status, why, **kw):
        if status == "suspended" and sha == old["check"]:
            raise RuntimeError("invented process stop")
        return real(sha, status, why, **kw)
    monkeypatch.setattr(lib, "set_status", cut)
    new = lib.add({"kind": "check", "decision": "heading", "conditions": [],
                   "source": "def check(ctx, out):\n    return 'accept'\n",
                   "origin": {"attempt": 9, "predecessor": old["check"]}})
    with pytest.raises(RuntimeError):
        co._change([(new, "adopted", "attempt 9 passed"), (old["check"], "suspended", f"replaced by {new}")],
                   "attempt 9 adopted", None)
    recovered, lib2 = restart(tmp_path)
    assert lib2.adopted("heading", "check") == [old["check"]]        # never two adopted checks
    recovered.reconcile_pairs()
    assert lib2.adopted("heading", "check") == [new] and states(lib2)[old["check"]] == "suspended"


def test_undo_does_not_revive_a_retired_predecessor(tmp_path, monkeypatch):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    e = adopt(co, b)
    lib.set_status(old["check"], "retired", "invented: retired on its own evidence")
    monkeypatch.setattr(GM, "replay_test", lambda *a, **k: {"wrong_accepted": [0]})
    events = co.confirm_wrong({"context": {"t": 0}, "output": {"heading_deg": 77}})
    assert states(lib)[old["check"]] == "retired"                     # not brought back
    assert states(lib)[e["paired_check"]] == "retired" and states(lib)[e["candidate"]] == "suspended"
    assert states(lib)[old["procedure"]] == "adopted"
    assert events[0] == {"kind": "undo", "retired": e["paired_check"], "restored": None}


def test_an_attempt_finishes_an_interrupted_group_before_it_starts(tmp_path, monkeypatch):
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
    assert lib2.open_groups()
    recovered.attempt(b)                                              # refused (no trial set), but reconciles first
    assert not lib2.open_groups()
    assert len(lib2.adopted("heading", "procedure")) == len(lib2.adopted("heading", "check")) == 1
    assert lib2.adopted("heading", "procedure") != [old["procedure"]]
