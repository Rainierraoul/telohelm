"""Independent status-cut observations for the exact grouped pair repair.

Invented library and trial/replay replies only; no world, model or sandbox flight.
"""

import telohelm.learning_monitor as GM  # replay qualification moved to the monitor
import pytest

import telohelm.learning as G
from .test_turn153_review_codex import setup; from .test_turn153_review_codex import adopt; from .test_turn153_review_codex import restart; from .test_turn153_review_codex import states


@pytest.mark.parametrize("cut_after", [1, 2, 3, 4])
def test_each_adoption_status_cut_keeps_prior_pair_then_completes_once(tmp_path, monkeypatch, cut_after):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    baseline = states(lib)
    real = lib.set_status
    n = 0
    def cut(sha, status, why, **kw):
        nonlocal n
        result = real(sha, status, why, **kw)
        if kw.get("group") is not None:
            n += 1
            if n == cut_after:
                raise RuntimeError("invented status cut")
        return result
    monkeypatch.setattr(lib, "set_status", cut)
    with pytest.raises(RuntimeError):
        adopt(co, b)
    recovered, again = restart(tmp_path)
    assert again.adopted("heading", "procedure") == [old["procedure"]]
    assert again.adopted("heading", "check") == [old["check"]]
    assert all(states(again)[sha] == status for sha, status in baseline.items())
    # Recovery completes from durable plans and originals; no callbacks or model work.
    monkeypatch.setattr(recovered, "trial", lambda *a, **k: pytest.fail("recovery repeated trial"))
    monkeypatch.setattr(recovered, "check_replay", lambda *a, **k: pytest.fail("recovery repeated replay"))
    recovered.reconcile_pairs()
    new_proc = again.adopted("heading", "procedure")
    new_check = again.adopted("heading", "check")
    assert len(new_proc) == len(new_check) == 1
    assert new_proc != [old["procedure"]] and new_check != [old["check"]]
    before = len(again.events)
    assert recovered.reconcile_pairs() == [] and len(again.events) == before
    third, last = restart(tmp_path)
    assert third.reconcile_pairs() == [] and len(last.events) == before
    assert third.pair_of == {new_check[0]: new_proc[0]}


@pytest.mark.parametrize("cut_after", [1, 2, 3, 4])
def test_each_linked_undo_cut_recovers_whole_original_disposition(tmp_path, monkeypatch, cut_after):
    co, lib, old, b, seen = setup(tmp_path, monkeypatch)
    event = adopt(co, b)
    monkeypatch.setattr(GM, "replay_test", lambda *a, **k: {"wrong_accepted": [0]})
    real = lib.set_status
    n = 0
    def cut(sha, status, why, **kw):
        nonlocal n
        result = real(sha, status, why, **kw)
        if kw.get("group") is not None:
            n += 1
            if n == cut_after:
                raise RuntimeError("invented undo cut")
        return result
    monkeypatch.setattr(lib, "set_status", cut)
    with pytest.raises(RuntimeError):
        co.confirm_wrong({"context": {"t": 0}, "output": {"heading_deg": 77}})
    recovered, again = restart(tmp_path)
    # Projection is atomic. This control explicitly completes pending undo BEFORE use;
    # it does not certify a live chooser that omits startup reconciliation.
    recovered.reconcile_pairs()
    assert states(again)[event["paired_check"]] == "retired"
    assert states(again)[event["candidate"]] == "suspended"
    assert again.adopted("heading", "procedure") == [old["procedure"]]
    assert again.adopted("heading", "check") == [old["check"]]
    before = len(again.events)
    third, last = restart(tmp_path)
    assert third.reconcile_pairs() == [] and len(last.events) == before
    assert third.pair_of == {}
