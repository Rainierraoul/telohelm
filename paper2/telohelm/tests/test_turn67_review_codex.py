"""TURN67: independent invented recovery/replay controls, without actual model calls."""
from __future__ import annotations

import json

import pytest

import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import verify
from .test_turn65_review_codex import MSG; from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit


# Fixed in TURN 69 (Claude): MG67.COST.FRESH: missing original charge silently accepted in fresh ledger
def test_recovery_cannot_hide_cost_when_original_charge_is_absent(gateway, tmp_path):
    journal = tmp_path / "shared"
    first, _, original, _ = gateway(journal=journal)
    submit(first)
    resumed, ledger, provider, _ = gateway(journal=journal)
    try:
        out = submit(resumed)
    except M.GatewayError:
        assert provider.calls == 0
        return  # Refusing absent original accounting before reliance is also safe.
    assert original.calls == 1 and provider.calls == 0 and out["recovered"]
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 7
    assert sum(c.tokens_out or 0 for c in ledger.calls) == 2


# Fixed in TURN 69 (Claude): MG67.COST.COUNT: reconciliation rows counted as new calls at endings
def test_reconciliation_does_not_inflate_calls_at_an_alternative_ending(gateway, tmp_path):
    journal = tmp_path / "shared"
    ledger = Ledger()
    first, _, _, _ = gateway(journal=journal, ledger=ledger)
    submit(first)
    before = ledger.ending_calls("episode-a", 12)
    for _ in range(2):
        resumed, _, provider, _ = gateway(journal=journal, ledger=ledger)
        assert submit(resumed)["recovered"] and provider.calls == 0
    assert ledger.ending_calls("episode-a", 12) == before


# Fixed in TURN 69 (Claude): MG67.ID.LEDGER: metadata mismatch calls provider before duplicate ledger id fails
def test_different_metadata_is_refused_early_or_gets_a_distinct_charge(gateway, tmp_path):
    journal = tmp_path / "shared"
    ledger = Ledger()
    first, _, _, _ = gateway(journal=journal, ledger=ledger)
    submit(first)
    resumed, _, provider, root = gateway(journal=journal, ledger=ledger)
    try:
        out = submit(resumed, goal=2)
    except M.GatewayError:
        assert provider.calls == 0 and len(ledger.calls) == 1
        return
    assert provider.calls == 1 and len(ledger.calls) == 2
    assert out["goal_version"] == 2 and not out["recovered"]
    assert len({c.request_id for c in ledger.calls}) == 2
    events = [json.loads(line)["payload"] for line in (root / "log").read_text().splitlines()[1:]]
    assert events[-1]["kind"] == "model_end" and verify(root / "log", root / "heads")["ok"]


# Fixed in TURN 69 (Claude): MG67.ID.REPLAY: stored provider answer lacks original goal metadata
def test_provider_replay_does_not_relabel_a_stored_old_goal_answer(gateway):
    first, ledger, _, _ = gateway()
    old = submit(first)
    rid = ledger.calls[0].request_id
    request = M.ModelRequest(rid, "attempt1", "episode-a", "condition-a", "stream-a", "small_model",
                             tuple(MSG), 256, 1, 10, 10, 2.4)
    stored = M.ModelResponse("answered", old["text"], 7, 2,
                             origin={"provider": "scripted", "reportable": False})
    replay, _, _, _ = gateway(provider=M.ReplayProvider({(rid, request.fingerprint): stored}), replay=True)
    try:
        out = submit(replay, goal=2)
    except M.GatewayError:
        return
    assert out["status"] != "answered" or out["goal_version"] == 1


# Fixed in TURN 69 (Claude): MG67.ORIGIN.SCRIPTED: response origin overrides known scripted nonreportability
def test_scripted_source_cannot_promote_its_output_with_an_origin_field(gateway):
    response = M.ModelResponse("answered", "invented", 7, 2,
                               origin={"provider": "local-model", "reportable": True})
    gw, _, _, root = gateway(provider=M.ScriptedProvider(lambda r: response))
    assert submit(gw)["reportable"] is False
    records = [json.loads(line)["payload"] for line in (root / "log").read_text().splitlines()[1:]]
    assert all(record["reportable"] is False for record in records)


@pytest.mark.parametrize("status", ["answered", "failed"])
def test_three_recoveries_preserve_charge_timing_and_protected_logs(gateway, tmp_path, status):
    journal = tmp_path / "shared"
    ledger = Ledger()
    ticks = iter([4., 6.5])
    response = M.ModelResponse(status, "invented" if status == "answered" else None, 7, 2)
    first, _, _, _ = gateway(journal=journal, ledger=ledger, clock=lambda: next(ticks),
                             provider=M.ScriptedProvider(lambda r: response))
    submit(first)
    for _ in range(3):
        resumed, _, provider, root = gateway(journal=journal, ledger=ledger)
        out = submit(resumed)
        assert provider.calls == 0 and out["recovered"] and out["status"] == status
        assert verify(root / "log", root / "heads")["ok"]
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 7
    assert sum(c.tokens_out or 0 for c in ledger.calls) == 2
    assert [c.wall_s for c in ledger.calls] == [2.5] * 4
    assert all(c.recovers == ledger.calls[0].request_id for c in ledger.calls[1:])


@pytest.mark.parametrize("origin", [{}, {"provider": "local-model"},
                                   {"provider": "", "reportable": True},
                                   {"provider": "local-model", "reportable": 1},
                                   {"provider": "local-model", "reportable": True, "extra": 1}])
def test_incomplete_or_mistyped_origin_is_unreportable_live_and_recovered(gateway, tmp_path, origin):
    journal = tmp_path / "shared"
    response = M.ModelResponse("answered", "invented", 7, 2, origin=origin)
    first, ledger, _, _ = gateway(journal=journal, provider=M.ScriptedProvider(lambda r: response))
    assert submit(first)["reportable"] is False
    resumed, _, provider, _ = gateway(journal=journal, ledger=ledger)
    assert submit(resumed)["reportable"] is False and provider.calls == 0
