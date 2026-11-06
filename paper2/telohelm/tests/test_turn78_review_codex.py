"""TURN78 independent attribution/provenance controls. Invented CPU data; no research collection."""
import pytest

import telohelm.runtime.attribution as A; import telohelm.runtime.episode_loop as L; import telohelm.runtime.models as M
from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify
from .test_attribution import fly; from .test_attribution import derive
from .test_delivery import joining_scenario
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit
from .test_turn70_review_codex import flight; from .test_turn70_review_codex import execute; from .test_turn70_review_codex import records
from .test_turn76_review_codex import InventedProducer


def signed_trace(tmp_path, change):
    """An authenticated but semantically incomplete trusted-writer fixture, not a sandbox forgery."""
    log, heads = tmp_path / "trace", tmp_path / "heads"
    writer = TrustedLog(log, heads, "invented")
    decision = {"kind": "decision", "decision": "heading", "source": "procedure",
                "method": "invented-procedure-sha", "request_id": None, "context_sha256": "a" * 64,
                "part": {"heading_deg": 90.}, "goal_version": 1}
    action = {"kind": "action", "action": {"heading_deg": 90., "goal_version": 1},
              "field_writers": {"heading_deg": "heading"}, "pending": {"heading": False}}
    world = {"kind": "world_step", "accepted": True, "ended": True, "action_steps": 1}
    if change == "accepted":
        world.pop("accepted")
    elif change == "pending":
        action["pending"] = {}
        action["action"] = {}
        action["field_writers"] = {}
    elif change == "writer":
        action["field_writers"] = {}
    elif change == "wrong_writer":
        action["field_writers"]["heading_deg"] = "report"
    elif change == "model_link":
        decision.update(source="small_model", method="model:missing", request_id=None)
    for row in [{"kind": "bundle", "bundle": {"t": 0}}, decision, action, world]:
        # Adapted in TURN 79 (Claude): every broker event carries its action step, which the derivation now checks;
        # test_attribution.py shows this trace derives cleanly when change is None.
        writer.append({"episode": "invented-episode", "t": 0, "action_step": 0, **row})
    assert verify(log, heads)["ok"]
    return log, heads


# Fixed in TURN 79 (Claude): AT78.PROV: missing or inconsistent semantic provenance is guessed as holding or accepted model source
@pytest.mark.parametrize("change", ["accepted", "pending", "writer", "wrong_writer", "model_link"])
def test_authenticated_trace_with_missing_semantic_provenance_fails(tmp_path, change):
    log, heads = signed_trace(tmp_path, change)
    with pytest.raises(A.AttributionError):
        A.derive(log, heads, "invented-episode")


# Fixed in TURN 79 (Claude): AT78.PROV: an unknown requested episode is returned as an empty result
def test_unknown_episode_is_not_fabricated_as_a_zero_length_episode(tmp_path, monkeypatch):
    _, root, _ = fly(tmp_path, monkeypatch, lambda t: 90., delayed=False, steps=1)
    with pytest.raises(A.AttributionError):
        A.derive(root / "blog", root / "bheads", "invented-unrecorded-episode")


# Fixed in TURN 79 (Claude): AT78.HELD: held_by keeps only the category and loses the original model request identity
def test_held_command_retains_original_model_request_identity(tmp_path, monkeypatch):
    parts = flight(tmp_path, monkeypatch, latency=0., steps=2)
    def choose(decision, ctx, *args, **kw):
        if decision == "heading" and ctx["t"] == 0:
            return {"climb": "missing_tool", "tried": []}
        return {"use": "invented-no-command", "output": {}, "tried": [], "checks": []}
    monkeypatch.setattr(L, "choose", choose)
    execute(parts, tmp_path)
    out = A.derive(tmp_path / "blog", tmp_path / "bheads", parts[0].episode_id)
    assert [s["category"] for s in out["steps"]] == ["small_model", "holding_none"]
    original = out["steps"][0]["request_id"]
    # Adapted in TURN 79 (Claude): the held command's provenance is nested under "held" (an accepted equivalent schema).
    assert original and out["steps"][1]["held"]["request_id"] == original


# Fixed in TURN 79 (Claude): AT78.POOL: incomplete prefixes are mixed into completed-episode shares
def test_pool_requires_explicit_separation_of_completed_and_incomplete_episodes(tmp_path, monkeypatch):
    _, aroot, ep = fly(tmp_path, monkeypatch, lambda t: 0., delayed=False, steps=3, name="completed", sc=joining_scenario())
    _, broot, ep2 = fly(tmp_path, monkeypatch, lambda t: None, delayed=False, steps=1, name="prefix")
    complete, prefix = derive(aroot, ep), derive(broot, ep2)
    assert complete["complete"] and not prefix["complete"]
    # A separate grouped output or an explicit mixed-prefix mode is an acceptable marked API adaptation;
    # the conservative default here requires a declared choice before pooling unlike scopes.
    with pytest.raises(A.AttributionError):
        A.pool([complete, prefix])


@pytest.mark.parametrize("origin", [{"reportable": True}, {"provider": "\ud800", "reportable": True}])
def test_malformed_origin_is_unreportable_with_safe_ids_and_once_only_recovery(gateway, tmp_path, origin):
    response = M.ModelResponse("answered", "\ud800", 11, 4, {"id": "safe-id", "model": "safe-model"}, origin)
    p = InventedProducer(response)
    journal = tmp_path / "shared"
    first, ledger, _, root = gateway(provider=p, journal=journal)
    assert submit(first)["reportable"] is False
    resumed, _, _, rroot = gateway(provider=p, ledger=ledger, journal=journal)
    out = submit(resumed)
    assert out["recovered"] and out["reportable"] is False and p.calls == 1
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 11
    assert verify(root / "log", root / "heads")["ok"] and verify(rroot / "log", rroot / "heads")["ok"]


def test_independent_report_refusal_cancels_an_otherwise_valid_heading_after_scoping(tmp_path, monkeypatch):
    parts = flight(tmp_path, monkeypatch, latency=1., steps=2)
    def choose(decision, ctx, *args, **kw):
        if decision == "heading":
            return {"climb": "missing_tool", "tried": []}
        return {"use": "invented-report", "output": {"report": ""}, "tried": [], "checks": []}
    monkeypatch.setattr(L, "choose", choose)
    execute(parts, tmp_path)
    out = A.derive(tmp_path / "blog", tmp_path / "bheads", parts[0].episode_id)
    assert [s["category"] for s in out["steps"]] == ["holding_pending", "holding_none"]
    replies = [e for e in records(tmp_path / "blog") if e["kind"] == "model_reply"]
    assert replies[0]["part"]["heading_deg"] == 123. and replies[0]["role"] == "small_model"
    world = [e for e in records(tmp_path / "blog") if e["kind"] == "world_step"]
    assert world[-1]["accepted"] is False


def test_actual_journal_replay_preserves_full_attribution_and_logical_cut_without_inference(tmp_path, monkeypatch):
    live, root, ep = fly(tmp_path, monkeypatch, lambda t: None, name="live", steps=5)
    original_gateway = M.ModelGateway
    providers = []
    def from_original_journal(*args, **kwargs):
        args = list(args)
        providers.append(args[3])
        args[6] = root / "journal"
        return original_gateway(*args, **kwargs)
    monkeypatch.setattr(M, "ModelGateway", from_original_journal)
    replay, rroot, rep = fly(tmp_path, monkeypatch, lambda t: None, name="replay", steps=5, replay=True)
    assert providers and all(p.calls == 0 for p in providers)
    assert derive(root, ep) == derive(rroot, rep)
    assert live["ending_cuts"][0]["logical_prefix_sha256"] == replay["ending_cuts"][0]["logical_prefix_sha256"]
