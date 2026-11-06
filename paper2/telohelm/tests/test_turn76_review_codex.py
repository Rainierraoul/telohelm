"""TURN76 CPU evidence: response salvage and prospective attribution boundaries. No actual provider work."""
from dataclasses import replace
import hashlib
import json

import pytest

import telohelm.runtime.episode_loop as L; import telohelm.runtime.models as M
from telohelm.runtime.trusted_log import verify
from .test_local_adapter import REQ; from .test_local_adapter import adapter; from .test_local_adapter import ok_body
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit
from .test_turn70_review_codex import execute; from .test_turn70_review_codex import flight; from .test_turn70_review_codex import records


class InventedProducer:
    name, relays, reportable = "invented-producer", False, True
    def __init__(self, response):
        self.response, self.calls = response, 0
    def complete(self, request):
        self.calls += 1
        return self.response


def malformed_response(kind, *, origin=None):
    r = M.ModelResponse("answered", "invented valid text", 11, 4,
                        {"id": "invented-response", "model": "invented-model"}, origin)
    if kind == "text":
        return replace(r, text="\ud800")
    if kind == "tokens":
        return replace(r, tokens_in=-1)
    return replace(r, status="invented-invalid-status")


# Fixed in TURN 77 (Claude): LA76.SALVAGE: discarding explicit nonreportable origin promotes invalid producer response
@pytest.mark.parametrize("kind", ["text", "tokens"])
def test_invalid_response_cannot_erase_nonreportable_origin_live_or_recovered(gateway, tmp_path, kind):
    p = InventedProducer(malformed_response(kind, origin={"provider": "invented-original", "reportable": False}))
    journal = tmp_path / "shared"
    first, ledger, _, _ = gateway(provider=p, journal=journal)
    out = submit(first)
    assert out["status"] == "failed" and out["text"] is None
    assert out["reportable"] is False
    resumed, _, _, _ = gateway(provider=p, ledger=ledger, journal=journal)
    recovered = submit(resumed)
    assert recovered["recovered"] and recovered["reportable"] is False and p.calls == 1


# Fixed in TURN 77 (Claude): LA76.SALVAGE: unrelated invalid response field discards valid original response identifiers
@pytest.mark.parametrize("kind", ["text", "tokens", "status"])
def test_gateway_salvage_preserves_safe_original_response_identifiers(gateway, tmp_path, kind):
    p = InventedProducer(malformed_response(kind))
    journal = tmp_path / "shared"
    gw, ledger, _, root = gateway(provider=p, journal=journal)
    out = submit(gw)
    assert out["status"] == "failed" and out["text"] is None
    saved = next(e for e in map(json.loads, journal.read_text().splitlines()) if e["type"] == "response")
    ids = saved["response"]["provider_ids"]
    assert ids.get("id") == "invented-response" and ids.get("model") == "invented-model"
    assert ledger.calls[0].tokens_out == 4 and verify(root / "log", root / "heads")["ok"]


def test_bad_adapter_text_has_declared_digest_and_safe_cost_and_recovery(gateway, tmp_path):
    bad = "\ud800"
    body = ok_body(text=bad)
    a = adapter(lambda *args: (200, body))
    journal = tmp_path / "shared"
    gw, ledger, _, root = gateway(provider=a, journal=journal)
    out = submit(gw)
    assert (out["status"], out["text"], out["tokens_in"], out["tokens_out"]) == ("failed", None, 11, 4)
    saved = next(e for e in map(json.loads, journal.read_text().splitlines()) if e["type"] == "response")
    ids = saved["response"]["provider_ids"]
    assert ids["text_sha256"] == hashlib.sha256(bad.encode("utf-8", "surrogatepass")).hexdigest()
    assert ids["id"] == "cmpl-invented" and ids["refused"] == "text encoding"
    assert verify(root / "log", root / "heads")["ok"]
    calls = []
    resumed, _, _, _ = gateway(provider=adapter(lambda *x: calls.append(x)), ledger=ledger, journal=journal)
    assert submit(resumed)["recovered"] and calls == []
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 11


@pytest.mark.parametrize("change", [{"model": None}, {"choices": []},
                                   {"choices": [{"finish_reason": "tool_calls"}]}])
def test_refused_reply_keeps_partial_valid_usage_unknown_is_not_zero(change):
    body = ok_body(); body.update(change)
    body["usage"] = {"prompt_tokens": True, "completion_tokens": 4}
    r = adapter(lambda *args: (200, body)).complete(REQ)
    assert r.status == "failed" and r.text is None
    assert r.tokens_in is None and r.tokens_out == 4
    assert r.provider_ids["id"] == "cmpl-invented"


def test_delivery_precedence_does_not_evaluate_an_alternative_procedure(tmp_path, monkeypatch):
    parts = flight(tmp_path, monkeypatch, latency=1., steps=2)
    calls = []
    def choose(decision, ctx, *args, **kw):
        calls.append((ctx["t"], decision))
        if decision == "heading" and ctx["t"] == 0:
            return {"climb": "missing_tool", "tried": []}
        return {"use": "invented-procedure", "output": {"heading_deg": 10.} if decision == "heading" else {},
                "tried": [], "checks": []}
    monkeypatch.setattr(L, "choose", choose)
    out = execute(parts, tmp_path)
    assert out["actions"][1]["heading_deg"] == 123.
    assert (1, "heading") not in calls
    decision = next(e for e in records(tmp_path / "log") if e["kind"] == "decision" and e["t"] == 1 and e["decision"] == "heading")
    assert decision["method"].startswith("model:") and decision["tried"] == []


@pytest.mark.parametrize("text", ['{"heading_deg":361}', '{"heading_deg":123,"report":""}'])
def test_proposed_heading_can_be_present_but_atomically_refused_by_world(tmp_path, monkeypatch, text):
    parts = flight(tmp_path, monkeypatch, text=text, steps=2)
    execute(parts, tmp_path)
    rows = records(tmp_path / "log")
    decision = next(e for e in rows if e["kind"] == "decision" and e["t"] == 1 and e["decision"] == "heading")
    assert "heading_deg" in decision["part"] and decision["method"].startswith("model:")
    world = [e for e in records(tmp_path / "blog") if e["kind"] == "world_step"]
    if "report" in text:
        # TURN 77 (Claude): parts are now scoped to their decision (MS76.ACCEPT), so the heading decision's stray
        # report field is dropped and recorded, and the world accepts the heading.
        assert decision["out_of_scope_fields"] == ["report"] and world[-1]["accepted"] is True
    else:
        assert world[-1]["accepted"] is False
    assert verify(tmp_path / "blog", tmp_path / "bheads")["ok"]
