"""TURN74 independent CPU controls. Invented transports; no HTTP, model or collection work."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import verify
from .test_local_adapter import REQ; from .test_local_adapter import ROLES; from .test_local_adapter import TIMEOUTS; from .test_local_adapter import adapter; from .test_local_adapter import ok_body
from .test_turn65_review_codex import gateway; from .test_turn65_review_codex import submit
from .test_turn70_review_codex import execute; from .test_turn70_review_codex import flight; from .test_turn70_review_codex import records


@pytest.mark.parametrize("role", ["small_model", "reasoning_model"])
def test_explicit_alias_keeps_observed_name_and_configuration_is_copied(role):
    names, limits = dict(ROLES), dict(TIMEOUTS)
    served = {r: [n + "-alias"] for r, n in names.items()}
    observed = served[role][0]
    body = ok_body(); body["model"] = observed
    seen = []
    def transport(url, payload, timeout):
        seen.append((payload, timeout)); return 200, body
    a = M.LocalChatProvider("http://invented.local", names, transport, limits, served_names_by_role=served)
    names[role] = "changed"; limits[role] = 0.; served[role].clear()
    response = a.complete(replace(REQ, role=role))
    assert response.status == "answered" and response.provider_ids == {"id": "cmpl-invented", "model": observed}
    assert seen[0][0]["model"] == ROLES[role] and seen[0][1] == TIMEOUTS[role]
    assert a.reportable is False


@pytest.mark.parametrize("kwargs", [{"temperature": True}, {"temperature": float("inf")},
                                   {"model_by_role": {**ROLES, "reasoning_model": "   "}},
                                   {"timeout_s_by_role": {**TIMEOUTS, "reasoning_model": True}}])
def test_other_role_and_sampling_invalid_values_fail_before_work(kwargs):
    calls = []
    params = {"base_url": "http://invented.local", "model_by_role": ROLES,
              "timeout_s_by_role": TIMEOUTS, "transport": lambda *a: calls.append(a)}
    params.update(kwargs)
    with pytest.raises(M.GatewayError):
        M.LocalChatProvider(**params)
    assert calls == []


@pytest.mark.parametrize("part", [{"heading_deg": {1}}, {"help": complex(1, 2)}, {"report": float("nan")}])
def test_nonserialisable_or_nonfinite_parser_values_refuse_with_original_evidence(tmp_path, monkeypatch, part):
    text = "invented raw reply"
    parts = flight(tmp_path, monkeypatch, text=text, parser=lambda d, t: part)
    result = execute(parts, tmp_path)
    assert len(result["actions"]) == 3
    reply = next(e for e in records(tmp_path / "log") if e["kind"] == "model_reply")
    assert reply["part"] is None and reply["refusal"] == "invalid_part"
    assert reply["reply_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert reply["goal_version"] == 1 and reply["issued_second"] == 0
    # Each later second may issue a fresh request once the previous reply is consumed;
    # one charge per request is the invariant, not one call for the whole episode.
    assert len({c.request_id for c in parts[1].calls}) == len(parts[1].calls) == parts[2].calls == 3
    assert verify(tmp_path / "log", tmp_path / "heads")["ok"]
    assert verify(tmp_path / "blog", tmp_path / "bheads")["ok"]


def test_refused_finish_retains_usage_through_recovery_without_new_transport(gateway, tmp_path):
    calls, ledger, journal = [], Ledger(), tmp_path / "shared"
    def transport(*args):
        calls.append(args); return 200, ok_body(text=None, finish="stop")
    provider = adapter(transport)
    first, _, _, _ = gateway(provider=provider, ledger=ledger, journal=journal)
    out = submit(first)
    assert out["status"] == "failed" and out["text"] is None and out["reportable"] is False
    for _ in range(2):
        resumed, _, _, root = gateway(provider=provider, ledger=ledger, journal=journal)
        assert submit(resumed)["recovered"]
        assert verify(root / "log", root / "heads")["ok"]
    assert len(calls) == 1 and sum(c.tokens_in or 0 for c in ledger.calls) == 11
    assert sum(c.tokens_out or 0 for c in ledger.calls) == 4
    assert len(ledger.ending_calls("episode-a", 14)["counted"]) == 1


# Fixed in TURN 75 (Claude): LA74.RETAIN: refusal before extraction discards observed usage and response identity
@pytest.mark.parametrize("change", [{"model": "invented-other-model"}, {"model": None}, {"choices": []}])
def test_rejected_reply_preserves_observed_usage_and_safe_identifiers(change):
    body = ok_body(); body.update(change)
    response = adapter(lambda *a: (200, body)).complete(REQ)
    assert response.status == "failed" and response.text is None
    assert (response.tokens_in, response.tokens_out) == (11, 4)
    assert response.provider_ids.get("id") == "cmpl-invented"
    assert response.provider_ids.get("model") == body["model"]


# Fixed in TURN 75 (Claude): LA74.RETAIN: gateway and recovery cannot restore usage already dropped by adapter
def test_rejected_identity_keeps_single_charge_in_live_and_recovered_ledger(gateway, tmp_path):
    body = ok_body(); body["model"] = "invented-other-model"
    calls, ledger, journal = [], Ledger(), tmp_path / "shared"
    def transport(*args):
        calls.append(args); return 200, body
    provider = adapter(transport)
    first, _, _, _ = gateway(provider=provider, ledger=ledger, journal=journal)
    assert submit(first)["status"] == "failed"
    for _ in range(2):
        resumed, _, _, _ = gateway(provider=provider, ledger=ledger, journal=journal)
        assert submit(resumed)["recovered"]
    assert len(calls) == 1
    assert any(c.tokens_in == 11 and c.tokens_out == 4 for c in ledger.calls)
    assert sum(c.tokens_in or 0 for c in ledger.calls) == 11
    ids = set(ledger.ending_calls("episode-a", 14)["charge_ids"])
    assert sum(c.tokens_in or 0 for c in ledger.calls if c.request_id in ids) == 11


# Fixed in TURN 75 (Claude): DL74.UTF8: finite escaped JSON validation does not establish trusted UTF8 checkpoint writability
@pytest.mark.parametrize("key", ["heading_deg", "help"])
def test_escaped_lone_surrogate_proposal_is_safely_refused_before_checkpoint(tmp_path, monkeypatch, key):
    # Raw response is printable ASCII JSON; its decoded string contains a lone surrogate.
    text = '{"' + key + '":"\\ud800"}'
    parts = flight(tmp_path, monkeypatch, text=text)
    result = execute(parts, tmp_path)
    assert len(result["actions"]) == 3 and key not in result["actions"][1]
    reply = next(e for e in records(tmp_path / "log") if e["kind"] == "model_reply")
    assert reply["part"] is None and reply["refusal"] is not None
    assert reply["reply_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert verify(tmp_path / "log", tmp_path / "heads")["ok"]
    assert verify(tmp_path / "blog", tmp_path / "bheads")["ok"]


# Fixed in TURN 75 (Claude): LA74.UTF8: unpaired surrogate in returned text aborts the gateway's journal after provider work
def test_bad_response_text_encoding_is_refused_without_losing_observed_charge(gateway):
    body = ok_body(text="\ud800")
    gw, ledger, _, root = gateway(provider=adapter(lambda *a: (200, body)))
    out = submit(gw)
    assert out["status"] == "failed" and out["text"] is None
    assert ledger.calls[0].tokens_in == 11 and ledger.calls[0].tokens_out == 4
    assert verify(root / "log", root / "heads")["ok"]
