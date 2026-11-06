"""Request identities stay unique when flight and a learning attempt both call the reasoning model for one episode
(found by the local rehearsal of 15 October: both gateways numbered from 1, and the pause refused the run). Invented
replies; no model."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from telohelm.ladder import pause_due
import telohelm.runtime.models as M
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog


def gateway(tmp_path, name, **kw):
    provider = M.ScriptedProvider(lambda request: M.ModelResponse("answered", "{}", 10, 2))
    return M.ModelGateway("s4401-p1-e04", "telohelm", "4401", provider,
                          TrustedLog(tmp_path / f"{name}.log", tmp_path / f"{name}.heads", "4401"), Ledger(),
                          tmp_path / f"{name}.journal", **kw)


def call(gw):
    return gw.submit("reasoning_model", [("user", "x")], 64, 1, 0, 0, 0.0)["request_id"]


def test_a_learning_gateway_never_reuses_flights_identifiers(tmp_path):
    flight = gateway(tmp_path, "flight")
    build = gateway(tmp_path, "build", purpose="build", id_scope="build/attempt1")
    a, b = call(flight), call(build)
    assert a != b
    assert json.loads(a) == ["4401", "s4401-p1-e04", "telohelm", "reasoning_model", 1]      # unchanged without scope
    assert json.loads(b) == ["4401", "s4401-p1-e04", "telohelm", "reasoning_model", "build/attempt1", 1]
    assert json.loads(b)[3] == "reasoning_model"                  # accounting.py reads the role at position 3
    rows = [asdict(c) for c in flight.ledger.calls] + [asdict(c) for c in build.ledger.calls]
    assert pause_due(rows, ["s4401-p1-e04"])["started"] == 2     # refused as a duplicate before the fix


def test_the_scope_must_be_a_non_empty_string(tmp_path):
    for bad in ("", 3):
        with pytest.raises(M.GatewayError):
            gateway(tmp_path, f"bad{bad!r}", id_scope=bad)


def test_the_learning_proposer_scopes_its_gateways(tmp_path):
    import telohelm.learning as G
    p = G.GatewayProposer("http://127.0.0.1:1", "Qwen/Qwen3.8-27B", tmp_path, "4401")
    p.provider = M.ScriptedProvider(lambda request: M.ModelResponse("answered", "{}", 10, 2))
    p.count_tokens = lambda messages: 10
    p([("system", "s"), ("user", "u")], {"attempt": 2, "call": 1, "episode": "s4401-p1-e09"})
    assert json.loads(p.ledger.calls[-1].request_id)[4] == "build/attempt2"
