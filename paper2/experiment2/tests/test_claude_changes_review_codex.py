"""Supplemental exact-snapshot controls; fixed invented worlds, no real model.

PAUSE172.CEILING is a prospective design-integration invariant, not a claim
that the code violates its literal first-20-episodes filter. The remaining
strict witnesses test the declared interface and common collection records.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

import paper2.experiment2.adviser as AD; import telohelm.learning as G; import paper2.experiment2.run_stream as RS; import telohelm.token_count as T
from telohelm.ladder import pause_due
import telohelm.runtime.models as M
from telohelm.runtime.ledger import Call
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P; import paper2.experiment2.streams as streams


def fixed_world(name="invented-adviser", deadline=18):
    return E.Scenario(name, P.ReceiverState(0., 0., 0., 0., 0., .1, .02), (0., 0.),
                       (E.Tanker("SHELL 1", P.Orbit(200., 100., 8., .075, 0.)),),
                       ((0, E.Goal(1, "SHELL 1", 5., deadline)),))


def budget_trace(out, outcome="budget"):
    schedule = [f"invented-{i:02d}" for i in range(25)]
    budget = G.AdaptiveBudget()
    events, rows, levels = [], [], []
    for attempt, ordinal in enumerate(range(5, 26, 5), 1):
        ep = schedule[ordinal - 1]
        level = budget.current()["level"]
        levels.append(level)
        events.append({"kind": "attempt_budget", "attempt": attempt, "episode": ep, "level": level})
        for call in range(3):
            rows.append(Call(request_id=f"attempt{attempt}/call{call}", kind="reasoning_model",
                             episode=ep, action_step=0, started_s=None, wall_s=1.,
                             tokens_in=None, tokens_out=None, status="timed_out", scope="build"))
        budget.observe(outcome)
    co = SimpleNamespace(budget=budget, events=events)
    excluded = []
    retained = RS._ledger_rows(out, schedule, SimpleNamespace(ledger=SimpleNamespace(calls=rows)), co, excluded)
    return {"schedule": schedule, "levels": levels, "retained": retained, "excluded": excluded,
            "raw_pause": pause_due([asdict(c) for c in rows], schedule),
            "adaptive_pause": pause_due(retained, schedule)}


@pytest.mark.xfail(strict=True, reason="PAUSE172.CEILING: prospective ceiling guard cannot observe its first ceiling attempt")
def test_ceiling_pause_observes_the_first_ceiling_timeout_attempt(tmp_path):
    trace = budget_trace(tmp_path)
    assert trace["levels"] == [0, 0, 1, 1, 2]
    assert len(trace["excluded"]) == 12 and len(trace["retained"]) == 3
    assert trace["raw_pause"]["started"] == 12 and trace["raw_pause"]["due"]
    assert trace["adaptive_pause"]["due"]


def test_flight_pause_and_ceiling_calls_are_not_removed_by_the_level_filter(tmp_path):
    trace = budget_trace(tmp_path)
    ceiling = trace["retained"]
    # Isolate the level filter from the separate, defective window interaction.
    assert pause_due(ceiling, trace["schedule"], window=25)["due"]
    p = tmp_path / trace["schedule"][0]
    p.mkdir()
    flight = {**ceiling[0], "episode": trace["schedule"][0], "scope": "running", "request_id": "flight"}
    (p / "ledger.jsonl").write_text(json.dumps(flight) + "\n")
    rows = RS._ledger_rows(tmp_path, trace["schedule"], None)
    assert rows == [flight] and pause_due(rows, trace["schedule"])["due"]


@pytest.mark.skipif(not (T.MODEL_FILES / "tokenizer.json").exists(), reason="pinned offline tokenizer unavailable")
def test_every_default_budget_level_renders_with_the_pinned_template():
    for level in G.BUDGET_LEVELS:
        assert T.count([("user", "fixed invented prompt")],
                       {"enable_thinking": True, "reasoning_effort": level["effort"]}) > 0


@pytest.mark.xfail(strict=True, reason="AD172.PAUSE: adviser CLI continues after a first-window timeout exceeds the pause threshold")
def test_adviser_driver_stops_when_its_first_window_pause_is_due(tmp_path, monkeypatch):
    provider = M.ScriptedProvider(lambda request: M.ModelResponse("timed_out"))
    monkeypatch.setattr(AD.local_model, "provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(streams, "solvable_episode", lambda seed, phase, index: SimpleNamespace(
        scenario=fixed_world(f"invented-p{phase}-e{index}"), attempts=[]))
    AD.main(["--stream", "4401", "--phases", "1", "--episodes", "3", "--out", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out" / "stream-summary.json").read_text())
    assert provider.calls == 1 and summary["episodes"] == 1
    assert json.loads((tmp_path / "out" / "paused.json").read_text())["due"]


@pytest.mark.xfail(strict=True, reason="AD172.GENERATION: adviser loses generation-cap attempt evidence before stopping")
def test_adviser_driver_persists_generation_cap_evidence_before_propagating(tmp_path, monkeypatch):
    provider = M.ScriptedProvider(lambda request: M.ModelResponse("answered", "{}", 1, 1))
    monkeypatch.setattr(AD.local_model, "provider", lambda *args, **kwargs: provider)
    attempts = [{"attempt": 0, "seed": 17, "solvable": False, "reason": "fixed invented cap witness"}]
    def cap(*args):
        raise streams.GenerationError("fixed invented cap", attempts)
    monkeypatch.setattr(streams, "solvable_episode", cap)
    out = tmp_path / "out"
    with pytest.raises(streams.GenerationError):
        AD.main(["--stream", "4401", "--phases", "1", "--episodes", "1", "--out", str(out)])
    assert provider.calls == 0
    path = out / "generation.jsonl"
    assert path.exists()
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["attempts"] == attempts and record["accepted"] is False


@pytest.mark.xfail(strict=True, reason="AD172.ZERO: zero-latency reply is stranded after its delivery phase has passed")
def test_zero_latency_is_delivered_in_its_second_or_refused_before_a_call(tmp_path):
    provider = M.ScriptedProvider(lambda request: M.ModelResponse(
        "answered", '{"heading_deg":90,"next_call_at_s":null}', 1, 1))
    try:
        record = AD.run_adviser_episode(fixed_world(), tmp_path, "4401", provider, latency_s=0., max_steps=5)
    except ValueError:
        assert provider.calls == 0
        return
    log = tmp_path / record["episode"] / "episode.log"
    rows = [json.loads(line)["payload"] for line in log.read_text().splitlines()[1:]]
    assert record["model_calls"] == 1
    assert any(r["kind"] == "adviser_reply" and r["t"] == 0 for r in rows)


def test_thirty_second_rationing_is_a_spacing_rule_not_full_horizon_coverage():
    ration = RS.SmallRation(30)
    calls = [(t, d) for t in range(430) for d in ("heading", "report")
             if ration.may_issue("small_model", d, t)]
    first_allowance = calls[:30]
    assert len(first_allowance) == 30 and first_allowance[-1][0] == 420
    assert E.HORIZON == 3600
