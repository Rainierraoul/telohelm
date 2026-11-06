"""The method ladder in flight (B36, D26): escalation to the reasoning model, adopted checks on model replies, flight
reasoning calls from the shared allowance, and one help request. Invented model answers on a development episode;
no real model, no scored stream."""
from __future__ import annotations

import json
import math

import pytest

import paper2.experiment2.run_stream as RS
from telohelm.ladder import ReasoningAllowance
import telohelm.runtime.models as M
from telohelm.runtime.escalation import FlightLadder
from telohelm.runtime.library import Library
import paper2.experiment2.streams as streams


def scenario():
    return streams.solvable_episode(4401, 1, 0).scenario


def answers(small: str | None, reasoning: str | None):
    """A stand-in provider by role: None answers with the bearing to the orbit centre."""
    def answer(request):
        body = json.loads(request.messages[-1][1])
        ctx, decision = body["context"], body["decision"]
        text = small if request.role == "small_model" else reasoning
        if text is None:
            if decision == "heading":
                h = math.degrees(math.atan2(ctx["tanker_centre_x"] - ctx["x"], ctx["tanker_centre_y"] - ctx["y"])) % 360
                text = json.dumps({"heading_deg": round(h, 3)})
            else:
                text = "{}"
        return M.ModelResponse("answered", text, 50, 5)
    return answer


def fly(tmp_path, monkeypatch, small, reasoning, library=None, allowance=None, steps=200, name="e"):
    monkeypatch.setattr(RS.seed, "stand_in_model", answers(small, reasoning))
    lib = library or Library(tmp_path / f"library-{name}")
    out = tmp_path / name
    out.mkdir()
    flight = {"allowance": allowance, "reasoning": None}
    done = RS.run_episode(scenario(), lib, out, "dev-test", steps, {"small_model": 2.0, "reasoning_model": 60.0},
                          None, flight)
    rows = [json.loads(l) for l in (out / scenario().episode_id / "ledger.jsonl").read_text().splitlines()]
    return done["record"], rows, out / scenario().episode_id


def roles(rows):
    return [r["role"] for r in rows if r.get("kind") == "model"] or [r.get("role") for r in rows]


def test_an_unusable_small_reply_escalates_its_decision_to_the_reasoning_model(tmp_path, monkeypatch):
    rec, rows, _ = fly(tmp_path, monkeypatch, small="not json at all", reasoning=None)
    kinds = [e["kind"] for e in rec["ladder"]]
    assert "escalated" in kinds
    first = next(e for e in rec["ladder"] if e["kind"] == "escalated")
    assert first["why"] == "unparseable"
    used = [r["kind"] for r in rows]
    assert "reasoning_model" in used and used.index("reasoning_model") > used.index("small_model")
    assert "released" in kinds                                  # a usable reasoning reply releases the latch


def test_an_adopted_check_judges_model_replies_too(tmp_path, monkeypatch):
    lib = Library(tmp_path / "library-c")
    check = lib.add({"kind": "check", "decision": "heading", "conditions": [],
                     "source": "def check(ctx, out):\n    return 'reject'\n", "origin": {"invented": True}})
    lib.set_status(check, "adopted", "invented: rejects every heading")
    rec, rows, ep = fly(tmp_path, monkeypatch, small=None, reasoning=None, library=lib, steps=120, name="c")
    assert any(e["kind"] == "escalated" and e["why"] == "check_rejected" for e in rec["ladder"])
    assert rec["share_of_seconds"]["small_model"] == 0.0           # no rejected reply ever flew


def test_flight_reasoning_calls_draw_on_the_shared_allowance_and_help_follows(tmp_path, monkeypatch):
    allowance = ReasoningAllowance(tmp_path / "allowance.jsonl")
    rec, rows, _ = fly(tmp_path, monkeypatch, small="not json", reasoning="not json either", allowance=allowance,
                       steps=600, name="a")
    key = allowance.key(RS.CONDITION, "dev-test", scenario().episode_id)
    flight = [a for a in allowance.allocations(key) if a["purpose"] == "flight"]
    assert 1 <= len(flight) <= 4 and allowance.attempt_budget(key) == 4 - len(flight)
    assert sum(r["kind"] == "reasoning_model" for r in rows) == len(flight)
    asked = [e for e in rec["ladder"] if e["kind"] == "help_asked"]
    assert len(asked) <= 1 and rec["help_requests"] == len(asked)
    if len(flight) == 4:
        assert any(e["kind"] == "no_reasoning_allowance" for e in rec["ladder"]) and len(asked) == 1


class _Gateway:
    def __init__(self, small_used=0, reasoning_used=0):
        self.used = {"small_model": small_used, "reasoning_model": reasoning_used}
        self.limits = {"small_model": 30, "reasoning_model": 4}


def test_role_choice_latch_release_and_goal_change():
    ladder = FlightLadder(_Gateway())
    ladder.at(1)
    assert ladder.role_for("heading", "missing_tool") == "small_model"
    ladder.unusable("heading", "reasoning_model", 1, "failed", 5)          # only small-model replies latch
    assert ladder.role_for("heading", "missing_tool") == "small_model"
    ladder.unusable("heading", "small_model", 0, "failed", 5)              # an old goal's reply never escalates
    assert ladder.role_for("heading", "missing_tool") == "small_model"
    ladder.unusable("heading", "small_model", 1, "unparseable", 6)
    assert ladder.role_for("heading", "missing_tool") == "reasoning_model"
    assert ladder.role_for("report", "missing_tool") == "small_model"     # per decision
    ladder.at(2)                                                           # a new goal version clears it
    assert ladder.role_for("heading", "missing_tool") == "small_model"
    ladder.at(1)
    ladder.usable("heading", 1, 9, "procedure")
    assert ladder.role_for("heading", "missing_tool") == "small_model"
    spent = FlightLadder(_Gateway(small_used=30))
    spent.at(1)
    assert spent.role_for("report", "missing_tool") == "reasoning_model"   # small allowance spent


def test_help_once_with_reason_and_suppressed_when_the_reply_would_be_late():
    ladder = FlightLadder(_Gateway(), help_delay_s=30, horizon_s=3600)
    ladder.at(1)
    fresh = {"t": 100, "report_age_s": 1.0, "deadline": None}
    assert ladder.help(fresh, {}, heading_by_procedure=False) is None
    late = {"t": 100, "report_age_s": 12.0, "deadline": 125}
    assert ladder.help(late, {}, heading_by_procedure=False) is None       # reply at 130 is not before 125
    assert [e["kind"] for e in ladder.events] == ["help_suppressed"]
    assert ladder.help(late, {}, heading_by_procedure=False) is None
    assert len(ladder.events) == 1                                         # suppression recorded once
    flown = {"t": 100, "report_age_s": 12.0, "deadline": None}
    assert ladder.help(flown, {}, heading_by_procedure=True) is None       # an accepted procedure is flying
    ask = ladder.help(flown, {}, heading_by_procedure=False)
    assert ask == {"reason": "missing_or_stale_evidence", "text": ask["text"]}
    ladder.help_outcome(100, ask, "ladder", None, accepted=False)          # refused with its action: not used
    assert ladder.help(flown, {}, heading_by_procedure=False) == ask
    ladder.help_outcome(100, ask, "ladder", None, accepted=True)
    assert ladder.help(flown, {"heading": "missing_tool"}, heading_by_procedure=False) is None   # never twice
    other = FlightLadder(_Gateway())
    other.at(1)
    ask = other.help(fresh, {"report": "unfamiliar_but_solvable"}, heading_by_procedure=True)
    assert ask["reason"] == "unfamiliar_but_solvable"


def test_reasoning_calls_need_an_allocation():
    allowance_rows = []

    class Allowance:
        def allocate(self, key, request, purpose, kind):
            if len(allowance_rows) >= 2:
                return None
            allowance_rows.append((key, request, purpose, kind))
            return len(allowance_rows)
    ladder = FlightLadder(_Gateway(), Allowance(), "k")
    ladder.at(1)
    assert ladder.may_issue("small_model", "heading", 3)
    assert ladder.may_issue("reasoning_model", "heading", 3) and ladder.may_issue("reasoning_model", "heading", 4)
    assert not ladder.may_issue("reasoning_model", "heading", 5)
    assert allowance_rows[0][2:] == ("flight", "provider")
    assert ("heading", 1) in ladder.no_reasoning
    capped = FlightLadder(_Gateway(reasoning_used=4))
    capped.at(1)
    assert not capped.may_issue("reasoning_model", "report", 7)            # the gateway's own limit


def test_a_world_refusal_escalates_the_small_reply_it_swallowed(tmp_path, monkeypatch):
    """LAD163.ACTION, the atomic case: an adopted procedure's invalid heading makes the world refuse the whole
    action, so the small model's valid report reply in it never acted; its decision escalates."""
    lib = Library(tmp_path / "library-w")
    proc = lib.add({"kind": "procedure", "decision": "heading", "conditions": [],
                    "source": 'def decide(ctx):\n    return {"heading_deg": 360.0}\n', "origin": {"invented": True}})
    check = lib.add({"kind": "check", "decision": "heading", "conditions": [],
                     "source": "def check(ctx, out):\n    return 'accept'\n", "origin": {"invented": True}})
    for sha in (proc, check):
        lib.set_status(sha, "adopted", "invented")
    rec, rows, _ = fly(tmp_path, monkeypatch, small='{"report_trigger_nm": 25}', reasoning=None, library=lib,
                       steps=8, name="w")
    assert any(e["kind"] == "escalated" and e["decision"] == "report" and e["why"] == "world_refused"
               for e in rec["ladder"])


def test_a_model_help_request_is_the_episodes_only_one_and_malformed_help_escalates(tmp_path, monkeypatch):
    """LAD165.HELP beyond Codex's witness: a valid help-only reply is delivered once and the ladder adds no second
    request; a later help in a model reply is refused as second_help; malformed help is unusable and escalates."""
    rec, rows, _ = fly(tmp_path, monkeypatch, small='{"help": {"reason": "missing_tool", "text": "need a tool"}}',
                       reasoning=None, steps=40, name="h1")
    assert rec["help_requests"] == 1
    assert [e["kind"] for e in rec["ladder"]].count("help_by_model") == 1
    assert not any(e["kind"] == "help_asked" for e in rec["ladder"])
    rec, rows, _ = fly(tmp_path, monkeypatch, small='{"help": {"reason": "because", "text": "x"}}', reasoning=None,
                       steps=10, name="h2")
    assert rec["help_requests"] == 0
    assert any(e["kind"] == "escalated" and e["decision"] == "heading" for e in rec["ladder"])


def test_flight_holds_at_most_one_reasoning_allocation_under_version_0_11(tmp_path):
    """D39 (proposed; Simon, 15 October): with the runner's flight limit, flight takes one allocation and the rewrite
    attempt after the episode keeps three."""
    from telohelm.runtime.escalation import FLIGHT_REASONING_LIMIT
    a = ReasoningAllowance(tmp_path / "allowance.jsonl")
    key = a.key("telohelm", "invented-unit", "fixed")
    gw = _Gateway()
    ladder = FlightLadder(gw, a, key, flight_limit=FLIGHT_REASONING_LIMIT)
    ladder.at(1)
    assert ladder.may_issue("reasoning_model", "heading", 2)
    gw.used["reasoning_model"] += 1
    assert not ladder.may_issue("reasoning_model", "heading", 3)
    assert not ladder.may_issue("reasoning_model", "report", 3)
    assert a.attempt_budget(key) == 3
    assert [e.get("why") for e in ladder.events if e["kind"] == "no_reasoning_allowance"] == ["flight_limit"] * 2
    assert ("heading", 1) in ladder.no_reasoning                            # so help can follow


def test_the_runner_rations_the_small_model_and_caps_flight_by_default(tmp_path):
    """Version 0.11's settings are the runner's defaults and are recorded in the stream summary; the command line
    defaults match."""
    done = RS.run_stream(1, 1, tmp_path / "out", max_steps=5)
    assert done["summary"]["flight"] == {"ladder": True, "small_interval_s": RS.SMALL_INTERVAL_S,
                                         "reasoning_limit": 1}
    assert RS.SMALL_INTERVAL_S == 30
    off = RS.run_stream(1, 1, tmp_path / "off", max_steps=5, ladder=False, small_interval_s=0)
    assert off["summary"]["flight"] == {"ladder": False, "small_interval_s": 0, "reasoning_limit": None}
