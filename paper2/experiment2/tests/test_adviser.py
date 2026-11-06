"""The reasoning model alone as a budgeted adviser (D3, D26 amendment 5): one episode-wide call at a time, at most 4,
triggered by events and its own schedule, replies held for the modelled latency, out-of-date and malformed replies
refused, and method-seconds with procedure and small model not applicable. Invented replies on development episodes;
no real model, no scored stream."""
from __future__ import annotations

import dataclasses
import json
import math

import pytest

import paper2.experiment2.adviser as AD
import telohelm.runtime.models as M
import telohelm.taskpacks.tanker.world.episode as E
import paper2.experiment2.streams as streams


def scenario(**changes):
    s = streams.solvable_episode(4401, 1, 0).scenario
    return dataclasses.replace(s, **changes) if changes else s


def request_of(request) -> dict:
    return json.loads(request.messages[-1][1].split("Request:\n", 1)[1])


class Script:
    """Answers each call in turn from a list of functions of the request body (or fixed strings), and keeps the
    bodies it was sent."""

    def __init__(self, *answers):
        self.answers, self.seen = list(answers), []

    def __call__(self, request):
        body = request_of(request)
        self.seen.append(body)
        a = self.answers[min(len(self.seen), len(self.answers)) - 1]
        text = a(body) if callable(a) else a
        return M.ModelResponse("answered", text, 300, 40)


def toward_centre(nxt=None, **extra):
    def answer(body):
        c = body["context"]
        h = math.degrees(math.atan2(c["tanker_centre_x"] - c["x"], c["tanker_centre_y"] - c["y"])) % 360
        return json.dumps({"heading_deg": round(h, 3), "next_call_at_s": nxt, **extra})
    return answer


def fly(tmp_path, script, steps=300, name="e", **changes):
    provider = M.ScriptedProvider(script)
    return AD.run_adviser_episode(scenario(**changes), tmp_path / name, "4401", provider, max_steps=steps)


def events(tmp_path, record, name="e"):
    rows = (tmp_path / name / record["episode"] / "episode.log").read_text().splitlines()[1:]
    return [json.loads(r)["payload"] for r in rows]


def test_first_call_at_zero_reply_held_for_the_latency_and_heading_held_after(tmp_path):
    script = Script(toward_centre())
    r = fly(tmp_path, script, steps=200)
    ev = events(tmp_path, r)
    calls = [e for e in ev if e["kind"] == "adviser_call"]
    assert [(c["t"], c["triggers"]) for c in calls] == [(0, ["start"])]
    assert calls[0]["available_second"] == 60 and script.seen[0]["calls_left_after_this"] == 3
    reply = next(e for e in ev if e["kind"] == "adviser_reply")
    assert reply["t"] == 60 and reply["refusal"] is None and reply.get("next_call_at_s") is None
    assert r["method_seconds"] == {"procedure": "not applicable", "small_model": "not applicable",
                                   "reasoning_model": 1, "holding_pending": 60, "holding_none": 139}
    assert r["seconds"] == 200 and r["model_calls"] == 1 and r["censored_by_step_limit"]
    assert r["shares"]["procedure"] == "not applicable" and r["shares"]["reasoning_model"] == 1 / 200


def test_the_reply_schedules_the_next_call_and_a_past_second_is_moved_after_delivery(tmp_path):
    r = fly(tmp_path, Script(toward_centre(100), toward_centre(30), toward_centre()), steps=260)
    ev = events(tmp_path, r)
    assert [(c["t"], c["triggers"]) for c in ev if c["kind"] == "adviser_call"] == [
        (0, ["start"]), (100, ["scheduled"]), (161, ["past_due"])]
    replies = [e for e in ev if e["kind"] == "adviser_reply"]
    assert replies[1]["scheduled"] == {"at": 161, "asked": 30, "past_due": True}
    assert r["model_calls"] == 3


def test_at_most_four_calls_and_the_fifth_is_never_made(tmp_path):
    script = Script(toward_centre(0))
    r = fly(tmp_path, script, steps=400)
    ev = events(tmp_path, r)
    assert [c["t"] for c in ev if c["kind"] == "adviser_call"] == [0, 61, 122, 183]
    assert [e["t"] for e in ev if e["kind"] == "adviser_no_calls_left"] == [244]
    assert r["model_calls"] == 4 and len(script.seen) == 4
    assert [b["calls_left_after_this"] for b in script.seen] == [3, 2, 1, 0]


def test_an_unusable_reply_acts_on_nothing_and_triggers_a_call_the_next_second(tmp_path):
    r = fly(tmp_path, Script("I would turn left.", toward_centre()), steps=200)
    ev = events(tmp_path, r)
    replies = [e for e in ev if e["kind"] == "adviser_reply"]
    assert replies[0]["refusal"] == "unparseable" and replies[0]["part"] is None
    assert [(c["t"], c["triggers"]) for c in ev if c["kind"] == "adviser_call"] == [(0, ["start"]), (61, ["reply_refused"])]
    assert not [e for e in ev if e["kind"] == "action" and e["t"] == 60]
    assert r["method_seconds"]["reasoning_model"] == 1                   # only the second reply's heading, at 121


@pytest.mark.parametrize("reply, why", [
    ({"heading_deg": 90, "next_call_at_s": 12.5}, "bad_schedule"),
    ({"heading_deg": 90, "next_call_at_s": -1}, "bad_schedule"),
    ({"heading_deg": 90, "next_call_at_s": True}, "bad_schedule"),
    ({"heading_deg": 90, "next_call_at_s": "soon"}, "bad_schedule"),
    ({"heading_deg": 90, "report": "Wedgetail, aircraft 2"}, "unknown_field"),
    ({"report_trigger_nm": 20, "next_call_at_s": None}, "no_heading"),
    ({"heading_deg": 360, "next_call_at_s": None}, "invalid_action"),
    ({"heading_deg": 90, "report_trigger_nm": 0}, "invalid_action"),
    ({"heading_deg": 90, "help": {"reason": "bored", "text": "x"}}, "invalid_action"),
])
def test_malformed_replies_are_refused_under_the_fixed_schema(reply, why):
    with pytest.raises(AD.AdviceRefused, match=why):
        AD.parse(json.dumps(reply), help_asked=False)


def test_well_formed_replies_parse_and_absent_schedule_is_null():
    assert AD.parse('{"heading_deg": 90}', False) == ({"heading_deg": 90}, None)
    assert AD.parse('<think>x</think>{"heading_deg": 1.5, "report_trigger_nm": null, "next_call_at_s": 300.0}',
                    False) == ({"heading_deg": 1.5, "report_trigger_nm": None}, 300)
    with pytest.raises(AD.AdviceRefused, match="second_help"):
        AD.parse(json.dumps({"heading_deg": 90, "help": {"reason": "missing_tool", "text": "x"}}), help_asked=True)


def test_a_goal_change_while_pending_waits_and_the_old_goal_reply_is_refused(tmp_path):
    s = scenario()
    first = s.goals[0][1]
    second = dataclasses.replace(first, version=first.version + 1)
    r = fly(tmp_path, Script(toward_centre(), toward_centre()), steps=200, goals=(s.goals[0], (30, second)))
    ev = events(tmp_path, r)
    replies = [e for e in ev if e["kind"] == "adviser_reply"]
    assert replies[0]["t"] == 60 and replies[0]["stale"] and replies[0]["refusal"] == "out_of_date"
    assert [(c["t"], c["triggers"], c["goal_version"]) for c in ev if c["kind"] == "adviser_call"] == [
        (0, ["start"], first.version), (61, ["goal"], second.version)]
    assert not [e for e in ev if e["kind"] == "action" and e["t"] == 60]


def test_stale_evidence_triggers_once_per_change_from_fresh(tmp_path):
    s = scenario()
    goal_tanker = s.goals[0][1].tanker
    r = fly(tmp_path, Script(toward_centre(), toward_centre()), steps=200, dropouts=((goal_tanker, 1, None),))
    ev = events(tmp_path, r)
    assert [(c["t"], c["triggers"]) for c in ev if c["kind"] == "adviser_call"] == [(0, ["start"]), (61, ["stale"])]


def test_help_is_asked_once_and_a_second_help_reply_is_refused(tmp_path):
    ask = {"help": {"reason": "missing_or_stale_evidence", "text": "where is the tanker?"}}
    r = fly(tmp_path, Script(toward_centre(100, **ask), toward_centre(None, **ask)), steps=200)
    ev = events(tmp_path, r)
    replies = [e for e in ev if e["kind"] == "adviser_reply"]
    assert replies[0]["refusal"] is None and replies[1]["refusal"] == "second_help"
    assert r["help_requests"] == 1


def test_a_call_pending_at_the_end_stays_charged_and_is_listed(tmp_path):
    r = fly(tmp_path, Script(toward_centre()), steps=30)
    ev = events(tmp_path, r)
    assert [e["request_id"] for e in ev if e["kind"] == "model_undelivered"] == [
        next(e["request_id"] for e in ev if e["kind"] == "adviser_call")]
    assert r["model_calls"] == 1 and r["tokens_in"] == 300
    assert r["method_seconds"]["holding_pending"] == 30


def test_the_adviser_is_told_its_latency_and_sees_telohelms_public_context(tmp_path):
    script = Script(toward_centre())
    seen = []
    provider = M.ScriptedProvider(lambda req: (seen.append(req), script(req))[1])
    AD.run_adviser_episode(scenario(), tmp_path / "e", "4401", provider, latency_s=30.0, max_steps=40)
    system = seen[0].messages[0][1]
    assert "applied 30 simulated seconds after" in system and system.startswith(AD.local_model.SYSTEM)
    assert seen[0].role == "reasoning_model" and seen[0].max_output_tokens == AD.OUTPUT_TOKENS
    assert "reports" not in script.seen[0]["context"] and "recent_reports" in script.seen[0]["context"]


def test_the_command_line_refuses_scored_streams(tmp_path):
    with pytest.raises(SystemExit):
        AD.main(["--stream", "1001", "--out", str(tmp_path / "x")])


def test_the_matched_draw_reproduces_codexs_reference_exactly():
    """D22: the 100 matched indices and their digest equal Codex's prospective reference (runtime-turn125.json),
    made independently before any scored scenario."""
    import hashlib
    from pathlib import Path
    path = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/reproduction/runtime-turn125.json'
    if not path.exists():
        pytest.skip("the reference record is in the paper2 folder, which a code-only copy does not have")
    ref = json.loads(path.read_text())["sampling"]
    draws = AD.matched_draws()
    assert draws == ref["matched_draws"]
    assert hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest() == AD.MATCHED_DRAWS_SHA256
    assert AD.MATCHED_DRAWS_SHA256 == ref["draws_sha256"]
    for seed in draws.values():
        assert sorted({i // 5 for i in seed["3"]["draw_order"][:4]}) == [0, 1, 2, 3]
        assert all(len(set(p["draw_order"])) == 5 for p in seed.values())
