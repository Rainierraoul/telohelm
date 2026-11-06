"""Learning between episodes (MVP step 3) with scripted proposals and invented cases. No model is called."""
from __future__ import annotations

import json

import pytest

import telohelm.learning as G
import paper2.experiment2.mvp_seed as seed
from telohelm.runtime.library import Library
import telohelm.taskpacks.tanker.world.phase5 as phase5

PROC = 'def decide(ctx):\n    return {"heading_deg": 123.0}\n'
ACCEPT_ALL = "def check(ctx, out):\n    return 'accept'\n"
REJECT_ALL = "def check(ctx, out):\n    return 'reject'\n"
BY_FLAG = "def check(ctx, out):\n    return 'accept' if ctx.t >= 0 and out.heading_deg < 200 else 'reject'\n"


def reply(source=PROC, kind="procedure", decision="heading"):
    return json.dumps({"kind": kind, "decision": decision, "conditions": [], "source": source,
                       "reason": {"code": "unfamiliar_but_solvable", "text": "invented"}})


class Scripted:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, messages, meta):
        self.calls.append((messages, meta))
        return self.texts.pop(0) if self.texts else "no json"


def coordinator(tmp_path, propose, fly=None, corpus=None, solvable=lambda s: True, **kw):
    lib = Library(tmp_path / "library")
    seed.stock(lib)
    return G.Coordinator(lib, tmp_path, propose, solvable, seed.stand_in_tool, corpus=corpus, fly=fly, **kw), lib


def outcome(goal_met=False, joined=False, report=False):
    return {"goal_met": goal_met, "joined": joined, "report_correct": report}


def finish(co, n, first=1, **o):
    events = []
    for i in range(first, first + n):
        events += co.after_episode(phase5.draw(i).world_a, outcome(**o), complete=True)
    return events


def passing(shadow, c, root):
    return {"episode": c.episode, "goal_met": True, "help": 0, "end_reason": "join"}


def failing(shadow, c, root):
    return {"episode": c.episode, "goal_met": False, "help": 0, "end_reason": "horizon"}


def test_a_prefix_is_not_learned_from(tmp_path):
    co, _ = coordinator(tmp_path, Scripted(reply()), fly=passing)
    for i in range(1, 6):
        co.after_episode(phase5.draw(i).world_a, outcome(), complete=False)
    assert co.completed == [] and co.attempts == 0 and [e["kind"] for e in co.events] == ["prefix_excluded"] * 5


def test_attempts_only_at_every_fifth_completed_episode_and_only_with_a_reason(tmp_path):
    p = Scripted(reply(), reply())
    co, _ = coordinator(tmp_path, p, fly=failing)
    finish(co, 4)
    assert co.attempts == 0
    finish(co, 1, first=5)
    assert co.attempts == 1
    finish(co, 5, first=6, goal_met=True, joined=True, report=True)       # no misses: nothing queued
    assert co.attempts == 1 and co.events[-1] == {"kind": "no_attempt", "ordinal": 10, "reason": "no queued reason"}


def test_a_passing_procedure_is_adopted_and_its_predecessor_suspended(tmp_path):
    co, lib = coordinator(tmp_path, Scripted(reply()), fly=passing)
    old = lib.adopted("heading", "procedure")
    finish(co, 5)
    result = co.events[-1]
    assert result["kind"] == "attempt_result" and result["passed"] is True and result["predecessor"] == old[0]
    assert lib.adopted("heading", "procedure") == [result["candidate"]]
    assert lib.projection()["entries"][old[0]]["status"] == "suspended"


def test_a_failing_or_empty_trial_leaves_the_candidate_unadopted(tmp_path):
    co, lib = coordinator(tmp_path, Scripted(reply()), fly=failing)
    old = lib.adopted("heading", "procedure")
    finish(co, 5)
    r = co.events[-1]
    assert r["passed"] is False and lib.adopted("heading", "procedure") == old
    assert lib.projection()["entries"][r["candidate"]]["status"] == "candidate"
    # Every episode asked for help because its evidence ran out, so none is a fair trial case.
    co2, _ = coordinator(tmp_path / "b", Scripted(reply()), fly=passing)
    for i in range(1, 6):
        co2.after_episode(phase5.draw(i).world_a, {**outcome(), "help_requests": [STALE]}, complete=True)
    assert co2.events[-1] == {"kind": "attempt_refused", "attempt": 1, "reason": "no eligible trial set"}


STALE = {"t": 100, "goal_version": 1, "reason": "missing_or_stale_evidence", "text": "invented"}


def _with_help(requests, summary_shape):
    """An episode record with these accepted help requests: a bare world outcome lists them; a flown episode's
    summary counts them at its top level and lists them in its nested outcome (Codex's TURN295 witness)."""
    if not summary_shape:
        return {**outcome(), "help_requests": requests}
    return {**outcome(), "help_requests": len(requests), "outcome": {**outcome(), "help_requests": requests}}


@pytest.mark.parametrize("summary_shape", [False, True])
def test_the_trial_set_is_the_last_eight_eligible_completed_episodes_declared_first(tmp_path, summary_shape):
    """Eligibility is Telohelm's own record (27 October): an accepted stale-evidence help request leaves an episode out;
    a request for want of a method, or the reference pilot's label, does not."""
    seen = []
    def fly(shadow, c, root):
        seen.append(c.episode)
        return passing(shadow, c, root)
    co, _ = coordinator(tmp_path, Scripted(reply(), reply()), fly=fly, solvable=lambda s: s.episode_id != "dev-4")
    for i in range(1, 11):
        help_requests = ([STALE] if i == 9 else
                         [{**STALE, "reason": "missing_tool"}] if i == 3 else [])
        co.after_episode(phase5.draw(i).world_a, _with_help(help_requests, summary_shape), complete=True)
    start = [e for e in co.events if e["kind"] == "attempt_start"][-1]
    assert start["trial_set"] == ["dev-2", "dev-3", "dev-4", "dev-5", "dev-6", "dev-7", "dev-8", "dev-10"]
    assert seen[-8:] == start["trial_set"]
    completed = {e["episode"]: e for e in co.events if e["kind"] == "completed"}
    assert completed["dev-9"]["eligible"] is False and completed["dev-4"]["solvable"] is False
    assert completed["dev-4"]["eligible"] is True and completed["dev-3"]["eligible"] is True


def test_format_repairs_use_at_most_three_calls(tmp_path):
    p = Scripted("not json", json.dumps({"kind": "procedure"}), reply())
    co, _ = coordinator(tmp_path, p, fly=passing)
    finish(co, 5)
    assert len(p.calls) == 3 and co.events[-1]["passed"] is True and co.events[-1]["calls"] == 3
    p2 = Scripted("x", "y", "z", reply())
    co2, _ = coordinator(tmp_path / "b", p2, fly=passing)
    finish(co2, 5)
    assert len(p2.calls) == 3 and co2.events[-1]["kind"] == "attempt_failed"


def test_the_proposer_sees_feedback_and_the_library_and_no_hidden_truth(tmp_path):
    p = Scripted(reply())
    co, _ = coordinator(tmp_path, p, fly=passing)
    finish(co, 5)
    body = json.loads(p.calls[0][0][1][1])
    assert set(body) == {"feedback", "library", "context_fields"}
    assert body["feedback"][0] == {"episode": "dev-1", "failed": "both"} and "heading procedure" in body["library"]
    assert "true" not in json.dumps(body).lower().replace("context_fields", "")


def corpus(n_wrong=5):
    ctx = {"t": 1}
    return G.Corpus(wrong=[{"context": ctx, "output": {"heading_deg": 300.0}} for _ in range(n_wrong)],
                    right=[{"context": ctx, "output": {"heading_deg": 100.0}} for _ in range(100)])


@pytest.mark.parametrize("source,adopted", [(BY_FLAG, True), (ACCEPT_ALL, False), (REJECT_ALL, False)])
def test_a_check_is_adopted_only_through_the_replay_test(tmp_path, source, adopted):
    co, lib = coordinator(tmp_path, Scripted(reply(source, kind="check")), fly=passing, corpus=corpus())
    finish(co, 5)
    r = co.events[-1]
    assert r["passed"] is adopted
    assert (lib.adopted("heading", "check") == [r["candidate"]]) is adopted


def test_a_check_with_no_stored_cases_is_refused(tmp_path):
    co, lib = coordinator(tmp_path, Scripted(reply(BY_FLAG, kind="check")), fly=passing)
    finish(co, 5)
    assert co.events[-1]["passed"] is False and co.events[-1]["detail"]["refused"]


def test_a_newly_confirmed_wrong_answer_undoes_a_check_that_accepts_it(tmp_path):
    co, lib = coordinator(tmp_path, Scripted(reply(BY_FLAG, kind="check")), fly=passing, corpus=corpus())
    old = lib.adopted("heading", "check")
    finish(co, 5)
    new = lib.adopted("heading", "check")[0]
    assert co.confirm_wrong({"context": {"t": 1}, "output": {"heading_deg": 150.0}}) == [
        {"kind": "undo", "retired": new, "restored": old[0]}]
    assert lib.adopted("heading", "check") == old
    assert co.confirm_wrong({"context": {"t": 1}, "output": {"heading_deg": 350.0}}) == []   # nothing left to undo


def test_the_floor_breaches_once_and_rearms_after_recovery(tmp_path):
    co, _ = coordinator(tmp_path, Scripted(*([reply()] * 20)), fly=failing, attempt_every=1000)
    finish(co, 20, goal_met=True, joined=True, report=True)
    finish(co, 2, first=21)                                               # 21st solvable: 19/20, a breach
    kinds = [e["kind"] for e in co.events]
    assert kinds.count("floor_breach") == 1
    finish(co, 20, first=23, goal_met=True, joined=True, report=True)
    assert [e["kind"] for e in co.events].count("floor_rearmed") == 1


def test_a_real_shadow_flight_uses_the_candidates_actions(tmp_path):
    co, lib = coordinator(tmp_path, Scripted(reply()), shadow_max_steps=3)
    finish(co, 5)
    trial = co.events[-1]["detail"]["trial"]
    assert len(trial) == 5 and all(t["goal_met"] is False for t in trial)        # 3 seconds cannot meet the goal
    assert all(a["heading_deg"] == 123.0 for t in trial for a in t["first_actions"])


def test_the_proposer_is_told_the_rules_the_checks_enforce(tmp_path):
    """13 October, first cluster stream: attempt 1's only answer was refused for getattr, a rule the prompt never
    stated. The prompt now states the rules a procedure or check needs, with numbers from the modules that enforce
    them, and no flying advice. Codex's MVP119.REPORT found the report tolerance wrong; it is now the world's."""
    import telohelm.runtime.procedures as R
    c, _ = coordinator(tmp_path, Scripted())
    system = dict(c.messages([]))["system"]
    assert system == G.PROPOSER_RULES
    for name in R.FORBIDDEN_NAMES + tuple(R.ALLOWED_IMPORTS) + tuple(R.REASON_CODES) + tuple(R.OPERATORS):
        assert name in system, name
    assert '{"heading_deg"' in system and '"report_trigger_nm"' in system and str(R.MAX_SOURCE_CHARS) in system
    for advice in ("intercept", "lead", "aim", "wind correction", "orbit centre"):
        assert advice not in system.lower(), advice


def test_the_reasoning_model_gets_the_protocols_output_cap():
    """Protocol section 5: at most 16,384 output tokens and 600 seconds per call. It was 8,192 until 13 October."""
    import inspect
    defaults = inspect.signature(G.GatewayProposer.__init__).parameters
    assert G.PROTOCOL_OUTPUT_TOKENS == 16_384
    assert defaults["max_output_tokens"].default == 16_384 and defaults["timeout_s"].default == 600.0



def test_the_report_rule_in_the_prompt_is_the_worlds():
    import telohelm.taskpacks.tanker.world.episode as E
    assert f"within {E.REPORT_TOLERANCE} NM of report_at_nm" in G.PROPOSER_RULES
    assert f"within {E.JOIN_TOLERANCE} NM of it" in G.PROPOSER_RULES and "0.15" not in G.PROPOSER_RULES


class Counting:
    """An invented provider that records what reaches it."""
    def __init__(self):
        self.sent = []

    def __call__(self, request):
        from telohelm.runtime.models import ModelResponse
        self.sent.append(request)
        return ModelResponse("answered", "{}", tokens_in=1, tokens_out=1)


def proposer_with(tmp_path, monkeypatch, count_tokens=None):
    import telohelm.local_model as LM
    from telohelm.runtime.models import ScriptedProvider
    seen = Counting()
    monkeypatch.setattr(LM, "provider", lambda *a, **kw: ScriptedProvider(seen))
    return G.GatewayProposer("http://127.0.0.1:1", "invented", tmp_path, "invented", count_tokens=count_tokens), seen


def preflight(tmp_path, n=1):
    return [json.loads(l) for l in (tmp_path / f"learning-attempt-{n}" / "preflight.jsonl").read_text().splitlines()]


def test_requests_within_the_input_cap_are_sent_and_every_count_is_recorded(tmp_path, monkeypatch):
    p, seen = proposer_with(tmp_path, monkeypatch)
    p([("system", G.PROPOSER_RULES), ("user", "{}")], {"attempt": 1, "call": 1, "episode": "e"})
    (row,) = preflight(tmp_path)
    assert len(seen.sent) == 1 and row["sent"] is True and 600 < row["input_tokens"] <= 4096 and row["cap"] == 4096


def test_requests_over_the_input_cap_or_uncountable_are_not_sent(tmp_path, monkeypatch):
    p, seen = proposer_with(tmp_path, monkeypatch, count_tokens=lambda m: 4097)
    assert p([("user", "x")], {"attempt": 1, "call": 1, "episode": "e"}) == ""
    def broken(m):
        raise OSError("no tokenizer")
    q, seen2 = proposer_with(tmp_path / "b", monkeypatch, count_tokens=broken)
    assert q([("user", "x")], {"attempt": 1, "call": 1, "episode": "e"}) == ""
    assert seen.sent == [] and seen2.sent == [] and p.ledger.calls == [] and q.ledger.calls == []
    assert preflight(tmp_path)[0]["sent"] is False and preflight(tmp_path / "b")[0]["error"] == "OSError"


def test_the_count_includes_the_template_and_a_repairs_history():
    import telohelm.token_count as T
    one = [("system", G.PROPOSER_RULES), ("user", "{}")]
    repaired = one + [("assistant", "not json " * 50), ("user", "Fix these problems and reply again: x")]
    assert T.count(repaired) > T.count(one) + 100
    assert T.render(one).startswith("<|im_start|>system\n") and T.render(one).endswith("<|im_start|>assistant\n<think>\n")


def test_the_prompt_states_the_prediction_tools_interface_and_limits_without_advice():
    """BACKLOG B35, after Codex's TURN 121 review: the interface, its bounds and its precision limit, from the
    modules that enforce them, and no way of using it."""
    import telohelm.runtime.broker as B; import telohelm.runtime.prediction as PT
    p = G.PROPOSER_RULES
    for text in ("predict(args)", "goal_version", "headings_deg", "wind_nm_s", "angle_rad", "positions_nm",
                 "closest_approach_nm", "closest_at_s", '"refusal"', f"1 to {B.MAX_PREDICTION_SECONDS}",
                 f"{B.PREDICTION_ALLOWANCE} predictions", f"{PT.MAX_EVALUATIONS:,} evaluations", "and nothing else"):
        assert text in p, text
    for advice in ("intercept", "aim", "lead", "steer", "pursuit", "wind correction"):
        assert advice not in p.lower(), advice


def test_the_prompt_states_the_triggers_re_arming_in_codexs_reviewed_words():
    """BACKLOG B38, wording accepted in Codex's TURN 129 review."""
    assert ("Each accepted non-null trigger setting re-arms it, even at the same distance; it cannot fire in its arming "
            "second, and omitting this field leaves it armed until it fires, is cleared, or the goal version changes."
            ) in G.PROPOSER_RULES


def test_the_stand_in_heading_procedure_never_returns_360():
    """B38 note: math.degrees of a tiny negative angle, modulo 360, is exactly 360.0, which the world refuses. The
    stand-in now returns 0.0 there, the same direction; other headings are unchanged."""
    import math
    from types import SimpleNamespace
    import paper2.experiment2.mvp_seed as seed
    ns = {}
    exec(seed.AIM, ns)
    ctx = SimpleNamespace(x=0.0, y=0.0, t=0, report_measured_at=0, report_x=-1e-16, report_y=11.0,
                          tanker_centre_x=-1e-16, tanker_centre_y=10.0, tanker_radius_nm=1.0, tanker_speed_nm_s=0.0,
                          wind_x_nm_s=0.0, wind_y_nm_s=0.0, tas_nm_s=0.125)
    assert math.degrees(math.atan2(-1e-16, 11.0)) % 360 == 360.0
    assert ns["decide"](ctx) == {"heading_deg": 0.0}
    east = SimpleNamespace(**{**vars(ctx), "report_x": 10.0, "tanker_centre_x": 10.0, "report_y": 1.0,
                              "tanker_centre_y": 0.0})
    assert ns["decide"](east)["heading_deg"] == pytest.approx(math.degrees(math.atan2(10.0, 1.0)))


def _o(**kw):
    base = {"end_reason": "join", "settled": True, "help_requests": [], "joined": True, "in_time": True,
            "refusals": [], "reports": [], "goal": {"version": 1}, "report_correct": False,
            "report_correct_options": [False]}
    return {**base, **kw}


def _d(t, decision, method, part, goal_version=1):
    # Decision records carry their goal tag, as real ones do; untagged records count for nothing (TRIAL147).
    return {"kind": "decision", "t": t, "decision": decision, "method": method, "part": part,
            "goal_version": goal_version}


def test_a_heading_candidate_passes_on_joining_with_its_own_heading_even_without_a_report():
    used = [_d(0, "heading", "cand", {"heading_deg": 45.0}), _d(1, "heading", "cand", {"heading_deg": 46.0})]
    v = G.component_trial("heading", "cand", _o(), used)
    assert v["passed"] and v["used_seconds"] == 2


@pytest.mark.parametrize("outcome,decisions,reason", [
    (_o(joined=False, end_reason="horizon"), None, "did not join"),
    (_o(in_time=False, end_reason="deadline"), None, "did not join"),
    (_o(settled=False), None, "not settled"),
    (_o(end_reason=None), None, "did not end naturally"),
    (_o(help_requests=[{"reason": "missing_tool"}]), None, "help"),
    (_o(), [_d(0, "heading", "other", {"heading_deg": 45.0})], "never used"),
    (_o(refusals=[{"t": 0, "reason": "goal version out of date"}]), [_d(0, "heading", "cand", {"heading_deg": 45.0})],
     "never used"),
])
def test_a_heading_candidate_fails_on_any_missing_component_evidence(outcome, decisions, reason):
    decisions = decisions if decisions is not None else [_d(0, "heading", "cand", {"heading_deg": 45.0})]
    v = G.component_trial("heading", "cand", outcome, decisions)
    assert not v["passed"] and reason in v["reason"]


def _rep(t, via_trigger=False, version=1):
    return {"t": t, "goal_version": version, "via_trigger": via_trigger}


def test_a_report_candidate_passes_only_with_one_correct_report_of_its_own():
    good = dict(report_correct=True, report_correct_options=[True])
    direct = G.component_trial("report", "cand", _o(reports=[_rep(9)], **good),
                               [_d(9, "report", "cand", {"report": "Wedgetail, aircraft 2, 20 NM from SHELL 1"})])
    assert direct["passed"]
    trig = G.component_trial("report", "cand", _o(reports=[_rep(30, via_trigger=True)], **good),
                             [_d(5, "report", "cand", {"report_trigger_nm": 20.0})])
    assert trig["passed"]
    other = G.component_trial("report", "cand", _o(reports=[_rep(30, via_trigger=True)], **good),
                              [_d(4, "report", "cand", {"report_trigger_nm": 20.0}),
                               _d(5, "report", "someone-else", {"report_trigger_nm": 20.0})])
    assert not other["passed"] and "did not come from the candidate" in other["reason"]


@pytest.mark.parametrize("kw,reason", [
    (dict(reports=[]), "0 reports"),
    (dict(reports=[_rep(9), _rep(12)], report_correct=False, report_correct_options=[False]), "2 reports"),
    (dict(reports=[_rep(9)], report_correct=False, report_correct_options=[False]), "not correct"),
    (dict(reports=[_rep(9)], report_correct=True, report_correct_options=[False, True]), "not correct"),
    (dict(reports=[_rep(9, version=1)], goal={"version": 2}), "0 reports"),
])
def test_a_report_candidate_fails_on_missing_duplicate_wrong_ambiguous_or_old_goal_reports(kw, reason):
    v = G.component_trial("report", "cand", _o(**kw), [_d(9, "report", "cand", {"report": "x"})])
    assert not v["passed"] and reason in v["reason"]


def test_a_real_shadow_flight_admits_a_heading_procedure_that_joins_with_a_check_in_place(tmp_path):
    """The candidate is the starting library's heading procedure, its check adopted beside it, flown in shadow on a
    generated phase 1 episode: it joins, makes no report (no report procedure), and passes its component trial."""
    from telohelm.runtime.library import Library
    import paper2.experiment2.streams as streams
    lib = Library(tmp_path / "library")
    lib.set_status(lib.add({"kind": "check", "decision": "heading", "conditions": [], "source": seed.HEADING_CHECK,
                            "origin": {"calls": []}}), "adopted", "invented")
    cand = lib.add({"kind": "procedure", "decision": "heading", "conditions": [], "source": seed.AIM,
                    "origin": {"calls": []}})
    co = G.Coordinator(lib, tmp_path, Scripted(), lambda s: True, seed.stand_in_tool)
    sc = streams.solvable_episode(4401, 1, 0).scenario
    c = G.Completed(1, sc.episode_id, sc, False, "both", True)
    passed, detail = co.trial(cand, None, "heading", [c])
    (r,) = detail["trial"]
    assert passed and r["component"]["used_seconds"] > 0 and r["outcome"]["joined"] and not r["goal_met"]


def test_trial_evidence_holds_the_scenario_library_ledger_and_settings_read_only(tmp_path):
    from pathlib import Path
    from telohelm.runtime.library import Library
    import paper2.experiment2.streams as streams
    lib = Library(tmp_path / "library")
    lib.set_status(lib.add({"kind": "check", "decision": "heading", "conditions": [], "source": seed.HEADING_CHECK,
                            "origin": {"calls": []}}), "adopted", "invented")
    cand = lib.add({"kind": "procedure", "decision": "heading", "conditions": [], "source": seed.AIM,
                    "origin": {"calls": []}})
    co = G.Coordinator(lib, tmp_path, Scripted(), lambda s: True, seed.stand_in_tool, shadow_max_steps=3)
    sc = streams.solvable_episode(4401, 1, 0).scenario
    c = G.Completed(1, sc.episode_id, sc, False, "both", True)
    _, first = co.trial(cand, None, "heading", [c])
    _, second = co.trial(cand, None, "heading", [c])
    a, b = Path(first["trial"][0]["evidence_dir"]), Path(second["trial"][0]["evidence_dir"])
    assert a != b and a.name == "trial-0000" and b.name == "trial-0001"
    for name in ("scenario.json", "ledger.jsonl", "settings.json", "log", "heads"):
        assert (a / name).exists(), name
    assert (a / "shadow-library").is_dir()
    assert not (a / "log").stat().st_mode & 0o222                       # read-only


PAIR_PROC = 'def decide(ctx):\n    return {"heading_deg": 77.0}\n'
PAIR_CHECK = "def check(ctx, out):\n    return 'accept' if out.heading_deg < 200 else 'reject'\n"


def pair_reply(proc=PAIR_PROC, check=PAIR_CHECK):
    return json.dumps({"kind": "pair", "decision": "heading", "procedure": {"conditions": [], "source": proc},
                       "check": {"conditions": [], "source": check},
                       "reason": {"code": "unfamiliar_but_solvable", "text": "invented"}})


def corpus100():
    return G.Corpus(wrong=[{"context": {"t": 0}, "output": {"heading_deg": 300.0}}],
                    right=[{"context": {"t": 0}, "output": {"heading_deg": float(i)}} for i in range(100)])


def status(lib, sha):
    return lib.projection()["entries"][sha]["status"]


def test_a_pair_whose_check_fails_the_replay_test_adopts_neither(tmp_path):
    co, lib = coordinator(tmp_path, Scripted(pair_reply()), fly=passing)          # no corpus: the check is refused
    finish(co, 5)
    e = co.events[-1]
    assert e["entry_kind"] == "pair" and e["passed"] is False and "procedure" not in e["detail"]
    assert status(lib, e["candidate"]) == "candidate" and status(lib, e["paired_check"]) == "candidate"
    assert status(lib, e["predecessor"]) == "adopted"


def test_a_pair_is_trialled_with_its_check_in_place_and_adopted_together(tmp_path):
    seen = []
    def fly(shadow, c, root):
        seen.append(set(shadow.adopted("heading", "check")))
        return {"episode": c.episode, "goal_met": True, "help": 0, "end_reason": "join"}
    co, lib = coordinator(tmp_path, Scripted(pair_reply()), fly=fly, corpus=corpus100())
    finish(co, 5)
    e = co.events[-1]
    assert e["passed"] is True
    assert all(s == {e["paired_check"]} for s in seen)                # the candidate check, not its predecessor
    assert status(lib, e["candidate"]) == status(lib, e["paired_check"]) == "adopted"
    assert status(lib, e["predecessor"]) == status(lib, e["predecessor_check"]) == "suspended"
    rows = [json.loads(l) for l in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["adopting", "adopted"]


def test_a_pair_whose_procedure_fails_its_trial_adopts_neither(tmp_path):
    co, lib = coordinator(tmp_path, Scripted(pair_reply()), fly=failing, corpus=corpus100())
    finish(co, 5)
    e = co.events[-1]
    assert e["passed"] is False and e["detail"]["check"]["replay"]["passed"] and "procedure" in e["detail"]
    assert status(lib, e["candidate"]) == status(lib, e["paired_check"]) == "candidate"
    assert status(lib, e["predecessor"]) == status(lib, e["predecessor_check"]) == "adopted"


def test_an_interrupted_pair_adoption_is_completed_and_never_left_half_done(tmp_path, monkeypatch):
    co, lib = coordinator(tmp_path, Scripted(pair_reply()), fly=passing, corpus=corpus100())
    real = lib.set_status
    def crash_on_procedure(sha, new, why, **kw):
        if "pair from attempt" in why and lib.entry(sha)["kind"] == "procedure":
            raise RuntimeError("invented crash")
        return real(sha, new, why, **kw)
    monkeypatch.setattr(lib, "set_status", crash_on_procedure)
    with pytest.raises(RuntimeError):
        finish(co, 5)
    monkeypatch.setattr(lib, "set_status", real)
    rows = [json.loads(l) for l in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["adopting"]
    co2 = G.Coordinator(lib, tmp_path, Scripted(), lambda s: True, seed.stand_in_tool)
    done = co2.reconcile_pairs()
    assert len(done) == 1
    assert status(lib, rows[0]["procedure"]) == status(lib, rows[0]["check"]) == "adopted"
    assert co2.reconcile_pairs() == []                                     # carried through once


def test_retiring_a_paired_check_suspends_its_procedure_and_restores_the_one_before(tmp_path):
    co, lib = coordinator(tmp_path, Scripted(pair_reply()), fly=passing, corpus=corpus100())
    finish(co, 5)
    e = co.events[-1]
    assert e["passed"] is True
    co.confirm_wrong({"context": {"t": 0}, "output": {"heading_deg": 150.0}})     # the paired check accepts it
    assert status(lib, e["paired_check"]) == "retired" and status(lib, e["candidate"]) == "suspended"
    assert status(lib, e["predecessor"]) == status(lib, e["predecessor_check"]) == "adopted"
    assert [x.get("reason") for x in co.events if x["kind"] == "undo"][-1] == "paired check retired"


def test_the_prompt_states_the_adoption_rules_and_the_pair_schema():
    p = G.PROPOSER_RULES
    for text in ("used only when its decision also has an adopted check", "at least 95 of 100 right ones",
                 '"kind": "pair"', "adopted together or not at all", "joins the goal's tanker in time"):
        assert text in p, text
