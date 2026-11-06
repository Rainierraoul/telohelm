"""Examples from the arm's own episodes in the learning request (study2/examples.py; B7 item 5; Codex's TURN 205)."""
import json
from pathlib import Path

import pytest

import telohelm.examples as X; import telohelm.learning as G; import paper2.experiment2.run_stream as RS
from telohelm.runtime.library import Library
from telohelm.runtime.procedures import CONTEXT_FIELDS


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("ex") / "run"
    RS.run_stream(3, 1, out, stream_seed=4401, phases=(1,), empty_library=True, max_steps=200)
    order = [json.loads(l)["episode"] for l in (out / "generation.jsonl").read_text().splitlines()]
    return out, order


def completed(order, failed):
    return [G.Completed(i + 1, ep, None, f is None, f, True) for i, (ep, f) in enumerate(zip(order, failed))]


def test_each_example_is_what_the_arm_saw_shortened_and_named_by_alias(run):
    out, order = run
    shown, receipts, omitted = X.select(out, completed(order, ["join", "join", "join"]),
                                        [{"episode": order[2], "failed": "join"}, {"episode": order[1], "failed": "join"}])
    assert omitted == [] and len(shown) == 4
    assert [(r["episode"], r["second"]) for r in receipts] == [(order[2], 120), (order[2], 199), (order[1], 120),
                                                               (order[1], 199)]
    text = json.dumps(shown)
    assert "s4401" not in text and all(r["alias"] in text for r in receipts)
    for e, r in zip(shown, receipts):
        assert set(e["context"]) <= CONTEXT_FIELDS and len(e["context"]["reports"]) <= X.REPORTS_SHOWN
        assert r["recorded_context_sha256"] == r["full_context_sha256"]
        assert set(e["outcome"]) == {"goal_met", "failed"} and e["outcome"]["goal_met"] is False
        assert set(e) <= {"episode", "second", "outcome", "context", "submitted", "applied", "report_transmitted"}


def test_report_failures_use_transmission_seconds_and_both_takes_the_union(run, tmp_path):
    out, order = run
    copy = tmp_path / "run"
    import shutil
    shutil.copytree(out, copy)
    s = json.loads((copy / order[0] / "summary.json").read_text())
    s["outcome"]["reports"] = [{"t": 150, "via_trigger": False, "true_range": 1.0, "correct": False}]
    (copy / order[0] / "summary.json").write_text(json.dumps(s))
    assert X.points(copy / order[0], "report", s["outcome"])[0] == [150]
    assert X.points(copy / order[0], "both", s["outcome"])[0] == [120, 150, 199]
    shown, receipts, _ = X.select(copy, completed(order, ["both", None, None]), [{"episode": order[0], "failed": "both"}])
    assert [(r["role"], r["second"]) for r in receipts] == [("failure", 120), ("failure", 150), ("failure", 199),
                                                            ("contrast", 120)]
    sent = shown[1]["report_transmitted"]
    assert sent == {"by_trigger": False}                     # the transmission and how, nothing else of the outcome
    assert "true_range" not in json.dumps(shown) and "correct" not in json.dumps(shown)


def test_an_episode_queued_twice_is_counted_once_and_the_limit_is_eight(run):
    out, order = run
    reasons = [{"episode": order[2], "failed": "join"}, {"episode": order[2], "failed": "floor"}]
    shown, receipts, _ = X.select(out, completed(order, ["join", "join", "join"]), reasons)
    assert [r["episode"] for r in receipts] == [order[2], order[2]]
    many = [G.Completed(i, order[i % 3], None, False, "join", True) for i in range(12)]
    shown, receipts, omitted = X.select(out, many, [{"episode": c.episode, "failed": "join"} for c in many])
    assert len(shown) <= X.MAX_EXAMPLES


def test_a_log_that_does_not_verify_gives_no_example(run, tmp_path):
    out, order = run
    import shutil
    copy = tmp_path / "run"
    shutil.copytree(out, copy)
    log = copy / order[1] / "broker.log"
    lines = log.read_text().splitlines()
    lines[5] = lines[5].replace('"t":', '"t" :', 1)
    log.write_text("\n".join(lines) + "\n")
    shown, receipts, omitted = X.select(copy, completed(order, ["join", "join", "join"]), [{"episode": order[1], "failed": "join"}])
    assert all(r["episode"] != order[1] for r in receipts)
    assert any(o["episode"] == order[1] and "does not verify" in o["reason"] for o in omitted)


def coordinator(out, examples, propose=None):
    lib = Library(out.parent / ("lib-on" if examples else "lib-off"))
    return G.Coordinator(lib, out, propose or (lambda *_: ""), lambda _: True, lambda *_: None, examples=examples)


def test_without_examples_the_request_is_unchanged(run):
    out, order = run
    reasons = [{"episode": order[2], "failed": "join"}]
    off = coordinator(out, False)
    plain = off.messages(reasons)
    assert "examples" not in json.loads(plain[1][1]) and X.RULES not in plain[0][1]
    assert json.loads(plain[1][1])["feedback"] == reasons


def test_with_examples_feedback_is_aliased_and_examples_are_fitted_oldest_first(run):
    out, order = run
    reasons = [{"episode": order[2], "failed": "join"}, {"episode": order[1], "failed": "join"}]

    class Counting:
        input_cap = None

        def count_tokens(self, messages):
            return sum(len(c) for _, c in messages)

        def __call__(self, *_):
            return ""
    p = Counting()
    co = coordinator(out, True, p)
    co.completed = completed(order, ["join", "join", "join"])
    co._select_examples(1, reasons)
    full = co.messages(reasons)
    body = json.loads(full[1][1])
    assert X.RULES in full[0][1] and len(body["examples"]) == 4
    assert body["feedback"] == [{"episode": X.alias(order[2]), "failed": "join"}, {"episode": X.alias(order[1]), "failed": "join"}]
    p.input_cap = p.count_tokens(full) - 1                 # one character too many: the oldest example goes
    log = {}
    fitted = co.messages(reasons, (), log)
    kept = [(r["episode"], r["second"]) for r in log["kept"]]
    assert [(r["episode"], r["second"]) for r in log["dropped"]] == [(order[1], 120)]
    assert kept == [(order[2], 120), (order[2], 199), (order[1], 199)]
    assert p.count_tokens(fitted) <= p.input_cap


def test_a_format_repair_is_refitted_with_its_tail(run):
    out, order = run
    reasons = [{"episode": order[2], "failed": "join"}]
    sent = []

    class Proposer:
        input_cap = None

        def count_tokens(self, messages):
            return sum(len(c) for _, c in messages)

        def __call__(self, messages, meta):
            sent.append(messages)
            return "not json" if len(sent) == 1 else ""
    p = Proposer()
    co = coordinator(out, True, p)
    co.completed = completed(order, ["join", "join", "join"])
    co._select_examples(1, reasons)
    first = co.messages(reasons)
    p.input_cap = p.count_tokens(first) + 10              # the first send fits; the repair's tail does not
    co.queue = list(reasons)
    co.attempt(co.completed[-1])
    assert len(sent) >= 2
    assert sent[1][-2][0] == "assistant" and sent[1][-1][0] == "user"
    assert p.count_tokens(sent[1]) <= p.input_cap
    assert len(json.loads(sent[1][1][1])["examples"]) < len(json.loads(sent[0][1][1])["examples"])
    rows = [json.loads(l) for l in (out / "learning-examples.jsonl").read_text().splitlines()]
    stages = [r["stage"] for r in rows]
    last = len(stages) - 1 - stages[::-1].index("selected")
    assert stages[last:] == ["selected"] + ["sent"] * len(sent)


def test_both_learning_conditions_get_the_same_selection_from_the_same_history(run):
    out, order = run
    reasons = [{"episode": order[2], "failed": "join"}]
    a = X.select(out, completed(order, ["join", "join", "join"]), reasons)
    b = X.select(out, completed(order, ["join", "join", "join"]), reasons)
    assert a == b


def test_a_held_heading_is_not_shown_as_sent_and_a_trigger_is_kept_apart_from_its_transmission():
    actions = {409: {"action": {"goal_version": 1, "heading_deg": 54.6, "report_trigger_nm": 32.8},
                     "field_writers": {"heading_deg": "heading", "report_trigger_nm": "report"}},
               410: {"action": {"goal_version": 1, "report_trigger_nm": 32.9},
                     "field_writers": {"report_trigger_nm": "report"}},
               411: {"action": {"goal_version": 1}, "field_writers": {}}}
    sources = {(409, "heading"): "procedure", (409, "report"): "procedure", (410, "report"): "small_model"}
    assert X.submitted_at(410, actions, sources) == {"report_trigger_nm": {"value": 32.9, "by": "small model"}}
    assert X.submitted_at(411, actions, sources) == {}               # the heading was held, not sent again
    steps = {409: {"accepted": True}, 410: {"accepted": True}, 411: {"accepted": True}}
    assert X.transmission(411, True, actions, sources, steps) == {"by_trigger": True, "trigger_set_at": 410,
                                                                  "trigger_nm": 32.9, "trigger_by": "small model"}
    assert X.transmission(411, False, actions, sources, steps) == {"by_trigger": False}


def _trigger(value, by="report"):
    return {"action": {"report_trigger_nm": value}, "field_writers": {"report_trigger_nm": by}}


def test_a_refused_replacement_does_not_take_the_credit_for_the_armed_trigger():
    # Codex's TURN207 witness: a trigger accepted at 10 survives a refused replacement at 20.
    actions = {10: _trigger(10), 20: _trigger(20)}
    sources = {(10, "report"): "procedure", (20, "report"): "reasoning_model"}
    shown = X.transmission(30, True, actions, sources, {10: {"accepted": True}, 20: {"accepted": False}})
    assert shown == {"by_trigger": True, "trigger_set_at": 10, "trigger_nm": 10, "trigger_by": "stored procedure"}


def test_no_trigger_is_named_when_the_logs_cannot_establish_which_was_armed():
    ok = {10: {"accepted": True}, 15: {"accepted": True}, 20: {"accepted": True}}
    sources = {(10, "report"): "procedure"}
    disarmed = {10: _trigger(10), 15: _trigger(None)}
    direct = {10: _trigger(10), 15: {"action": {"report": "x"}, "field_writers": {"report": "report"}}}
    for actions in (disarmed, direct):
        assert "trigger_unattributed" in X.transmission(30, True, actions, sources, ok)
    assert "trigger_unattributed" in X.transmission(30, True, {10: _trigger(10)}, sources, ok, goal_changes=[20])
    assert "trigger_unattributed" in X.transmission(30, True, {10: _trigger(10)}, sources, ok, fired=[25])
    assert "trigger_unattributed" in X.transmission(30, True, {10: _trigger(10)}, sources, {})    # no world step
    later = X.transmission(30, True, {10: _trigger(10), 22: _trigger(12)}, sources | {(22, "report"): "small_model"},
                           ok | {22: {"accepted": True}}, goal_changes=[20])
    assert later["trigger_set_at"] == 22                   # set after the goal change, so it is the one armed


def test_an_action_the_world_refused_is_marked_not_applied(run, tmp_path):
    out, order = run
    import shutil
    copy = tmp_path / "run"
    shutil.copytree(out, copy)
    (shown, receipt), = X.build(copy / order[0], order[0], [120], {"goal_met": False, "failed": "join"}, {})
    assert shown["applied"] is True
    assert X.applied_at(120, {120: {"accepted": False}}) is False
    assert X.applied_at(120, {120: {"accepted": "yes"}}) is False      # only an exact True counts as applied
    assert X.applied_at(121, {120: {"accepted": True}}) is None
