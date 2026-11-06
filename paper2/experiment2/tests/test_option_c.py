"""Option C (paper2/notes/OPTION-C-PLAN.md): learned checks beside permanent guards, and cases labelled by outcome."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import telohelm.learning as G; import paper2.experiment2.outcomecases as O; import paper2.experiment2.run_stream as RS
from telohelm.runtime.library import Library

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
HEADING = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/taskpacks/tanker/guards/poc-heading-guard.py'
REPORT = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/taskpacks/tanker/guards/poc-report-guard.py'
CHECK = "def check(ctx, out):\n    return 'accept'\n"


def guarded(tmp_path, **kw):
    lib = Library(tmp_path / "library")
    hg = RS.install_trusted_heading_guard(lib, HEADING)
    rg = RS.install_trusted_heading_guard(lib, REPORT, decision="report")
    co = G.Coordinator(lib, tmp_path, lambda *_: "", lambda _: True, lambda *_: None, **kw)
    return lib, co, hg, rg


def test_a_guard_is_never_a_learned_checks_predecessor(tmp_path):
    lib, co, hg, rg = guarded(tmp_path)
    assert co._is_guard(hg) and co._is_guard(rg)
    assert co._learned("heading", "check") == [] and co._learned("report", "check") == []
    learned = lib.add({"kind": "check", "decision": "heading", "conditions": [], "source": CHECK, "origin": {"attempt": 1}})
    lib.set_status(learned, "adopted", "test")
    assert co._learned("heading", "check") == [learned]
    assert set(lib.adopted("heading", "check")) == {hg, learned}          # both run; the guard stays


def test_an_adopted_learned_check_runs_beside_the_guard_and_the_guard_is_never_suspended(tmp_path):
    lib, co, hg, rg = guarded(tmp_path)
    reply = {"kind": "check", "decision": "heading", "conditions": [], "source": CHECK,
             "reason": {"code": "missing_tool", "text": "x"}}
    co.propose = lambda *_: json.dumps(reply)
    co.check_replay = lambda *a, **k: (True, {"replay": {"passed": True}})
    for i in range(1, 6):
        co.after_episode(SimpleNamespace(episode_id=f"e{i}"),
                         {"goal_met": False, "joined": False, "report_correct": False}, True)
    adopted = lib.adopted("heading", "check")
    assert hg in adopted and len(adopted) == 2
    assert lib.projection()["entries"][hg]["status"] == "adopted"
    result = next(e for e in co.events if e["kind"] == "attempt_result")
    assert result["passed"] and result["predecessor"] is None


def test_the_prompt_lists_guards_and_learned_checks_apart(tmp_path):
    lib, co, hg, rg = guarded(tmp_path)
    src = co._sources()
    assert "heading guard (fixed)" in src and "report guard (fixed)" in src and "heading check" not in src


def test_learning_checks_needs_both_guards_and_a_case_store(tmp_path):
    with pytest.raises(ValueError):
        RS.run_stream(1, 1, tmp_path / "a", max_steps=2, empty_library=True, trusted_heading_guard=HEADING,
                      trusted_report_guard=REPORT, learn_checks=True,
                      learn={"base_url": "http://127.0.0.1:1", "model": "m"})


def test_the_option_c_rules_state_the_label_rule_and_allow_checks():
    assert "120 seconds" in RS.POC_RULES_C and "15 seconds" in RS.POC_RULES_C
    assert "You may propose procedures and checks" in RS.POC_RULES_C and "65,536 bytes" in RS.POC_RULES_C


@pytest.fixture(scope="module")
def moment_rows():
    ms = O.moments(4400, 1, phases=(1,))
    return O._label_moment(ms[2])


def test_outcome_labels_follow_the_rule_at_a_far_moment(moment_rows):
    by = {(r["decision"], r["evidence"]["construction"], r["evidence"].get("turn_deg", r["evidence"].get("offset_nm")))
          : r["disposition"] for r in moment_rows}
    assert by[("heading", "reference", None)] == "right"
    assert by[("heading", "turned", 180.0)] == "wrong" and by[("heading", "turned", 90.0)] == "wrong"
    assert by[("report", "nothing", None)] == "right" and by[("report", "trigger_at_order", None)] == "right"
    assert by[("report", "report_now", None)] == "wrong"


def test_the_verifier_recomputes_each_label_from_scratch_and_refuses_a_flipped_one(moment_rows):
    cases = [r for r in moment_rows if r["disposition"] in ("right", "wrong")]     # marginal rows never become cases
    for r in cases[:3] + cases[-2:]:
        case = O._case(4401, "stored_right" if r["disposition"] == "right" else "stored_wrong", r, 0)
        assert O.verify(case) is True
        flipped = json.loads(json.dumps(case))
        flipped["label"]["disposition"] = "wrong" if r["disposition"] == "right" else "right"
        assert O.verify(flipped) is False


def test_a_case_with_a_changed_context_goal_or_origin_fails_the_verifier(moment_rows):
    r = moment_rows[0]
    case = O._case(4401, "stored_right", r, 0)
    assert O.verify(case) is True
    for change in (lambda c: c["context"].update(x=c["context"]["x"] + 0.5),
                   lambda c: c["goal"].update(report_at_nm=c["goal"]["report_at_nm"] + 1),
                   lambda c: c["origin"].update(second=c["origin"]["second"] + 1)):
        bad = json.loads(json.dumps(case))
        change(bad)
        assert O.verify(bad) is False


def test_a_store_with_another_label_rule_is_refused_when_learning_checks(tmp_path):
    from telohelm.qualification.store import CaseStore
    store = CaseStore(tmp_path / "store", verifier=lambda c: True, verifier_name="test", development=True)
    assert O.store_rules(store) == set()
    with pytest.raises(ValueError, match="labelled only by"):
        RS.run_stream(1, 1, tmp_path / "a", max_steps=2, empty_library=True, trusted_heading_guard=HEADING,
                      trusted_report_guard=REPORT, learn_checks=True, dev_cases=tmp_path / "store", stream_seed=4401,
                      learn={"base_url": "http://127.0.0.1:1", "model": "m"})


def test_the_wording_names_the_reference_continuation_not_a_best_course():
    assert "best course" not in RS.POC_RULES_C
    assert "reference pilot's course" in RS.POC_RULES_C and "no trigger is armed" in RS.POC_RULES_C


def test_the_no_replay_arm_adopts_checks_without_replay_and_never_undoes(tmp_path):
    lib, co, hg, rg = guarded(tmp_path, no_replay=True)
    ok, detail = co.check_replay(CHECK, None, "heading", 1)
    assert ok is True and detail["replay"]["skipped"] is True
    assert "not run" in detail["replay"]["reason"]
    events = co.confirm_wrong({"split": "confirmed_wrong"})
    assert [e["kind"] for e in events] == ["undo_skipped"]


def test_the_no_replay_arm_needs_learned_checks(tmp_path):
    with pytest.raises(ValueError, match="needs --learn-checks"):
        RS.run_stream(1, 1, tmp_path / "a", max_steps=2, empty_library=True, trusted_heading_guard=HEADING,
                      trusted_report_guard=REPORT, no_replay=True, learn={"base_url": "http://127.0.0.1:1", "model": "m"})


def test_both_learning_rule_sets_say_a_report_trial_needs_an_adopted_heading_to_reach_range():
    for rules in (RS.POC_RULES_BOTH, RS.POC_RULES_C):
        assert "may never come within reporting range" in rules and "adopted heading procedure" in rules
