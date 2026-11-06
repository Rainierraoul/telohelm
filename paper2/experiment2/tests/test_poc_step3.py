"""Step 3 of the proof-of-concept work: trusted guards for both decisions, and the model writes procedures for
whichever part of the goal failed."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import telohelm.learning as G; import paper2.experiment2.run_stream as RS
import telohelm.runtime.sandbox as S
from telohelm.runtime.library import Library
from telohelm.runtime.procedures import reply_errors

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
HEADING = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/taskpacks/tanker/guards/poc-heading-guard.py'
REPORT = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/telohelm/src/telohelm/taskpacks/tanker/guards/poc-report-guard.py'
ns: dict = {}
exec(REPORT.read_text(), ns)
check = ns["check"]


class Rec:
    def __init__(self, **d):
        self._d = d

    def __getattr__(self, k):
        if k in self.__dict__.get("_d", {}):
            return self._d[k]
        raise AttributeError(k)


@pytest.mark.parametrize("fields,expected", [
    ({}, "accept"), ({"report": "Wedgetail, aircraft 2, 30 NM from SHELL 1"}, "accept"),
    ({"report_trigger_nm": None}, "accept"), ({"report_trigger_nm": 30}, "accept"), ({"report_trigger_nm": 1000.0}, "accept"),
    ({"report": "x", "report_trigger_nm": 25.5}, "accept"),
    ({"report": ""}, "reject"), ({"report": "   "}, "reject"), ({"report": "x" * 201}, "reject"), ({"report": 30}, "reject"),
    ({"report_trigger_nm": 0}, "reject"), ({"report_trigger_nm": -1}, "reject"), ({"report_trigger_nm": 1000.5}, "reject"),
    ({"report_trigger_nm": math.nan}, "reject"), ({"report_trigger_nm": math.inf}, "reject"),
    ({"report_trigger_nm": True}, "reject"), ({"report_trigger_nm": "30"}, "reject"),
])
def test_the_report_guard_accepts_exactly_the_public_report_actions(fields, expected):
    assert check(None, Rec(**fields)) == expected


def test_the_report_guard_is_a_valid_report_check_and_runs_contained(tmp_path):
    src = REPORT.read_text()
    assert reply_errors({"kind": "check", "decision": "report", "conditions": [], "source": src,
                         "reason": {"code": "missing_tool", "text": "x"}}) == []
    if S.available():
        assert S.run(src, "check", [{"t": 0}, {"report_trigger_nm": 30.0}], context_args=True).result == "accept"
        assert S.run(src, "check", [{"t": 0}, {"report": ""}], context_args=True).result == "reject"


def test_a_stream_with_both_guards_learns_procedures_for_both_decisions(tmp_path, monkeypatch):
    seen = {}

    class Built(Exception):
        pass

    def spy(*a, **k):
        seen.update(k)
        raise Built

    monkeypatch.setattr(G, "Coordinator", spy)
    with pytest.raises(Built):
        RS.run_stream(1, 1, tmp_path / "out", max_steps=2, empty_library=True, trusted_heading_guard=HEADING,
                      trusted_report_guard=REPORT, learn={"base_url": "http://127.0.0.1:1", "model": "m"})
    assert seen["learn_decisions"] == ("heading", "report") and seen["procedures_only"] is True
    assert seen["extra_rules"] == RS.POC_RULES_BOTH
    lib = Library(tmp_path / "out" / "library")
    assert len(lib.adopted("heading", "check")) == 1 and len(lib.adopted("report", "check")) == 1
    assert lib.entry(lib.adopted("report", "check")[0])["source"] == REPORT.read_text()


def test_a_report_guard_needs_the_heading_guard(tmp_path):
    with pytest.raises(ValueError):
        RS.run_stream(1, 1, tmp_path / "out", max_steps=2, empty_library=True, trusted_report_guard=REPORT)


def test_procedures_only_refuses_checks_and_pairs_when_learning_both(tmp_path):
    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, lambda *_: "", lambda _: True, lambda *_: None,
                       procedures_only=True)
    proc = {"kind": "procedure", "decision": "report", "conditions": [], "source": "def decide(ctx):\n    return {}\n",
            "reason": {"code": "missing_tool", "text": "x"}}
    assert co._scope_errors(proc) == []
    assert co._scope_errors({**proc, "decision": "heading"}) == []
    assert co._scope_errors({**proc, "kind": "check"}) and co._scope_errors({"kind": "pair", "decision": "report"})
    plain = G.Coordinator(Library(tmp_path / "l2"), tmp_path, lambda *_: "", lambda _: True, lambda *_: None)
    assert plain._scope_errors({**proc, "kind": "check"}) == []          # the default run is unchanged


def test_the_step3_rules_keep_the_interface_clarification_and_the_byte_bound():
    assert 'decision \\"heading\\" or \\"report\\"'.replace("\\\\", "") in RS.POC_RULES_BOTH.replace("\\\\", "") or \
        'decision "heading" or "report"' in RS.POC_RULES_BOTH
    assert "x is east and y is north" in RS.POC_RULES_BOTH and "65,536 bytes" in RS.POC_RULES_BOTH
    assert "Propose heading procedures only" not in RS.POC_RULES_BOTH


def test_no_pause_is_a_command_line_option_that_reaches_the_stream(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(RS, "run_stream", lambda *a, **k: seen.update(k) or {"summary": {
        "completed": 0, "episodes": 0, "goal_met_of_completed": 0, "joined_of_completed": 0, "model_calls": 0}})
    RS.main(["--out", str(tmp_path / "o"), "--no-pause"])
    assert seen["no_pause"] is True
    RS.main(["--out", str(tmp_path / "p")])
    assert seen["no_pause"] is False
