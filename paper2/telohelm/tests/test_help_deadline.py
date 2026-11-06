"""The stale-evidence help rule with a procedure flying (study2/runtime/escalation.py; STACK-WALKTHROUGH.md)."""
import telohelm.runtime.escalation as X


def ladder():
    return X.FlightLadder(None)


def ctx(age, deadline):
    return {"t": 100, "report_age_s": age, "deadline": deadline}


def test_a_procedure_no_longer_silences_stale_evidence_when_the_goal_has_a_deadline():
    assert ladder().help(ctx(40, 600), {}, heading_by_procedure=True)["reason"] == "missing_or_stale_evidence"


def test_without_a_deadline_a_flying_procedure_still_waits_for_the_reports():
    assert ladder().help(ctx(40, None), {}, heading_by_procedure=True) is None


def test_fresh_evidence_or_a_reply_that_would_come_too_late_asks_nothing():
    assert ladder().help(ctx(5, 600), {}, heading_by_procedure=True) is None
    assert ladder().help(ctx(40, 120), {}, heading_by_procedure=True) is None      # reply at 130 s, after the deadline


def test_without_a_procedure_the_rule_is_unchanged():
    assert ladder().help(ctx(40, None), {}, heading_by_procedure=False)["reason"] == "missing_or_stale_evidence"
