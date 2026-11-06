"""Tests for the replay test and the method chooser, on made-up cases and procedures."""
from __future__ import annotations

import pytest

import telohelm.runtime.chooser as C
import telohelm.runtime.library as L
import telohelm.runtime.replay as R
import telohelm.runtime.sandbox as S

pytestmark = pytest.mark.skipif(not S.available(), reason="macOS sandbox route only")

AGE_CHECK = "def check(ctx, out):\n    return 'accept' if ctx.report_age_s <= 10 else 'reject'\n"
LOOSE_CHECK = "def check(ctx, out):\n    return 'accept' if ctx.report_age_s <= 1000 else 'reject'\n"
REJECT_ALL = "def check(ctx, out):\n    return 'reject'\n"
CRASHING = "def check(ctx, out):\n    return 1 / 0\n"


def case(age):
    return {"context": {"report_age_s": age}, "output": {"heading_deg": 90}}


WRONG = [case(30 + i) for i in range(20)]           # stale evidence treated as current
RIGHT = [case(i % 10) for i in range(100)]


def test_a_sound_check_passes():
    r = R.replay_test(AGE_CHECK, WRONG, RIGHT)
    assert r["passed"] and r["right_accepted"] == 100 and r["wrong_accepted"] == []


def test_a_weakened_check_is_caught():
    r = R.replay_test(LOOSE_CHECK, WRONG, RIGHT, old_source=AGE_CHECK)
    assert not r["passed"] and len(r["wrong_accepted"]) == 20


def test_rejecting_everything_cannot_pass():
    r = R.replay_test(REJECT_ALL, WRONG, RIGHT)
    assert not r["passed"] and r["right_accepted"] == 0


def test_fewer_right_answers_than_the_old_check_fails():
    stricter = "def check(ctx, out):\n    return 'accept' if ctx.report_age_s <= 8 else 'reject'\n"
    r = R.replay_test(stricter, WRONG, RIGHT, old_source=AGE_CHECK, min_right=50)
    assert not r["passed"] and r["right_accepted"] == 90 and r["old_right_accepted"] == 100


def test_errors_count_as_not_accepting():
    r = R.replay_test(CRASHING, WRONG, RIGHT)
    assert not r["passed"] and r["wrong_accepted"] == [] and r["right_accepted"] == 0


def entry(kind, source, conditions=()):
    return {"kind": kind, "decision": "heading", "conditions": list(conditions), "source": source,
            "origin": {"calls": []}}


AIM = ("import math\ndef decide(ctx):\n    return {'goal_version': ctx.goal_version, "
       "'heading_deg': math.degrees(math.atan2(ctx.report_x - ctx.x, ctx.report_y - ctx.y)) % 360}\n")
CTX = {"x": 0.0, "y": 0.0, "goal_version": 1, "report_x": 1.0, "report_y": 0.0, "report_age_s": 2}


def test_chooser_uses_an_adopted_procedure_whose_output_passes(tmp_path):
    lib = L.Library(tmp_path)
    p = lib.add(entry("procedure", AIM, [{"field": "report_age_s", "op": "<=", "value": 10}]))
    c = lib.add(entry("check", AGE_CHECK))
    lib.set_status(p, "adopted", "test")
    lib.set_status(c, "adopted", "test")
    r = C.choose("heading", CTX, lib)
    assert r["use"] == p and r["output"] == {"goal_version": 1, "heading_deg": 90.0}


def test_chooser_climbs_with_its_reason(tmp_path):
    lib = L.Library(tmp_path)
    assert C.choose("heading", {**CTX, "report_x": None}, lib)["climb"] == "missing_or_stale_evidence"
    assert C.choose("heading", CTX, lib)["climb"] == "missing_tool"
    p = lib.add(entry("procedure", AIM))
    lib.set_status(p, "adopted", "test")
    assert C.choose("heading", CTX, lib)["climb"] == "missing_tool"               # no adopted check
    c = lib.add(entry("check", REJECT_ALL))
    lib.set_status(c, "adopted", "test")
    r = C.choose("heading", CTX, lib)
    assert r["climb"] == "unfamiliar_but_solvable" and r["tried"][0]["accepted"] is False


def test_chooser_skips_procedures_whose_conditions_fail(tmp_path):
    lib = L.Library(tmp_path)
    p = lib.add(entry("procedure", AIM, [{"field": "report_age_s", "op": "<=", "value": 1}]))
    c = lib.add(entry("check", LOOSE_CHECK))
    lib.set_status(p, "adopted", "test")
    lib.set_status(c, "adopted", "test")
    assert C.choose("heading", CTX, lib)["climb"] == "missing_tool"


def test_a_rejected_procedure_keeps_every_check_verdict(tmp_path):
    lib = L.Library(tmp_path)
    p = lib.add(entry("procedure", AIM))
    c1 = lib.add(entry("check", AGE_CHECK))
    c2 = lib.add(entry("check", REJECT_ALL))
    for sha in (p, c1, c2):
        lib.set_status(sha, "adopted", "test")
    r = C.choose("heading", CTX, lib)
    checks = r["tried"][0]["checks"]
    assert r["climb"] == "unfamiliar_but_solvable"
    assert [(c["check"], c["verdict"], c["accepted"]) for c in checks] == [(c1, "accept", True), (c2, "reject", False)]
    assert all(c["wall_seconds"] > 0 and c["cpu_seconds"] is not None for c in checks)


def test_a_check_applies_only_where_its_conditions_hold(tmp_path):
    lib = L.Library(tmp_path)
    p = lib.add(entry("procedure", AIM))
    c = lib.add(entry("check", REJECT_ALL, [{"field": "report_age_s", "op": ">", "value": 100}]))
    for sha in (p, c):
        lib.set_status(sha, "adopted", "test")
    assert C.choose("heading", CTX, lib)["climb"] == "missing_tool"     # the only check does not apply here
