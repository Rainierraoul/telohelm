"""Tests for the reply checks and the read-only context of backlog note B7. Made-up replies only."""
from __future__ import annotations

import copy

import pytest

import telohelm.runtime.procedures as R
import telohelm.runtime.sandbox as S

GOOD = {
    "kind": "procedure", "decision": "heading",
    "conditions": [{"field": "report_age_s", "op": "<=", "value": 10}],
    "source": ("import math\n"
               "def decide(ctx):\n"
               "    dx, dy = ctx.report_x - ctx.x, ctx.report_y - ctx.y\n"
               "    return {'goal_version': ctx.goal_version, 'heading_deg': math.degrees(math.atan2(dx, dy)) % 360}\n"),
    "reason": {"code": "unfamiliar_but_solvable", "text": "no procedure for this decision yet"},
}
CHECK = {
    "kind": "check", "decision": "heading", "conditions": [],
    "source": "def check(ctx, out):\n    return 'accept' if ctx.report_age_s <= 10 and 0 <= out.heading_deg < 360 else 'reject'\n",
    "reason": {"code": "missing_tool", "text": "no check yet"},
}
CTX = {"t": 12, "x": 0.0, "y": 0.0, "goal_version": 1, "report_x": 3.0, "report_y": 3.0, "report_age_s": 2,
       "reports": [{"x": 3.0, "y": 3.0, "measured_at": 10}]}


def bad(**changes):
    r = copy.deepcopy(GOOD)
    r.update(changes)
    return r


def test_good_replies_pass():
    assert R.reply_errors(GOOD) == []
    assert R.reply_errors(CHECK) == []


@pytest.mark.parametrize("reply, fragment", [
    ("not an object", "JSON object"),
    ({k: v for k, v in GOOD.items() if k != "reason"}, "exactly the fields"),
    (bad(kind="tool"), "kind"),
    (bad(decision="orbit"), "decision"),
    (bad(conditions=[{"field": "true_range", "op": "<", "value": 1}]), "unknown field"),
    (bad(conditions=[{"field": "report_age_s", "op": "~", "value": 1}]), "unknown operator"),
    (bad(conditions=[{"field": "report_age_s", "op": "<", "value": True}]), "value must be"),
    (bad(reason={"code": "guess", "text": ""}), "4 codes"),
    (bad(source="def decide(ctx):\n    return ("), "does not parse"),
    (bad(source="def act(ctx):\n    return {}\n"), "decide() exactly once"),
    (bad(source="import os\ndef decide(ctx):\n    return {}\n"), "import not allowed: os"),
    (bad(source="from pathlib import Path\ndef decide(ctx):\n    return {}\n"), "import not allowed: pathlib"),
    (bad(source="def decide(ctx):\n    return object.__getattribute__(ctx, '_d')\n"), "double-underscore attribute"),
    (bad(source="def decide(ctx):\n    return __import__('os')\n"), "double-underscore name"),
    (bad(source="x = 1\n" * 2000 + "def decide(ctx):\n    return {}\n"), "longer than"),
])
def test_refusals(reply, fragment):
    errors = R.reply_errors(reply)
    assert errors and any(fragment in e for e in errors), errors


def test_conditions_hold():
    assert R.conditions_hold(GOOD["conditions"], {"report_age_s": 3})
    assert not R.conditions_hold(GOOD["conditions"], {"report_age_s": 11})
    assert not R.conditions_hold(GOOD["conditions"], {"report_age_s": None})
    assert not R.conditions_hold(GOOD["conditions"], {})


def test_weakening_fields_are_context_or_output_fields():
    for names in R.WEAKENING_FIELDS.values():
        assert names <= R.CONTEXT_FIELDS | R.OUTPUT_FIELDS


sandbox_only = pytest.mark.skipif(not S.available(), reason="macOS sandbox route only")


@sandbox_only
def test_procedure_and_check_run_on_the_context_object():
    r = S.run(GOOD["source"], "decide", [CTX], context_args=True)
    assert r.status == "ok" and r.result == {"goal_version": 1, "heading_deg": 45.0}
    c = S.run(CHECK["source"], "check", [CTX, {"heading_deg": 45.0}], context_args=True)
    assert c.status == "ok" and c.result == "accept"


@sandbox_only
@pytest.mark.parametrize("body", ["return ctx['report_age_s']", "ctx.report_age_s = 0\n    return 1",
                                  "return ctx.true_range", "return ctx.reports[0]['x']"])
def test_context_allows_attribute_reads_only(body):
    r = S.run(f"def decide(ctx):\n    {body}\n", "decide", [CTX], context_args=True)
    assert r.status == "error"


@sandbox_only
def test_history_is_read_by_attribute():
    r = S.run("def decide(ctx):\n    return ctx.reports[-1].measured_at\n", "decide", [CTX], context_args=True)
    assert r.status == "ok" and r.result == 10


# --- R22.7: fail closed on malformed replies; the missing-value rule for every operator --------------

@pytest.mark.parametrize("field, value", [("kind", []), ("decision", {}), ("reason", {"code": [], "text": ""})])
def test_unhashable_values_are_refused_not_raised(field, value):
    errors = R.reply_errors(bad(**{field: value}))
    assert errors


@pytest.mark.parametrize("cond", [{"field": [], "op": "<", "value": 1}, {"field": "report_age_s", "op": [], "value": 1},
                                  {"field": "report_age_s", "op": "<", "value": float("inf")},
                                  {"field": "report_age_s", "op": "<", "value": None}])
def test_malformed_conditions_are_refused(cond):
    assert R.reply_errors(bad(conditions=[cond]))


@pytest.mark.parametrize("op", ["<", "<=", "==", "!=", ">=", ">"])
def test_a_missing_or_null_field_fails_every_operator(op):
    c = [{"field": "report_age_s", "op": op, "value": 0}]
    assert not R.conditions_hold(c, {}) and not R.conditions_hold(c, {"report_age_s": None})


@pytest.mark.parametrize("source", [
    'def decide(ctx):\n    b = globals()["__builtins__"]\n    return 1\n',
    'def check(ctx, out):\n    print("x")\n    return "reject"\n',
    'def decide(ctx):\n    return getattr(ctx, "x")\n',
])
def test_codex_probe_patterns_are_refused_by_the_reply_checks(source):
    kind = "check" if "def check" in source else "procedure"
    assert R.source_errors(source, kind)
