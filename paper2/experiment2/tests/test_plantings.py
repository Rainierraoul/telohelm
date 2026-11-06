"""The 10 mutation operators of protocol section 6 (study2/plantings.py)."""
import pytest

import paper2.experiment2.plantings as P

CHECK = '''def helper(x):
    return 'reject'

def check(ctx, out):
    if ctx.report_age_s > 10 and ctx.goal_version == 1:
        return 'reject'
    if abs(ctx.observed_range_rate_nm_s) <= 0.05:
        return 'reject'
    if out.report_tanker != ctx.goal_tanker:
        return 'reject'
    return 'accept'
'''


def mutated(op):
    m = P.mutate(CHECK, op)
    assert m is not None and m.before_sha256 != m.after_sha256
    compile(m.source, "<planting>", "exec")
    return m


def test_each_operator_makes_the_change_the_protocol_names():
    assert "ctx.report_age_s > 100" in mutated(1).source                     # first numeric constant in a comparison
    assert "ctx.report_age_s > 10 and True" in mutated(2).source             # last operand of the first and
    assert "ctx.report_age_s <= 10" in mutated(3).source                     # first comparison reversed
    assert "report_age_s" not in mutated(4).source.split("def check")[1].split("goal_version")[0]
    assert "if ctx.report_age_s > 10 and True" in mutated(5).source          # goal-version comparison is True
    assert "except Exception" in mutated(6).source and "return 'accept'" in mutated(6).source
    body = mutated(7).source.split("def check")[1]
    assert "'reject'" not in body and "def helper(x):\n    return 'reject'" in mutated(7).source
    assert "<= 0.5" in mutated(8).source                                     # small constant becomes 0.5
    assert "if True:" in mutated(9).source                                   # range-rate comparison is True
    assert "out.report_tanker" not in mutated(10).source                     # tanker comparison is True


def test_first_follows_pre_order_and_the_path_is_recorded():
    m = P.mutate(CHECK, 3)
    node = P._get(__import__("ast").parse(CHECK), m.path)
    assert type(node).__name__ == "Compare" and node.left.attr == "report_age_s"
    assert m.path[0] == "body"


def test_an_operator_that_does_not_apply_returns_none():
    plain = "def check(ctx, out):\n    return 'accept'\n"
    assert all(P.mutate(plain, op) is None for op in (1, 2, 3, 4, 5, 8, 9, 10))
    assert P.mutate(plain, 7) is None                                         # already returns accept: no change
    assert P.mutate("def decide(ctx):\n    return {}\n", 6) is None           # no check function to wrap


def test_operators_are_tried_from_the_counter_and_wrap():
    assert [m.operation for m in P.in_order(CHECK, 9)][:3] == [9, 10, 1]
    assert len(list(P.in_order(CHECK, 1))) == 10


def test_attribute_names_must_match_exactly():
    near = "def check(ctx, out):\n    if ctx.report_age_seconds > 5:\n        return 'reject'\n    return 'accept'\n"
    assert P.mutate(near, 4) is None
