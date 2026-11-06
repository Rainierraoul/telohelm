"""Property tests of the planting operators (study2/plantings.py, protocol section 6), with Hypothesis.

Development tests only. Hypothesis generates small checks in the style the learner writes (comparisons on context
and output fields, joined by and, or and not, with helper and nested functions), and each test states a rule the
operators must keep for every one of them. Runs are derandomised so they repeat exactly.
"""
import ast
import hashlib
from types import SimpleNamespace

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st          # noqa: E402

import paper2.experiment2.plantings as PL                                 # noqa: E402

SETTINGS = settings(derandomize=True, max_examples=300, deadline=None)

CTX_FIELDS = ("report_age_s", "goal_version", "observed_range_rate_nm_s", "goal_tanker", "observed_range_nm", "t")
OUT_FIELDS = ("heading_deg", "report_tanker", "report_trigger_nm")
NUMBERS = st.sampled_from(["0", "0.05", "0.1", "0.25", "1", "1e-3", "15", "30.0", "360"])
OPS = st.sampled_from(["<", "<=", ">", ">=", "==", "!="])

operand = st.one_of(st.sampled_from([f"ctx.{f}" for f in CTX_FIELDS] + [f"out.{f}" for f in OUT_FIELDS]), NUMBERS)
comparison = st.builds(lambda a, op, b: f"{a} {op} {b}", operand, OPS, operand)
condition = st.recursive(
    comparison,
    lambda c: st.one_of(st.builds(lambda a, b: f"({a} and {b})", c, c),
                        st.builds(lambda a, b: f"({a} or {b})", c, c),
                        st.builds(lambda a: f"not ({a})", c)),
    max_leaves=5)
verdict = st.sampled_from(["'accept'", "'reject'"])


@st.composite
def checks(draw):
    lines = []
    if draw(st.booleans()):                         # a helper before check, as the reference checks have
        lines += ["def helper(ctx):", f"    return {draw(condition)}", ""]
    lines.append("def check(ctx, out):")
    if draw(st.booleans()):                         # a nested function whose returns operator 7 must leave alone
        lines += ["    def inner(x):", f"        return {draw(verdict)}"]
    for _ in range(draw(st.integers(0, 3))):
        lines += [f"    if {draw(condition)}:", f"        return {draw(verdict)}"]
    if draw(st.booleans()):
        lines.append(f"    return 'accept' if {draw(condition)} else 'reject'")
    else:
        lines.append(f"    return {draw(verdict)}")
    return "\n".join(lines) + "\n"


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@SETTINGS
@given(checks(), st.integers(1, 10))
def test_a_mutation_is_well_formed_and_records_what_it_changed(source, op):
    m = PL.mutate(source, op)
    if m is None:
        return
    ast.parse(m.source)                                             # still compiles
    assert m.operation == op and m.before_sha256 == _sha(source) and m.after_sha256 == _sha(m.source)
    assert m.source != ast.unparse(ast.parse(source)) + "\n"        # a real change, not a reformatting
    PL._get(ast.parse(source), m.path)                              # the path names a node of the original


@SETTINGS
@given(checks())
def test_a_field_operator_applies_exactly_when_a_comparison_reads_its_field(source):
    tree = ast.parse(source)
    for op, key in ((4, "report_age"), (5, "goal_version"), (9, "range_rate"), (10, "tanker")):
        reads = any(isinstance(n, ast.Compare) and PL._reads(n, PL.FIELDS[key]) for n, _, _ in PL.walk(tree))
        assert (PL.mutate(source, op) is not None) == reads


@SETTINGS
@given(checks(), st.integers(1, 10))
def test_the_order_starts_at_the_counter_wraps_and_skips_only_what_does_not_apply(source, counter):
    order = [m.operation for m in PL.in_order(source, counter)]
    expected = [(counter - 1 + k) % 10 + 1 for k in range(10)]
    assert order == [op for op in expected if PL.mutate(source, op) is not None]


def _values():
    return st.one_of(st.integers(-5, 400), st.floats(-5, 400, allow_nan=False), st.none(),
                     st.sampled_from(["SHELL 1", "SHELL 2"]))


@st.composite
def inputs(draw):
    ctx = SimpleNamespace(**{f: draw(_values()) for f in CTX_FIELDS})
    out = SimpleNamespace(**{f: draw(_values()) for f in OUT_FIELDS})
    return ctx, out


def _call(source, ctx, out):
    space = {}
    exec(compile(source, "<check>", "exec"), space)
    try:
        return ("value", space["check"](ctx, out))
    except Exception as e:                                          # a comparison of a string with a number, say
        return ("error", type(e).__name__)


@SETTINGS
@given(checks(), inputs())
def test_operator_6_turns_every_error_into_accept_and_changes_nothing_else(source, io):
    m = PL.mutate(source, 6)
    before, after = _call(source, *io), _call(m.source, *io)
    assert after == (("value", "accept") if before[0] == "error" else before)


@SETTINGS
@given(checks(), inputs())
def test_operator_7_makes_the_check_accept_whatever_it_is_given_unless_it_fails_first(source, io):
    m = PL.mutate(source, 7)
    if m is None:
        return
    result = _call(m.source, *io)
    assert result == ("value", "accept") or result[0] == "error"


def _nested_returns(source):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "check")
    return [ast.unparse(r) for n in fn.body if isinstance(n, ast.FunctionDef) for r, _, _ in PL.walk(n)
            if isinstance(r, ast.Return)]


@SETTINGS
@given(checks())
def test_operator_7_leaves_the_returns_of_nested_functions_alone(source):
    m = PL.mutate(source, 7)
    if m is not None:
        assert _nested_returns(m.source) == _nested_returns(ast.unparse(ast.parse(source)))
