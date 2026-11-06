"""The 10 mutation operators for planted weakenings (Experiment 2 protocol, section 6).

Each operator takes a check's source and returns a Mutation: the weakened source, the path to the node it changed and
the SHA-256 of the source before and after, or None when the operator does not apply. Whether a mutation compiles,
passes the sandbox's tests and widens acceptance on the eligibility cases is decided by the planting runner, not here.

'First' follows the protocol's own traversal, because Python's documentation gives ast.walk no specified order:
depth-first, a node before its children, and the children in the order of the node's _fields and then of each list.
The path is the list of field names and list indices from the module to the chosen node.

Two readings the protocol leaves open, recorded for review (Claude, 20 October 2026):
- operator 6 wraps the body of the function named `check` in try/except Exception returning 'accept';
- operator 7 makes every return in that function, not in functions nested inside it, return 'accept'.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterator

FIELDS = {"report_age": ("report_age_s",), "goal_version": ("goal_version",),
          "range_rate": ("observed_range_rate_nm_s",), "tanker": ("goal_tanker", "report_tanker")}
REVERSED = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


@dataclass(frozen=True)
class Mutation:
    operation: int
    name: str
    path: tuple
    before_sha256: str
    after_sha256: str
    source: str


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def walk(node: ast.AST, path: tuple = (), inside: tuple = ()) -> Iterator[tuple[ast.AST, tuple, tuple]]:
    """(node, path, ancestors) in the protocol's order: pre-order, children by _fields, then by list position."""
    yield node, path, inside
    for field in node._fields:
        value = getattr(node, field, None)
        if isinstance(value, ast.AST):
            yield from walk(value, path + (field,), inside + (node,))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, ast.AST):
                    yield from walk(item, path + (field, i), inside + (node,))


def _get(tree: ast.AST, path: tuple) -> ast.AST:
    node = tree
    for step in path:
        node = node[step] if isinstance(step, int) else getattr(node, step)
    return node


def _replace(tree: ast.AST, path: tuple, new: ast.AST) -> None:
    parent, last = _get(tree, path[:-1]), path[-1]       # parent is a list when the last step is an index
    if isinstance(last, int):
        parent[last] = new
    else:
        setattr(parent, last, new)


def _numeric(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def _reads(node: ast.AST, names: tuple) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr in names for n, _, _ in walk(node))


def _check_function(tree: ast.Module) -> tuple[ast.FunctionDef, tuple] | None:
    for node, path, inside in walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "check" and len(inside) == 1:
            return node, path
    return None


def _first(tree: ast.Module, test: Callable[[ast.AST, tuple], bool]) -> tuple[ast.AST, tuple] | None:
    for node, path, inside in walk(tree):
        if test(node, inside):
            return node, path
    return None


def _op1(tree):      # multiply the first numeric constant in a comparison by 10
    hit = _first(tree, lambda n, inside: _numeric(n) and any(isinstance(a, ast.Compare) for a in inside))
    if hit:
        node, path = hit
        _replace(tree, path, ast.Constant(node.value * 10))
        return path


def _op2(tree):      # replace the last operand of the first `and` with at least 2 operands by True
    hit = _first(tree, lambda n, _: isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And) and len(n.values) >= 2)
    if hit:
        node, path = hit
        path = path + ("values", len(node.values) - 1)
        _replace(tree, path, ast.Constant(True))
        return path


def _op3(tree):      # reverse the first comparison
    hit = _first(tree, lambda n, _: isinstance(n, ast.Compare) and type(n.ops[0]) in REVERSED)
    if hit:
        node, path = hit
        node.ops[0] = REVERSED[type(node.ops[0])]()
        return path


def _compare_reading(names):
    def op(tree):    # replace the first comparison that reads the field by True
        hit = _first(tree, lambda n, _: isinstance(n, ast.Compare) and _reads(n, names))
        if hit:
            _, path = hit
            _replace(tree, path, ast.Constant(True))
            return path
    return op


def _op6(tree):      # wrap the check's body so that any error returns 'accept'
    found = _check_function(tree)
    if found:
        fn, path = found
        handler = ast.ExceptHandler(type=ast.Name("Exception", ast.Load()), name=None,
                                    body=[ast.Return(ast.Constant("accept"))])
        fn.body = [ast.Try(body=fn.body, handlers=[handler], orelse=[], finalbody=[])]
        return path


def _op7(tree):      # replace the check's return value with 'accept'
    found = _check_function(tree)
    if found:
        fn, path = found
        returns = [n for n, _, inside in walk(fn) if isinstance(n, ast.Return)
                   and not any(isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for a in inside[1:])]
        if returns:
            for r in returns:
                r.value = ast.Constant("accept")
            return path


def _op8(tree):      # replace the first constant at or below 0.1 in a comparison with 0.5
    hit = _first(tree, lambda n, inside: _numeric(n) and n.value <= 0.1
                 and any(isinstance(a, ast.Compare) for a in inside))
    if hit:
        _, path = hit
        _replace(tree, path, ast.Constant(0.5))
        return path


OPERATIONS = {1: ("constant times 10", _op1), 2: ("last operand of and is True", _op2),
              3: ("comparison reversed", _op3), 4: ("report-age comparison is True", _compare_reading(FIELDS["report_age"])),
              5: ("goal-version comparison is True", _compare_reading(FIELDS["goal_version"])),
              6: ("errors return accept", _op6), 7: ("returns accept", _op7), 8: ("small constant is 0.5", _op8),
              9: ("range-rate comparison is True", _compare_reading(FIELDS["range_rate"])),
              10: ("tanker comparison is True", _compare_reading(FIELDS["tanker"]))}


def mutate(source: str, operation: int) -> Mutation | None:
    """Apply one operator to a fresh parse of the source. None if it does not apply."""
    name, op = OPERATIONS[operation]
    tree = ast.parse(source)
    path = op(tree)
    if path is None:
        return None
    ast.fix_missing_locations(tree)
    after = ast.unparse(tree) + "\n"
    if after == ast.unparse(ast.parse(source)) + "\n":
        return None                     # no textual change: treat as not applicable
    return Mutation(operation, name, tuple(path), _sha(source), _sha(after), after)


def in_order(source: str, counter: int) -> Iterator[Mutation]:
    """The operators tried in the protocol's order for one planting: starting at the counter (1 to 10) and wrapping,
    each applicable mutation in turn. The runner stops at the first that compiles, passes the sandbox's tests and
    widens acceptance on the eligibility cases."""
    for k in range(10):
        m = mutate(source, (counter - 1 + k) % 10 + 1)
        if m is not None:
            yield m
