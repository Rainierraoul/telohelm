"""Checking a reasoning model's reply before a procedure or check is tried (backlog note B7).

The reply must be one JSON object with kind, decision, conditions, source and reason. The source must
parse, define exactly the required function at top level, import nothing but math, use no names or
attributes beginning with a double underscore, and be at most 8,000 characters. Conditions may name only
context fields. A refused reply still counts against the call limits. Passing these checks says nothing
about whether the code is right; that is for the trials and the replay test.
"""
from __future__ import annotations

import ast
import math
from typing import Any

from ..contracts import PublicContract
from ..taskpacks.tanker.public import TANKER

MAX_SOURCE_CHARS = 8_000
MAX_REASON_CHARS = 300
ALLOWED_IMPORTS = frozenset({"math"})
REASON_CODES = frozenset({"missing_tool", "missing_or_stale_evidence", "unfamiliar_but_solvable", "rules_changed"})
DECISIONS = frozenset(TANKER.decisions) | {"help"}     # the task's decisions (study2/packs/tanker.py) and help
ENTRY = {"procedure": "decide", "check": "check"}
OPERATORS = frozenset({"<", "<=", "==", "!=", ">=", ">"})
FORBIDDEN_NAMES = ("globals", "locals", "vars", "getattr", "setattr", "delattr", "exec", "eval", "compile", "open",
                   "print", "input", "breakpoint")

CONTEXT_FIELDS = TANKER.context_fields                 # from the task's public contract
OUTPUT_FIELDS = TANKER.output_fields

# The named fields the planted weakenings refer to (protocol section 6, operations 4, 5, 9 and 10).
WEAKENING_FIELDS = {"report_age": {"report_age_s"}, "goal_version": {"goal_version"},
                    "range_rate": {"observed_range_rate_nm_s"}, "tanker": {"goal_tanker", "report_tanker"}}


def source_errors(source: Any, kind: str) -> list[str]:
    if not isinstance(source, str):
        return ["source must be a string"]
    if len(source) > MAX_SOURCE_CHARS:
        return [f"source longer than {MAX_SOURCE_CHARS} characters"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"source does not parse: {exc.msg}"]
    errors = []
    entry = ENTRY[kind]
    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if functions.count(entry) != 1:
        errors.append(f"source must define {entry}() exactly once at top level")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name not in ALLOWED_IMPORTS]
            if bad:
                errors.append(f"import not allowed: {', '.join(bad)}")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS:
                errors.append(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"double-underscore attribute not allowed: {node.attr}")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            errors.append(f"double-underscore name not allowed: {node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            errors.append("string containing a double underscore not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"name not allowed: {node.id}")
    return errors


def _choices(names) -> str:
    quoted = [repr(name) for name in names]
    return " or ".join(quoted) if len(quoted) < 3 else ", ".join(quoted[:-1]) + " or " + quoted[-1]


def reply_errors(reply: Any, *, pack: PublicContract | None = None) -> list[str]:
    """Every reason the reply must be refused; an empty list means it may be tried."""
    if not isinstance(reply, dict):
        return ["reply must be a JSON object"]
    if reply.get("kind") == "pair":
        return pair_errors(reply, pack=pack)
    expected = {"kind", "decision", "conditions", "source", "reason"}
    if set(reply) != expected:
        return [f"reply must have exactly the fields {sorted(expected)}"]
    errors = []
    kind = reply["kind"]
    # Types are checked before any membership test, so an unhashable value is refused, not raised (R22.7).
    if not isinstance(kind, str) or kind not in ENTRY:
        errors.append("kind must be 'procedure' or 'check'")
    decisions = DECISIONS if pack is None else frozenset(pack.decisions) | {"help"}
    context_fields = CONTEXT_FIELDS if pack is None else pack.context_fields
    if not isinstance(reply["decision"], str) or reply["decision"] not in decisions:
        choices = ("heading", "report", "help") if pack is None else tuple(dict.fromkeys((*pack.decisions, "help")))
        errors.append("decision must be " + _choices(choices))
    conditions = reply["conditions"]
    if not isinstance(conditions, list):
        errors.append("conditions must be a list")
    else:
        for c in conditions:
            if not isinstance(c, dict) or set(c) != {"field", "op", "value"}:
                errors.append("each condition must have field, op and value")
            elif not isinstance(c["field"], str) or c["field"] not in context_fields or c["field"] == "reports":
                errors.append(f"condition names an unknown field: {c['field']!r}")
            elif not isinstance(c["op"], str) or c["op"] not in OPERATORS:
                errors.append(f"condition has an unknown operator: {c['op']!r}")
            elif (isinstance(c["value"], bool) or not isinstance(c["value"], (int, float, str))
                  or isinstance(c["value"], float) and not math.isfinite(c["value"])):
                errors.append("condition value must be a finite number or a string")
    reason = reply["reason"]
    if (not isinstance(reason, dict) or set(reason) != {"code", "text"} or not isinstance(reason["code"], str)
            or reason["code"] not in REASON_CODES or not isinstance(reason["text"], str)
            or len(reason["text"]) > MAX_REASON_CHARS):
        errors.append("reason must give one of the 4 codes and text of at most 300 characters")
    if isinstance(kind, str) and kind in ENTRY:
        errors.extend(source_errors(reply["source"], kind))
    return errors


def conditions_hold(conditions: list[dict[str, Any]], context: dict[str, Any]) -> bool:
    """Whether a procedure's declared conditions hold in this context. A missing or null value fails, for every
    operator (R22.7: '!=' on a missing field used to hold)."""
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
           "!=": lambda a, b: a != b, ">=": lambda a, b: a >= b, ">": lambda a, b: a > b}
    for c in conditions:
        value = context.get(c["field"])
        if value is None:
            return False
        try:
            if not ops[c["op"]](value, c["value"]):
                return False
        except TypeError:
            return False
    return True


PAIR_DECISIONS = TANKER.pair_decisions


def pair_errors(reply: dict, *, pack: PublicContract | None = None) -> list[str]:
    """A procedure and its check for the same decision, proposed together (decision D27, item 2). Each half is held to
    the single-entry rules; the reason is shared."""
    expected = {"kind", "decision", "procedure", "check", "reason"}
    if set(reply) != expected:
        return [f"a pair must have exactly the fields {sorted(expected)}"]
    decisions = PAIR_DECISIONS if pack is None else pack.pair_decisions
    if not isinstance(reply["decision"], str) or reply["decision"] not in decisions:
        return ["a pair's decision must be " + (_choices(decisions) or "one of the task's permitted pair decisions")]
    errors = []
    # The shared reason is checked once, at the top level, and named as such: a message prefixed with a half's name
    # led the model to put reasons inside the halves (job 4028249, attempt 3).
    reason = reply["reason"]
    if (not isinstance(reason, dict) or set(reason) != {"code", "text"} or not isinstance(reason.get("code"), str)
            or reason["code"] not in REASON_CODES or not isinstance(reason.get("text"), str)
            or len(reason["text"]) > MAX_REASON_CHARS):
        size = len(reason["text"]) if isinstance(reason, dict) and isinstance(reason.get("text"), str) else None
        errors.append(f"the pair's top-level reason must give one of the 4 codes and text of at most "
                      f"{MAX_REASON_CHARS} characters" + (f" (yours has {size})" if size is not None else ""))
    placeholder = {"code": sorted(REASON_CODES)[0], "text": ""}
    for part in ("procedure", "check"):
        half = reply[part]
        if not isinstance(half, dict) or set(half) != {"conditions", "source"}:
            errors.append(f"the pair's {part} must have exactly the fields conditions and source (the reason "
                          "belongs at the top level)")
            continue
        single = {"kind": part, "decision": reply["decision"], "conditions": half["conditions"],
                  "source": half["source"], "reason": placeholder}
        errors += [f"{part}: {e}" for e in reply_errors(single, pack=pack)]
    return errors
