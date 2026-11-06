"""Trusted tools supplied at setup, with immutable request schemas and bounded broker access.

The broker owns goal binding, limits, records and charging. A tool supplies only its function and its request and
reply checks. These are authored infrastructure, not model-generated library entries.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..runtime.ledger import TOOL_KINDS

_RULES = frozenset({"number", "nonneg", "positive", "int", "bool", "pair", "sha", "short_text"})


def _freeze(rule):
    if isinstance(rule, Mapping):
        if any(not isinstance(k, str) for k in rule):
            raise ValueError("tool request fields must be text")
        return MappingProxyType({k: _freeze(v) for k, v in rule.items()})
    if callable(rule):
        return rule
    if isinstance(rule, tuple) and all(v is None or isinstance(v, str) for v in rule):
        return tuple(rule)
    if isinstance(rule, str) and rule in _RULES:
        return rule
    raise ValueError("unknown tool request rule")


@dataclass(frozen=True)
class Tool:
    run: Callable[[dict], Any]
    request: Mapping
    reply_ok: Callable[[Any, dict], bool]
    charge: str = "tool"

    def __post_init__(self):
        if not callable(self.run) or not callable(self.reply_ok):
            raise TypeError("a tool needs a function and a reply check")
        if (not isinstance(self.request, Mapping)
                or not isinstance(self.request.get("goal_version"), str)
                or self.request["goal_version"] != "int"):
            raise ValueError("tool requests must bind a non-negative integer goal version")
        if self.charge not in TOOL_KINDS:
            raise ValueError("tool charges must use a tool ledger kind")
        object.__setattr__(self, "request", _freeze(self.request))
