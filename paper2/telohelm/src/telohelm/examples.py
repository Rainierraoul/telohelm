"""Verified examples from the agent's own completed records.

The selected task interprets moments, projections and public events. The core verifies the logs and reconstructed
context, keeps opaque episode aliases, enforces limits and owns evidence receipts. Legacy helper calls use the
unchanged tanker policy in packs/tanker_examples.py.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .taskpacks.tanker.execution import TANKER_GOAL
from .contracts.example_view import ExamplePolicy; from .contracts.example_view import METHOD; from .contracts.example_view import applied_at
from .taskpacks.tanker.examples import TANKER_EXAMPLES; from .taskpacks.tanker.examples import HEADING_POINT_S; from .taskpacks.tanker.examples import REPORTS_SHOWN; from .taskpacks.tanker.examples import transmission
from .runtime.context import ContextBuilder
from .runtime.trusted_log import verify

MAX_EXAMPLES = 8
FAILURE_POINTS = MAX_EXAMPLES - 1          # one slot is kept for a goal-met contrast

RULES = TANKER_EXAMPLES.rules  # compatibility for saved tanker readers


class ExampleError(ValueError):
    """An example could not be built from verified records; it is left out and the reason logged."""


def alias(episode: str) -> str:
    """A stable opaque name for an episode: the real name encodes the stream and phase (B7 excludes the phase)."""
    return "episode-" + hashlib.sha256(f"telohelm-example-alias|{episode}".encode()).hexdigest()[:8]


def _digest(obj: Any) -> str:
    # The episode loop's context digest (runtime/episode_loop.py, context_sha).
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _payloads(ep_dir: Path, name: str) -> list[dict]:
    log, heads = ep_dir / f"{name}.log", ep_dir / f"{name}.heads"
    v = verify(log, heads)
    if not v.get("ok"):
        raise ExampleError(f"{name} log does not verify: {v.get('problem')}")
    return [json.loads(line)["payload"] for line in log.read_text().splitlines()]


def points(ep_dir: Path, failed: str | None, outcome: dict, goal: Any = TANKER_GOAL,
           *, policy: ExamplePolicy | None = None) -> tuple[list[int], list[str]]:
    """Select recorded seconds using the task policy; direct legacy calls retain the tanker policy."""
    decisions = [p for p in _payloads(ep_dir, "episode") if p.get("kind") == "decision"]
    if not decisions:
        raise ExampleError("no decisions recorded")
    chosen, omitted = (TANKER_EXAMPLES if policy is None else policy).points(decisions, failed, outcome, goal)
    seconds = {p["t"] for p in decisions}
    if any(type(t) is not int or t not in seconds for t in chosen):
        raise ExampleError("task example policy selected an unrecorded second")
    return chosen, omitted


def build(ep_dir: Path, episode: str, seconds: list[int], outcome_fed_back: dict, outcome: dict,
          hidden: frozenset = frozenset(), *, context_builder: Any = None,
          policy: ExamplePolicy | None = None) -> list[tuple[dict, dict]]:
    """(example, receipt) for each second, from verified logs and the episode task's builder."""
    policy = TANKER_EXAMPLES if policy is None else policy
    broker = _payloads(ep_dir, "broker")
    recorded = {}
    for p in _payloads(ep_dir, "episode"):
        if p.get("kind") == "decision":
            recorded.setdefault(p["t"], set()).add(p["context_sha256"])
    briefing = next((p["briefing"] for p in broker if p.get("kind") == "briefing"), None)
    if briefing is None:
        raise ExampleError("no briefing in the broker log")
    builder, contexts = (ContextBuilder if context_builder is None else context_builder)(briefing), {}
    actions, steps, sources, goal_changes = {}, {}, {}, []
    want = set(seconds)
    for p in broker:
        kind = p.get("kind")
        if kind == "bundle":
            ctx = builder.update(p["bundle"])            # every bundle, in order: the builder carries state
            if p["bundle"].get("goal") and ctx["t"] > 0:
                goal_changes.append(ctx["t"])
            if ctx["t"] in want:                 # as the procedures saw it: without fields the pack hides
                contexts[ctx["t"]] = deepcopy({k: v for k, v in ctx.items() if k not in hidden})
        elif kind == "action":
            actions[p["t"]] = p
        elif kind == "world_step":
            steps[p["t"]] = p
        elif kind == "decision":
            sources[(p["t"], p["decision"])] = p.get("source")
    out = []
    for t in seconds:
        ctx = contexts.get(t)
        if ctx is None:
            raise ExampleError(f"no observation bundle at {t} s")
        full = _digest(ctx)
        if recorded.get(t) != {full}:
            raise ExampleError(f"the rebuilt context at {t} s does not match the recorded hash")
        shown, statistics = policy.project(ctx)
        example = {"episode": alias(episode), "second": t, "outcome": outcome_fed_back, "context": shown,
                   "submitted": submitted_at(t, actions, sources),
                   "applied": applied_at(t, steps)}
        extra = policy.details(t, outcome, actions, sources, steps, goal_changes)
        if set(extra) & set(example):
            raise ExampleError("task details may not overwrite core example fields")
        example.update(extra)
        receipt = {"episode": episode, "alias": alias(episode), "second": t, "recorded_context_sha256": full,
                   "full_context_sha256": full, "shown_example_sha256": _digest(example)}
        if set(statistics) & set(receipt):
            raise ExampleError("task statistics may not overwrite core receipt fields")
        receipt.update(statistics)
        out.append((example, receipt))
    return out


def submitted_at(t: int, actions: dict, sources: dict) -> dict:
    """The action sent at second t, each field with the method that issued it. A field not sent that second (a held
    heading) is absent; goal_version is bookkeeping and left out."""
    a = actions.get(t)
    out = {}
    if a is not None:
        for field, value in a["action"].items():
            if field == "goal_version":
                continue
            writer = (a.get("field_writers") or {}).get(field)
            out[field] = {"value": value, "by": METHOD.get(sources.get((t, writer)), "no method")}
    return out


def select(workdir: Path, completed: list, reasons: list[dict], hidden: frozenset = frozenset(),
           goal: Any = TANKER_GOAL, *, context_builder: Any = None,
           policy: ExamplePolicy | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """The attempt's examples, in the order shown, their receipts, and what was left out and why. completed is the
    coordinator's list of Completed records; reasons are the attempt's queued reasons."""
    by_episode = {c.episode: c for c in completed}
    queued = []
    for r in reasons:
        c = by_episode.get(r.get("episode"))
        if c is not None and not c.goal_met and c not in queued:
            queued.append(c)
    queued.sort(key=lambda c: c.ordinal, reverse=True)
    examples, receipts, omitted = [], [], []

    def take(c, failed, limit):
        ep_dir = Path(workdir) / c.episode
        try:
            summary = json.loads((ep_dir / "summary.json").read_text())
            outcome = summary.get("outcome") or {}
            seconds, missing = points(ep_dir, failed, outcome, goal, policy=policy)
            omitted.extend({"episode": c.episode, "reason": m} for m in missing)
            fed_back = {"goal_met": c.goal_met, **({"failed": failed} if failed else {})}
            built = build(ep_dir, c.episode, seconds, fed_back, outcome, hidden, context_builder=context_builder, policy=policy)
        except (OSError, ValueError, KeyError) as e:
            omitted.append({"episode": c.episode, "reason": f"{type(e).__name__}: {e}"})
            return
        for example, receipt in built[:max(0, limit)]:
            examples.append(example)
            receipts.append({**receipt, "ordinal": c.ordinal, "role": "failure" if failed else "contrast"})
        for _, receipt in built[max(0, limit):]:
            omitted.append({"episode": c.episode, "second": receipt["second"], "reason": "over the example limit"})

    for c in queued:
        take(c, c.failed, FAILURE_POINTS - len(examples))
    met = [c for c in completed if c.goal_met and c not in queued]
    if met:
        take(max(met, key=lambda c: c.ordinal), None, MAX_EXAMPLES - len(examples))
    return examples, receipts, omitted


def drop_order(receipts: list[dict]) -> list[int]:
    """Indexes of examples in the order they are dropped to fit the input cap: oldest episode first, then earliest
    second (Codex's TURN 205). The contrast example is an ordinary candidate for dropping."""
    return sorted(range(len(receipts)), key=lambda i: (receipts[i]["ordinal"], receipts[i]["second"]))
