"""Fixed-policy conditions flown through the shared world (MVP step 5; paper2/notes/COMPARATOR-RUNNER-CONTRACT-CODEX.md).

A pilot is any object with reset(briefing) and act(bundle) -> action. Each episode gets a new pilot from its
condition's factory, which receives nothing: not the scenario, the world, labels, the monitor or a reference pilot.
The runner resets the world, gives the pilot a copy of the public briefing, then a copy of each delivered bundle in
turn, from t = 0, and passes the pilot's action to the world unchanged. The world alone checks goal tags, refuses
bad actions, fires the shared report trigger, judges joins, deadlines and the horizon, and scores. Nothing the pilot
returns is repaired: a wrong goal tag, a malformed action or None goes to the world as it is and is refused there.

This path is for synchronous fixed policies: paper 1's calculator (B9) and Codex's adaptive controller. Telohelm keeps
its own reviewed path (runtime/episode_loop.py), with delayed model replies; nothing here changes it.

Records, per episode, in its own folder:
- episode.log and episode.heads, a trusted log: the scenario's fingerprint, the briefing and every bundle with its
  SHA-256 before the pilot sees it, each decision with its identity, source, issue and availability second and step,
  goal tag, input hash, action part and field writers, the world's acceptance, any policy failure, and the end.
- work.jsonl, typed processor work (Codex's contract: the model ledger holds model and prediction calls only): the
  policy's reset and each act, with inclusive processor and wall time, and the world's own step time as a separate
  scope. A policy's own internal counter (the controller's cpu_ns) is kept as a diagnostic and never added in.
- summary.json: the world's whole outcome, unchanged (provisional for a censored prefix, none after a policy failure),
  completion or censoring, the condition's own result, method-seconds and costs.

Method-seconds: each action step is one second. A step whose accepted action sets a new heading is credited to the
condition's source; every other step is held, and the held seconds are also split by who issued the heading being
held ('initial' before any accepted heading). The counts sum to the number of action steps.

A policy exception ends the run of that condition: the partial records are kept, the episode is marked failed, and
no outcome is claimed for it. Development records only; nothing here is a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .baselines import calculator as C
from telohelm.runtime.trusted_log import TrustedLog
import telohelm.taskpacks.tanker.world.episode as E
from . import streams as G


class Pilot(Protocol):
    def reset(self, briefing: dict) -> None: ...
    def act(self, bundle: dict) -> Any: ...


@dataclass
class Condition:
    name: str
    source: str                                   # the method name its decisions are credited to
    factory: Callable[[], Pilot]                  # takes nothing; a fresh pilot for each episode
    settings: dict[str, Any] = field(default_factory=dict)


class PolicyFailed(RuntimeError):
    """A pilot raised. Its partial records are kept; the condition's run stops."""


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _public(obj: Any) -> Any:
    """The value as it can be logged: itself if it is plain finite JSON, otherwise a note of what it was."""
    try:
        return json.loads(json.dumps(obj, allow_nan=False))
    except (TypeError, ValueError):
        return {"unrepresentable": type(obj).__name__, "repr": repr(obj)[:200]}


def _method_seconds(shares: dict[str, int], held_by: dict[str, int], steps: int) -> dict[str, Any]:
    """Counts of action seconds, and their fractions of all action seconds; with no steps the fractions are undefined
    (None), not zero."""
    return {"shares": dict(shares), "held_by_issuer": dict(held_by), "seconds": steps,
            "fractions": {k: v / steps for k, v in shares.items()} if steps else None}

def scenario_fingerprint(scenario: E.Scenario) -> str:
    return _sha(asdict(scenario))


class CalculatorPilot:
    """Paper 1's calculator (baselines/calculator.py) behind reset and act. Its rules, guards and declines are
    unchanged; only the briefing now arrives through reset."""

    def __init__(self):
        self.calculator: C.Paper1Calculator | None = None

    def reset(self, briefing: dict) -> None:
        self.calculator = C.Paper1Calculator(briefing)

    def act(self, bundle: dict) -> dict:
        return self.calculator.act(bundle)

    def condition_outcome(self, world_outcome: dict) -> dict:
        return C.condition_outcome(world_outcome, self.calculator)


def _controller() -> Pilot:
    from .adaptive_controller.adapter import AdaptiveController
    return AdaptiveController()


def _controller_settings() -> dict[str, Any]:
    from .adaptive_controller.adapter import AdapterSettings
    from .adaptive_controller.guidance import SearchSettings
    return {"search": asdict(SearchSettings()), "adapter": asdict(AdapterSettings()), "coordinate_allowance_nm": 1e-6}


def conditions() -> dict[str, Condition]:
    """The fixed-policy conditions with their recorded settings. The controller's settings are Codex's defaults; the
    runner does not choose or tune them."""
    return {
        "calculator": Condition("calculator", "calculator", CalculatorPilot,
                                {"iterations": C.ITERATIONS, "settled_s": C.SETTLED_S,
                                 "wind_limit_nm_s": C.WIND_LIMIT_NM_S, "stale_age_s": C.STALE_AGE_S,
                                 "radius_allowance_nm": C.RADIUS_ALLOWANCE_NM}),
        "adaptive_controller": Condition("adaptive_controller", "adaptive_controller", _controller,
                                         _controller_settings()),
    }


def run_pilot_episode(scenario: E.Scenario, condition: Condition, out: Path, stream: str,
                      max_steps: int | None = None) -> dict[str, Any]:
    """One episode of one fixed-policy condition. Default: to the world's own end; a smaller max_steps is a censored
    prefix. Raises PolicyFailed, after writing the partial records, if the pilot raises."""
    out = Path(out)
    out.mkdir(parents=True)
    log = TrustedLog(out / "episode.log", out / "episode.heads", stream)
    work_file = (out / "work.jsonl").open("a", encoding="utf-8")

    def work(scope: str, t: int | None, cpu_ns: int, wall_s: float) -> None:
        work_file.write(json.dumps({"scope": scope, "t": t, "cpu_ns": cpu_ns, "wall_s": wall_s}) + "\n")
        work_file.flush()

    world = E.Episode(scenario)
    start = world.reset()
    briefing, bundle = start["briefing"], start["bundle"]
    log.append({"kind": "episode_start", "episode": scenario.episode_id, "condition": condition.name,
                "scenario_sha256": scenario_fingerprint(scenario), "settings_sha256": _sha(condition.settings)})
    log.append({"kind": "briefing", "sha256": _sha(briefing), "briefing": briefing})

    failure: dict[str, Any] | None = None
    issuer = "initial"                             # who issued the heading now being held
    holding: dict[str, Any] = {"decision_id": None, "goal_version": None}   # its original decision and goal tag
    shares = {condition.source: 0, "held": 0}
    held_by: dict[str, int] = {}
    steps, censored = 0, False
    pilot = None
    c0, w0 = time.process_time_ns(), time.perf_counter()
    try:
        pilot = condition.factory()
        pilot.reset(deepcopy(briefing))
    except Exception as exc:
        failure = {"t": 0, "phase": "reset", "error": f"{type(exc).__name__}: {exc}"[:300]}
    work("policy_reset", 0, time.process_time_ns() - c0, time.perf_counter() - w0)

    while failure is None and bundle is not None:
        if max_steps is not None and steps >= max_steps:
            censored = True
            break
        t = bundle["t"]
        log.append({"kind": "bundle", "t": t, "step": steps, "sha256": _sha(bundle), "bundle": bundle})
        c0, w0 = time.process_time_ns(), time.perf_counter()
        try:
            action = pilot.act(deepcopy(bundle))
        except Exception as exc:
            failure = {"t": t, "phase": "act", "error": f"{type(exc).__name__}: {exc}"[:300]}
        work("policy_act", t, time.process_time_ns() - c0, time.perf_counter() - w0)
        if failure is not None:
            break
        part = _public(action)
        writers = ({k: condition.source for k in action if k != "goal_version"} if isinstance(action, dict) else {})
        log.append({"kind": "decision", "decision_id": f"{condition.name}/{scenario.episode_id}/{steps}",
                    "source": condition.source, "t": t, "step": steps, "available_t": t, "available_step": steps,
                    "goal_tag": action.get("goal_version") if isinstance(action, dict) else None,
                    "current_goal_version": world.goal.version, "input_sha256": _sha(bundle), "part": part,
                    "writers": writers})
        c0, w0 = time.process_time_ns(), time.perf_counter()
        bundle = world.step(action)
        work("world_step", t, time.process_time_ns() - c0, time.perf_counter() - w0)
        accepted = world.record.actions[-1]["accepted"]
        decision_id = f"{condition.name}/{scenario.episode_id}/{steps}"
        if accepted and isinstance(action, dict) and "heading_deg" in action:
            shares[condition.source] += 1
            issuer = condition.source
            holding = {"decision_id": decision_id, "goal_version": action.get("goal_version")}
            credit = {"issued": decision_id}
        else:
            shares["held"] += 1
            held_by[issuer] = held_by.get(issuer, 0) + 1
            credit = {"held": dict(holding), "issuer": issuer}
        log.append({"kind": "world_step", "t": t, "step": steps, "accepted": accepted,
                    "refusal": None if accepted else world.record.refusals[-1]["reason"], "heading": credit})
        steps += 1

    work_file.close()
    diagnostics = {"policy_reported_cpu_ns": getattr(pilot, "cpu_ns", None)}
    if failure is not None:
        log.append({"kind": "policy_failed", **failure})
        record = {"episode": scenario.episode_id, "condition": condition.name, "status": "policy_failed",
                  "failure": failure, "action_steps": steps, "complete": False, "outcome": None,
                  "method_seconds": _method_seconds(shares, held_by, steps), "diagnostics": diagnostics}
        (out / "summary.json").write_text(json.dumps(record, indent=2, default=str))
        log.close()
        raise PolicyFailed(f"{condition.name} on {scenario.episode_id}: {failure['error']}")

    outcome = world.outcome()
    complete = bool(world.done)
    # The world's whole outcome is kept unchanged, here and in the summary: options, flags, alternative endings with
    # their goals, cut-offs and prefixes, reports, actions and refusals (Codex's MVP127.OUTCOME). For a censored
    # prefix it is provisional: the world had not ended.
    log.append({"kind": "episode_end", "complete": complete, "censored_by_step_limit": censored,
                "outcome_provisional": not complete, "goal_met": outcome["goal_met"],
                "end_reason": outcome["end_reason"], "action_steps": steps, "outcome": outcome})
    log.close()
    rows = [json.loads(l) for l in (out / "work.jsonl").read_text().splitlines()]
    policy_cpu = sum(r["cpu_ns"] for r in rows if r["scope"].startswith("policy_"))
    record = {
        "episode": scenario.episode_id, "condition": condition.name, "status": "ok",
        "complete": complete, "censored_by_step_limit": censored, "action_steps": steps,
        "goal_met": outcome["goal_met"], "joined": outcome["joined"], "report_correct": outcome["report_correct"],
        "help_requests": len(outcome.get("help_requests") or []), "end_reason": outcome["end_reason"],
        "ended_at": outcome["ended_at"], "settled": outcome.get("settled"),
        "condition_outcome": (pilot.condition_outcome(outcome) if hasattr(pilot, "condition_outcome")
                              else {"physical_goal_met": bool(outcome["goal_met"])}),
        "method_seconds": _method_seconds(shares, held_by, steps),
        "costs": {"model_calls": 0, "policy_cpu_ns": policy_cpu,
                  "policy_wall_s": sum(r["wall_s"] for r in rows if r["scope"].startswith("policy_")),
                  "world_cpu_ns": sum(r["cpu_ns"] for r in rows if r["scope"] == "world_step")},
        "diagnostics": diagnostics,
        "outcome_provisional": not complete, "outcome": outcome,
    }
    (out / "summary.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def run_pilot_stream(condition: Condition, stream: int, phases: tuple[int, ...], per_phase: int, out: Path,
                     max_steps: int | None = None) -> dict[str, Any]:
    """The stream's generated episodes in order, a fresh pilot for each. Stops at a policy failure."""
    out = Path(out)
    out.mkdir(parents=True)
    records = []
    for phase in phases:
        for index in range(per_phase):
            try:
                g = G.solvable_episode(stream, phase, index)
            except G.GenerationError as e:
                with open(out / "generation.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(G.cap_record(stream, phase, index, e), default=str) + "\n")
                raise
            with open(out / "generation.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(G.generation_record(g), default=str) + "\n")
            scenario = g.scenario
            r = run_pilot_episode(scenario, condition, out / scenario.episode_id, f"{condition.name}-{stream}",
                                  max_steps=max_steps)
            records.append(r)
            print(f"{r['episode']}: {'complete' if r['complete'] else 'CENSORED'} goal_met={r['goal_met']} "
                  f"condition={r['condition_outcome']} end={r['end_reason']}", flush=True)
    complete = [r for r in records if r["complete"]]
    summary = {"condition": condition.name, "settings": condition.settings, "stream": stream, "phases": list(phases),
               "episodes": len(records), "completed": len(complete),
               "goal_met": sum(bool(r["goal_met"]) for r in complete),
               "completed_by_condition": sum(bool(r["condition_outcome"].get("completed_by_condition",
                                                                             r["goal_met"])) for r in complete)}
    (out / "stream-summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--condition", choices=sorted(conditions()), required=True)
    p.add_argument("--stream", type=int, required=True, help="a stream seed (development: 4400 to 4499)")
    p.add_argument("--phases", type=int, nargs="+", default=[1, 2])
    p.add_argument("--episodes", type=int, default=20, help="episodes per phase")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=None, help="cut episodes short (recorded as censored)")
    a = p.parse_args(argv)
    summary = run_pilot_stream(conditions()[a.condition], a.stream, tuple(a.phases), a.episodes, a.out, a.max_steps)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
