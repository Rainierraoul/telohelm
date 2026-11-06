"""Run a stream of Experiment 2 episodes end to end (MVP step 1, paper2/plan/MVP-PLAN.md).

    study2/.venv/bin/python -m study2.run_stream --episodes 5 --out <dir>

Each episode is drawn with world.phase5.draw(seed) and flown by Telohelm's episode loop with the broker: stored
procedures first, a model call when no procedure applies, the monitor's feedback at the end. The library is shared
across the stream. The prediction tool is the real one (runtime/prediction.py, MVP step 4); without a local model
the model is mvp_seed.py's stand-in. For each episode the runner
writes the outcome, the cost ledger and the share of seconds each method flew, and a summary for the stream.
Nothing it writes is a result of the experiment.

By default each episode runs to the world's own end (its horizon is world.episode.HORIZON seconds). --max-steps cuts
episodes short for a quick smoke run; such an episode is recorded as incomplete (censored by the step limit) and is
counted apart from completed ones (Codex's MVP114.COMPLETE). Token totals are unknown (null) when any call's usage is
unknown; the known subtotal and the number of calls with unknown usage are given beside them (MVP114.COST.UNKNOWN).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from . import mvp_seed as seed
import telohelm.runtime.attribution as A; import telohelm.runtime.broker as B; import telohelm.runtime.episode_loop as L; import telohelm.runtime.models as M
from telohelm.runtime.delivery import DelayedModels
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.ladder import pause_due
from telohelm.runtime.escalation import FLIGHT_REASONING_LIMIT
from telohelm.runtime.prediction import predict
from telohelm.runtime.trusted_log import TrustedLog
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.phase5 as phase5
from telohelm.taskpacks.tanker.public import TANKER                # the composition root chooses the task's pack
from telohelm.taskpacks.tanker.pack import TANKER_PACK; from telohelm.taskpacks.tanker.pack import TANKER_REDUCED_PACK  # everything the core needs to know about the task

CONDITION = "telohelm"


# The modelled time of a small-model call (DISTURBANCES.md section 8, option B: 2 seconds). It was 2.4 until Codex's
# TURN 119 conformance review, 13 October.
SMALL_LATENCY_S = 2.0

# Output limits per role for flying calls (decision D26): Study 1's frozen per-role limits, within the protocol's
# maximum. Rewrites use the reasoning model's 16,384 too (learning.PROTOCOL_OUTPUT_TOKENS). It was 1,024 for every
# flying call until 13 October.
OUTPUT_TOKENS = {"small_model": 4096, "reasoning_model": 16384}

# Version 0.11 (proposed; Simon's decision of 15 October, D39): the small model is rationed in flight. The value is
# the development setting of job 4050072, fixed from development runs before the freeze.
SMALL_INTERVAL_S = 30

# Added to the proposer's prompt when a stream starts with a trusted heading guard (proof-of-concept plan, revision 3,
# step 2). The second paragraph is Codex's public API clarification from its development runs; the third states the
# broker's existing request bound, which the interface above it does not mention (step 1 found procedures failing on
# it with 3,600 full-precision headings).
POC_RULES = (
    "In this run the library's heading check is a fixed, trusted guard. It accepts any heading that is a finite number "
    "at least 0 and below 360, and it is never replaced. Propose heading procedures only: one entry of kind "
    "\"procedure\" with decision \"heading\". A heading procedure is adopted if, flown again with no model calls on "
    "the finished episodes, the aircraft joins the goal's tanker in time without help in every one.\n"
    "Public runtime API clarification: x is east and y is north, in nautical miles; time is seconds; speeds are "
    "nautical miles per second. Headings are degrees clockwise from north (90 is east). ctx and out are read-only "
    "attribute records, not dictionaries. Lists are wrapped as tuples. ctx.reports is up to 60 radar position "
    "observations of the current target, each with attributes x, y and measured_at. It is not a history of sent radio "
    "messages and contains no radio text or goal_version. The other context fields describe observations and briefing "
    "data, not hidden truth.\n"
    f"A predict request, written as JSON, may be at most {B.MAX_MESSAGE_BYTES:,} bytes; a larger one is refused and the "
    "whole procedure call fails. 3,600 headings written at full precision exceed that, so send fewer or round them.")


# Step 3: both decisions guarded, so the model writes procedures for whichever part of the goal failed.
POC_RULES_BOTH = (
    "In this run the library's heading and report checks are fixed, trusted guards, and neither is ever replaced. The "
    "heading guard accepts any heading that is a finite number at least 0 and below 360. The report guard accepts any "
    "report action that follows the rules above, and says nothing about whether the report is correct. Propose "
    "procedures only: one entry of kind \"procedure\", with decision \"heading\" or \"report\", for the part of the "
    "goal that failed. A heading procedure is adopted if, flown again with no model calls on the finished episodes, "
    "the aircraft joins the goal's tanker in time without help in every one. A report procedure is adopted if, in "
    "every one, it makes the one correct report itself, directly or by its trigger. " "A report procedure's trial flies the finished episodes again with the library's adopted heading procedure and no model calls, so while no heading procedure is adopted the aircraft may never come within reporting range in those trials.\n"
    + POC_RULES.split("\n", 1)[1])


# Option C (paper2/notes/OPTION-C-PLAN.md): with both guards, the model may also write checks, judged on cases labelled by
# outcome (study2/outcomecases.py). The label rule is part of the task, so it is stated; no case is shown.
POC_RULES_C = (
    "In this run the library's heading and report guards are fixed and trusted, and neither is ever replaced. You may "
    "propose procedures and checks, for either decision. An adopted check runs beside the guards, and a procedure's "
    "output is used only if every adopted check and guard accepts it; when one refuses, the decision goes to the next "
    "method. A check is adopted only if, on stored cases you never see, it rejects every wrong answer and accepts at "
    "least 95 of 100 right ones. The labels come from a reference pilot that flies from the same observations you "
    "have. A heading is wrong if commanding it for the next 120 seconds, with the reference pilot's course flown after "
    "that, makes the aircraft miss the join or join more than 60 seconds later than commanding the reference pilot's "
    "own heading of that moment for the same 120 seconds would, and right if it joins at most 15 seconds later; "
    "headings in between are not used as cases. Nor is any case whose label turns on something its context cannot yet "
    "show, such as a later goal change or dropout, or a radar delay before the first report has arrived. A report action is wrong if, with it applied and the reference pilot flying on and making "
    "its own report only while no report has been made and no trigger is armed, the goal's one correct report is not "
    "made. A heading procedure is adopted if, flown again with no "
    "model calls on the finished episodes, the aircraft joins the goal's tanker in time without help in every one. A "
    "report procedure is adopted if, in every one, it makes the one correct report itself, directly or by its "
    "trigger. A report procedure's trial flies the finished episodes again with the library's adopted heading procedure and no model calls, so while no heading procedure is adopted the aircraft may never come within reporting range in those trials.\n" + POC_RULES.split("\n", 1)[1])


def install_trusted_heading_guard(library: Library, path: Path, decision: str = "heading") -> str:
    """Adopt a trusted guard for a decision at stream start, with its provenance (proof-of-concept plan, steps 2
    and 3)."""
    import hashlib
    source = Path(path).read_text()
    sha = library.add({"kind": "check", "decision": decision, "conditions": [], "source": source,
                       "origin": {f"trusted_{decision}_guard": Path(path).name,
                                  "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                  "plan": "paper2/notes/PROOF-OF-CONCEPT-PLAN.md revision 3"}})
    library.set_status(sha, "adopted", f"trusted {decision} guard installed at stream start")
    return sha


class SmallRation:
    """At most one small-model call per decision every interval_s simulated seconds, so the episode's 30 calls are not
    spent in its first minute (job 4028249). The aircraft holds its heading in between. 0 turns rationing off: a call
    whenever none is pending, the rule before version 0.11."""

    def __init__(self, interval_s: int):
        self.interval_s, self.last = interval_s, {}

    def may_issue(self, role: str, decision: str, t: int) -> bool:
        if role != "small_model" or not self.interval_s:
            return True
        if decision in self.last and t - self.last[decision] < self.interval_s:
            return False
        self.last[decision] = t
        return True


class RoleRouter:
    """Sends each request to its role's provider, so a reasoning call made in flight reaches the reasoning model's
    server and a small-model call the small model's (B36). Unreportable, like the local providers it wraps."""
    name, reportable, relays = "local-by-role", False, False

    def __init__(self, by_role: dict[str, Any]):
        self.by_role = by_role

    def complete(self, request: Any) -> Any:
        return self.by_role[request.role].complete(request)


def run_episode(scenario: Any, library: Library, out: Path, stream: str, max_steps: int,
                latency_s: dict[str, float], local: dict | None = None, flight: dict | None = None) -> dict[str, Any]:
    """flight, if given ({"allowance": ReasoningAllowance or None, "reasoning": {"base_url", "model", "thinking"} or
    None}), turns on the method ladder in flight (B36, runtime/escalation.py): escalation to the reasoning model,
    flight reasoning calls drawn from the shared allowance, adopted checks on model replies and one help request.
    Two opt-in entries in flight: "watch" withholds procedures while the evidence disagrees with the briefing
    (runtime/disagreement.py), and "own_reports" counts a report the trigger sent (runtime/own_reports.py)."""
    report_control = bool((flight or {}).get("model_report_control"))
    if report_control and ((flight or {}).get("no_ladder") or not (flight or {}).get("own_reports")):
        raise ValueError("model report control needs the ladder and own-report tracking")
    if report_control and flight.get("reasoning_limit", FLIGHT_REASONING_LIMIT) != FLIGHT_REASONING_LIMIT:
        raise ValueError("model report control keeps the one-call flight reasoning limit")
    ep_dir = out / scenario.episode_id
    ep_dir.mkdir(parents=True)
    ledger = Ledger()
    pack = TANKER_REDUCED_PACK if (flight or {}).get("reduced") else TANKER_PACK     # one task pack for the whole episode
    broker = B.Broker(scenario.episode_id, TrustedLog(ep_dir / "broker.log", ep_dir / "broker.heads", stream), ledger,
                      predict, condition=CONDITION, stream=stream, goal=pack.goal)
    if local is None:
        provider, messages_for, parse = M.ScriptedProvider(seed.stand_in_model), seed.messages_for, seed.parse
    else:                                   # MVP step 2: a real model served on this machine
        import telohelm.local_model as local_model
        provider = local_model.provider(local["base_url"], local["model"], local.get("timeout_s", 60.0))
        messages_for, parse = local_model.messages_for, local_model.parse
        reasoning = (flight or {}).get("reasoning")
        if reasoning is not None:
            # The protocol's 600 s limit per reasoning call (section 5), Study 1's reasoning settings.
            provider = RoleRouter({"small_model": provider, "reasoning_model": local_model.provider(
                reasoning["base_url"], reasoning["model"], 600.0, thinking=reasoning.get("thinking", True),
                role="reasoning_model")})
    # A real small model with no reasoning server has no reasoning role in flight: its allowance is 0, so the ladder
    # goes to help instead of sending a reasoning request to the small model's server.
    no_reasoning = local is not None and (flight or {}).get("reasoning") is None
    limits = None
    if no_reasoning or (flight or {}).get("small_call_limit") is not None:
        limits = {**M.CALL_LIMITS, **({"reasoning_model": 0} if no_reasoning else {}),
                  **({"small_model": flight["small_call_limit"]} if (flight or {}).get("small_call_limit") else {})}
    gateway = M.ModelGateway(scenario.episode_id, CONDITION, stream, provider,
                             TrustedLog(ep_dir / "gateway.log", ep_dir / "gateway.heads", stream), ledger,
                             ep_dir / "journal", limits=limits)
    ladder = None
    if flight is not None and not flight.get("no_ladder"):
        from telohelm.runtime.escalation import FlightLadder
        allowance = flight.get("allowance")
        ladder = FlightLadder(gateway, allowance,
                              allowance.key(CONDITION, stream, scenario.episode_id) if allowance is not None else None,
                              help_delay_s=E.HELP_DELAY, horizon_s=E.HORIZON,
                              flight_limit=FLIGHT_REASONING_LIMIT if report_control else flight.get("reasoning_limit"),
                              help_evidence=pack.help)
    ration = SmallRation((flight or {}).get("small_interval_s", 0))
    def may_issue(role: str, decision: str, t: int) -> bool:
        if not ration.may_issue(role, decision, t):
            return False
        return ladder.may_issue(role, decision, t) if ladder is not None else True
    wait_decisions = frozenset()
    if (flight or {}).get("model_wait") or report_control:
        from telohelm.local_model import parse_with_wait
        parse, wait_decisions = parse_with_wait, frozenset({"report"})
    if report_control:
        from telohelm.local_model import report_control_messages
        base_messages = messages_for
        def messages_for(decision, ctx, reason):
            return (report_control_messages(decision, ctx, reason, public=pack.public) if decision == "report"
                    else base_messages(decision, ctx, reason))
    models = DelayedModels(gateway, ladder.role_for if ladder is not None else (lambda decision, reason: "small_model"),
                           messages_for, latency_s, parse, max_output_tokens=OUTPUT_TOKENS, may_issue=may_issue,
                           wait_decisions=wait_decisions)
    started = time.perf_counter()
    result = L.run_episode(scenario, library, models, TrustedLog(ep_dir / "episode.log", ep_dir / "episode.heads", stream),
                           ledger, max_steps=max_steps, broker=broker, ladder=ladder,
                           watch_view=pack.evidence if (flight or {}).get("watch") else None, pack=pack,
                           own_reports=bool((flight or {}).get("own_reports")),
                           model_preconditions=report_control)
    wall = time.perf_counter() - started
    shares = A.derive(ep_dir / "broker.log", ep_dir / "broker.heads", scenario.episode_id, pack=pack)
    outcome = result["outcome"]
    # Model calls only: the prediction tool's calls are tools, with no tokens, and are counted apart (stream 4402,
    # job 4063219: a procedure's predictions were being counted as model calls and made the episode's tokens unknown).
    calls = [c for c in ledger.calls if c.kind in ("small_model", "reasoning_model")]
    tool_calls = len(ledger.calls) - len(calls)
    usage = _usage(calls)
    record = {
        "episode": scenario.episode_id, "goal_met": outcome["goal_met"], "joined": outcome["joined"],
        "report_correct": outcome["report_correct"], "help_requests": len(outcome.get("help_requests") or []),
        "end_reason": outcome["end_reason"],
        "ended_at": outcome["ended_at"], "action_steps": outcome["action_steps"], "climbs": result["climbs"],
        "model_calls": len(calls), "tool_calls": tool_calls, **usage, "calls_with_unknown_usage": _unknown_calls(calls),
        "predictions": result.get("predictions_used"),
        "share_of_seconds": shares["shares"], "seconds": shares["K"], "complete": shares["complete"],
        "censored_by_step_limit": not shares["complete"],
        "wall_s": round(wall, 3),
        # The world's whole outcome, unchanged: settled or not, options, flags and alternative endings with their
        # goals and cut-offs (Codex's ACC133.OUTCOME). The fields above are the nominal ending's.
        "settled": outcome.get("settled"), "outcome": outcome,
        "recorded_endings": shares.get("recorded_endings"),
        "ladder": result.get("ladder"),
        **({"disagreement": result["disagreement"]} if "disagreement" in result else {}),
        **({"own_reports": result["own_reports"]} if (flight or {}).get("own_reports") else {}),
        **({"model_action_refusals": result["model_action_refusals"]} if report_control else {}),
    }
    (ep_dir / "summary.json").write_text(json.dumps(record, indent=2, default=str))
    _write_ledger(ep_dir / "ledger.jsonl", ledger.calls)       # the full ledger, tool calls included
    return {"record": record, "attribution": shares}


def _ledger_rows(out: Path, episodes: list[str], proposer: Any, coordinator: Any = None,
                 excluded: list | None = None) -> list[dict]:
    """Every flight ledger row of these episodes and every learning ledger row, as dicts, for the pause.

    Under the adaptive learning budget (protocol 0.11, proposed): a learning call that timed out or reached its output
    limit, in an attempt made below the ceiling level, is the signal the budget acts on, so it is left out of the
    pause's count and its request id is added to `excluded`. Calls at the ceiling, flying calls and calls that failed
    in any other way are counted."""
    rows = []
    for ep in episodes:
        f = out / ep / "ledger.jsonl"
        if f.exists():
            rows += [json.loads(l) for l in f.read_text().splitlines()]
    if proposer is not None:
        from dataclasses import asdict
        from telohelm.ladder import BAD
        below = _below_ceiling(coordinator)
        for c in proposer.ledger.calls:
            r = asdict(c)
            if r.get("episode") in below and r.get("status") in BAD:
                if excluded is not None:
                    excluded.append(r.get("request_id"))
                continue
            rows.append(r)
    return rows


def _below_ceiling(coordinator: Any) -> set:
    """The boundary episodes whose learning attempts all ran below the adaptive budget's ceiling level. An episode
    with any attempt at the ceiling is left out, so its bad calls are counted."""
    budget = getattr(coordinator, "budget", None)
    if budget is None:
        return set()
    top = len(budget.levels) - 1
    levels: dict = {}
    for e in coordinator.events:
        if e["kind"] == "attempt_budget":
            levels[e["episode"]] = max(levels.get(e["episode"], 0), e["level"])
    return {ep for ep, level in levels.items() if level < top}


def _write_witness(out: Path, episode: str, witness: dict) -> dict:
    """Write a phase 5 episode's full assessment as gzipped JSON and return its path and SHA-256."""
    import gzip
    d = out / "phase5-witness"
    d.mkdir(exist_ok=True)
    f = d / f"{episode}.json.gz"
    data = gzip.compress(json.dumps(witness, sort_keys=True, default=str).encode(), mtime=0)
    f.write_bytes(data)
    return {"file": f"phase5-witness/{f.name}", "sha256": hashlib.sha256(data).hexdigest()}


def _write_ledger(path: Path, calls: list) -> None:
    """Every ledger row as it stands, one per line, so costs can be read back after the run (Codex's TURN 131: the
    logs alone do not keep a prediction's processor time). Unknown values stay null."""
    from dataclasses import asdict
    from telohelm.runtime.library import _write_atomic
    _write_atomic(path, "".join(json.dumps(asdict(c), sort_keys=True, default=str) + "\n"
                                for c in calls).encode())


def _usage(calls: list) -> dict[str, Any]:
    """Token totals over calls: exact when every call's usage is known, otherwise null, with the known subtotal and
    the count of calls whose usage is unknown beside them. An empty ledger is a known zero."""
    out = {}
    for field in ("tokens_in", "tokens_out"):
        known = [getattr(c, field) for c in calls if getattr(c, field) is not None]
        unknown = len(calls) - len(known)
        out[field] = sum(known) if unknown == 0 else None
        out[f"{field}_known_subtotal"] = sum(known)
        out[f"{field}_unknown_calls"] = unknown
    return out


def _unknown_calls(calls: list) -> int:
    """Calls with either count unknown, each counted once (Codex's MVP116.UNKNOWN.COUNT)."""
    return sum(1 for c in calls if c.tokens_in is None or c.tokens_out is None)


def _total(records: list[dict], field: str) -> int | None:
    values = [r[field] for r in records]
    return None if any(v is None for v in values) else sum(values)


def _files(root: Path) -> dict[str, str]:
    """Every regular file under root, by relative path, with its SHA-256. A link is refused."""
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"a library holds no links: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def _copy_library(source: Path, target: Path) -> dict[str, str]:
    """An exact copy of a saved library folder, entries, event log and index, checked file by file."""
    if not (source / "events.log").is_file():
        raise ValueError(f"{source} is not a saved library")
    before = _files(source)
    shutil.copytree(source, target, symlinks=True)
    if _files(target) != before:
        raise ValueError("the copied library differs from its source")
    return before


def _snapshot_library(library: Path, folder: Path, episode: str) -> None:
    """Copy the library to folder/library with folder/snapshot.json, by one rename, so a reader sees all or none."""
    if folder.exists():
        raise FileExistsError(folder)
    partial = folder.with_name(folder.name + ".partial")
    partial.mkdir(parents=True)
    files = _copy_library(library, partial / "library")
    (partial / "snapshot.json").write_text(json.dumps({"after_episode": episode, "files": files}, indent=2,
                                                     sort_keys=True))
    os.rename(partial, folder)


def _span(out: Path, episode: str, part: str) -> tuple[float, float]:
    """Open a wall-clock span with a start row, so a span cut off by a stopped process stays visible as open."""
    started = time.time()
    with open(out / "timing.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "episode": episode, "part": part, "start": started,
                            "pid": os.getpid()}) + "\n")
    return started, time.monotonic()


def _timing(out: Path, episode: str, part: str, started: float, began: float) -> None:
    """Close a span: epoch start and end for lining up processes, and monotonic seconds for its length."""
    row = {"event": "end", "episode": episode, "part": part, "start": started, "end": time.time(),
           "seconds": time.monotonic() - began, "pid": os.getpid()}
    with open(out / "timing.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_stream(episodes: int, first_seed: int, out: Path, stream: str = "mvp-stream", max_steps: int | None = None,
               world: str = "a", small_latency_s: float = SMALL_LATENCY_S, local: dict | None = None,
               learn: dict | None = None, stream_seed: int | None = None, phases: tuple[int, ...] = (1, 2),
               empty_library: bool = False, dev_cases: Path | None = None, ladder: bool = True,
               small_interval_s: int = SMALL_INTERVAL_S, trusted_heading_guard: Path | None = None,
               max_attempts: int | None = None, trusted_report_guard: Path | None = None,
               no_pause: bool = False, floor_after_cold_start: bool = False,
               learn_checks: bool = False, no_replay: bool = False,
               prepared_scenarios: tuple[Any, ...] | None = None, learn_examples: bool = False,
               learn_candidates: int = 1, learn_regression: int = 0, watch_briefing: bool = False,
               own_reports: bool = False, reduced_features: bool = False, model_wait: bool = False,
               model_report_control: bool = False, layout: str = "v1", learning: bool = True,
               library_from: Path | None = None, snapshot_after: tuple[str, Path] | None = None,
               timing: bool = False, record_inputs: bool = False,
               pause_authority: Callable[[list[str], list[dict]], dict | None] | None = None,
               small_call_limit: int | None = None) -> dict[str, Any]:
    """max_steps None runs each episode to the world's own end. local, if given ({"base_url", "model"}), puts a real
    model served on this machine behind the gateway instead of the stand-in. dev_cases, if given, is a development
    store built by study2.devcases for this stream (Simon, 14 October): the monitor's replay test then has cases, so
    a rewritten check can be adopted. Its labels are relative to the reference pilot and are development only.

    For the M1 timing rehearsal (paper2/development/m1-timing-rehearsal/): layout picks the stream generator's layout;
    learning=False keeps the reasoning model for flight but runs no learning between episodes; library_from starts from
    an exact copy of a saved library and its event log; snapshot_after=(episode, folder) copies the library, whole and
    atomically, once that episode and its learning boundary are done; timing appends wall-clock spans to timing.jsonl,
    each opened before and closed after its work; record_inputs writes each flown scenario's fingerprint to inputs.jsonl
    before it flies, so conditions can be checked for identical inputs.

    pause_authority, if given, replaces this stream's own pause with a judge shared by every stream of a condition
    (the scored-study runner's). It is called before the first episode and after each episode and its learning
    boundary, with the episodes seen so far and this stream's pause rows, and returns the condition's lasting stop
    record or None. A stop record ends the stream before its next episode.

    small_call_limit, if given, replaces the small model's per-episode allowance of CALL_LIMITS (30 calls). With
    small_interval_s 0 it is the unrationed no-learning condition C-full (protocol amendment 7): set to the horizon
    in seconds, it cannot run out, because each decision has at most one call pending and each takes 2 modelled
    seconds."""
    if small_call_limit is not None and (type(small_call_limit) is not int or small_call_limit < 1):
        raise ValueError("the small model's allowance is a positive whole number of calls")
    if pause_authority is not None and no_pause:
        raise ValueError("a condition-wide pause replaces the stream's own pause; leave no_pause off")
    if layout not in ("v1", "v2") or (layout != "v1" and stream_seed is None):
        raise ValueError("layout v2 needs a stream seed; the layouts are v1 and v2")
    if not learning:
        if learn is None:
            raise ValueError("learning off keeps the reasoning model for flight; give its settings")
        if (learn_checks or no_replay or dev_cases is not None or learn_examples or learn_candidates != 1
                or learn_regression or max_attempts is not None or floor_after_cold_start or learn.get("adaptive_budget")):
            raise ValueError("learning options need learning; drop them when learning is off")
    if library_from is not None and (not empty_library or trusted_heading_guard is not None
                                     or trusted_report_guard is not None):
        raise ValueError("a copied library is used as it is: give an empty library and no trusted guards")
    if model_report_control and not (ladder and own_reports):
        raise ValueError("model report control needs the ladder and own-report tracking")
    model_wait = model_wait or model_report_control
    if prepared_scenarios is not None:
        if stream_seed is None or not prepared_scenarios:
            raise ValueError("prepared scenarios need a stream seed and a nonempty schedule")
        if any(not isinstance(s, E.Scenario) for s in prepared_scenarios):
            raise ValueError("prepared scenarios must be Scenario objects")
        if len({s.episode_id for s in prepared_scenarios}) != len(prepared_scenarios):
            raise ValueError("prepared scenario identifiers must be unique")
    if max_steps is None:
        max_steps = E.HORIZON + 1          # the world ends itself by its horizon; this is only a backstop
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    chosen = TANKER_REDUCED_PACK if reduced_features else TANKER_PACK
    if library_from is not None:
        copied = _copy_library(Path(library_from), out / "library")
        (out / "library-from.json").write_text(json.dumps({"source": str(Path(library_from).resolve()),
                                                           "files": copied}, indent=2, sort_keys=True))
    library = Library(out / "library", pack=chosen.public)
    if not empty_library:                   # the protocol starts scored streams empty (section 4)
        seed.stock(library)
    if trusted_report_guard is not None and trusted_heading_guard is None:
        raise ValueError("a trusted report guard is used only with a trusted heading guard")
    if trusted_heading_guard is not None:
        install_trusted_heading_guard(library, trusted_heading_guard)
    if trusted_report_guard is not None:
        install_trusted_heading_guard(library, trusted_report_guard, decision="report")
    both = trusted_report_guard is not None
    if learn_checks and not (both and dev_cases is not None):
        raise ValueError("learning checks needs both trusted guards and an outcome-labelled --dev-cases store")
    if no_replay and not learn_checks:
        raise ValueError("the no-replay arm differs only in how learned checks are adopted; it needs --learn-checks")
    # The replay test still gives checks the full stored contexts, so a check could qualify on fields it will not see in
    # flight; until replay projects them through the same view, reduced features go with procedure-only learning
    # (Codex's TURN237 review).
    procedures_only = (both and not learn_checks) or (not both and trusted_heading_guard is not None)
    if reduced_features and learn is not None and (learn_checks or no_replay or not procedures_only):
        raise ValueError("--reduced-features supports procedure-only learning (both trusted guards, no --learn-checks): "
                         "learned checks would be replayed on fields they cannot see in flight")
    coordinator = proposer = allowance = None
    if learn is not None:
        from telohelm.ladder import ReasoningAllowance
        # One allowance object for flight and learning, so both read the same spent room (D26 amendment 3).
        allowance = ReasoningAllowance(out / "reasoning-allowance.jsonl")
    if learn is not None and learning:      # MVP step 3: learning between episodes
        import telohelm.learning as G
        proposer = G.GatewayProposer(learn["base_url"], learn["model"], out, stream, allowance=allowance,
                                     thinking=learn.get("thinking", True),
                                     max_output_tokens=learn.get("max_output_tokens", G.PROTOCOL_OUTPUT_TOKENS),
                                     timeout_s=learn.get("timeout_s", 600.0), effort=learn.get("effort"),
                                     input_cap=learn.get("input_cap"))
        corpus = None
        if dev_cases is not None:
            from . import devcases as devcases
            from telohelm.qualification.store import CaseStore
            if stream_seed is None:
                raise ValueError("development cases serve one development stream; give --stream")
            if learn_checks:                # option C: the outcome-labelled store and its recomputing verifier
                from . import outcomecases as outcomecases
                corpus = CaseStore(dev_cases, verifier=outcomecases.verify, verifier_name=outcomecases.VERIFIER_NAME,
                                   development=True)
                rules = outcomecases.store_rules(corpus)
                if rules != {outcomecases.RULE}:
                    raise ValueError(f"learning checks needs a store labelled only by {outcomecases.RULE}, not {sorted(map(str, rules))}")
            else:
                corpus = CaseStore(dev_cases, verifier=devcases.verify, verifier_name=devcases.VERIFIER_NAME,
                                   development=True)
            if not corpus.cases(stream_seed):
                raise ValueError(f"{dev_cases} holds no development cases for stream {stream_seed}")
        coordinator = G.Coordinator(library, out, proposer, G.reference_solvable, predict, corpus=corpus,
                                    stream=stream_seed,
                                    budget=G.AdaptiveBudget() if learn.get("adaptive_budget") else None,
                                    max_attempts=max_attempts,
                                    extra_rules=POC_RULES_C if learn_checks else POC_RULES_BOTH if both
                                    else POC_RULES if trusted_heading_guard is not None else "",
                                    learn_decisions=("heading", "report") if both or trusted_heading_guard is None
                                    else ("heading",), procedures_only=both and not learn_checks,
                                    floor_after_cold_start=floor_after_cold_start, no_replay=no_replay,
                                    examples=learn_examples, candidates=learn_candidates,
                                    regression=learn_regression, pack=chosen.public, goal=chosen.goal,
                                    flight_pack=chosen)
    latency = {"small_model": small_latency_s, "reasoning_model": 60.0}
    records, attributions = [], []
    if prepared_scenarios is not None:
        # The comparison parent owns generation/witness records. No condition regenerates its input.
        from .comparators import scenario_fingerprint
        (out / "prepared-inputs.json").write_text(json.dumps([
            {"episode": s.episode_id, "scenario_sha256": scenario_fingerprint(s)}
            for s in prepared_scenarios], indent=2))
        scenarios = iter(prepared_scenarios)
    elif stream_seed is not None:             # protocol phases from the stream generator (phases 1 to 5)
        from . import streams as streams
        plan = [(p, i) for p in phases for i in range(episodes)]

        def generated():
            # Every generated episode's attempts are kept beside the collection before it is flown (Codex's GEN141).
            with open(out / "generation.jsonl", "a", encoding="utf-8") as f:
                for p, i in plan:
                    try:
                        g = streams.episode(stream_seed, p, i, layout)
                    except streams.GenerationError as e:
                        # The cap's evidence is kept before the stop (Codex's GEN143.CAP.RECORD).
                        f.write(json.dumps(streams.cap_record(stream_seed, p, i, e), default=str) + "\n")
                        f.flush()
                        raise
                    row = streams.generation_record(g)
                    if g.witness is not None:
                        # A phase 5 episode's full certificate and with-help witness, flight traces included, is kept
                        # in its own file, written before the episode flies; the generation record holds its hash.
                        row["witness"] = _write_witness(out, g.scenario.episode_id, g.witness)
                    f.write(json.dumps(row, default=str) + "\n")
                    f.flush()
                    yield g.scenario
        scenarios = generated()
    else:                                   # the earlier development source: phase-5 help-case draws
        scenarios = ((phase5.draw(s).world_a if world == "a" else phase5.draw(s).world_b)
                     for s in range(first_seed, first_seed + episodes))
    seen: list[str] = []
    for scenario in scenarios:
        if pause_authority is not None and not seen and (stop := pause_authority([], [])) is not None:
            (out / "paused.json").write_text(json.dumps({"after": None, "authority": stop}, indent=2, default=str))
            print("PAUSED before the first episode by the condition-wide pause", flush=True)
            break
        flight = None
        if ladder:
            flight = {"allowance": allowance, "reasoning_limit": FLIGHT_REASONING_LIMIT,
                      "reasoning": None if learn is None else {"base_url": learn["base_url"], "model": learn["model"],
                                                               "thinking": learn.get("thinking", True)}}
        if small_interval_s:
            flight = {**(flight or {"allowance": None, "reasoning": None, "no_ladder": True}),
                      "small_interval_s": small_interval_s}
        if small_call_limit is not None:
            flight = {**(flight or {"allowance": None, "reasoning": None, "no_ladder": True}),
                      "small_call_limit": small_call_limit}
        if watch_briefing or own_reports or reduced_features or model_wait or model_report_control:
            flight = {**(flight or {"allowance": None, "reasoning": None, "no_ladder": True}),
                      **({"watch": True} if watch_briefing else {}),
                      **({"own_reports": True} if own_reports else {}),
                      **({"model_wait": True} if model_wait else {}),
                      **({"model_report_control": True} if model_report_control else {}),
                      **({"reduced": True} if reduced_features else {})}
        if record_inputs:
            from .comparators import scenario_fingerprint
            with open(out / "inputs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"episode": scenario.episode_id,
                                    "scenario_sha256": scenario_fingerprint(scenario)}) + "\n")
        started, began = _span(out, scenario.episode_id, "flight") if timing else (time.time(), time.monotonic())
        done = run_episode(scenario, library, out, stream, max_steps, latency, local, flight)
        if timing:
            _timing(out, scenario.episode_id, "flight", started, began)
        records.append(done["record"])
        attributions.append(done["attribution"])
        r = done["record"]
        if coordinator is not None:
            started, began = (_span(out, scenario.episode_id, "after_episode") if timing
                              else (time.time(), time.monotonic()))
            for e in coordinator.after_episode(scenario, r, r["complete"]):
                if e["kind"].startswith("attempt") or e["kind"] in ("floor_breach", "floor_rearmed", "undo"):
                    print(f"  learning: {json.dumps(e, default=str)[:300]}", flush=True)
            if timing:
                _timing(out, scenario.episode_id, "after_episode", started, began)
        if snapshot_after is not None and scenario.episode_id == snapshot_after[0]:
            started, began = _span(out, scenario.episode_id, "snapshot") if timing else (time.time(), time.monotonic())
            _snapshot_library(out / "library", Path(snapshot_after[1]), scenario.episode_id)
            if timing:
                _timing(out, scenario.episode_id, "snapshot", started, began)
        print(f"{r['episode']}: {'complete' if r['complete'] else 'CENSORED by step limit'} goal_met={r['goal_met']} "
              f"end={r['end_reason']} at={r['ended_at']} calls={r['model_calls']} wall={r['wall_s']}s", flush=True)
        # The protocol's pause (section 10, D26): after each episode and its rewrite attempt.
        seen.append(r["episode"])
        excluded: list = []
        rows = _ledger_rows(out, seen, proposer if coordinator is not None else None, coordinator, excluded)
        if pause_authority is not None:
            stop = pause_authority(list(seen), rows)
            if stop is not None:
                (out / "paused.json").write_text(json.dumps({"after": r["episode"], "authority": stop,
                                                             "note": "stopped by the condition-wide pause; nothing "
                                                                     "resumes automatically"}, indent=2, default=str))
                print(f"PAUSED after {r['episode']} by the condition-wide pause", flush=True)
                break
            continue
        pause = pause_due(rows, seen)
        pause = {**pause, "pause_excluded": excluded}
        if pause["due"] and no_pause:
            # Development (D41): the pause is a protocol safeguard. Record when it would have stopped the run, once,
            # and go on.
            if not (out / "pause-would-be-due.json").exists():
                (out / "pause-would-be-due.json").write_text(json.dumps({"after": r["episode"], **pause}, indent=2))
                print(f"pause would be due after {r['episode']} (development run continues): {pause}", flush=True)
        elif pause["due"]:
            (out / "paused.json").write_text(json.dumps({"after": r["episode"], **pause,
                                                         "note": "stopped for Simon; nothing resumes automatically"},
                                                        indent=2))
            print(f"PAUSED after {r['episode']}: {pause}", flush=True)
            break
    pooled = A.pool(attributions, mixed=True)
    completed = [r for r in records if r["complete"]]
    summary = {"episodes": len(records), "completed": len(completed), "censored_by_step_limit": len(records) - len(completed),
               "goal_met_of_completed": sum(bool(r["goal_met"]) for r in completed),
               "joined_of_completed": sum(bool(r["joined"]) for r in completed),
               "model_calls": sum(r["model_calls"] for r in records),
               "tokens_in": _total(records, "tokens_in"), "tokens_out": _total(records, "tokens_out"),
               "tokens_in_known_subtotal": sum(r["tokens_in_known_subtotal"] for r in records),
               "tokens_out_known_subtotal": sum(r["tokens_out_known_subtotal"] for r in records),
               "calls_with_unknown_usage": sum(r["calls_with_unknown_usage"] for r in records),
               "share_of_seconds": pooled,
               "stand_ins": ["model"] if local is None else [],
               "model": "stand-in" if local is None else local["model"],
               "note": "MVP development run; not a result",
               "flight": {"ladder": ladder, "small_interval_s": small_interval_s,
                          **({"small_call_limit": small_call_limit} if small_call_limit is not None else {}),
                          "reasoning_limit": FLIGHT_REASONING_LIMIT if ladder else None,
                          **({"watch_briefing": {"tolerance": TANKER_PACK.evidence.tolerance}} if watch_briefing else {}),
                          **({"own_reports": True} if own_reports else {}),
                          **({"model_wait": True} if model_wait else {}),
                          **({"model_report_control": True} if model_report_control else {}),
                          **({"reduced_features": sorted(TANKER_REDUCED_PACK.hidden)} if reduced_features else {})}}
    if learn is not None and not learning:
        summary["learning"] = {"off": True, "note": "reasoning model for flight only; no learning between episodes"}
    if library_from is not None:
        summary["library_from"] = str(Path(library_from).resolve())
    if layout != "v1":
        summary["layout"] = layout
    if coordinator is not None:
        lc = proposer.ledger.calls
        summary["learning"] = {"attempts": coordinator.attempts,
                               "adopted": [e["candidate"] for e in coordinator.events
                                           if e["kind"] == "attempt_result" and e["passed"]],
                               "reasoning_calls": len(lc), **_usage(lc), "calls_with_unknown_usage": _unknown_calls(lc),
                               "input_token_cap": f"{proposer.input_cap:,}, counted before each request (token_count.py)",
                               "profile": {"max_output_tokens": proposer.max_output_tokens, "timeout_s": proposer.timeout_s,
                                           "effort": proposer.effort, "input_cap": proposer.input_cap,
                                           "max_attempts": coordinator.max_attempts,
                                           "floor_after_cold_start": coordinator.floor_after_cold_start,
                                           "no_replay": coordinator.no_replay,
                                           "examples": coordinator.examples,
                                           "candidates": coordinator.candidates,
                                           "regression": coordinator.regression,
                                           "learn_decisions": list(coordinator.learn_decisions),
                                           "extra_rules_sha256": hashlib.sha256(coordinator.extra_rules.encode()).hexdigest()
                                           if coordinator.extra_rules else None},
                               "reasoning_thinking": "on" if proposer.thinking else "off (development setting)"}
        (out / "learning-events.json").write_text(json.dumps(coordinator.events, indent=2, default=str))
        if callable(getattr(proposer, "persist_costs", None)):
            proposer.persist_costs()
        else:
            _write_ledger(out / "learning-ledger.jsonl", lc)
    (out / "stream-summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"summary": summary, "episodes": records}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=None,
                   help="cut episodes short for a smoke run (recorded as censored); default: run to the world's end")
    p.add_argument("--world", choices=("a", "b"), default="a")
    p.add_argument("--local-model", default=None, help="a model served on this machine, e.g. "
                   "mlx-community/Qwen3-4B-Instruct-2507-8bit; default: the stand-in")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--learn-model", default=None, help="the reasoning model that writes procedures and checks between "
                   "episodes, e.g. mlx-community/Qwen3.8-27B-4bit; default: no learning")
    p.add_argument("--learn-url", default="http://127.0.0.1:8001")
    p.add_argument("--learn-no-thinking", action="store_true", help="development setting: ask the reasoning "
                   "model's chat template to skip thinking (on this Mac it cannot answer within 600 s otherwise)")
    p.add_argument("--stream", type=int, default=None, help="a stream seed (development: 4400 to 4499); runs "
                   "--episodes episodes of each of --phases from the stream generator instead of phase-5 draws")
    p.add_argument("--phases", type=int, nargs="+", default=[1, 2], choices=[1, 2, 3, 4, 5],
                   help="stream phases; 5 is the help cases")
    p.add_argument("--empty-library", action="store_true", help="start with no procedures, as scored streams do")
    p.add_argument("--learn-max-tokens", type=int, default=None, help="development: the reasoning model's output "
                   "cap for learning calls (protocol 16,384)")
    p.add_argument("--learn-timeout", type=float, default=None, help="development: seconds per learning call "
                   "(protocol 600)")
    p.add_argument("--learn-effort", default=None, help="development: reasoning effort for learning calls "
                   "(protocol medium)")
    p.add_argument("--adaptive-budget", action="store_true", help="Telohelm adapts its learning budget between "
                   "attempts (learning.BUDGET_LEVELS); not with the fixed --learn-max-tokens/timeout/effort")
    p.add_argument("--small-interval", type=int, default=SMALL_INTERVAL_S, help="at most one small-model call per "
                   "decision every this many simulated seconds (protocol 0.11; 0 turns rationing off)")
    p.add_argument("--no-ladder", action="store_true", help="development: the earlier flight path (small model only, "
                   "no checks on model replies, no help)")
    p.add_argument("--dev-cases", type=Path, default=None, help="a development case store built by study2.devcases "
                   "for --stream; labels relative to the reference pilot, development only")
    p.add_argument("--trusted-heading-guard", type=Path, default=None, help="development: adopt this trusted heading "
                   "check at stream start and ask for heading procedures only (proof-of-concept plan, step 2)")
    p.add_argument("--max-attempts", type=int, default=None, help="development: stop rewrite attempts after this many")
    p.add_argument("--floor-after-cold-start", action="store_true", help="development: exclude the first 20 solvable "
                   "episodes from the floor; require 20 later ones before its first assessment")
    p.add_argument("--no-replay", action="store_true", help="the comparison arm Telohelm without the replay test "
                   "(D5): learned checks are adopted without replay and never retired on replay evidence")
    p.add_argument("--learn-candidates", type=int, default=1, help="development: separate candidates per learning "
                   "attempt, 1 to 3, each trialled until one passes (the default 1 keeps format repairs)")
    p.add_argument("--learn-regression", type=int, default=0, help="development: a candidate must also keep the goal "
                   "on up to this many earlier episodes the library met by itself (0, the default, keeps the trial as before)")
    p.add_argument("--watch-briefing", action="store_true", help="development: while the goal tanker's newest radar "
                   "report is more than the pack's tolerance from the briefed orbit, withhold the learned procedures and "
                   "climb to a model told the gap (runtime/disagreement.py; needs the ladder)")
    p.add_argument("--reduced-features", action="store_true", help="development: procedures, checks and models see "
                   "the observations without the quantities the context builder derives (wind, range, closing rate, "
                   "report age), so Telohelm's own code must work them out (packs/tanker_pack.py)")
    p.add_argument("--own-reports", action="store_true", help="development: count a report the trigger sent, so the "
                   "report decision is not asked again under that goal version (runtime/own_reports.py)")
    p.add_argument("--model-wait", action="store_true", help="development: retain and check an explicit {} model "
                   "reply as waiting for this report decision, rather than no answer; budgets are unchanged")
    p.add_argument("--model-report-control", action="store_true", help="development: show existing report trigger "
                   "actions, check direct model reports against simultaneous observations and feed back refusals; "
                   "implies --model-wait and needs --own-reports and the ladder")
    p.add_argument("--learn-examples", action="store_true", help="development: each learning request carries up to 8 "
                   "examples from this arm's own completed episodes (B7 item 5, study2/examples.py)")
    p.add_argument("--learn-checks", action="store_true", help="development, option C: with both guards and an "
                   "outcome-labelled --dev-cases store, the model may also write checks")
    p.add_argument("--no-pause", action="store_true", help="development (D41): record when the protocol's pause "
                   "would stop the run, and continue")
    p.add_argument("--trusted-report-guard", type=Path, default=None, help="development: also adopt this trusted "
                   "report check at stream start and learn procedures for both decisions (proof-of-concept step 3)")
    p.add_argument("--learn-input-cap", type=int, default=None, help="development: input tokens per learning call "
                   "(protocol 4,096)")
    p.add_argument("--layout", choices=("v1", "v2"), default="v1", help="the stream generator's layout; v2 makes "
                   "phase 3 an orbit block then a late-information block of ten (M1 timing rehearsal)")
    p.add_argument("--no-learning", action="store_true", help="with --learn-model: the reasoning model serves flight "
                   "only, and nothing is learned between episodes")
    p.add_argument("--library-from", type=Path, default=None, help="start from an exact copy of this saved library "
                   "folder (needs --empty-library and no trusted guards)")
    p.add_argument("--snapshot-library-after", nargs=2, metavar=("EPISODE", "FOLDER"), default=None,
                   help="copy the library to FOLDER once EPISODE and its learning boundary are done")
    p.add_argument("--timing", action="store_true", help="append wall-clock spans of flights and learning "
                   "boundaries to timing.jsonl")
    p.add_argument("--record-inputs", action="store_true", help="write each flown scenario's fingerprint to "
                   "inputs.jsonl before it flies")
    a = p.parse_args(argv)
    local = None if a.local_model is None else {"base_url": a.base_url, "model": a.local_model}
    learn = None if a.learn_model is None else {"base_url": a.learn_url, "model": a.learn_model,
                                                "thinking": not a.learn_no_thinking,
                                                **({"max_output_tokens": a.learn_max_tokens} if a.learn_max_tokens else {}),
                                                **({"timeout_s": a.learn_timeout} if a.learn_timeout else {}),
                                                **({"effort": a.learn_effort} if a.learn_effort else {}),
                                                **({"input_cap": a.learn_input_cap} if a.learn_input_cap else {}),
                                                **({"adaptive_budget": True} if a.adaptive_budget else {})}
    if a.adaptive_budget and (a.learn_max_tokens or a.learn_timeout or a.learn_effort):
        p.error("--adaptive-budget sets the learning budget itself; drop the fixed --learn-* settings")
    done = run_stream(a.episodes, a.first_seed, a.out, max_steps=a.max_steps, world=a.world, local=local, learn=learn,
                      stream_seed=a.stream, phases=tuple(a.phases), empty_library=a.empty_library,
                      dev_cases=a.dev_cases, ladder=not a.no_ladder, small_interval_s=a.small_interval,
                      trusted_heading_guard=a.trusted_heading_guard, max_attempts=a.max_attempts,
                      trusted_report_guard=a.trusted_report_guard, no_pause=a.no_pause,
                      floor_after_cold_start=a.floor_after_cold_start, learn_checks=a.learn_checks,
                      no_replay=a.no_replay, learn_examples=a.learn_examples,
                      learn_candidates=a.learn_candidates, learn_regression=a.learn_regression,
                      watch_briefing=a.watch_briefing, own_reports=a.own_reports,
                      reduced_features=a.reduced_features, model_wait=a.model_wait,
                      model_report_control=a.model_report_control, layout=a.layout,
                      learning=not a.no_learning, library_from=a.library_from,
                      snapshot_after=None if a.snapshot_library_after is None else (a.snapshot_library_after[0],
                                                                                   Path(a.snapshot_library_after[1])),
                      timing=a.timing, record_inputs=a.record_inputs)
    s = done["summary"]
    print(f"stream: {s['completed']}/{s['episodes']} episodes completed; of those, {s['goal_met_of_completed']} met the "
          f"goal and {s['joined_of_completed']} joined; {s['model_calls']} model calls")


if __name__ == "__main__":
    main()
