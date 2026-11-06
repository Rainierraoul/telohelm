"""The reasoning model alone, a budgeted adviser (decision D3; D26 amendment 5 in
paper2/notes/MVP-TURN141-REVIEW-CODEX.md; protocol section 5; paper2/notes/B36-FLIGHT-LADDER-PLAN.md section 3).

One episode-wide request at a time goes to the reasoning model. Its reply gives a heading, optionally a report trigger,
optionally a help request, and next_call_at_s: the absolute whole simulated second of its next call, or null. There is
no library, no small model and no learning. The aircraft holds the last accepted heading between replies.

Calls, at most 4 per episode (D3). The first is made at t = 0. After that a call is made when one is due and none is
pending. A call is due for any of these, each an event, never a state that stays true:
- the second the last usable reply asked for. A second already reached at delivery is moved to the second after
  delivery and recorded as past due; null asks for no timed call;
- a new goal version;
- a newly observed refusal: the world refused the last action, or a delivered reply could not be used;
- the goal tanker's newest measured report turning stale (over 10 s old, or missing), once per change from fresh.
While a call is pending, new events are merged with it and wait for its delivery; the pending call is never cancelled
or relabelled. No call is made in the second a reply is delivered, so the earliest is the next second. Issuing a call
consumes the events waiting and any timed call not yet reached, since the new reply sets the next one. A reply issued
under an older goal version is delivered marked out of date and refused, and nothing from it acts.

Each call goes through the episode's model gateway, which charges it once, keeps its tokens and wall time in the
ledger and refuses a fifth. Its reply is held until the modelled latency has passed (60 s; 30 s for the D2
sensitivity run). A reply acts only if it is a JSON object with heading_deg, no fields beyond the four above, a
well-formed next_call_at_s (absent is taken as null), values within the world's public action rules
(escalation.part_error), and help only if none has been asked this episode. A call still pending when the episode
ends stays charged and is listed as undelivered.

Method-seconds follow the rule of runtime/attribution.py: a second whose accepted action carries a heading goes to the
reasoning model, and any other second is holding, with a call pending immediately before the action or with none.
Stored procedure and small model are not applicable to this condition and are reported so, never as zero.

The prompt is a development draft, like local_model's; the frozen prompts belong to backlog item B7. Development
records only; nothing here is a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import telohelm.local_model as local_model
import telohelm.runtime.models as M
from telohelm.runtime.context import ContextBuilder
from telohelm.runtime.escalation import HELP_REASONS; from telohelm.runtime.escalation import STALE_S; from telohelm.runtime.escalation import part_error
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.trusted_log import TrustedLog
import telohelm.taskpacks.tanker.world.episode as E

CONDITION = "reasoning_alone"
CALLS = 4                                   # D3
LATENCY_S = 60.0                            # D2; 30.0 for the sensitivity run
OUTPUT_TOKENS = 16_384                      # D26 amendment 6, as for Telohelm's flying reasoning calls
TIMEOUT_S = 600.0                           # protocol section 5, flying calls
ADVICE_KEYS = frozenset({"heading_deg", "report_trigger_nm", "help", "next_call_at_s"})
NOT_APPLICABLE = "not applicable"

ADVISER = f"""You advise the receiver's autopilot. Each reply is applied {{latency}} simulated seconds after the time t
in your context, and the autopilot holds your last heading until your next reply is applied. You have at most {CALLS}
replies in the episode; calls_left_after_this says how many remain. Reply with one JSON object with these fields:
- "heading_deg": the heading to fly, in [0, 360);
- "report_trigger_nm" (optional): arms a one-shot radio report. The first second after it is applied at which the
  observed range to the goal tanker is decreasing and at most this many NM, the report
  "<control>, aircraft 2, <report_at_nm> NM from <goal_tanker>" is sent. null disarms it; a new goal clears it.
  Only one report counts, scored on the true range;
- "help" (optional, at most once an episode): {{"reason": one of {list(HELP_REASONS)}, "text": at most 500
  characters}}. 30 s later the goal tanker's exact position is reported to you;
- "next_call_at_s": the whole simulated second at which to call you again, or null for no timed call. You are also
  called when the goal changes, when an action of yours is refused, and when the goal tanker's newest report becomes
  more than {int(STALE_S)} s old, while calls remain."""


# The matched episodes the adviser flies (protocol section 4; D22, accepted by Codex in TURN 125): one NumPy
# Generator(PCG64(seed)) per scored stream, seeds 6001 to 6004, used through phases 1 to 5 in order. Phases 1, 2, 4 and
# 5 choose 5 of the 20 indices without replacement; phase 3 chooses one index from each block of 5 in order, then one
# of the 16 left, in ascending order. Codex's reference draw (paper2/reproduction/runtime-turn125.json) has digest
# MATCHED_DRAWS_SHA256 over the compact JSON of matched_draws().
MATCH_SEEDS = (6001, 6002, 6003, 6004)
MATCHED_DRAWS_SHA256 = "eed2617ebc170ab61f42c80c2aea5c83f6b22584a0e9f12d6af857ce4a2882de"


def matched_indices(seed: int) -> dict[str, dict[str, list[int]]]:
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(seed))
    out = {}
    for phase in (1, 2, 3, 4, 5):
        if phase != 3:
            order = [int(i) for i in rng.choice(20, size=5, replace=False)]
        else:
            order = [5 * block + int(rng.integers(0, 5)) for block in range(4)]
            left = [i for i in range(20) if i not in order]
            order.append(left[int(rng.integers(0, len(left)))])
        out[str(phase)] = {"chronological_indices": sorted(order), "draw_order": order}
    return out


def matched_draws() -> dict[str, dict]:
    return {str(seed): matched_indices(seed) for seed in MATCH_SEEDS}


class AdviceRefused(ValueError):
    """A reply that cannot act; its message is the public refusal recorded."""


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def messages_for(ctx: dict, triggers: list[str], calls_left: int, latency_s: float) -> list:
    request = {"context": local_model.compact(ctx), "triggers": triggers, "calls_left_after_this": calls_left}
    return [("system", local_model.SYSTEM + "\n" + ADVISER.replace("{latency}", str(int(math.ceil(latency_s))))),
            ("user", "Request:\n" + json.dumps(request, default=str, sort_keys=True))]


def parse(text: str | None, help_asked: bool) -> tuple[dict, int | None]:
    """The action part and next_call_at_s of a reply, or AdviceRefused naming why it cannot act."""
    if text is None:
        raise AdviceRefused("no_text")
    try:
        obj = local_model.first_object(text)
    except ValueError:
        raise AdviceRefused("unparseable") from None
    if not set(obj) <= ADVICE_KEYS:
        raise AdviceRefused("unknown_field")
    if "heading_deg" not in obj:
        raise AdviceRefused("no_heading")
    nxt = obj.get("next_call_at_s")
    if nxt is not None:
        whole = (isinstance(nxt, int) and not isinstance(nxt, bool)) or (
            isinstance(nxt, float) and math.isfinite(nxt) and nxt.is_integer())
        if not whole or nxt < 0:
            raise AdviceRefused("bad_schedule")
        nxt = int(nxt)
    part = {k: v for k, v in obj.items() if k != "next_call_at_s"}
    if part_error("heading", part) is not None:          # checks every action field present, whatever the decision
        raise AdviceRefused("invalid_action")
    if "help" in part and help_asked:
        raise AdviceRefused("second_help")
    return part, nxt


def run_adviser_episode(scenario: E.Scenario, out: Path, stream: str, provider: Any, latency_s: float = LATENCY_S,
                        max_steps: int | None = None, calls: int = CALLS) -> dict[str, Any]:
    """One episode of the reasoning model alone, to the world's own end (or a censored prefix of max_steps)."""
    ep_dir = Path(out) / scenario.episode_id
    ep_dir.mkdir(parents=True)
    ledger = Ledger()
    gateway = M.ModelGateway(scenario.episode_id, CONDITION, stream, provider,
                             TrustedLog(ep_dir / "gateway.log", ep_dir / "gateway.heads", stream), ledger,
                             ep_dir / "journal", limits={"reasoning_model": calls, "small_model": 0})
    log = TrustedLog(ep_dir / "episode.log", ep_dir / "episode.heads", stream)
    ep = E.Episode(scenario)
    start = ep.reset()
    builder = ContextBuilder(start["briefing"])
    bundle = start["bundle"]
    log.append({"kind": "episode_start", "episode": scenario.episode_id, "condition": CONDITION,
                "latency_s": latency_s, "calls": calls})
    pending: dict | None = None
    waiting: list[str] = ["start"]
    scheduled: dict | None = None             # {"at": second, "asked": value returned, "past_due": bool}
    last_version, was_stale, help_asked, out_of_calls = None, False, False, False
    counts = {"reasoning_model": 0, "holding_pending": 0, "holding_none": 0}
    started = time.perf_counter()
    for _ in range(max_steps if max_steps is not None else E.HORIZON + 1):
        ctx = builder.update(bundle)
        t, version = ctx["t"], ctx["goal_version"]
        # Events observed this second.
        if last_version is not None and version != last_version:
            waiting.append("goal")
        last_version = version
        if bundle.get("refusals"):
            waiting.append("refusal")
        stale = ctx["report_age_s"] is None or ctx["report_age_s"] > STALE_S
        if stale and not was_stale:
            waiting.append("stale")
        was_stale = stale
        # Reply phase: a pending reply due now.
        action: dict[str, Any] = {}
        delivered = False
        if pending is not None and pending["available_second"] == t:
            p, pending, delivered = pending, None, True
            rec = {"kind": "adviser_reply", "t": t, "request_id": p["request_id"], "goal_version": p["goal_version"],
                   "issued_second": p["issued_second"], "status": p["status"], "triggers": p["triggers"],
                   "reply_sha256": None if p["text"] is None else hashlib.sha256(
                       p["text"].encode("utf-8", "surrogatepass")).hexdigest()}
            if p["goal_version"] != version:
                rec.update(stale=True, refusal="out_of_date", part=None)
            else:
                try:
                    if p["status"] != "answered":
                        raise AdviceRefused(p["status"])
                    part, nxt = parse(p["text"], help_asked)
                except AdviceRefused as e:
                    rec.update(stale=False, refusal=str(e), part=None)
                    waiting.append("reply_refused")
                else:
                    rec.update(stale=False, refusal=None, part=part, next_call_at_s=nxt)
                    action = {**part, "goal_version": version}
                    scheduled = None if nxt is None else {"at": max(nxt, t + 1), "asked": nxt, "past_due": nxt <= t}
                    if scheduled is not None:
                        rec["scheduled"] = scheduled
            log.append(rec)
        if scheduled is not None and t >= scheduled["at"]:
            waiting.append("past_due" if scheduled["past_due"] else "scheduled")
            scheduled = None
        # Issue one call if one is due, none is pending, no reply came this second and calls remain.
        if waiting and pending is None and not delivered:
            if gateway.used["reasoning_model"] >= calls:
                if not out_of_calls:
                    out_of_calls = True
                    log.append({"kind": "adviser_no_calls_left", "t": t, "triggers": sorted(set(waiting))})
                waiting = []
            else:
                triggers = sorted(set(waiting))
                left = calls - gateway.used["reasoning_model"] - 1
                got = gateway.submit("reasoning_model", messages_for(ctx, triggers, left, latency_s), OUTPUT_TOKENS,
                                     version, t, len(ep.record.actions), latency_s)
                if got["status"] == "refused":
                    log.append({"kind": "adviser_call_refused", "t": t, "refusal": got.get("refusal")})
                else:
                    pending = {"request_id": got["request_id"], "goal_version": version, "issued_second": t,
                               "available_second": got["available_second"], "status": got["status"],
                               "text": got["text"], "triggers": triggers, "context_sha256": _sha(ctx)}
                    log.append({"kind": "adviser_call", "t": t, "request_id": got["request_id"], "goal_version": version,
                                "triggers": triggers, "available_second": got["available_second"],
                                "context_sha256": pending["context_sha256"]})
                    scheduled = None
                waiting = []
        pending_before_action = pending is not None
        bundle = ep.step(action)
        accepted = ep.record.actions[-1]["accepted"]
        if action:
            log.append({"kind": "action", "t": t, "action": action, "accepted": accepted})
        if accepted and "help" in action:
            help_asked = True
        if accepted and "heading_deg" in action:
            counts["reasoning_model"] += 1
        else:
            counts["holding_pending" if pending_before_action else "holding_none"] += 1
        if bundle is None:
            break
    if pending is not None:
        log.append({"kind": "model_undelivered", "request_id": pending["request_id"],
                    "available_second": pending["available_second"], "issued_second": pending["issued_second"],
                    "reason": "episode ended before its second"})
    wall = time.perf_counter() - started
    outcome = ep.outcome()
    complete = outcome["end_reason"] is not None
    log.append({"kind": "episode_end", "episode": scenario.episode_id, "goal_met": outcome["goal_met"],
                "settled": outcome["settled"], "end_reason": outcome["end_reason"]})
    K = sum(counts.values())
    seconds = {"procedure": NOT_APPLICABLE, "small_model": NOT_APPLICABLE, **counts}
    calls_made = ledger.calls
    record = {
        "episode": scenario.episode_id, "condition": CONDITION, "goal_met": outcome["goal_met"],
        "joined": outcome["joined"], "report_correct": outcome["report_correct"],
        "help_requests": len(outcome.get("help_requests") or []), "end_reason": outcome["end_reason"],
        "ended_at": outcome["ended_at"], "action_steps": outcome["action_steps"],
        "model_calls": len(calls_made),
        "tokens_in": _sum(calls_made, "tokens_in"), "tokens_out": _sum(calls_made, "tokens_out"),
        "method_seconds": seconds, "seconds": K,
        "shares": {k: (v / K if K else None) if isinstance(v, int) else v for k, v in seconds.items()},
        "complete": complete, "censored_by_step_limit": not complete, "latency_s": latency_s,
        "wall_s": round(wall, 3), "settled": outcome.get("settled"), "outcome": outcome,
    }
    (ep_dir / "summary.json").write_text(json.dumps(record, indent=2, default=str))
    (ep_dir / "ledger.jsonl").write_text("".join(json.dumps(asdict(c), sort_keys=True, default=str) + "\n"
                                                 for c in calls_made))
    return record


def _sum(calls: list, field: str) -> int | None:
    """The total of a usage field, or None if any call's is unknown."""
    values = [getattr(c, field) for c in calls]
    return None if any(v is None for v in values) else sum(values)


def main(argv: list[str] | None = None) -> None:
    """Development runs of the reasoning model alone on a development stream (4400 to 4499)."""
    from . import streams as streams
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stream", type=int, required=True)
    p.add_argument("--phases", type=int, nargs="+", default=[1, 2])
    p.add_argument("--episodes", type=int, default=5, help="episodes per phase, from index 0")
    p.add_argument("--base-url", default="http://127.0.0.1:8001")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--latency", type=float, default=LATENCY_S)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if not 4400 <= a.stream <= 4499:
        p.error("development streams only (4400 to 4499); matched scored episodes come from the frozen schedule")
    provider = local_model.provider(a.base_url, a.model, TIMEOUT_S, role="reasoning_model")
    records = []
    for phase in a.phases:
        for i in range(a.episodes):
            g = streams.solvable_episode(a.stream, phase, i)
            r = run_adviser_episode(g.scenario, a.out, str(a.stream), provider, a.latency)
            records.append(r)
            print(f"{r['episode']}: goal_met={r['goal_met']} end={r['end_reason']} at={r['ended_at']} "
                  f"calls={r['model_calls']} seconds={r['method_seconds']}", flush=True)
    (a.out / "stream-summary.json").write_text(json.dumps(
        {"episodes": len(records), "goal_met": sum(bool(r["goal_met"]) for r in records),
         "model_calls": sum(r["model_calls"] for r in records), "note": "development run; not a result"}, indent=2))


if __name__ == "__main__":
    main()
