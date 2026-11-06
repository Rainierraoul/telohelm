"""Telohelm's loop for one episode (plan section 5; architecture sections 5 to 10). Model calls are behind
an interface that is stubbed until the model layer is built.

Each second: build the context, ask the chooser for a heading and, until a report has been sent under the
current goal version, for a report. When it climbs, ask the model layer, which may return an action part or
nothing, in which case the aircraft holds its heading. Every decision is written to the trusted log, and every
model call to the ledger. A refused action is not yet fed back here; that comes with the model layer.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from ..taskpacks.tanker.world import episode as E
from .chooser import check_output; from .chooser import choose
from .escalation import own_part; from .escalation import part_error
from .disagreement import Watch; from .disagreement import models_exhausted
from .own_reports import OwnTrigger
from ..taskpacks.tanker.context import ContextBuilder  # compatibility import; execution uses the supplied pack
from ..taskpacks.tanker.pack import TANKER_PACK
from .delivery import DECISION_FIELDS; from .delivery import DelayedModels; from .delivery import delivery_record
from .ledger import Call; from .ledger import Ledger
from .library import Library
from .environment import environment_for
from .model_actions import ModelActions
from .trusted_log import TrustedLog


class ModelLayer(Protocol):
    def decide(self, decision: str, ctx: dict[str, Any], reason: str) -> tuple[dict[str, Any] | None, Call | None]:
        """An action part for the decision, or None, and the ledger record of any call made."""


class NoModels:
    """The stub: never answers and makes no call, so a climb means holding the heading."""
    def decide(self, decision, ctx, reason):
        return None, None


def run_episode(scenario: E.Scenario, library: Library, models: ModelLayer, log: TrustedLog,
                ledger: Ledger, max_steps: int = 3600, run: Any = None, broker: Any = None,
                ladder: Any = None, watch_view: Any = None, own_reports: bool = False,
                pack: Any = TANKER_PACK, model_preconditions: bool = False) -> dict[str, Any]:
    """run: the sandbox call to use (sandbox.run by default, or a SandboxWorker's run).

    broker (TURN 56): if given, procedures and checks run with the broker channel. Before each second's decisions
    the broker is told the simulated second, the goal version and the action step, which is the number of actions
    the world has recorded so far, so the step is the world's own position in its event sequence. After each world
    step the broker's event-prefix fingerprint is kept for that step. At the end the broker passes on the
    episode-end feedback, and the result lists, for the nominal ending and each alternative ending, the calls counted
    before it, those excluded, the straddling ones and the event prefix at its place."""
    # ladder (B36, D26): a runtime.escalation.FlightLadder for this episode. With it, a model's reply passes the
    # applicable adopted checks before it acts, an unusable current small-model reply escalates its decision to the
    # reasoning model, reasoning calls draw on the shared allowance, and one help request may be made. Without it the
    # loop behaves as before. It needs the delayed model layer.
    # watch_view (opt-in, runtime/disagreement.py): the task pack's evidence view. While the newest observation
    # disagrees with the briefing beyond its tolerance, the decisions that read it are withheld from their procedures
    # and climb to a model, which is told the gap. It needs the ladder. own_reports (opt-in, runtime/own_reports.py):
    # a report the trigger sent counts as sent, so the report decision is not asked again under that goal version.
    if ladder is not None and not isinstance(models, DelayedModels):
        raise ValueError("the flight ladder needs the delayed model layer")
    if watch_view is not None and ladder is None:
        raise ValueError("the disagreement watch needs the flight ladder")
    if model_preconditions and ladder is None:
        raise ValueError("model action preconditions need the flight ladder")
    model_actions = ModelActions(pack.model_precondition) if model_preconditions else None
    if broker is not None and broker.episode != scenario.episode_id:
        raise ValueError("the broker belongs to another episode")
    if broker is not None and broker.ledger is not ledger:
        raise ValueError("the broker charges a different ledger")
    delayed = isinstance(models, DelayedModels)
    decision_fields = pack.public.decision_fields
    action_keys = frozenset().union(*decision_fields.values())
    if any(not fields <= decision_fields.get(decision, frozenset())
           for decision, fields in pack.roles.completion_fields.items()):
        raise ValueError("completion fields must belong to their task decision")
    if delayed:
        if not models.wait_decisions <= set(decision_fields):
            raise ValueError("wait decisions must belong to the selected task")
        # The gateway must belong to this episode, and to the broker's condition and stream, before any work
        # (Codex's DL70.BIND); a shared ledger alone does not establish that.
        gw = models.gateway
        if gw.ledger is not ledger:
            raise ValueError("the model gateway charges a different ledger")
        if gw.episode != scenario.episode_id:
            raise ValueError("the model gateway belongs to another episode")
        if broker is not None and (gw.condition != broker.condition or gw.stream != broker.stream):
            raise ValueError("the model gateway belongs to another condition or stream")

    def deliver(p, ctx, subphase):
        """Record one delivery in the trusted log and the broker's sequence; return the part it proposes, if any."""
        part, stale, refusal = models.part_for(p, ctx["goal_version"], action_keys=action_keys)
        rec = delivery_record(p, stale, part, subphase, refusal)
        log.append({"kind": "model_reply", "t": ctx["t"], **rec})
        if broker is not None:
            broker.checkpoint("model_reply", **rec)
        if (ladder is not None and part is None and p.goal_version == ctx["goal_version"]
                and (refusal is not None or p.status != "answered")):
            ladder.unusable(p.decision, p.role, p.goal_version, refusal or p.status, ctx["t"])
        return part

    def context_sha(ctx):
        return hashlib.sha256(json.dumps(ctx, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                         allow_nan=False).encode()).hexdigest()
    environment = environment_for(pack)
    ep = environment.create(scenario)
    start = ep.reset()
    if broker is not None:
        # The exact public inputs go into the broker's log before they are used, so a replay can be rebuilt from it
        # (Codex's BR59.STORE): the briefing here, each bundle below.
        broker.checkpoint("briefing", briefing=start["briefing"])
    prefix_after_step = {0: (broker.event_prefix, broker.logical_prefix)} if broker is not None else {}
    builder = pack.context_builder(start["briefing"])     # the supplied task builds its observations
    roles = pack.roles
    watch = Watch(watch_view, start["briefing"]) if watch_view is not None else None
    trig = OwnTrigger() if own_reports or watch is not None else None
    bundle = start["bundle"]
    log.append({"kind": "episode_start", "episode": scenario.episode_id,
                **({"environment": environment.record()} if getattr(pack, "environment", None) is not None else {})})
    climbs = 0
    once_issued: dict[str, tuple[int, int]] = {}
    for _ in range(max_steps):
        # A whole-action refusal means none of its once-per-goal decisions acted; allow each to retry.
        if bundle.get("refusals"):
            once_issued = {d: mark for d, mark in once_issued.items() if mark[1] != bundle["t"] - 1}
        ctx = builder.update(bundle)
        if model_actions is not None:
            model_actions.at(ctx["goal_version"])
        if ladder is not None:
            ladder.at(ctx["goal_version"])
        if trig is not None:
            trig.at(ctx["goal_version"])
        if watch is not None:
            for e in watch.update(ctx):
                log.append(dict(e))
            if watch.open is not None:
                # A trigger a withheld procedure left armed needs withdrawing; one no longer armed by a procedure
                # (fired, replaced by another method, or dropped with its goal) no longer does. The replica tracks the
                # report trigger, the tanker's one standing order.
                armed = trig.armed
                by_procedure = armed is not None and armed["by"] == "procedure"
                for decision in watch.view.withdraw:
                    w = watch.withdrawal(decision)
                    if not watch.withholds(decision, ctx["goal_version"]):
                        continue
                    if w is None and by_procedure:
                        log.append(dict(watch.withdrawal_needed(decision, armed, ctx["t"])))
                    elif w is not None and w["state"] == "pending" and not by_procedure:
                        log.append(dict(watch.withdrawal_superseded(decision, ctx["t"],
                                                                    "no longer armed by a procedure")))
        unresolved: dict[str, str] = {}
        used_sources: dict[str, tuple[str, str | None]] = {}
        heading_by_procedure = False
        action: dict[str, Any] = {}
        # The task decides which fields complete a once-per-goal decision. The opt-in tanker trigger replica
        # supplies its existing report completion separately; arming that trigger is not a report.
        acted = {d for d, mark in once_issued.items() if mark[0] == ctx["goal_version"]}
        if own_reports and trig.reported(ctx["goal_version"]):
            acted.add("report")
        decisions = roles.asked(acted)
        if broker is not None:
            broker.at(t=ctx["t"], action_step=len(ep.record.actions), goal_version=ctx["goal_version"])
            broker.checkpoint("bundle", bundle=bundle, bundle_sha256=hashlib.sha256(json.dumps(
                bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest())
        # Reply phase (DISTURBANCES section 8): after this second's termination checks, goal changes and reports,
        # which ep.step has already applied, and before the actions. Replies come in the order their calls were made;
        # the first usable part for a decision is the one used.
        replied: dict[str, tuple[str, str, dict]] = {}
        asked_at: dict[str, int] = {}
        if delayed:
            for p in models.due(ctx["t"]):
                got = deliver(p, ctx, "reply phase")
                if got is not None and p.decision not in replied:
                    replied[p.decision] = (p.request_id, p.role, got)
                    asked_at[p.decision] = p.issued_second
        # What procedures, checks and models see: the full context, or with the pack's hidden fields removed under
        # the reduced-feature option. The runtime's own rules (help, the briefing check, own reports) read ctx.
        mctx = {k: v for k, v in ctx.items() if k not in pack.hidden} if pack.hidden else ctx
        ctx_digest = context_sha(mctx)
        writers: dict[str, str] = {}
        for decision in decisions:
            # The source of the part used: procedure, small_model, reasoning_model, or none (Codex's M776.RECORD).
            # A delivered reply is used without running the chooser, so no rival offer is observed for that decision.
            source, request_id = "none", None
            withheld = watch is not None and watch.withholds(decision, ctx["goal_version"])
            if withheld and models_exhausted(ladder, decision):
                log.append(dict(watch.lift(decision, ctx["goal_version"], ctx["t"], "no model can be asked")))
                withheld = False
            set_aside = None
            if decision in replied and withheld and asked_at[decision] < watch.open["opened_at"]:
                # Asked before the incident, without the gap: delivered and charged, but it does not act, and its
                # setting aside is no sign the model failed, so nothing escalates. A fresh request under the incident
                # follows within the same limits (Codex's TURN231 item 2).
                rid, role, _ = replied.pop(decision)
                set_aside = {"request_id": rid, "source": role, "refused": "asked_before_incident",
                             "asked_at": asked_at[decision]}
            if decision in replied:
                request_id, source, part = replied[decision]
                choice = {"use": f"model:{request_id}", "output": part, "tried": [], "checks": [],
                          **({"withheld": watch.open["incident"]} if withheld else {})}
            elif withheld:
                # Withheld before selection: no procedure runs, and the climb goes to a model told the gap.
                choice = {"climb": "unfamiliar_but_solvable", "tried": [], "checks": [],
                          "withheld": watch.open["incident"]}
            elif broker is not None:
                choice = choose(decision, mctx, library, broker=broker, needs=roles.needs, evidence=ctx)
            else:
                choice = (choose(decision, mctx, library, needs=roles.needs, evidence=ctx) if run is None
                          else choose(decision, mctx, library, run=run, needs=roles.needs, evidence=ctx))
            part = choice.get("output")
            if part is not None and source == "none":
                source = "procedure"
            if part is None:
                climbs += 1
                if delayed:
                    asked = {**mctx, "briefing_disagreement": watch.evidence()} if withheld else mctx
                    if model_actions is not None:
                        asked = model_actions.context(decision, asked)
                        role = models.role_for(decision, choice["climb"])
                        asked = {**asked, "reply_delay_s": models.latency_s[role]}
                    issued = models.issue(decision, asked, choice["climb"], ctx["t"], len(ep.record.actions))
                    if issued is not None and issued.available_second == ctx["t"]:
                        # Zero latency: an explicit post-issuance subphase in the same second, since this second's
                        # reply phase has already passed (Codex's TURN 67 condition).
                        for p in models.due(ctx["t"]):
                            got = deliver(p, ctx, "post-issuance")
                            if got is not None and p.decision == decision and part is None:
                                part, source, request_id = got, p.role, p.request_id
                                choice = {**choice, "use": f"model:{p.request_id}"}
                else:
                    part, call = models.decide(decision, mctx, choice["climb"])
                    if call is not None:
                        ledger.add(call)
            reply_checks = set_aside
            waiting = (delayed and part == {} and source in ("small_model", "reasoning_model")
                       and decision in models.wait_decisions)
            if ladder is not None:
                if part is not None and source in ("small_model", "reasoning_model"):
                    # A reply is usable only through a field its own decision may write, valid under the public
                    # action rules (LAD163.FIELD, LAD163.ACTION); otherwise it is unusable and escalates, as a refusal
                    # would, and nothing from it acts.
                    own = own_part(decision, part, fields=decision_fields)
                    why = None if waiting else ("no_own_field" if not own else
                                                part_error(decision, own, task_rules=pack.part_error))
                    if why is None and "help" in own and ladder.help_asked:
                        why = "second_help"           # one help request per episode (the world would refuse it)
                    if why is not None:
                        reply_checks = {"request_id": request_id, "source": source, "refused": why}
                        ladder.unusable(decision, source, ctx["goal_version"], why, ctx["t"])
                        part, source, request_id = None, "none", None
                    elif model_actions is not None:
                        refusal = model_actions.refuse(decision, mctx, own, source, request_id)
                        if refusal is not None:
                            log.append({"kind": "model_action_refused", **refusal})
                            if broker is not None:
                                broker.checkpoint("model_action_refused", **refusal)
                            reply_checks = {"request_id": request_id, "source": source,
                                            "refused": refusal["reason"], "evidence": refusal["evidence"]}
                            ladder.unusable(decision, source, ctx["goal_version"], refusal["reason"], ctx["t"])
                            part, source, request_id = None, "none", None
                control = waiting or (bool(part) and any(k != "help" for k in own_part(
                    decision, part, fields=decision_fields)))
                if part is not None and source in ("small_model", "reasoning_model") and control:
                    # Every output passes the checks that apply to it, a model's reply included (B36). A help-only
                    # reply is information for the person: no learned check judges it, and it neither escalates nor
                    # releases the decision (LAD165.HELP).
                    verdict = (check_output(decision, mctx, part, library, broker=broker) if broker is not None else
                               check_output(decision, mctx, part, library) if run is None else
                               check_output(decision, mctx, part, library, run=run))
                    reply_checks = {"request_id": request_id, "source": source, "checks": verdict["checks"],
                                    "accepted": verdict["accepted"]}
                    if verdict["accepted"]:
                        ladder.usable(decision, ctx["goal_version"], ctx["t"], source)
                    else:
                        # A rejected reply is not used: the decision has no method this second. The rejected request
                        # is kept in reply_checks, not credited as the decision's source.
                        ladder.unusable(decision, source, ctx["goal_version"], "check_rejected", ctx["t"])
                        part, source, request_id = None, "none", None
                elif part and source == "procedure":
                    ladder.usable(decision, ctx["goal_version"], ctx["t"], "procedure")
                if decision == roles.flying and part and source == "procedure":
                    heading_by_procedure = True
                if ((part is None or (not part and not waiting)) and not models.pending_for(decision)
                        and (decision, ctx["goal_version"]) in ladder.no_reasoning):
                    unresolved[decision] = choice.get("climb") or "missing_tool"
            discarded = None
            if isinstance(part, dict) and "goal_version" in part and part["goal_version"] != ctx["goal_version"]:
                discarded, part = part, None          # never overwrite a version the method returned (R22.8)
            out_of_scope = None
            if isinstance(part, dict):
                # A decision writes only its own fields (MS76.ACCEPT); anything else is dropped and recorded.
                allowed = decision_fields[decision] | {"goal_version"}
                if set(part) - allowed:
                    out_of_scope = sorted(set(part) - allowed)
                    part = {k: v for k, v in part.items() if k in allowed}
            log.append({"kind": "decision", "t": ctx["t"], "decision": decision, "goal_version": ctx["goal_version"],
                        "method": choice.get("use") or f"climb:{choice.get('climb')}", "part": part,
                        "tried": choice.get("tried"), "checks": choice.get("checks"),
                        "discarded_for_version": discarded, "source": source, "request_id": request_id,
                        "context_sha256": ctx_digest, "out_of_scope_fields": out_of_scope,
                        **({"reply_checks": reply_checks} if reply_checks is not None else {}),
                        **({"withheld": choice["withheld"]} if "withheld" in choice else {})})
            if broker is not None:
                # The procedures and checks tried, with their outcomes and verdicts in order, and no timings, so
                # the logical prefix covers what was refused and why (Codex's RP61.OUTCOME).
                tried = [{"procedure": a.get("procedure"), "status": a.get("status"), "accepted": a.get("accepted"),
                          "checks": [{"check": c.get("check"), "status": c.get("status"), "verdict": c.get("verdict"),
                                      "accepted": c.get("accepted")} for c in a.get("checks", [])]}
                         for a in (choice.get("tried") or [])]
                broker.checkpoint("decision", decision=decision, method=choice.get("use") or f"climb:{choice.get('climb')}",
                                  goal_version=ctx["goal_version"], part=part, discarded_for_version=discarded, tried=tried, source=source,
                                  request_id=request_id, context_sha256=ctx_digest, out_of_scope_fields=out_of_scope,
                                  **({"withheld": choice["withheld"]} if "withheld" in choice else {}))
            if part:
                used_sources[decision] = (source, request_id)
                for k, v in part.items():
                    if k != "goal_version":
                        action[k] = v
                        writers[k] = decision
        attempted = []
        if watch is not None and watch.open is not None:
            # A pending withdrawal is tried in every second in which no method writes that decision's fields itself,
            # until the world accepts it or it is superseded (Codex's TURN231 item 1).
            for decision, cancel in watch.view.withdraw.items():
                w = watch.withdrawal(decision)
                if w is None or w["state"] != "pending":
                    continue
                if not watch.withholds(decision, ctx["goal_version"]):        # lifted: the procedure has it again
                    log.append(dict(watch.withdrawal_superseded(decision, ctx["t"], "withholding lifted")))
                    continue
                if set(decision_fields[decision]) & set(action):
                    continue
                for k, v in cancel.items():
                    action[k], writers[k] = v, "watch"
                log.append(dict(watch.withdrawal_attempted(decision, cancel, ctx["t"])))
                attempted.append(decision)
        if ladder is not None and "help" not in action:
            ask = ladder.help(ctx, unresolved, heading_by_procedure)
            if ask is not None:
                action["help"] = ask
                writers["help"] = "ladder"
                log.append({"kind": "help_request", "t": ctx["t"], "goal_version": ctx["goal_version"], **ask})
        if action:
            action["goal_version"] = ctx["goal_version"]
        for decision, fields in roles.completion_fields.items():
            if any(field in action and writers.get(field) == decision for field in fields):
                once_issued[decision] = (ctx["goal_version"], ctx["t"])
        # Observed immediately before the action, after this second's deliveries and any zero-latency subphase.
        pending = {roles.flying: bool(delayed and models.pending_for(roles.flying))}
        if broker is not None:
            broker.checkpoint("action", action=action, field_writers=writers, pending=pending)
        bundle = ep.step(action)
        for decision in attempted:
            log.append(dict(watch.withdrawal_verdict(decision, ctx["t"], ep.record.actions[-1]["accepted"])))
        if trig is not None:
            writer = writers.get("report_trigger_nm")
            by = "watch" if writer == "watch" else used_sources.get(writer, (None, None))[0]
            n = len(trig.events)
            trig.after(ctx["t"], ctx, action, ep.record.actions[-1]["accepted"], by)
            for e in trig.events[n:]:
                log.append(dict(e))
        if ladder is not None and "help" in action:
            # Only now is it known whether the help request was made: the world accepts or refuses the whole action.
            by = "ladder" if writers.get("help") == "ladder" else "model"
            ladder.help_outcome(ctx["t"], action["help"], by, None if by == "ladder" else writers.get("help"),
                                ep.record.actions[-1]["accepted"])
        if ladder is not None and action and not ep.record.actions[-1]["accepted"]:
            # The world refused the whole action: a current small-model reply that was part of it never acted, so
            # its decision escalates as if the reply had been refused (LAD163.ACTION). Linked by the writers of this
            # second's action, under the goal version it was sent for.
            for field, decision in writers.items():
                if decision in used_sources and used_sources[decision][0] == "small_model":
                    ladder.unusable(decision, "small_model", action.get("goal_version"), "world_refused", ctx["t"])
        if broker is not None:
            last = ep.record.actions[-1]
            broker.checkpoint("world_step", accepted=last["accepted"], ended=bundle is None,
                              world_t=ep.t, action_steps=len(ep.record.actions))
            prefix_after_step[len(ep.record.actions)] = (broker.event_prefix, broker.logical_prefix)
        if bundle is None:
            break
    if delayed:
        # Replies due after the ending are never delivered; their calls stay charged and straddle the ending.
        for p in models.undelivered():
            rec = {"request_id": p.request_id, "decision": p.decision, "available_second": p.available_second,
                   "issued_second": p.issued_second, "reason": "episode ended before its second"}
            log.append({"kind": "model_undelivered", **rec})
            if broker is not None:
                broker.checkpoint("model_undelivered", **rec)
    outcome = ep.outcome()
    log.append({"kind": "episode_end", "episode": scenario.episode_id, "goal_met": outcome["goal_met"],
                "settled": outcome["settled"], "end_reason": outcome["end_reason"]})
    result = {"outcome": outcome, "climbs": climbs, "actions": [a["actions"] for a in ep.record.actions]}
    if ladder is not None:
        result["ladder"] = list(ladder.events)
    if model_actions is not None:
        result["model_action_refusals"] = list(model_actions.events)
    if watch is not None:
        result["disagreement"] = watch.summary()
    if trig is not None:
        result["own_reports"] = [e for e in trig.events if e["kind"] == "own_report"]
    if broker is not None:
        # The endings the world recorded, before the episode closes, as protected evidence for the cuts below and for
        # attribution (Codex's AT78 cut evidence). The nominal ending is recorded only if the world ended.
        endings = ([("nominal", outcome)] if outcome["end_reason"] is not None else []) + [
            (f"alternative {i}", a) for i, a in enumerate(outcome["alternatives"])]
        for name, ending in endings:
            broker.checkpoint("ending", ending=name, end_reason=ending["end_reason"], ended_at=ending["ended_at"],
                              action_steps=ending["action_steps"])
        broker.end_episode()
        broker.feedback("episode_end", goal_met=bool(outcome["goal_met"]), failed_part=pack.goal.failed(outcome))
        cuts = []
        for name, ending in [("nominal", outcome)] + [(f"alternative {i}", a)
                                                      for i, a in enumerate(outcome["alternatives"])]:
            k = ending["action_steps"]
            prefixes = prefix_after_step.get(k, (None, None))
            cuts.append({"ending": name, "end_reason": ending["end_reason"], "ended_at": ending["ended_at"],
                         "action_steps": k, "event_prefix_sha256": prefixes[0], "logical_prefix_sha256": prefixes[1],
                         **ledger.ending_calls(scenario.episode_id, k)})
        result["ending_cuts"] = cuts
        result["predictions_used"] = broker.predictions_used
    return result
