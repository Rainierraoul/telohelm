"""Learning coordinator: queue observed failures, request candidates and remember permitted feedback.

The protected monitor owns scenarios, reference labels, replay cases, qualification and library changes. Proposal
logic uses its public projections; detailed results are returned only to the host's records. Existing host-side
inspection and fault-injection names remain as explicit compatibility properties at the end of Coordinator.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .qualification.store import CaseStore; from .qualification.store import StoreError; from .qualification.store import case_errors; from .qualification.store import snapshot
from .local_model import first_object
from .runtime import broker as B; from .runtime import episode_loop as L
from .runtime import prediction as PT; from .runtime import sandbox as S
from .runtime.ledger import Ledger
from .runtime.library import Library; from .runtime.library import _write_atomic
from .runtime.model_actions import learner_feedback as model_action_feedback
from .runtime import procedures as R
from .runtime.procedures import CONTEXT_FIELDS; from .runtime.procedures import reply_errors
from .runtime.replay import replay_test
from .runtime.trusted_log import TrustedLog
from .contracts import PublicContract; from .contracts import TaskPack
from .taskpacks.tanker.public import TANKER
from .contracts.goal_parts import GoalParts
from .taskpacks.tanker.execution import TANKER_GOAL
from .taskpacks.tanker.pack import TANKER_PACK
from .taskpacks.tanker.world import episode as E
from .learning_monitor import Monitor; from .learning_monitor import PublicEpisode; from .learning_monitor import Completed; from .learning_monitor import Corpus; from .learning_monitor import FLOOR_WINDOW; from .learning_monitor import FLOOR_RATE; from .learning_monitor import RIGHT_CASES; from .learning_monitor import MIN_RIGHT_CASES; from .learning_monitor import _trial_ok; from .learning_monitor import _regression_ok; from .learning_monitor import _two_facts; from .learning_monitor import reference_solvable; from .learning_monitor import component_trial; from .learning_monitor import _source_sha; from .learning_monitor import _new_evidence_dir; from .learning_monitor import _plain  # compatibility imports for saved readers and host controls

ATTEMPT_EVERY = 5
TRIAL_SET = 8
CALLS_PER_ATTEMPT = 3


def _aliased(text: str, names: dict) -> str:
    """text with every real episode name replaced by its alias (study2/examples.py), longest names first."""
    for real in sorted(names, key=len, reverse=True):
        text = text.replace(real, names[real])
    return text










Proposer = Callable[[list, dict], str]

FEEDBACK_ATTEMPTS = 3          # earlier attempts shown to the proposer
FEEDBACK_MAX_CHARS = 9000      # above this the latest refused source is left out, to stay inside the input cap


# The adaptive learning budget (Simon, 15 October; paper2/notes/ADAPTIVE-LEARNING-BUDGET.md). Level 0 is the
# protocol's budget; the step values are development settings until fixed from development evidence before the freeze.
BUDGET_LEVELS = (
    {"max_output_tokens": 16_384, "timeout_s": 600.0, "effort": "medium"},
    {"max_output_tokens": 24_576, "timeout_s": 1_200.0, "effort": "medium"},
    {"max_output_tokens": 24_576, "timeout_s": 1_800.0, "effort": "xhigh"},   # the pinned template has no "high" (job 4051059)
)
BUDGET_STEP_AFTER = 2          # consecutive unadopted attempts (budget or merit) before a step up
BUDGET_EASE_AFTER = 2          # consecutive adoptions before a step down


class AdaptiveBudget:
    """Telohelm's learning budget for its next attempt. Two attempts in a row that end without adoption, for lack of
    time or tokens or because the replay test or trial refused them, raise the budget one level, up to the ceiling;
    two adoptions in a row lower it one level, down to the protocol's level 0. A format refusal neither counts nor
    resets, since a larger budget does not mend a reply's form. Every level used is recorded with each attempt."""

    def __init__(self, levels: tuple = BUDGET_LEVELS, step_after: int = BUDGET_STEP_AFTER,
                 ease_after: int = BUDGET_EASE_AFTER):
        self.levels, self.step_after, self.ease_after = tuple(levels), step_after, ease_after
        self.level, self.misses, self.hits = 0, 0, 0

    def current(self) -> dict:
        return {"level": self.level, **self.levels[self.level]}

    def observe(self, outcome: str) -> dict | None:
        """outcome: adopted, budget, merit or format. Returns the change, if the level moved."""
        before = self.level
        if outcome == "adopted":
            self.misses, self.hits = 0, self.hits + 1
            if self.hits >= self.ease_after and self.level > 0:
                self.level, self.hits = self.level - 1, 0
        elif outcome in ("budget", "merit"):
            self.hits, self.misses = 0, self.misses + 1
            if self.misses >= self.step_after and self.level < len(self.levels) - 1:
                self.level, self.misses = self.level + 1, 0
        return None if self.level == before else {"from": before, "to": self.level, "after": outcome}




# The proposer's instructions. They state the rules reply_errors and the world enforce that a procedure or check needs,
# and the task as the small model is told it; they give no advice on how to fly. Numbers are taken from the modules
# that enforce them, so the text cannot drift from the checks (13 October: attempt 1 of the first cluster stream was
# refused for using getattr, a rule the prompt had not stated; Codex's MVP119.REPORT then found the report tolerance
# given as 0.15 NM where the world allows 0.25). The prediction tool's interface is stated (BACKLOG B35, after Codex's
# TURN 121 review of the tool), with its limits and its precision, and no way of using it. Not stated, because these
# runs do not use it: the help decision, which the checker accepts.
def learner_limits() -> dict:
    """The core's and the tool implementation's limits that the learner's text states."""
    return {"max_prediction_seconds": B.MAX_PREDICTION_SECONDS, "gap_nm": PT.GAP_NM, "max_evaluations": PT.MAX_EVALUATIONS,
            "prediction_allowance": B.PREDICTION_ALLOWANCE, "max_requests_per_call": B.MAX_REQUESTS_PER_CALL,
            "cpu_seconds": S.CPU_SECONDS, "wall_seconds": S.WALL_SECONDS}


def render_learner_rules(pack, limits: dict) -> str:
    """Telohelm's learner rules: the core's text, the same for every task, joined with the pack's task parts."""
    return (
        f"You improve a library of Python procedures and checks that {pack.learner_purpose}. The library is used "
        "during flight; you are called between episodes, with feedback on the episodes that failed.\n"
        "A procedure defines decide(ctx) and returns a dict holding its decision's action:\n"
        + pack.learner_actions
        + pack.learner_goal
        + pack.learner_tools(limits)
        + "A check defines check(ctx, out), where out is a procedure's returned action, and returns 'accept' or 'reject'.\n"
        "A procedure is used only when its decision also has an adopted check, and only if every applicable adopted check "
        "accepts its output; with no adopted check for its decision, a procedure never runs. A check is adopted only if, "
        "on stored cases you never see, it rejects every wrong answer and accepts at least 95 of 100 right ones, and no "
        "fewer than the check it replaces. The stored wrong answers are well-formed actions that do the wrong thing, so a "
        "check that only tests the format or range of an output accepts them and is refused. A procedure is tried on "
        "episodes already finished, with no model calls: a "
        + pack.learner_trial_rule
        + " You may propose a procedure and its check together as a pair; the check is tested first, then "
        "the procedure with that check in place, and they are adopted together or not at all.\n"
        "previous_attempts lists your last attempts and the stage that refused each: format (your reply broke a rule "
        "below), replay_test (the stored-case test of a check), trial (a procedure's test on finished episodes) or "
        "no_answer; for the latest, your own refused source is included. Nothing about the stored cases is given.\n"
        f"Read ctx and out by attribute, for example {pack.learner_attribute_example}. The context fields are listed in "
        "context_fields.\n"
        f"The source must define decide (for a procedure) or check (for a check) exactly once at top level, be at most "
        f"{R.MAX_SOURCE_CHARS} characters, and import only {', '.join(sorted(R.ALLOWED_IMPORTS))}. No name or "
        "attribute may begin with a double underscore, and no string may contain one. These names may not appear: "
        f"{', '.join(R.FORBIDDEN_NAMES)}.\n"
        "conditions limit when the procedure is used: a list, possibly empty, of {\"field\": <a context field other than "
        f"reports>, \"op\": <one of {' '.join(sorted(R.OPERATORS))}>, \"value\": <number or string>}}; all must hold, "
        "and a missing or null field fails.\n"
        f"reason.code is one of {', '.join(sorted(R.REASON_CODES))}; reason.text is at most {R.MAX_REASON_CHARS} "
        "characters.\n"
        "Reply with one JSON object only, either one entry: "
        f'{{"kind": "procedure" or "check", "decision": {pack.learner_decision_choices}, "conditions": [...], '
        '"source": "<python>", "reason": {"code": "<code>", "text": "<why>"}}, or a pair: '
        f'{{"kind": "pair", "decision": {pack.learner_decision_choices}, "procedure": {{"conditions": [...], "source": "<python>"}}, '
        '"check": {"conditions": [...], "source": "<python>"}, "reason": {"code": "<code>", "text": "<why>"}}.')


# What Telohelm is told, for the tanker pack: the module name is kept for every existing caller (D42 fingerprints it).
PROPOSER_RULES = render_learner_rules(TANKER, learner_limits())


def briefing_disagreement(record: dict) -> dict:
    """Opt-in (runtime/disagreement.py): when the episode's evidence disagreed with its briefing, the learner's feedback
    for it says when that began and how large the gap grew. It is evidence from that episode, not a prior about which
    disturbances exist (Codex's TURN229 ruling 1; recorded under D42). Nothing is added otherwise."""
    watch = record.get("disagreement") or {}
    incidents = watch.get("incidents") or []
    if not incidents:
        return {}
    unit = str(watch.get("unit", "")).lower()
    return {"briefing_disagreement": {"first_at_s": incidents[0]["opened_at"],
                                      f"largest_gap_{unit}": round(max(i["largest"] for i in incidents), 2),
                                      f"tolerance_{unit}": watch.get("tolerance")}}


class Coordinator:
    def __init__(self, library: Library, workdir: Path, propose: Proposer, solvable: Callable[[Any], bool | None],
                 tool: Callable, corpus: Corpus | None = None, attempt_every: int = ATTEMPT_EVERY,
                 trial_set: int = TRIAL_SET, shadow_max_steps: int | None = None,
                 fly: Callable[[Library, "Completed", Path], dict] | None = None, stream: int | None = None,
                 budget: AdaptiveBudget | None = None, max_attempts: int | None = None, extra_rules: str = "",
                 learn_decisions: tuple[str, ...] | None = None, procedures_only: bool = False,
                 floor_after_cold_start: bool = False, no_replay: bool = False, examples: bool = False,
                 candidates: int = 1, regression: int = 0, pack: PublicContract | None = None,
                 goal: GoalParts | None = None, flight_pack: TaskPack = TANKER_PACK):
        # The learner and its trial must use the same task. Explicit legacy arguments may confirm that selection,
        # but cannot silently give the learner different fields or a different goal from its trials.
        if pack is not None and pack != flight_pack.public:
            raise ValueError("learner contract must match the selected flight pack")
        if goal is not None and goal != flight_pack.goal:
            raise ValueError("learner goal must match the selected flight pack")
        pack, goal = flight_pack.public, flight_pack.goal
        if examples and flight_pack.example_policy is None:
            raise ValueError("the selected task must supply an example policy when examples are enabled")
        self.example_rules = flight_pack.example_policy.rules if flight_pack.example_policy is not None else ""
        self.workdir = Path(workdir)
        # The task's public contract (slice 1 of the task pack): the learner's rules are the core's text joined with it.
        self.pack = pack
        self.rules = render_learner_rules(pack, learner_limits())
        # The task's goal in parts (slice 2(a)): which part failed, as the learner is told it, and which decisions
        # that points to.
        self.goal = goal
        # The whole task pack trial flights use, so a trial sees what live flight sees (the reduced-feature option
        # hides derived fields in both).
        self.flight_pack = flight_pack
        # Development settings of the proof-of-concept plan (paper2/notes/PROOF-OF-CONCEPT-PLAN.md, revision 3): a cap
        # on rewrite attempts, and rules added to the proposer's prompt for a run that starts with a trusted guard.
        self.max_attempts, self.extra_rules = max_attempts, extra_rules
        # The decisions this run learns. With heading alone, a missed report is outside learning's scope, so it queues
        # no attempt, and the floor counts joins without help (the step 2 rehearsal found every report-only miss
        # asking the model for another heading procedure).
        self.learn_decisions = tuple(pack.decisions if learn_decisions is None else learn_decisions)
        # Procedures only: a run whose checks are trusted guards takes no checks or pairs from the model. Learning some
        # decisions only implies it (step 2); a run with guards for both decisions sets it (step 3).
        self.procedures_only = procedures_only or set(self.learn_decisions) != set(pack.pair_decisions)
        if flight_pack.hidden and not self.procedures_only:
            # The replay test gives checks the full stored contexts; with fields hidden in flight a check could qualify
            # on inputs it will not have (Codex's TURN237 review). Refused until replay projects through the same view.
            raise ValueError("a pack that hides fields supports procedure-only learning, not learned checks or pairs")
        self.budget = budget              # the adaptive learning budget, or None for a fixed one
        self.propose = propose
        self.attempt_every = attempt_every
        self.queue: list[dict] = []
        # Telohelm without the replay test (protocol section 2, decision D5): a learned check is adopted without the
        # replay before adoption, and replay evidence never retires one. Everything else is kept.
        # Examples from the arm's own completed episodes in each learning request (B7 item 5; study2/examples.py).
        # Off by default, so earlier runs and their prompts are unchanged.
        self.examples = examples
        self._attempt_examples: tuple[list, list] = ([], [])
        # Independent candidates per attempt (21 October 2026, after the step 4 rerun with examples): with more than one,
        # an attempt spends its calls on separate samples of the same request instead of format repairs, trials each
        # valid one and adopts the first that passes. One, the default, keeps the original attempt unchanged.
        if type(candidates) is not int or not 1 <= candidates <= CALLS_PER_ATTEMPT:
            raise ValueError(f"candidates must be an integer from 1 to {CALLS_PER_ATTEMPT}")
        self.candidates = candidates
        # A regression check in the trial (22 October 2026, after job 4070764; paper2/notes/REGRESSION-TRIAL-PLAN.md):
        # with n > 0 a candidate must also keep the goal on up to n earlier episodes the library met by itself. Zero,
        # the default, keeps the trial as before.
        if type(regression) is not int or regression < 0:
            raise ValueError("regression must be a non-negative integer")
        self.history: list[dict] = []     # public outcome of each attempt, fed back to the proposer (job 4028249)
        self.attempts = 0
        self.monitor = Monitor(library, self.workdir, solvable, tool, corpus, pack=pack, goal=goal,
                               flight_pack=flight_pack, learn_decisions=self.learn_decisions, trial_set=trial_set,
                               shadow_max_steps=shadow_max_steps, fly=fly, stream=stream, regression=regression,
                               floor_after_cold_start=floor_after_cold_start, no_replay=no_replay)



    # -- dependencies, read from the library so that a restart keeps them (Codex's PAIR153.RESTORE) ----------------




    # -- after each episode -------------------------------------------------------------------

    def after_episode(self, scenario: Any, outcome: dict, complete: bool) -> list[dict]:
        start = len(self.events)
        c = self.monitor.record(scenario, outcome, complete)
        if c is None:
            return self.events[start:]
        failed = c.failed
        if not c.goal_met:
            concerns = self.goal.concerns(failed)
            if concerns & set(self.learn_decisions):
                self.queue.append({"episode": c.episode, "failed": failed, **briefing_disagreement(outcome),
                                   **model_action_feedback(outcome)})
            else:
                self._event("not_queued", episode=c.episode, failed=failed, reason="outside the decisions learned")
        floor_reason = self.monitor.floor_reason()
        if floor_reason is not None:
            self.queue.append(floor_reason)
        if c.ordinal % self.attempt_every == 0:
            if self.max_attempts is not None and c.ordinal > self.attempt_every * self.max_attempts:
                # With a cap, learning happens only at the first max_attempts boundaries (5, 10, 15 and 20 for a cap of
                # 4); a boundary skipped for want of a reason is not made up later (Codex's POC177.BOUNDARIES).
                self._event("no_attempt", ordinal=c.ordinal, reason="past the last learning boundary")
            elif self.max_attempts is not None and self.attempts >= self.max_attempts:
                self._event("no_attempt", ordinal=c.ordinal, reason="attempt cap")
            elif not self.queue:
                self._event("no_attempt", ordinal=c.ordinal, reason="no queued reason")
            else:
                self.attempt(c)
        return self.events[start:]


    # -- one rewrite attempt --------------------------------------------------------------------




    def messages(self, reasons: list[dict], tail: list | tuple = (), log: dict | None = None) -> list:
        """The proposer's prompt: feedback, the current library and the context fields. No hidden truth. tail is the
        attempt's format repairs so far. With examples on, the attempt's examples are added and fitted to the input
        cap for this exact send, tail included, dropping the oldest first; log, if given, receives what was kept."""
        system = self.rules + ("\n" + self.extra_rules if self.extra_rules else "")
        previous = [dict(h) for h in self.history[-FEEDBACK_ATTEMPTS:]]
        for h in previous[:-1]:
            h.pop("source", None)                  # the refused source of the latest attempt only
        body = {"feedback": reasons, "library": self.monitor.sources(), "context_fields": sorted(self.pack.context_fields)}
        if self.examples:
            from . import examples as X
            system += "\n" + self.example_rules
            names = {episode: X.alias(episode) for episode in self.monitor.episode_names()}
            body["feedback"] = [{**r, "episode": names.get(r.get("episode"), r.get("episode"))} for r in reasons]
            for h in previous:
                if "why" in h:
                    h["why"] = [_aliased(w, names) for w in h["why"]]
        if previous:
            body["previous_attempts"] = previous
        user = json.dumps(body, sort_keys=True)
        if len(user) > FEEDBACK_MAX_CHARS and previous and "source" in previous[-1]:
            # Keep within the input cap: the latest refused source goes first, and its omission is stated.
            previous[-1] = {**{k: v for k, v in previous[-1].items() if k != "source"}, "source_omitted": True}
            user = json.dumps({**body, "previous_attempts": previous}, sort_keys=True)
        if self.examples:
            from . import examples as X
            shown, receipts = self._attempt_examples
        else:
            shown, receipts = [], []
        keep = list(range(len(shown)))
        # A server's admission counter may probe its endpoint. Fitting uses its
        # separate pure local counter, when supplied, and never relaxes admission.
        count = getattr(self.propose, "fit_tokens", getattr(self.propose, "count_tokens", None))
        cap = getattr(self.propose, "input_cap", None)
        base = {**json.loads(user)}

        def render(kept):
            body = {**base, "examples": [shown[i] for i in kept]} if self.examples else base
            return [("system", system), ("user", json.dumps(body,
                                                            sort_keys=True))] + list(tail)

        def fitted_count(messages):
            if not callable(count) or type(cap) is not int:
                return None
            try:
                value = count(messages)
                if type(value) is not int or value < 0:
                    raise ValueError("input token count must be a nonnegative integer")
                return value
            except Exception as exc:
                if log is not None:
                    log["token_fit_error"] = type(exc).__name__
                return None                    # normal admission retains/refuses an uncountable request

        messages = render(keep)
        dropped = []
        tokens = fitted_count(messages)
        source_omitted = False
        history = base.get("previous_attempts") or []
        if tokens is not None and tokens > cap and history and "source" in history[-1]:
            history[-1] = {**{k: v for k, v in history[-1].items() if k != "source"}, "source_omitted": True}
            source_omitted = True
            messages = render(keep)
            tokens = fitted_count(messages)
        if self.examples and tokens is not None and tokens > cap:
            for i in X.drop_order(receipts):
                if tokens is None or tokens <= cap:
                    break
                keep.remove(i)
                dropped.append(i)
                messages = render(keep)
                tokens = fitted_count(messages)
        if log is not None:
            log.update(kept=[receipts[i] for i in keep], dropped=[receipts[i] for i in dropped],
                       input_tokens=tokens, source_omitted_for_tokens=source_omitted)
        return messages

    def _scope_errors(self, reply: dict) -> list[str]:
        """A run that learns some decisions only takes procedures for those decisions and nothing else, so its installed
        checks (the trusted heading guard) stay as they are. Anything else is refused before any trial or library
        change, as a format problem within the attempt's own calls (Codex's POC177.SCOPE). The default run learns both
        decisions and every kind, unchanged."""
        if not self.procedures_only:
            return []
        if reply.get("kind") != "procedure" or reply.get("decision") not in self.learn_decisions:
            wanted = " or ".join(f'"{d}"' for d in self.learn_decisions)
            return [f'this run takes only kind "procedure" with decision {wanted}']
        return []

    def attempt(self, boundary: PublicEpisode) -> dict:
        self.reconcile_pairs()            # finish any interrupted change before anything new is tried
        self.attempts += 1
        n = self.attempts
        eligible = self.monitor.begin_attempt(n, boundary, self.queue)
        reasons, self.queue = self.queue, []
        # The attempt's budget comes from the original episode's shared reasoning allowance where there is one
        # (decision D26): min(3, what flight left). Without an allowance authority it is the attempt limit.
        budget_for = getattr(self.propose, "attempt_budget", None)
        budget = budget_for(boundary.episode) if callable(budget_for) else CALLS_PER_ATTEMPT
        if budget <= 0:
            return self._event("attempt_refused", attempt=n, reason="reasoning allowance spent")
        if not eligible:
            return self._event("attempt_refused", attempt=n, reason="no eligible trial set")
        if self.budget is not None and callable(getattr(self.propose, "use", None)):
            self.propose.use(self.budget.current())
            self._event("attempt_budget", attempt=n, episode=boundary.episode, **self.budget.current())
        if self.examples:
            self._select_examples(n, reasons)
        if self.candidates > 1:
            return self._attempt_candidates(n, budget, boundary, reasons)
        reply, calls, tail = None, 0, []
        errors, text = [], ""
        while calls < budget:
            calls += 1
            fitted: dict = {}
            # Built afresh for every send, so a format repair is fitted to the cap with its examples (TURN 205).
            messages = self.messages(reasons, tail, fitted)
            if self.examples:
                self._log_examples(n, calls, fitted)
            text = self.propose(messages, {"attempt": n, "call": calls, "episode": boundary.episode})
            try:
                candidate = first_object(text)
                errors = reply_errors(candidate, pack=self.pack) or self._scope_errors(candidate)
            except ValueError as e:
                candidate, errors = None, [str(e)]
            if not errors:
                reply = candidate
                break
            tail = tail + [("assistant", text), ("user", "Fix these problems and reply again: " + "; ".join(errors))]
        if reply is None:
            event = self._event("attempt_failed", attempt=n, calls=calls, reason="no valid candidate")
            if not (text or "").strip():
                status = self._last_status()
                self._remember(n, None, None, "no_answer", [status or "no reply"], None)
            else:
                self._remember(n, None, None, "format", errors, None)
            return event
        return self.monitor.host_event(self._submit(n, calls, reply, boundary, reasons).event_index)

    def _submit(self, n, calls, reply, boundary, reasons, call=None):
        verdict = self.monitor.submit(n, calls, reply, boundary, reasons, call=call)
        if verdict.feedback is None:
            return verdict
        source = ({"procedure": reply["procedure"]["source"], "check": reply["check"]["source"]}
                  if reply["kind"] == "pair" else reply["source"])
        self._remember_verdict(verdict.feedback, source)
        return verdict

    def _try_candidate(self, n, calls, reply, boundary, trial, reasons, call=None):
        # Earlier host-side tests supply their declared trial directly. Normal attempts use begin_attempt.
        self.monitor._attempt_trial = trial
        return self.monitor.host_event(self._submit(n, calls, reply, boundary, reasons, call=call).event_index)

    def _attempt_candidates(self, n: int, budget: int, boundary: PublicEpisode, reasons: list) -> dict:
        """Up to min(candidates, budget) separate samples of one request, built once so that no candidate sees
        another's outcome. Malformed replies are not repaired; valid ones are tried in order until one passes."""
        fitted: dict = {}
        messages = self.messages(reasons, (), fitted)
        tried = []
        k = min(self.candidates, budget)
        for call in range(1, k + 1):
            if self.examples:
                self._log_examples(n, call, fitted)       # the same request each call, logged for each (TURN 207)
            text = self.propose(messages, {"attempt": n, "call": call, "episode": boundary.episode})
            try:
                reply = first_object(text)
                errors = reply_errors(reply, pack=self.pack) or self._scope_errors(reply)
            except ValueError as e:
                reply, errors = None, [str(e)]
            if errors:
                stage = "format" if (text or "").strip() else "no_answer"
                self._event("candidate_refused", attempt=n, call=call, stage=stage, errors=errors)
                self._remember(n, None, None, stage, errors if stage == "format" else [self._last_status() or "no reply"],
                               None)
                continue
            verdict = self._submit(n, call, reply, boundary, reasons, call=call)
            if verdict.candidate is not None:
                tried.append(verdict.candidate)
            if verdict.passed:
                return self.monitor.host_event(verdict.event_index)
        return self._event("attempt_failed", attempt=n, calls=k, reason="no candidate passed",
                           tried=tried)

    def _select_examples(self, n: int, reasons: list[dict]) -> None:
        self._attempt_examples = self.monitor.examples(n, reasons)

    def _log_examples(self, n: int, call: int, fitted: dict) -> None:
        """The host-side record of which examples one send carried. The learner never sees it."""
        with (self.workdir / "learning-examples.jsonl").open("a") as f:
            f.write(json.dumps({"attempt": n, "call": call, "stage": "sent",
                                "kept": [(r["episode"], r["second"]) for r in fitted.get("kept", [])],
                                "dropped_for_cap": [(r["episode"], r["second"]) for r in fitted.get("dropped", [])]},
                               sort_keys=True) + "\n")

    # -- what the proposer is told about its earlier attempts (job 4028249; LEARNING-FEEDBACK-PROPOSAL.md) -------
    def _last_status(self) -> str | None:
        if hasattr(self.propose, "last_status"):
            return self.propose.last_status      # belongs to the current logical request, including an unsent one
        ledger = getattr(self.propose, "ledger", None)
        calls = getattr(ledger, "calls", None) or []
        return calls[-1].status if calls else None

    def _remember(self, n: int, kind: str | None, decision: str | None, stage: str, why: list[str],
                  source: Any) -> None:
        self.history.append({"attempt": n, "kind": kind, "decision": decision, "refused_at": stage, "why": why,
                             **({"source": source} if source is not None else {})})
        outcome = {"format": "format", "replay_test": "merit", "trial": "merit"}.get(stage)
        if stage == "no_answer":
            outcome = "budget" if any(w in ("timed_out", "cut_off") for w in why) else "format"
        self._budget_outcome(n, outcome)

    def _budget_outcome(self, n: int, outcome: str | None) -> None:
        if self.budget is None or outcome is None:
            return
        change = self.budget.observe(outcome)
        if change is not None:
            self._event("budget_changed", attempt=n, **change, now=self.budget.current())

    def _remember_verdict(self, feedback, source):
        if feedback is None:
            return
        n, kind, decision = feedback["attempt"], feedback["kind"], feedback["decision"]
        if feedback["adopted"]:
            self.history.append({"attempt": n, "kind": kind, "decision": decision, "adopted": True})
            self._budget_outcome(n, "adopted")
        else:
            self._remember(n, kind, decision, feedback["stage"], list(feedback["why"]), source)

    def _remember_result(self, event, source):
        # Host compatibility for earlier analyses; proposal logic receives Verdict.feedback instead.
        self._remember_verdict(self.monitor.feedback(event), source)






    # -- undo on new evidence ---------------------------------------------------------------------







    def _floor(self):
        reason = self.monitor.floor_reason()
        if reason is not None:
            self.queue.append(reason)

    # Host diagnostics compatibility. These are not used to construct proposals; the monitor owns the values.
    stream = property(lambda self: self.monitor.stream,
                       lambda self, value: setattr(self.monitor, "stream", value))
    trial_set_size = property(lambda self: self.monitor.trial_set_size,
                       lambda self, value: setattr(self.monitor, "trial_set_size", value))
    shadow_max_steps = property(lambda self: self.monitor.shadow_max_steps,
                       lambda self, value: setattr(self.monitor, "shadow_max_steps", value))
    floor_after_cold_start = property(lambda self: self.monitor.floor_after_cold_start,
                       lambda self, value: setattr(self.monitor, "floor_after_cold_start", value))
    no_replay = property(lambda self: self.monitor.no_replay,
                       lambda self, value: setattr(self.monitor, "no_replay", value))
    regression = property(lambda self: self.monitor.regression,
                       lambda self, value: setattr(self.monitor, "regression", value))
    library = property(lambda self: self.monitor.library,
                       lambda self, value: setattr(self.monitor, "library", value))
    completed = property(lambda self: self.monitor.completed,
                       lambda self, value: setattr(self.monitor, "completed", value))
    corpus = property(lambda self: self.monitor.corpus,
                       lambda self, value: setattr(self.monitor, "corpus", value))
    solvable = property(lambda self: self.monitor.solvable,
                       lambda self, value: setattr(self.monitor, "solvable", value))
    tool = property(lambda self: self.monitor.tool,
                       lambda self, value: setattr(self.monitor, "tool", value))
    fly = property(lambda self: self.monitor.fly,
                       lambda self, value: setattr(self.monitor, "fly", value))
    armed = property(lambda self: self.monitor.armed,
                       lambda self, value: setattr(self.monitor, "armed", value))
    breaches = property(lambda self: self.monitor.breaches,
                       lambda self, value: setattr(self.monitor, "breaches", value))
    _regression_set = property(lambda self: self.monitor._regression_set,
                       lambda self, value: setattr(self.monitor, "_regression_set", value))
    _candidate = property(lambda self: self.monitor._candidate,
                       lambda self, value: setattr(self.monitor, "_candidate", value))
    events = property(lambda self: self.monitor.events,
                       lambda self, value: setattr(self.monitor, "events", value))
    _event = property(lambda self: self.monitor._event,
                       lambda self, value: setattr(self.monitor, "_event", value))
    replaced = property(lambda self: self.monitor.replaced,
                       lambda self, value: setattr(self.monitor, "replaced", value))
    pair_of = property(lambda self: self.monitor.pair_of,
                       lambda self, value: setattr(self.monitor, "pair_of", value))
    _change = property(lambda self: self.monitor._change,
                       lambda self, value: setattr(self.monitor, "_change", value))
    _is_guard = property(lambda self: self.monitor._is_guard,
                       lambda self, value: setattr(self.monitor, "_is_guard", value))
    _learned = property(lambda self: self.monitor._learned,
                       lambda self, value: setattr(self.monitor, "_learned", value))
    _sources = property(lambda self: self.monitor.sources,
                       lambda self, value: setattr(self.monitor, "sources", value))
    trial = property(lambda self: self.monitor.trial,
                       lambda self, value: setattr(self.monitor, "trial", value))
    regression_set = property(lambda self: self.monitor.regression_set,
                       lambda self, value: setattr(self.monitor, "regression_set", value))
    _fly = property(lambda self: self.monitor._fly,
                       lambda self, value: setattr(self.monitor, "_fly", value))
    check_replay = property(lambda self: self.monitor.check_replay,
                       lambda self, value: setattr(self.monitor, "check_replay", value))
    fresh_set = property(lambda self: self.monitor.fresh_set,
                       lambda self, value: setattr(self.monitor, "fresh_set", value))
    _attempt_pair = property(lambda self: self.monitor._attempt_pair,
                       lambda self, value: setattr(self.monitor, "_attempt_pair", value))
    _adopt_pair = property(lambda self: self.monitor._adopt_pair,
                       lambda self, value: setattr(self.monitor, "_adopt_pair", value))
    _pair_log = property(lambda self: self.monitor._pair_log,
                       lambda self, value: setattr(self.monitor, "_pair_log", value))
    reconcile_pairs = property(lambda self: self.monitor.reconcile_pairs,
                       lambda self, value: setattr(self.monitor, "reconcile_pairs", value))
    confirm_wrong = property(lambda self: self.monitor.confirm_wrong,
                       lambda self, value: setattr(self.monitor, "confirm_wrong", value))





PROTOCOL_OUTPUT_TOKENS = 16_384          # the reasoning model's output cap per call (protocol section 5)

class GatewayProposer:
    """The reasoning model behind its own gateway, one per rewrite attempt, bound to the completed episode that
    prompted it (no aircraft-control second). At most 4 reasoning calls per such gateway, as the protocol allows per
    episode; the coordinator allows 3 per attempt. Costs go to one learning ledger, reported apart from flying, with the
    purpose 'build' and the completed episode that prompted the attempt as their episode (Codex's ACC133.BUILD).
    Output is capped at the protocol's 16,384 tokens and 600 seconds per call (Experiment 2 protocol, section 5);
    until 13 October it was 8,192 by mistake, and 10 of the first cluster stream's 12 calls stopped there.

    Input is capped at the protocol's 4,096 tokens per request, counted before the request is sent with the pinned
    model's own template and tokenizer (token_count.py; Codex's MVP119.INPUT), including every earlier message of a
    repair. A request over the cap, or one that cannot be counted, is not sent: the proposer returns no text, so the
    coordinator counts it as a failed call, and preflight.jsonl in the attempt's folder records every count, sent or
    not. No tokens are charged for a request that was not sent."""

    def __init__(self, base_url: str, model: str, workdir: Path, stream: str, condition: str = "telohelm",
                 max_output_tokens: int = PROTOCOL_OUTPUT_TOKENS, timeout_s: float = 600.0, thinking: bool = True,
                 count_tokens: Callable[[list], int] | None = None, allowance: Any = None,
                 effort: str | None = None, input_cap: int | None = None):
        from . import local_model as local_model; from . import token_count as token_count
        self.allowance = allowance              # ladder.ReasoningAllowance, shared with flight (D26), or None
        self.thinking, self.effort, self.timeout_s, self.base_url = thinking, effort, timeout_s, base_url
        self._custom_counter = count_tokens is not None
        self.provider = local_model.provider(base_url, model, timeout_s, thinking=thinking, role="reasoning_model",
                                             effort=effort)
        self.workdir, self.stream, self.condition = Path(workdir), stream, condition
        self.max_output_tokens = max_output_tokens
        self.ledger = Ledger()
        self.last_status = None
        self.gateways: dict[int, Any] = {}
        kwargs = (local_model.SETTINGS["reasoning_model"]["chat_template_kwargs"] if thinking
                  else {"enable_thinking": False})
        if thinking and effort is not None:
            kwargs = {**kwargs, "reasoning_effort": effort}       # counted with the template actually sent
        self.count_tokens = count_tokens or (lambda messages: token_count.count(messages, kwargs))
        self.model = model
        # The protocol's cap unless a development run sets another (the proof-of-concept plan's step 2).
        self.input_cap = token_count.INPUT_TOKEN_CAP if input_cap is None else input_cap

    def __call__(self, messages: list, meta: dict) -> str:
        from .runtime import models as M
        self.last_status = None
        n = meta["attempt"]
        d = self.workdir / f"learning-attempt-{n}"
        # The same logical request (attempt, call, episode) asked again returns its first answer and makes no new
        # model call; a different message under the same identity is refused (Codex's ALLOW143.REPEAT). Answers are
        # kept on disk, so a restart recovers them too.
        logical = json.dumps([meta.get("episode"), n, meta.get("call")])
        fingerprint = hashlib.sha256(json.dumps([list(m) for m in messages]).encode()).hexdigest()
        config = self._config()
        answers = self.workdir / "learning-answers.jsonl"
        if answers.exists():
            for row in (json.loads(l) for l in answers.read_text().splitlines()):
                if row["logical"] == logical:
                    if row["fingerprint"] != fingerprint:
                        raise ValueError(f"learning request {logical} was made with different messages")
                    if row.get("config") != config:
                        # A saved answer belongs to its condition, stream, model, output limit and purpose
                        # (Codex's ALLOW145.CONFIG).
                        raise ValueError(f"learning request {logical} was made under another configuration")
                    saved_calls = row.get("ledger_rows", [])
                    status = saved_calls[-1]["status"] if saved_calls else (
                        "preflight_refused" if not row["sent"] else "no_reply")
                    if "call_status" in row and row["call_status"] != status:
                        raise ValueError(f"learning request {logical} has contradictory saved status")
                    self.last_status = status
                    # The original call's charge comes back into this ledger once, unchanged, with no new call or
                    # allocation (Codex's ALLOW145.COST.FRESH); unknown values stay unknown.
                    from .runtime.ledger import Call
                    have = {c.request_id for c in self.ledger.calls}
                    for saved in row.get("ledger_rows", []):
                        if saved["request_id"] not in have:
                            self.ledger.add(Call(**{**saved, "gpus": tuple(saved["gpus"])
                                                    if saved.get("gpus") is not None else None}))
                    self.persist_costs()
                    return row["text"]
        if n not in self.gateways:
            d.mkdir(parents=True)
            self.gateways[n] = M.ModelGateway(meta["episode"], self.condition, self.stream, self.provider,
                                              TrustedLog(d / "gateway.log", d / "gateway.heads", self.stream),
                                              self.ledger, d / "journal",
                                              limits={"reasoning_model": 4, "small_model": 0}, purpose="build",
                                              id_scope=f"build/attempt{n}")
        messages = [tuple(m) for m in messages]
        try:
            tokens = self.count_tokens(messages)
            if type(tokens) is not int or tokens < 0:
                raise ValueError("input token count must be a nonnegative integer")
            error = None
        except Exception as exc:                  # an uncountable request is refused, never sent uncounted
            tokens, error = None, type(exc).__name__
        sent = tokens is not None and tokens <= self.input_cap
        ordinal = None
        if self.allowance is not None:
            # One allocation per logical request: a refused preflight takes one and costs nothing; with no room
            # left the request is not made and takes none (decision D26, item 3).
            key = self.allowance.key(self.condition, self.stream, meta["episode"])
            ordinal = self.allowance.allocate(key, f"build/attempt{n}/call{meta.get('call')}", "build",
                                              "provider" if sent else "preflight_refused")
            if ordinal is None:
                sent = False
        with open(d / "preflight.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"attempt": n, "call": meta.get("call"), "input_tokens": tokens, "cap": self.input_cap,
                                "sent": sent, "error": error, "allocation": ordinal,
                                "no_allowance": self.allowance is not None and ordinal is None},
                               sort_keys=True) + "\n")
        text = ""
        before = len(self.ledger.calls)
        if sent:
            out = self.gateways[n].submit("reasoning_model", messages, self.max_output_tokens, 1, 0, 0, 0.0)
            text = out.get("text") or ""
        from dataclasses import asdict
        charged = [asdict(c) for c in self.ledger.calls[before:]]
        self.last_status = charged[-1]["status"] if charged else ("preflight_refused" if not sent else "no_reply")
        with open(answers, "a", encoding="utf-8") as f:
            f.write(json.dumps({"logical": logical, "fingerprint": fingerprint, "sent": sent, "allocation": ordinal,
                                "config": config, "ledger_rows": charged, "text": text, "call_status": self.last_status},
                               sort_keys=True, default=list) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.persist_costs()
        return text

    def persist_costs(self) -> None:
        """Keep completed charges after every answer, including cached answers.

        A request is charged once. Earlier rows survive a later instance; a
        conflicting row stops the caller rather than replacing the first charge.
        In-flight requests remain in their gateway logs with unknown usage.
        """
        from dataclasses import asdict
        path = self.workdir / "learning-ledger.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        saved = {r["request_id"]: r for r in rows}
        if len(saved) != len(rows):
            raise ValueError("duplicate request in saved learning ledger")
        for call in self.ledger.calls:
            row = json.loads(json.dumps(asdict(call), default=str))
            if row["request_id"] in saved:
                if saved[row["request_id"]] != row:
                    raise ValueError("conflicting charge in saved learning ledger")
            else:
                saved[row["request_id"]] = row
                rows.append(row)
        _write_atomic(path, "".join(json.dumps(r, sort_keys=True, default=str) + "\n"
                                    for r in rows).encode())


def _attempt_budget(self, episode: str) -> int:
    if self.allowance is None:
        return CALLS_PER_ATTEMPT
    return min(CALLS_PER_ATTEMPT, self.allowance.attempt_budget(self.allowance.key(self.condition, self.stream, episode)))


GatewayProposer.attempt_budget = _attempt_budget


def _config(self) -> dict:
    """What a saved learning answer is bound to."""
    return {"condition": self.condition, "stream": self.stream, "model": self.model,
            "max_output_tokens": self.max_output_tokens, "purpose": "build", "thinking": self.thinking,
            **({"effort": self.effort} if getattr(self, "effort", None) is not None else {}),
            **({"timeout_s": self.timeout_s} if getattr(self, "timeout_s", 600.0) != 600.0 else {}),
            **({"input_cap": self.input_cap} if getattr(self, "input_cap", 4_096) != 4_096 else {})}


GatewayProposer._config = _config


def _use(self, level: dict) -> None:
    """Set the next attempt's budget (adaptive learning budget): output cap, call timeout and reasoning effort. The
    provider and the token counter are rebuilt so the template counted is the one sent."""
    from . import local_model as local_model; from . import token_count as token_count
    self.max_output_tokens, self.timeout_s, self.effort = level["max_output_tokens"], level["timeout_s"], level["effort"]
    self.provider = local_model.provider(self.base_url, self.model, self.timeout_s, thinking=self.thinking,
                                         role="reasoning_model", effort=self.effort)
    if self.thinking and not getattr(self, "_custom_counter", False):
        kwargs = {**local_model.SETTINGS["reasoning_model"]["chat_template_kwargs"], "reasoning_effort": self.effort}
        self.count_tokens = lambda messages: token_count.count(messages, kwargs)


GatewayProposer.use = _use








