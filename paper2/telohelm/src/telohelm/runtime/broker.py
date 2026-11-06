"""The trusted broker (architecture sections 6, 9 and 10; paper2/notes/BROKER-CONTRACT.md).

Code a model wrote never touches the world, the logs, the ledger, the clock or the monitor. During a procedure call
it can only send JSON request messages to the broker and read JSON replies. The broker runs in the trusted runtime.
It checks each request against a fixed public schema, answers or refuses with a public refusal code, counts
all tool calls against the episode's allowance, and writes every event to the trusted log itself, with the
action step and a running fingerprint of the episode's event sequence (backlog note B10). After the episode it passes
on the goal monitor's restricted feedback, and nothing else.

The trusted setup supplies a fixed table of tools; omitting it retains the historical prediction callable. The
channel carries JSON text over the contained procedure’s broker pipe. Event names retain the historical
``prediction_*`` spelling; the saved request and request ID identify the actual capability.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Any, Callable

from ..taskpacks.tanker.execution import TANKER_GOAL
from ..taskpacks.tanker.tools import MAX_PREDICTION_SECONDS; from ..taskpacks.tanker.tools import predict_tool  # noqa: F401 (kept for its readers)
from ..taskpacks.tanker.tools import result_ok as _result_ok              # noqa: F401 (kept for its readers)
from ..contracts.tools import Tool
from .ledger import Call; from .ledger import Ledger
from .trusted_log import TrustedLog

PREDICTION_ALLOWANCE = 1100            # calls per episode, shared by all conditions (D13, revised 11 October)
MAX_MESSAGE_BYTES = 65_536             # a larger request is refused unread
MAX_REQUESTS_PER_CALL = 256            # requests one procedure call may make, answered or refused
MAX_REPLY_BYTES = 4_194_304            # a reply above this is refused, never truncated; a full-horizon reply is
                                       # about 2.2 MB (D19)

# The only reasons the goal monitor may give for an undo (Codex's TURN 55: a schema alone cannot stop meaning
# leaking through free text). Each says only which public test failed.
UNDO_REASONS = ("replay test failed", "trial on completed episodes failed", "rewrite allowance exceeded",
                "procedure refused by format checks")

REFUSALS = ("malformed", "too_large", "unknown_capability", "stale_goal_version", "over_allowance",
            "request_limit", "call_closed", "tool_failed", "reply_too_large")

# A tool's request form and reply check come with the tool (the tool table, packs/tools.py); the tanker's prediction
# tool, its form, its heading-list limit and its reply check are in packs/tanker_tools.py.


FEEDBACK = {                           # the only feedback the monitor can pass to Telohelm (section 9)
    "episode_end": {"goal_met": "bool", "failed_part": TANKER_GOAL.labels()},   # a broker may be given its pack's
    "replay_test": {"entry": "sha", "passed": "bool"},
    "trial": {"entry": "sha", "passed": "bool"},
    "undo": {"entry": "sha", "reason": UNDO_REASONS},
}


# Events that describe what happened, as opposed to how it was accounted. The logical prefix chains only these, without
# their sequence numbers, so a replay of the same inputs and actions has the same logical prefix even though its
# accounting events (replay mode, call bookkeeping) differ (Codex's TURN 59).
LOGICAL_KINDS = ("briefing", "bundle", "decision", "action", "world_step", "prediction_start", "prediction_end",
                 "prediction_interrupted", "reply", "call_close", "feedback", "episode_end", "model_reply",
                 "model_undelivered", "ending")
# TURN 79 (Codex's AT78 cut evidence): each ending the world recorded is a logical checkpoint, so a share-of-seconds
# cut can be checked against the protected log instead of trusting the caller's step count.
# TURN 62 (Codex's RP61.OUTCOME): procedure outcomes (call_close), prediction outcomes and the monitor's public
# feedback are logical too. The monitor's restricted outcomes live in its own store and are outside this comparison.


class BrokerError(ValueError):
    """A trusted-side misuse, such as monitor feedback outside the schema. Never shown to model-written code."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _is_number(v: Any) -> bool:
    """A finite JSON number. An integer too large for a double is refused, not raised (Codex's BR51.2)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    """JSON objects with a repeated key are refused, so a request cannot say two things (BR51.7)."""
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def _check(value: Any, rule: Any) -> bool:
    """Type first, then range. A dict rule means exactly those keys; a callable rule is a tool's own check."""
    if callable(rule):
        return bool(rule(value))
    if isinstance(rule, (dict, MappingProxyType)):
        return isinstance(value, dict) and set(value) == set(rule) and all(_check(value[k], r) for k, r in rule.items())
    if isinstance(rule, tuple):
        return (value is None or isinstance(value, str)) and value in rule
    if rule == "number":
        return _is_number(value)
    if rule == "nonneg":
        return _is_number(value) and value >= 0
    if rule == "positive":
        return _is_number(value) and value > 0
    if rule == "int":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if rule == "bool":
        return isinstance(value, bool)
    if rule == "pair":
        return isinstance(value, list) and len(value) == 2 and all(_is_number(v) for v in value)
    if rule == "sha":
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if rule == "short_text":
        return isinstance(value, str) and 0 < len(value) <= 200
    raise BrokerError(f"unknown rule {rule!r}")


class ProcedureCall:
    """One procedure call's view of the broker. request() is the only thing model-written code reaches."""

    def __init__(self, broker: "Broker", call_id: str, purpose: str):
        self._broker, self.call_id, self.purpose = broker, call_id, purpose
        self.requests, self.closed = 0, False

    def request(self, text: str) -> str:
        return self._broker._handle(self, text)

    def __enter__(self) -> "ProcedureCall":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._broker._close(self, "interrupted" if exc_type is not None else "completed",
                            None if exc_type is None else f"{exc_type.__name__}: {exc}")
        return False


class Broker:
    """One episode's broker. The runtime sets the simulated second, action step and goal version before each
    procedure call. replay marks a replay-test episode; it never changes a reply (live/replay parity)."""

    def __init__(self, episode: str, log: TrustedLog, ledger: Ledger, predict: Callable[[dict], dict],
                 replay: bool = False, allowance: int = PREDICTION_ALLOWANCE, clock: Callable[[], float] = time.perf_counter,
                 condition: str | None = None, stream: str | None = None,
                 cpu_clock: Callable[[], float] = time.process_time, goal: Any = None,
                 tools: Any = None):
        """cpu_clock measures the trusted tool's processor time in this process; a sandboxed procedure's own
        processor time is measured separately, from its child process, so neither is counted twice."""
        self.episode, self._log, self._ledger, self._predict = episode, log, ledger, predict
        # The tools procedures may call, by capability name (packs/tools.py). Without a table, the tanker's prediction
        # tool with the function given, as before; every tool draws on the one allowance below.
        registry = dict(tools) if tools is not None else {"predict": predict_tool(predict)}
        if any(not isinstance(k, str) or not k or not isinstance(v, Tool) for k, v in registry.items()):
            raise ValueError("tools must map capability names to trusted Tool entries")
        self._tools = MappingProxyType(registry)
        # The labels the episode-end feedback may carry come from the task's goal (its pack); the rest of the
        # schema is the core's. Without a goal, the tanker's labels, as before.
        self._feedback_schema = FEEDBACK if goal is None else {
            **FEEDBACK, "episode_end": {"goal_met": "bool", "failed_part": goal.labels()}}
        self.condition, self.stream, self._cpu_clock = condition, stream, cpu_clock
        self._replay, self.allowance, self._clock = replay, allowance, clock
        self.predictions_used = 0
        self.t, self.action_step, self.goal_version = 0, 0, 0
        self.ended = False
        self._seq, self._prefix, self._calls = 0, "0" * 64, 0
        self._logical = "0" * 64
        self._feedback: list[dict] = []
        self._open: list[ProcedureCall] = []
        self._event({"kind": "broker_open", "allowance": allowance, "replay": replay, "condition": condition,
                     "stream": stream})

    # --- trusted runtime side -------------------------------------------------------------------------------
    def at(self, t: int, action_step: int, goal_version: int) -> None:
        if (not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (t, action_step, goal_version))
                or action_step < self.action_step or t < self.t or goal_version < self.goal_version):
            raise BrokerError("simulated time, action step and goal version must be non-negative and never go back")
        self.t, self.action_step, self.goal_version = t, action_step, goal_version

    def open_call(self, purpose: str) -> ProcedureCall:
        if self.ended:
            raise BrokerError("the episode has ended")
        self._calls += 1
        call = ProcedureCall(self, f"{self.episode}/call{self._calls}", purpose)
        self._open.append(call)
        self._event({"kind": "call_open", "call": call.call_id, "purpose": purpose})
        return call

    @property
    def ledger(self) -> Ledger:
        """The ledger this broker charges, so the runtime can refuse a mismatch (Codex's BR57.LEDGER)."""
        return self._ledger

    def checkpoint(self, kind: str, **fields: Any) -> None:
        """Trusted side: put a world or decision event into the broker's ordered event sequence, so the event-prefix
        fingerprint commits to what happened, not only to broker traffic (Codex's BR57.PREFIX)."""
        if kind not in ("briefing", "bundle", "decision", "action", "world_step", "model_reply", "model_undelivered",
                        "model_action_refused", "ending"):
            raise BrokerError(f"unknown checkpoint {kind}")
        _canonical(fields)
        self._event({"kind": kind, **fields})

    def close_call(self, call: ProcedureCall, status: str, error: str | None = None) -> None:
        """Trusted side: close a call with the sandbox's outcome (for example 'timeout' or 'bad_reply')."""
        self._close(call, status, error)

    def end_episode(self) -> None:
        """Ends the episode and closes every procedure call still open, so none can predict afterwards (BR51.4)."""
        for call in list(self._open):
            self._close(call, "episode_ended", None)
        self.ended = True
        self._event({"kind": "episode_end", "predictions_used": self.predictions_used})

    def feedback(self, kind: str, **fields: Any) -> None:
        """The monitor's restricted feedback. Anything outside the schema is a trusted-side error and is not passed
        on, so a fault in the monitor cannot leak a label or the truth through this channel."""
        if kind not in self._feedback_schema or not _check(fields, self._feedback_schema[kind]):
            raise BrokerError(f"feedback outside the schema: {kind}")
        if kind == "episode_end" and not self.ended:
            raise BrokerError("episode feedback comes only after the episode ends")
        item = {"kind": kind, **fields}
        self._feedback.append(item)
        self._event({"kind": "feedback", "feedback": item})

    def take_feedback(self) -> list[dict]:
        items, self._feedback = self._feedback, []
        return items

    @property
    def event_prefix(self) -> str:
        return self._prefix

    @property
    def logical_prefix(self) -> str:
        return self._logical

    # --- internal --------------------------------------------------------------------------------------------
    def _event(self, payload: dict) -> None:
        self._seq += 1
        body = {"episode": self.episode, "condition": self.condition, "stream": self.stream, "seq": self._seq,
                "t": self.t, "action_step": self.action_step, **payload}
        self._prefix = hashlib.sha256((self._prefix + _canonical(body)).encode()).hexdigest()
        if payload.get("kind") in LOGICAL_KINDS:
            logical = {k: v for k, v in body.items() if k != "seq"}
            self._logical = hashlib.sha256((self._logical + _canonical(logical)).encode()).hexdigest()
        self._log.append({**body, "event_prefix_sha256": self._prefix})

    def _reply(self, call: ProcedureCall, reply: dict, request_digest: str | None) -> str:
        self._event({"kind": "reply", "call": call.call_id, "request_sha256": request_digest, "reply": reply})
        return _canonical(reply)

    def _refuse(self, call: ProcedureCall, code: str, request_digest: str | None) -> str:
        return self._reply(call, {"ok": False, "refusal": code}, request_digest)

    def _close(self, call: ProcedureCall, status: str, error: str | None) -> None:
        if not call.closed:
            call.closed = True
            if call in self._open:
                self._open.remove(call)
            self._event({"kind": "call_close", "call": call.call_id, "status": status, "requests": call.requests,
                         "error": error})

    def _handle(self, call: ProcedureCall, text: Any) -> str:
        if call.closed:
            return self._refuse(call, "call_closed", None)
        call.requests += 1
        if call.requests > MAX_REQUESTS_PER_CALL:
            return self._refuse(call, "request_limit", None)
        if not isinstance(text, str):
            return self._refuse(call, "malformed", None)
        raw = text.encode("utf-8", "surrogatepass")
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) > MAX_MESSAGE_BYTES:
            return self._refuse(call, "too_large", digest)
        try:
            msg = json.loads(text, parse_constant=_no_constant, object_pairs_hook=_unique_keys)
        except (ValueError, RecursionError):
            return self._refuse(call, "malformed", digest)
        if not isinstance(msg, dict) or set(msg) != {"capability", "args"} or not isinstance(msg["capability"], str):
            return self._refuse(call, "malformed", digest)
        tool = self._tools.get(msg["capability"])
        if tool is None:
            return self._refuse(call, "unknown_capability", digest)
        args = msg["args"]
        try:
            valid = _check(args, tool.request)
        except Exception:
            valid = False
        if not valid:
            return self._refuse(call, "malformed", digest)
        if args["goal_version"] != self.goal_version:
            return self._refuse(call, "stale_goal_version", digest)
        if self.predictions_used >= self.allowance:
            return self._refuse(call, "over_allowance", digest)
        self.predictions_used += 1
        n = self.predictions_used
        request_id = f"{call.call_id}/{msg['capability']}{n}"
        scope = "replay" if self._replay else "running"
        # The exact validated request is logged before the tool runs, so it can be replayed and an interrupted
        # prediction leaves a start (Codex's BR51.1 and BR51.6).
        self._event({"kind": "prediction_start", "call": call.call_id, "request_id": request_id,
                     "request_sha256": digest, "request": msg})
        # Accounting is completed whatever happens from here (Codex's BR53.RET): the prediction already counts, so it
        # gets exactly one ledger row and one terminal event. An interruption anywhere, in the tool or in either clock
        # read, records status 'interrupted' and is then re-raised unchanged. If the tool was interrupted, the wall time
        # is measured once more; if a clock read was interrupted, the wall time is None. The tool is never retried. A
        # trusted clock that raises an ordinary exception leaves the wall time unknown.
        started = wall = cpu = None
        phase = "start clock"
        status, reply = "failed", {"ok": False, "refusal": "tool_failed"}
        try:
            try:
                started = self._clock()
            except Exception:
                started = None
            phase = "tool"
            try:
                cpu0 = self._cpu_clock()
            except Exception:
                cpu0 = None
            try:
                result = tool.run(args)
                if tool.reply_ok(result, args):
                    candidate = {"ok": True, "result": result, "remaining": self.allowance - n}
                    if len(_canonical(candidate).encode()) > MAX_REPLY_BYTES:
                        status, reply = "failed", {"ok": False, "refusal": "reply_too_large"}
                    else:
                        status, reply = "answered", candidate
            except Exception:
                pass                                       # status stays failed; nothing from the tool is passed on
            try:
                cpu = None if cpu0 is None else self._cpu_clock() - cpu0
            except Exception:
                cpu = None
            phase = "end clock"
            if started is not None:
                try:
                    wall = self._clock() - started
                except Exception:
                    wall = None
        except BaseException as exc:
            if phase == "tool" and started is not None:
                try:
                    wall = self._clock() - started
                except Exception:
                    wall = None
            self._ledger.add(Call(request_id, tool.charge, self.episode, self.action_step, started, wall,
                                  None, None, "interrupted", scope=scope, cpu_s=None, condition=self.condition,
                                  stream=self.stream, sim_second=self.t, answered_step=None))
            self._event({"kind": "prediction_interrupted", "call": call.call_id, "request_id": request_id,
                         "phase": phase, "error": f"{type(exc).__name__}: {exc}"})
            raise
        self._ledger.add(Call(request_id, tool.charge, self.episode, self.action_step, started, wall,
                              None, None, status, scope=scope, cpu_s=cpu, condition=self.condition,
                              stream=self.stream, sim_second=self.t, answered_step=self.action_step))
        self._event({"kind": "prediction_end", "call": call.call_id, "request_id": request_id, "status": status})
        return self._reply(call, reply, digest)


def load_inputs(log_path: Any, heads_path: Any, episode: str) -> dict:
    """Rebuild an episode's public inputs from its broker log: the briefing and every bundle, in order (Codex's
    BR59.STORE). The protected heads are required: the whole log is first verified against them, so an edited
    briefing, a bundle rehashed with its digest, or a removed record is refused (RP61.AUTH). Then the episode's broker
    events must run in unbroken sequence from 1, there must be exactly one briefing, and the bundles must be at action
    steps 0, 1, 2, … in order, each matching the SHA-256 recorded with it. Raises BrokerError otherwise."""
    from pathlib import Path
    from .trusted_log import verify
    check = verify(Path(log_path), Path(heads_path))
    if not check.get("ok"):
        raise BrokerError(f"the log does not verify against its protected heads: {check}")
    briefing, bundles, seqs, briefings = None, [], [], 0
    for line in Path(log_path).read_text().splitlines()[1:]:
        e = json.loads(line)["payload"]
        if e.get("episode") != episode:
            continue
        seqs.append(e.get("seq"))
        if e["kind"] == "briefing":
            briefing, briefings = e["briefing"], briefings + 1
        elif e["kind"] == "bundle":
            if "bundle" not in e or hashlib.sha256(_canonical(e["bundle"]).encode()).hexdigest() != e["bundle_sha256"]:
                raise BrokerError(f"bundle at action step {e['action_step']} is missing or does not match its digest")
            bundles.append({"action_step": e["action_step"], "t": e["t"], "bundle": e["bundle"]})
    if briefings != 1:
        raise BrokerError("the log must hold exactly one briefing for the episode")
    if seqs != list(range(1, len(seqs) + 1)):
        raise BrokerError("the episode's broker events are not in unbroken order")
    if [b["action_step"] for b in bundles] != list(range(len(bundles))):
        raise BrokerError("bundles are missing or out of order")
    return {"briefing": briefing, "bundles": bundles}

