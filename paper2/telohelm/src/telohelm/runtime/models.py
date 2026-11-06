"""The model layer's trusted gateway (architecture section 3; DISTURBANCES.md section 8; paper2/notes/MODEL-LAYER.md).

Every model call goes through ModelGateway.submit. The gateway enforces the episode's call limits, writes the full
request to the trusted log before the provider is called, recovers a saved answer from its journal instead of calling
again, never retries, and charges every call that starts with exactly one ledger row and one terminal event, whatever
interrupts it. It computes when the answer becomes available in simulated time: the request's second plus the modelled
latency rounded up to a whole second.

This is the interface with stub providers only: a scripted provider for tests, marked so that its output can never be
reported as a result, and a replay provider that returns stored answers matched by request identity and content
fingerprint. No real provider exists here, and nothing in this module contacts a server.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from .exchange import ExchangeError; from .exchange import ExchangeStore; from .exchange import body_sha256; from .exchange import record_sha256; from .exchange import summary
from . import startup as startup
from .manifest import CheckedConfig
from .ledger import Call; from .ledger import Ledger
from .transport import interpret
from .trusted_log import TrustedLog; from .trusted_log import verify

ROLES = ("small_model", "reasoning_model")
CALL_LIMITS = {"reasoning_model": 4, "small_model": 30}     # per episode, repairs included (DISTURBANCES section 8)
STATUSES = ("answered", "failed", "timed_out", "cut_off", "interrupted")
MAX_MESSAGES_BYTES = 262_144                                # one request's messages, as canonical JSON (class D)


class GatewayError(ValueError):
    """A trusted-side misuse of the gateway, such as an impossible request. Never shown to model-written code."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    attempt_id: str
    episode: str
    condition: str
    stream: str
    role: str
    messages: tuple                 # ((speaker, text), ...)
    max_output_tokens: int
    goal_version: int
    sim_second: int
    action_step: int
    modelled_latency_s: float

    @property
    def fingerprint(self) -> str:
        """SHA-256 of what the model is asked, not of who asked or when; used to match stored answers."""
        return hashlib.sha256(_canonical({"role": self.role, "messages": [list(m) for m in self.messages],
                                          "max_output_tokens": self.max_output_tokens}).encode()).hexdigest()


@dataclass(frozen=True)
class ModelResponse:
    status: str
    text: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    provider_ids: dict = field(default_factory=dict)
    # Where the answer came from, when the provider is not its producer (a replay): exactly {"provider": name,
    # "reportable": bool}. None means the provider produced it. Anything partial makes it unreportable (MG65.ORIGIN).
    origin: dict | None = None


def utf8_ok(text: Any) -> bool:
    """True if text is a string that can be written as UTF-8. A lone surrogate cannot, and would break the trusted
    logs and the journal (Codex's DL74.UTF8 and LA74.UTF8)."""
    if not isinstance(text, str):
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def raw_sha256(text: str) -> str:
    """A digest of any string, lone surrogates included, so evidence of an unusable text can be kept."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def origin_reportable(origin: Any) -> bool:
    """True only for a complete, well-typed origin record that says reportable."""
    return (isinstance(origin, dict) and set(origin) == {"provider", "reportable"}
            and isinstance(origin["provider"], str) and origin["provider"] != ""
            and origin["reportable"] is True)


class Provider(Protocol):
    name: str
    reportable: bool

    def complete(self, request: ModelRequest) -> ModelResponse:
        ...


class ScriptedProvider:
    """For tests only. reportable is False, and the gateway marks every record it makes unreportable; an origin field
    in its responses is ignored, since it produces them (Codex's MG67.ORIGIN.SCRIPTED)."""
    name, reportable, relays = "scripted", False, False

    def __init__(self, answer: Callable[[ModelRequest], ModelResponse]):
        self._answer, self.calls = answer, 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return self._answer(request)


def request_meta(request: ModelRequest) -> dict:
    """The request details an answer belongs to besides its identity and content: goal version, issue second and
    step, modelled latency, attempt and output limit."""
    return {"goal_version": request.goal_version, "sim_second": request.sim_second,
            "action_step": request.action_step, "modelled_latency_s": request.modelled_latency_s,
            "attempt": request.attempt_id, "max_output_tokens": request.max_output_tokens}


class ReplayProvider:
    """Returns stored answers, matched by request identity, content fingerprint and the original request's details;
    anything else is not answered, so an answer stored for one goal version is never returned for another (Codex's
    MG67.ID.REPLAY). Each stored entry is {"response": ModelResponse, "meta": request_meta(original request)}. It
    relays answers it did not produce, so it is unreportable itself: a stored answer is reportable only if it carries a
    complete origin record saying so (MG65.ORIGIN)."""
    name, reportable, relays = "replay", False, True

    def __init__(self, stored: dict[tuple[str, str], dict]):
        self._stored, self.calls = stored, 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        entry = self._stored.get((request.request_id, request.fingerprint))
        if not isinstance(entry, dict) or set(entry) != {"response", "meta"}:
            raise LookupError("no stored answer, with its request details, for this request and fingerprint")
        if entry["meta"] != request_meta(request):
            raise LookupError("the stored answer belongs to a request with different details")
        return entry["response"]


class ProtectedRoots:
    """The protected gateway logs that a reader may treat as original evidence (Codex's ADM90.ROOT). Only trusted code
    adds to it: every ModelGateway registers its own log and heads when it is built, and a run that resumes in another
    process registers earlier runs' logs from its own protected records. An exchange's sidecar mark may point at a log,
    but that log gives admission only if it is registered here; the mark cannot add to the registry. Paths are compared
    after resolving them. This is a premise about the trusted caller, not a defence against code in the same process."""

    def __init__(self):
        self._roots: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _key(log: Path, heads: Path) -> tuple[str, str]:
        return (str(Path(log).resolve()), str(Path(heads).resolve()))

    def register(self, log: Path, heads: Path) -> None:
        with self._lock:
            self._roots.add(self._key(log, heads))

    def trusts(self, log: Path, heads: Path) -> bool:
        with self._lock:
            return self._key(log, heads) in self._roots

    # Restoring the registry in a new process (TURN 99, Codex's request after SG96.1). Each gateway can append its log,
    # heads and log anchor to a registry file kept in protected storage; a later process restores the registry from that
    # file. The file's protection is a premise. Restoration fails closed on any entry whose log does not verify against
    # its heads or whose anchor differs from the one recorded, so a substituted log or heads file is refused.

    @staticmethod
    def record(registry: Path, log: Path, heads: Path) -> None:
        """Append a root to the registry only after its log verifies against its heads and has an anchor; on refusal
        the registry is left unchanged (Codex's SG100.ROOT.RECORD)."""
        anchor = startup.log_anchor(Path(heads))
        try:
            ok = verify(Path(log), Path(heads)).get("ok")
        except (OSError, ValueError):
            ok = False
        if not ok or not anchor:
            raise GatewayError("a root is recorded only if its log verifies against its heads and has an anchor")
        entry = {"log": str(Path(log).resolve()), "heads": str(Path(heads).resolve()), "anchor": anchor}
        with Path(registry).open("ab") as f:
            f.write(_canonical(entry).encode("utf-8") + b"\n")
            f.flush()
            os.fsync(f.fileno())

    @classmethod
    def restore(cls, registry: Path) -> "ProtectedRoots":
        roots = cls()
        try:
            data = Path(registry).read_bytes()
        except OSError as e:
            raise GatewayError(f"no protected roots registry at {registry}") from e
        if data and not data.endswith(b"\n"):
            raise GatewayError("the roots registry ends without a line terminator")
        for n, line in enumerate(data.splitlines(), start=1):
            try:
                entry = json.loads(line)
                log, heads, anchor = Path(entry["log"]), Path(entry["heads"]), entry["anchor"]
            except (ValueError, KeyError, TypeError) as e:
                raise GatewayError(f"roots registry line {n} is not an entry") from e
            if not verify(log, heads).get("ok") or startup.log_anchor(heads) != anchor or not anchor:
                raise GatewayError(f"roots registry line {n}: the log does not verify with its recorded anchor")
            roots.register(log, heads)
        return roots


PROTECTED_ROOTS = ProtectedRoots()


class LocalChatProvider:
    """An adapter for a local model server of the chat-completions kind (architecture section 3, A1), as Study 1's
    LocalBackend used with vLLM. Design stub for review (TURN 71, revised TURN 73): constructing it contacts nothing,
    and the transport is injected as transport(url, body, timeout_s) -> (http_status, json_body). The real HTTP
    transport with a whole-request deadline, the servers, the pinned model names, aliases, weights and prompts are a
    later reviewed step. It is unreportable unless constructed with reportable=True, which only a reviewed
    configuration should do.

    Configuration is checked before anything else (Codex's LA72.CONFIG): non-empty base URL and model names, a finite
    positive timeout for each role, a finite temperature, reportable exactly True or False, and for each role a
    non-empty set of served names the server may report (default: the requested model name; vLLM reports its first
    served alias, so a reviewed alias list can be given). No timeout value is chosen here.

    Completion policy, text only and not streaming (LA72.FINISH, MODEL, CUTOFF): the reply must come from a model whose
    reported name is one of the role's served names, or it fails closed. finish_reason "stop" with string content is a
    usable answer; "length" is cut_off, with no text passed on and the reported usage kept; any other finish, null
    included, is failed. A transport TimeoutError is timed_out; any other transport error, a non-200 status or a body
    of the wrong shape is failed. Usage counts that are not non-negative integers are unknown (None).

Exchange evidence (TURN 83, revised TURN 85; a draft for cross-review): given an ExchangeStore and a CheckedConfig from
manifest.checked(), the adapter asks the transport for the exchange's raw evidence, saves it to the store before
parsing, and only then interprets it. Every response then names its exchange, the configuration fingerprint, the
digest of the reply bytes, whether the wire reply was complete and the transport's outcome, so the gateway's trusted log
holds those links; the gateway also writes an anchor with the exchange's digests into its trusted log when an exchange
is saved or reused (verify_anchors). If the store already holds the exchange for this exact request (the same
identifier, attempt, episode, condition, stream, content, issue details, endpoint, body and configuration), the saved
bytes are interpreted again and the server is not asked a second time; any other request with that identifier fails.
A reused exchange keeps the reportability it was saved with, which can only lower the adapter's own, and is marked so
that the gateway records its wall time as unknown. If the evidence cannot be saved, the call fails with no text.

Fail closed (Codex's MF84.GATE): a reportable adapter needs a store and a CheckedConfig; a bare digest is refused; the
adapter's base URL, models, served names, timeouts, temperature and its transport's origin and reply limit must equal
the configuration's; and a request for an output limit above the configured one fails before any contact."""
    name, relays = "local-chat", False
    SPEAKERS = ("system", "user", "assistant")

    def __init__(self, base_url: str, model_by_role: dict[str, str], transport: Callable[[str, dict, float], tuple],
                 timeout_s_by_role: dict[str, float], temperature: float = 0.0, reportable: bool = False,
                 served_names_by_role: dict[str, tuple] | None = None, exchanges: ExchangeStore | None = None,
                 config: CheckedConfig | None = None, roots: ProtectedRoots | None = None,
                 extra_body: dict | None = None):
        def finite(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        if not isinstance(base_url, str) or not base_url.strip():
            raise GatewayError("the base URL must be non-empty text")
        if not isinstance(model_by_role, dict) or set(model_by_role) != set(ROLES) or not all(
                isinstance(v, str) and v.strip() for v in model_by_role.values()):
            raise GatewayError("each role needs a non-empty model name")
        if not isinstance(timeout_s_by_role, dict) or set(timeout_s_by_role) != set(ROLES) or not all(
                finite(v) and v > 0 for v in timeout_s_by_role.values()):
            raise GatewayError("each role needs a finite positive timeout")
        if not finite(temperature) or temperature < 0:
            raise GatewayError("the temperature must be finite and non-negative")
        if not isinstance(reportable, bool):
            raise GatewayError("reportable must be exactly True or False")
        served = served_names_by_role if served_names_by_role is not None else {r: (m,) for r, m in model_by_role.items()}
        if not isinstance(served, dict) or set(served) != set(ROLES) or not all(
                isinstance(v, (tuple, list)) and v and all(isinstance(n, str) and n.strip() for n in v)
                for v in served.values()):
            raise GatewayError("each role needs a non-empty list of served names")
        self.base_url, self.model_by_role, self._transport = base_url.rstrip("/"), dict(model_by_role), transport
        self.timeout_s_by_role, self.temperature, self.reportable = dict(timeout_s_by_role), float(temperature), reportable
        self.served_names_by_role = {r: tuple(v) for r, v in served.items()}
        self._seed = None
        # Extra request-body fields for a development server setting, such as turning a local model's thinking off
        # (MVP, TURN 117). Not allowed with an exchange store: a reportable configuration must pin every setting.
        if extra_body is not None and (not isinstance(extra_body, dict) or exchanges is not None
                                       or set(extra_body) & {"model", "messages", "max_tokens", "temperature", "seed"}):
            raise GatewayError("extra_body is a dict of other fields, for an adapter without an exchange store")
        self._extra_body = dict(extra_body or {})
        if exchanges is not None:
            if not isinstance(exchanges, ExchangeStore) or not callable(getattr(transport, "exchange", None)):
                raise GatewayError("an exchange store needs a transport that returns exchange evidence")
            if not isinstance(config, CheckedConfig):
                raise GatewayError("an exchange store needs a configuration made by manifest.checked()")
            self._bind(config, transport)
        elif config is not None:
            raise GatewayError("a configuration is recorded only through an exchange store")
        if reportable and exchanges is None:
            raise GatewayError("a reportable adapter needs an exchange store and a checked configuration")
        self._exchanges, self._config = exchanges, config
        self._anchor: Callable[[dict], None] | None = None
        self._anchor_log: tuple[Path, Path] | None = None
        self._roots = PROTECTED_ROOTS if roots is None else roots

    # The checked configuration and its fingerprint are one snapshot, fixed at construction; neither can be replaced
    # or relabelled afterwards (Codex's CFG88.SNAPSHOT).
    @property
    def config(self) -> CheckedConfig | None:
        return self._config

    @property
    def config_sha256(self) -> str | None:
        return None if self._config is None else self._config.fingerprint

    def _bind(self, config: CheckedConfig, transport: Any) -> None:
        """The adapter's own arguments must be the configuration's (Codex's MF84.GATE)."""
        t, roles, names, sampling = (config.section("transport"), config.section("roles"), config.section("names"),
                                     config.section("sampling"))
        pairs = [("base URL", self.base_url, t["origin"].rstrip("/")),
                 ("transport origin", getattr(transport, "origin", None), t["origin"].rstrip("/")),
                 ("reply limit", getattr(transport, "max_response_bytes", None), t["max_response_bytes"]),
                 ("timeouts", self.timeout_s_by_role, t["timeout_s"]),
                 ("models", self.model_by_role, roles["model"]),
                 ("served names", {r: list(v) for r, v in self.served_names_by_role.items()}, names["served"]),
                 ("temperature", self.temperature, sampling["temperature"])]
        for name, actual, pinned in pairs:
            if actual != pinned:
                raise GatewayError(f"the adapter's {name} is not the configuration's")
        # Every pinned setting is either sent or refused (Codex's CFG86.EFFECTIVE): a client seed goes in the request;
        # reasoning settings are not supported by this adapter, so a non-empty one is refused here.
        if any(roles["reasoning"][r] for r in ROLES):
            raise GatewayError("this adapter does not support reasoning settings; the configuration pins some")
        self._output_limits, self._latency_s = sampling["max_output_tokens"], roles["modelled_latency_s"]
        self._seed = sampling["seed"]

    def bind_anchor(self, anchor: Callable[[dict], None], log_ref: tuple[Path, Path] | None = None) -> None:
        """The gateway gives the adapter a way to write an exchange's record into its trusted log, and names that log and
        its protected heads, so a later run can find the original anchor. One adapter serves one gateway at a time."""
        self._anchor, self._anchor_log = anchor, log_ref

    def _quarantine(self, response: ModelResponse, reason: str) -> ModelResponse:
        """Evidence kept, observed usage kept, no usable text and never reportable (ANCH86.GATE, ANCH88)."""
        return replace(response, status="failed", text=None,
                       provider_ids={**(response.provider_ids or {}), "refused": reason},
                       origin={"provider": self.name, "reportable": False})

    def _refusal(self, saved: dict) -> str | None:
        """None if the stored exchange is admitted; otherwise why not."""
        if self._exchanges.anchor_ref(saved["exchange_id"]) is None:
            return "unanchored receipt"
        root = self._root(saved)
        if root is None:
            return "original not admitted"
        if not self._original_started(saved, root):
            return "original startup not admitted"
        if _code_void(root[0]):
            return "original code identity void"
        return self._wire_problem(saved)

    def reuse_wall(self, request: ModelRequest) -> float | None:
        """For the gateway, after a store reuse: the original call's total time from its protected, joined model_end,
        or None if there is no valid original completion (Codex's COST100.REUSE.TIME). Never the time spent reading
        the store."""
        if self._exchanges is None:
            return None
        _, _, digest, identity = self._parts(request)
        try:
            saved = self._exchanges.get(request.request_id, request.attempt_id, self.config_sha256, digest, identity)
        except ExchangeError:
            return None
        root = None if saved is None else self._root(saved)
        return None if root is None else self._original_wall(saved, root)

    def _original_started(self, saved: dict, root: tuple[Path, Path]) -> bool:
        """The original run's own log holds, before its first model call, a start record that is admitted for that log,
        the exchange's scope and this configuration (SG96.4), recomputed from the record and not from its flag."""
        event, before = startup.first_start(root[0])
        ident = saved.get("request") or {}
        return event is not None and before and not startup.decide(
            event, startup.log_anchor(root[1]), ident.get("episode"), ident.get("condition"), ident.get("stream"),
            self._config, before_first_call=True)

    def _root(self, saved: dict) -> tuple[Path, Path] | None:
        """The registered protected log that admits this stored exchange, or None. The mark names the log; the registry
        decides whether it is trusted (ADM90.ROOT); the log must verify and hold a 'saved' anchor whose record equals
        this record and is joined to the original request (ANCH88.PREUSE)."""
        ref = self._exchanges.anchor_ref(saved["exchange_id"])
        if not ref or "log" not in ref or "heads" not in ref:
            return None
        log, heads = Path(ref["log"]), Path(ref["heads"])
        if not self._roots.trusts(log, heads):
            return None
        record = {k: v for k, v in summary(saved).items() if k != "anchored"}
        return None if _anchor_problems([record], [(log, heads)]) else (log, heads)

    def _wire_problem(self, saved: dict) -> str | None:
        """The request bytes actually sent must carry the configured model, temperature and seed, the request's output
        limit and its messages, and must hash to the stored body digest (the pre-use join Codex asked for in TURN 90)."""
        raw, ident = saved.get("request_wire"), saved.get("request") or {}
        if not isinstance(raw, bytes) or not ident:
            return "request bytes missing"
        try:
            body = json.loads(raw.partition(b"\r\n\r\n")[2].decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return "request bytes unreadable"
        if not isinstance(body, dict) or body_sha256(body) != saved["body_sha256"]:
            return "request bytes do not match their body digest"
        role, meta = saved["role"], ident.get("meta") or {}
        want = {"model": self._config.section("roles")["model"][role], "max_tokens": meta.get("max_output_tokens"),
                "temperature": self._config.section("sampling")["temperature"]}
        seed = self._config.section("sampling")["seed"]
        if seed is not None:
            want["seed"] = seed
        if any(body.get(k) != v for k, v in want.items()) or (seed is None and "seed" in body):
            return "request bytes do not match the configuration"
        try:
            messages = [[m["role"], m["content"]] for m in body["messages"]]
        except (KeyError, TypeError):
            return "request bytes do not carry the messages"
        fingerprint = hashlib.sha256(_canonical({"role": role, "messages": messages,
                                                 "max_output_tokens": meta.get("max_output_tokens")}).encode()).hexdigest()
        return None if fingerprint == ident.get("fingerprint") else "request bytes do not carry the request's messages"

    def _protected_end(self, saved: dict, log: Path, heads: Path) -> dict | None:
        """The protected original model_end for exactly this stored exchange, or None (Codex's COST94.IDENTITY). The log
        must be registered and verify. The model_end must follow, in the same log, a model_start for the same original
        request (identifier, attempt, role, episode, condition, stream, content fingerprint and issue details), must not
        be a recovery, and its response must name this exchange and this configuration. A request identifier alone is
        not enough: runs with other goals or details reuse the same sequential identifiers."""
        try:
            if not self._roots.trusts(log, heads) or not verify(log, heads).get("ok"):
                return None
            lines = log.read_text().splitlines()[1:]
        except (OSError, ValueError):
            return None
        record = {k: v for k, v in summary(saved).items() if k != "anchored"}
        started = False
        for line in lines:
            e = json.loads(line)["payload"]
            if e.get("kind") == "model_start" and (e.get("request") or {}).get("request_id") == saved["request_id"]:
                started = _start_matches(e, record)
            elif (started and e.get("kind") == "model_end" and e.get("request_id") == saved["request_id"]
                  and e.get("recovered") is False and isinstance(e.get("response"), dict)):
                ids = e["response"].get("provider_ids") or {}
                if ids.get("exchange_id") == saved["exchange_id"] and ids.get("config_sha256") == saved["config_sha256"]:
                    return e
                started = False
        return None

    def _protected_usage(self, saved: dict) -> tuple[int | None, int | None]:
        """The original call's token counts from its protected model_end, joined to the full original request, or
        (None, None). Used when the stored bytes are not admitted, so their own counts cannot be trusted."""
        ref = self._exchanges.anchor_ref(saved["exchange_id"]) or {}
        if "log" not in ref or "heads" not in ref:
            return None, None
        end = self._protected_end(saved, Path(ref["log"]), Path(ref["heads"]))
        if end is None:
            return None, None
        counts = [end["response"].get(k) for k in ("tokens_in", "tokens_out")]
        return tuple(v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None for v in counts)

    def _held_unverified(self, response: ModelResponse, reason: str, saved: dict,
                         claimed: dict | None = None) -> ModelResponse:
        """Quarantine a reply whose stored bytes were not admitted (COST92.UNVERIFIED). Its usage is the protected
        original's if a registered log recorded it, otherwise unknown. The counts the stored bytes report, and any the
        journal claimed, are kept apart as claims and never charged."""
        protected = self._protected_usage(saved)
        ids = {**(response.provider_ids or {}), **(claimed or {}),
               "stored_tokens_in": None if response.tokens_in is None else str(response.tokens_in),
               "stored_tokens_out": None if response.tokens_out is None else str(response.tokens_out),
               "usage_source": "protected original" if protected != (None, None) else "unknown"}
        return self._quarantine(replace(response, tokens_in=protected[0], tokens_out=protected[1], provider_ids=ids),
                                reason)

    def _original_wall(self, saved: dict, root: tuple[Path, Path]) -> float | None:
        """The original call's total wall time from its protected model_end, joined to the full original request as for
        usage, or None (COST90.TIME, COST94.IDENTITY)."""
        end = self._protected_end(saved, *root)
        w = None if end is None else end.get("wall_s")
        return w if isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) else None

    def _parts(self, request: ModelRequest) -> tuple:
        url = f"{self.base_url}/v1/chat/completions"
        body = self.body(request)
        # The full original request identity, so a stored exchange is reused only for the same question asked at the
        # same point (Codex's EX84.IDENTITY).
        identity = {"episode": request.episode, "condition": request.condition, "stream": request.stream,
                    "fingerprint": request.fingerprint, "meta": request_meta(request), "endpoint": url}
        return url, body, body_sha256(body), identity

    def admit_journal(self, request: ModelRequest, response: ModelResponse) -> tuple[ModelResponse, dict]:
        """Called by the gateway before it uses any response recovered from its journal, whether or not the response
        names an exchange (Codex's ANCH88.JOURNAL and ADM90.REQUIRED). It is used only if the store holds the exchange
        for this exact request, the exchange is admitted against a registered protected original, its request bytes
        match the configuration, and its bytes give the same response again, link included. Otherwise it is
        quarantined. Returns the response and what was verified: the original wall time from the protected log, or
        None when it cannot be verified (COST90.TIME).

        A quarantined response's token counts come from the stored bytes, never from the journal, and the journal's
        own counts are kept apart as claimed values (COST90.SOURCE). With no stored exchange the counts are unknown."""
        claimed = {"claimed_tokens_in": None if response.tokens_in is None else str(response.tokens_in),
                   "claimed_tokens_out": None if response.tokens_out is None else str(response.tokens_out)}
        _, _, digest, identity = self._parts(request)
        try:
            saved = self._exchanges.get(request.request_id, request.attempt_id, self.config_sha256, digest, identity)
        except ExchangeError:
            saved = None
        if saved is None:
            held = replace(response, tokens_in=None, tokens_out=None,
                           provider_ids={**(response.provider_ids or {}), **claimed})
            return self._quarantine(held, "journal exchange not found"), {"wall_s": None}
        got = self.reextract(saved)
        again = got if _response_ok(got) else _salvage(got)
        refusal = self._refusal(saved)
        # The protected original admits the bytes and supplies the verified time even when a run-level check (the
        # original's start or code identity, or the wire check) withholds the text (SG100.VOID.ORIGINAL).
        root = self._root(saved) if self._exchanges.anchor_ref(saved["exchange_id"]) is not None else None
        verified = {"wall_s": None if root is None else self._original_wall(saved, root)}
        journalled_ids = {k: v for k, v in (response.provider_ids or {}).items() if k != "exchange_reused"}
        if refusal is None and (again.status, again.text, again.tokens_in, again.tokens_out, again.provider_ids,
                                again.origin) != (response.status, response.text, response.tokens_in,
                                                  response.tokens_out, journalled_ids, response.origin):
            refusal = "journal differs from its exchange"
        if refusal is None:
            return response, verified
        if root is None:
            # The stored bytes were not admitted: their usage is a claim, like the journal's (COST92.UNVERIFIED).
            return self._held_unverified(again, refusal, saved, claimed), verified
        # The stored bytes were admitted against a protected original; only the journal disagreed, so the bytes'
        # usage is the verified usage and the journal's is kept as a claim.
        held = replace(again, provider_ids={**(again.provider_ids or {}), **claimed, "usage_source": "protected"})
        return self._quarantine(held, refusal), verified

    def body(self, request: ModelRequest) -> dict:
        body = {"model": self.model_by_role[request.role],
                "messages": [{"role": speaker, "content": text} for speaker, text in request.messages],
                "max_tokens": request.max_output_tokens, "temperature": self.temperature}
        if self._seed is not None:
            body["seed"] = self._seed
        body.update(self._extra_body)
        return body

    def complete(self, request: ModelRequest) -> ModelResponse:
        if any(speaker not in self.SPEAKERS for speaker, _ in request.messages):
            return ModelResponse("failed", provider_ids={"refused": "speaker"})
        if self._exchanges is not None:
            # The settings are checked against the frozen configuration again before every request, so a setting
            # changed after construction never reaches the server (Codex's CFG86.MUTABLE).
            try:
                self._bind(self.config, self._transport)
            except GatewayError:
                return ModelResponse("failed", provider_ids={"refused": "configuration changed"})
            if (request.max_output_tokens > self._output_limits[request.role]
                    or request.modelled_latency_s != self._latency_s[request.role]):
                return ModelResponse("failed", provider_ids={"refused": "outside the configuration"})
        url, body, timeout = (f"{self.base_url}/v1/chat/completions", self.body(request),
                              self.timeout_s_by_role[request.role])
        if self._exchanges is None:
            try:
                status, data = self._transport(url, body, timeout)
            except TimeoutError:
                return ModelResponse("timed_out")
            except OSError:
                return ModelResponse("failed")
            return self._interpret(request.role, status, data, {})
        url, body, digest, identity = self._parts(request)
        try:
            saved = self._exchanges.get(request.request_id, request.attempt_id, self.config_sha256, digest, identity)
        except ExchangeError:
            return ModelResponse("failed", provider_ids={"refused": "exchange mismatch"})
        reused = saved is not None
        if not reused:
            evidence = self._transport.exchange(url, body, timeout)
            try:                          # saved before anything parses the reply
                stored = self._exchanges.save(request.request_id, request.attempt_id, self.config_sha256,
                                              request.role, digest, evidence, identity, self.reportable)
            except (OSError, ExchangeError):
                return ModelResponse("failed", provider_ids={"refused": "evidence not saved"})
            saved = {**stored, "request_wire": evidence.get("request_wire"), "reply_wire": evidence.get("reply_wire")}
        anchored = False
        if self._anchor is not None:
            # The anchor carries the whole original record except the bytes, whose digests it holds (ANCH86.BIND).
            record = {k: v for k, v in summary(saved).items() if k != "anchored"}
            self._anchor({"use": "reused" if reused else "saved", "exchange_id": saved["exchange_id"],
                          "record": record, "record_sha256": record_sha256(record)})
            if not reused:
                try:
                    log, heads = self._anchor_log if self._anchor_log is not None else (None, None)
                    self._exchanges.mark_anchored(saved["exchange_id"], log, heads)
                    anchored = True
                except (OSError, ExchangeError):
                    anchored = False
        response = self.reextract(saved)
        wire_problem = None if reused else self._wire_problem(saved)
        if reused:
            # A stored exchange gives usable text only once its original protected anchor is verified (ANCH88.PREUSE).
            # Its bytes are then unverified, so their usage is not observed cost (COST92.UNVERIFIED).
            refusal = self._refusal(saved)
            if refusal is not None:
                response = self._held_unverified(response, refusal, saved)
        elif wire_problem is not None:
            # A fresh reply is released only after the request bytes actually sent are checked against the body, the
            # configuration and the messages (WIRE92.FRESH). Its usage was observed live and is kept.
            response = self._quarantine(response, wire_problem)
        elif self._anchor is not None and not anchored:
            # Saved but its mark could not be written (ANCH86.GATE): kept, quarantined, and not asked again.
            response = self._quarantine(response, "unanchored receipt")
        elif self._anchor is None:
            # Called directly, with no gateway to anchor it (ADM90.DIRECT). A reportable adapter must go through a
            # gateway, so its reply is quarantined; an unreportable one may be used directly, for development only, and
            # its reply is labelled as such. Neither is ever admitted for later reuse.
            if self.reportable:
                response = self._quarantine(response, "no gateway anchor")
            else:
                response = replace(response, provider_ids={**(response.provider_ids or {}), "scope": "direct"})
        if reused:
            # The gateway times the call; a reuse took no server time, so its wall time is unknown (EX84.TIME).
            response = replace(response, provider_ids={**(response.provider_ids or {}), "exchange_reused": "true"})
        return response

    def reextract(self, saved: dict) -> ModelResponse:
        """The response that the saved exchange evidence gives, with its links. Used for a fresh exchange, for one
        recovered from the store, and to check a journalled response against the bytes behind it."""
        links = {"exchange_id": saved["exchange_id"], "config_sha256": saved["config_sha256"],
                 "wire_sha256": saved["reply_sha256"], "wire_complete": "true" if saved["reply_complete"] else "false",
                 "wire_outcome": saved["outcome"]}
        # The reportability the exchange was saved with travels as its origin, which can only lower the adapter's own
        # (Codex's EX84.ORIGIN).
        origin = {"provider": self.name, "reportable": saved["reportable"]}
        try:
            status, data = interpret(saved)
        except TimeoutError:
            return ModelResponse("timed_out", provider_ids=links, origin=origin)
        except OSError:
            return ModelResponse("failed", provider_ids=links, origin=origin)
        return replace(self._interpret(saved["role"], status, data, links), origin=origin)

    def _interpret(self, role: str, status: Any, data: Any, links: dict) -> ModelResponse:
        """The chat-completions mapping from an HTTP status and parsed body to a response."""
        if status != 200 or not isinstance(data, dict):
            return ModelResponse("failed", provider_ids={**links,
                                                         "http_status": status if isinstance(status, int) else None})
        # Observed usage and safe identifiers are extracted first, so every refusal below keeps them; whether the
        # text is usable is decided separately (Codex's LA74.RETAIN). An identifier is kept only if it is UTF-8 text.
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        tokens = [v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
                  for v in (usage.get("prompt_tokens"), usage.get("completion_tokens"))]
        rid, model = data.get("id"), data.get("model")
        ids = {**links, "id": rid if utf8_ok(rid) else None, "model": model if utf8_ok(model) else None}
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return ModelResponse("failed", None, tokens[0], tokens[1], {**ids, "refused": "choices"})
        choice = choices[0]
        if not isinstance(model, str) or model not in self.served_names_by_role[role]:
            return ModelResponse("failed", None, tokens[0], tokens[1], {**ids, "refused": "model identity"})
        finish = choice.get("finish_reason")
        if finish == "length":
            return ModelResponse("cut_off", None, tokens[0], tokens[1], ids)
        message = choice.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if finish != "stop" or not isinstance(text, str):
            return ModelResponse("failed", None, tokens[0], tokens[1], ids)
        if not utf8_ok(text):
            return ModelResponse("failed", None, tokens[0], tokens[1],
                                 {**ids, "refused": "text encoding", "text_sha256": raw_sha256(text)})
        return ModelResponse("answered", text, tokens[0], tokens[1], ids)


def verify_extraction(provider: "LocalChatProvider", store: ExchangeStore, journal: Path) -> list[str]:
    """Check each journalled response against the exchange bytes behind it: the exchange must exist, belong to the same
    request, and give the same response when its saved bytes are interpreted again (Codex's TURN 82 requirement to
    verify the extraction chain). Returns the problems found, empty if none. The three digests involved are kept apart:
    the exchange's reply_sha256 covers the whole wire reply, the journal holds the extracted response, and a delivery's
    reply_sha256 covers only the extracted text."""
    problems = []
    exchanges = {r["exchange_id"]: r for r in store.records()}
    for n, line in enumerate(Path(journal).read_text().splitlines(), start=1):
        rec = json.loads(line)
        if rec.get("type", "response") != "response":
            continue
        saved = rec.get("response") or {}
        xid = (saved.get("provider_ids") or {}).get("exchange_id")
        ex = exchanges.get(xid)
        if ex is None:
            problems.append(f"journal line {n}: no stored exchange {xid!r}")
            continue
        if ex["request_id"] != rec.get("request_id"):
            problems.append(f"journal line {n}: exchange {xid} belongs to request {ex['request_id']!r}")
            continue
        # The journal's request details and content fingerprint must be the exchange's original ones (EX84.IDENTITY).
        identity = ex.get("request") or {}
        if identity.get("meta") != rec.get("meta") or identity.get("fingerprint") != rec.get("fingerprint"):
            problems.append(f"journal line {n}: the request details differ from exchange {xid}'s original request")
            continue
        got = provider.reextract(ex)
        again = asdict(got if _response_ok(got) else _salvage(got))
        journalled = {**saved, "provider_ids": {k: v for k, v in (saved.get("provider_ids") or {}).items()
                                                if k != "exchange_reused"}}
        for k in ("status", "text", "tokens_in", "tokens_out", "provider_ids", "origin"):
            if again.get(k) != journalled.get(k):
                problems.append(f"journal line {n}: {k} differs from what exchange {xid} gives")
    return problems


def _start_matches(start: dict, rec: dict) -> bool:
    """Whether a gateway model_start event is the original request a stored record names."""
    req, ident = start.get("request") or {}, rec.get("request") or {}
    meta = {"goal_version": req.get("goal_version"), "sim_second": req.get("sim_second"),
            "action_step": req.get("action_step"), "modelled_latency_s": req.get("modelled_latency_s"),
            "attempt": req.get("attempt_id"), "max_output_tokens": req.get("max_output_tokens")}
    return (bool(ident) and req.get("request_id") == rec.get("request_id") and req.get("attempt_id") == rec.get("attempt_id")
            and req.get("role") == rec.get("role") and start.get("fingerprint") == ident.get("fingerprint")
            and all(req.get(k) == ident.get(k) for k in ("episode", "condition", "stream")) and meta == ident.get("meta"))


def _anchor_problems(records: list[dict], logs: list[tuple[Path, Path]]) -> list[str]:
    """The anchor check for stored records (without their bytes) against protected gateway logs."""
    problems, anchors = [], {}
    for log, heads in logs:
        try:
            check = verify(Path(log), Path(heads))
        except (OSError, ValueError) as e:
            check = {"ok": False, "problem": f"{type(e).__name__}"}
        if not check.get("ok"):
            problems.append(f"{log}: {check.get('problem')}")
            continue
        starts: dict[Any, list] = {}
        for line in Path(log).read_text().splitlines()[1:]:
            e = json.loads(line)["payload"]
            if e.get("kind") == "model_start":
                starts.setdefault((e.get("request") or {}).get("request_id"), []).append(e)
            elif e.get("kind") == "exchange_anchor" and e.get("use") == "saved" and isinstance(e.get("record"), dict):
                rec = e["record"]
                joined = any(_start_matches(st, rec) for st in starts.get(rec.get("request_id"), []))
                anchors.setdefault(e.get("exchange_id"), []).append((rec, e.get("record_sha256"), joined))
    for want in records:
        found = anchors.get(want["exchange_id"], [])
        if not found:
            problems.append(f"exchange {want['exchange_id']} has no anchor in a protected log")
        elif not any(r == want and h == record_sha256(want) for r, h, _ in found):
            problems.append(f"exchange {want['exchange_id']} differs from its protected anchor")
        elif not any(r == want and joined for r, _, joined in found):
            problems.append(f"exchange {want['exchange_id']} is not joined to its original request in the same log")
    return problems


def verify_anchors(store: ExchangeStore, logs: list[tuple[Path, Path]]) -> list[str]:
    """Check every stored exchange against the protected gateway logs (TURN 85, revised TURN 87 for Codex's ANCH86.BIND
    and JOIN86.IDENTITY). Each log must verify against its protected heads. Each stored exchange needs a 'saved' anchor
    whose record equals the whole stored record except the bytes (so its request identity, endpoint, body digest, role,
    wire time, outcome and byte digests all match), written after a model_start event in the same log for the same
    original request. The store's own digests show only that a record agrees with itself; the protected anchor and its
    join to the original request show that the trusted gateway saw it. Returns the problems found."""
    return _anchor_problems([{k: v for k, v in summary(r).items() if k != "anchored"} for r in store.records()], logs)


def verify_exchange_evidence(provider: "LocalChatProvider", store: ExchangeStore, journal: Path,
                             logs: list[tuple[Path, Path]]) -> list[str]:
    """The joined check to rely on (Codex's JOIN86.IDENTITY): the store against the protected originals, then the
    journal against the store. verify_extraction alone shows only that the store and journal agree, which both could
    if both were changed."""
    return verify_anchors(store, logs) + verify_extraction(provider, store, journal)


def _code_void(log: Path) -> bool:
    """Whether a run's protected log records that its code identity was voided: a code_identity_void event, a close
    that recorded a void identity, or any refusal that gave code drift or a void identity as its reason (the last
    covers a run whose dedicated void event could not be written, Codex's VOID102.WRITE). A run that stopped after
    finding drift but before writing any of these leaves no evidence in its log; that case needs an independent
    checkpoint and is not covered here."""
    for line in Path(log).read_text().splitlines()[1:]:
        e = json.loads(line)["payload"]
        if e.get("kind") == "code_identity_void" or (e.get("kind") == "run_close" and e.get("code_identity_void")):
            return True
        if e.get("kind") == "startup_refused" and any(
                isinstance(p, str) and (p == "code identity void" or p.startswith("code drift:"))
                for p in (e.get("problems") or [])):
            return True
    return False


def compare_attribution(original: tuple[Path, Path], later: tuple[Path, Path]) -> list[str]:
    """Check that a later run's recovered or reused replies are attributed as the original run decided them (TURN 99,
    Codex's request for replay and M7 original attribution). Both logs must verify. For each reply in the later run
    that was recovered from a journal or reused from the store, there must be, in the original log, a model call with
    the same request identifier, attempt, role, content fingerprint, goal version, issue second and step and modelled
    latency, ending without recovery; and any text the later run used must be the original's text. A quarantined later
    reply (no text) matches; it used nothing. Returns the problems found. A new start in the later run never relabels
    the original: only the original log's own records are compared."""
    def calls(log: Path) -> list[tuple[dict, dict]]:
        out, starts = [], {}
        for line in Path(log).read_text().splitlines()[1:]:
            e = json.loads(line)["payload"]
            if e.get("kind") == "model_start":
                starts[(e.get("request") or {}).get("request_id")] = e
            elif e.get("kind") == "model_end" and e.get("request_id") in starts:
                out.append((starts.pop(e["request_id"]), e))
        return out

    def key(start: dict) -> tuple | None:
        """The call's identity as typed values, or None if any part is missing or of the wrong type (Codex's
        ATTR102.TYPE and ATTR102.REQUIRED): 13 and 13.0 are different, and two missing fields never match."""
        r = start.get("request") or {}
        whole = (r.get("goal_version"), r.get("sim_second"), r.get("action_step"), start.get("available_second"),
                 start.get("available_step"))
        texts = (r.get("request_id"), r.get("attempt_id"), r.get("role"), start.get("fingerprint"))
        latency = r.get("modelled_latency_s")
        # Latency must be a finite, non-negative number on both sides; zero is legitimate (Codex's ATTR104.DOMAIN).
        if (not all(type(v) is int and v >= 0 for v in whole) or not all(isinstance(v, str) and v for v in texts)
                or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
            return None
        return texts + whole + ((type(latency).__name__, latency),)

    problems = []
    for log, heads in (original, later):
        check = verify(Path(log), Path(heads))
        if not check.get("ok"):
            problems.append(f"{log}: {check.get('problem')}")
    if problems:
        return problems
    originals = {key(st): end for st, end in calls(original[0]) if end.get("recovered") is False and key(st) is not None}
    for st, end in calls(later[0]):
        ids = (end.get("response") or {}).get("provider_ids") or {}
        if not (end.get("recovered") is True or ids.get("exchange_reused") == "true"):
            continue
        rid = (st.get("request") or {}).get("request_id")
        if key(st) is None:
            problems.append(f"{rid}: the later call's identity is incomplete or wrongly typed")
            continue
        orig = originals.get(key(st))
        if orig is None:
            problems.append(f"{rid}: no original call with the same request, attempt, content and issue details")
            continue
        text = (end.get("response") or {}).get("text")
        if text is not None and text != (orig.get("response") or {}).get("text"):
            problems.append(f"{rid}: the later run used text the original did not give")
    return problems


def _salvage(r: Any) -> ModelResponse:
    """A provider response that cannot be used as given becomes a failure that keeps what can be kept safely (Codex's
    LA74 and LA76.SALVAGE): valid usage counts, provider identifiers that are UTF-8 text, a digest of text that could
    not be written, and the response's origin as a downward constraint. A well-formed origin is kept as it is; an
    origin that is present but malformed becomes an explicit unreportable marker, so it can never disappear and leave
    the answer to be promoted by its provider's own reportability."""
    if not isinstance(r, ModelResponse):
        return ModelResponse("failed")
    tokens = [v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None for v in (r.tokens_in, r.tokens_out)]
    ids = {}
    if isinstance(r.provider_ids, dict):
        ids = {k: v for k, v in r.provider_ids.items() if isinstance(k, str) and utf8_ok(k) and (v is None or utf8_ok(v))}
    if isinstance(r.text, str) and not utf8_ok(r.text):
        ids.update({"refused": "text encoding", "text_sha256": raw_sha256(r.text)})
    origin = None
    if r.origin is not None:
        well_formed = (isinstance(r.origin, dict) and set(r.origin) == {"provider", "reportable"}
                       and utf8_ok(r.origin.get("provider")) and isinstance(r.origin.get("reportable"), bool))
        origin = dict(r.origin) if well_formed else {"provider": "malformed origin", "reportable": False}
    return ModelResponse("failed", None, tokens[0], tokens[1], ids, origin)


def _response_ok(r: Any) -> bool:
    if not isinstance(r, ModelResponse) or r.status not in STATUSES or r.status == "interrupted":
        return False
    if r.text is not None and not utf8_ok(r.text):
        return False
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0) for v in (r.tokens_in, r.tokens_out)):
        return False
    if not isinstance(r.provider_ids, dict) or (r.origin is not None and not isinstance(r.origin, dict)):
        return False
    try:
        _canonical(r.provider_ids).encode("utf-8")
        _canonical(r.origin).encode("utf-8")
    except (TypeError, ValueError):
        return False
    return r.status != "answered" or isinstance(r.text, str)


class ModelGateway:
    """One episode's model gateway, for one condition in one stream.

    The journal holds three kinds of record, each fsynced as written (Codex's MG61 and MG65 reviews):
    - response: written as soon as a provider returns, before the end clock, with the request's metadata, the
      response and its origin;
    - completion: written once the call's charge is in a ledger, naming the row that carried it and the measured
      wall time;
    - recovery: written each time a later run recovers the response instead of calling again.
    A saved response is recovered only for the same identifier, fingerprint and request metadata (goal version, issue
    second and step, latency, attempt, output limit), so an answer to an old goal is never relabelled (MG65.ID).
    Tokens are charged once: by the original row if it completed, otherwise by the first recovering row. A recovery
    row links to the charged row and carries no tokens once they are charged; its wall time, copied from the
    completion record, is informational and is left out of time-based cost shares (ledger.recovers)."""

    def __init__(self, episode: str, condition: str, stream: str, provider: Provider, log: TrustedLog, ledger: Ledger,
                 journal: Path, replay: bool = False, limits: dict[str, int] | None = None,
                 clock: Callable[[], float] = time.perf_counter, purpose: str = "running",
                 id_scope: str | None = None):
        # The ledger purpose of this gateway's calls: 'running' for flying, 'build' for learning between episodes
        # (Codex's ACC133.BUILD). A replay gateway's calls are 'replay' whatever this says.
        if purpose not in ("running", "build", "trial", "planting", "setup"):
            raise GatewayError(f"unknown gateway purpose {purpose!r}")
        self._purpose = purpose
        # id_scope, if given, goes into every request identifier after the role, so a second gateway for the same
        # episode, condition and stream (a learning attempt's, beside flight's) can never reuse flight's identifiers.
        # Rehearsal of 15 October: both numbered from 1, and the pause refused the duplicate identity.
        if id_scope is not None and (not isinstance(id_scope, str) or not id_scope):
            raise GatewayError("id_scope must be a non-empty string")
        self._id_scope = id_scope
        self._episode, self._condition, self._stream = episode, condition, stream
        self._provider, self._log, self._ledger, self._journal_path = provider, log, ledger, Path(journal)
        self._replay, self._clock = replay, clock
        self.limits = dict(CALL_LIMITS if limits is None else limits)
        self.used = {role: 0 for role in ROLES}
        self._seq = 0
        self._responses: dict[tuple, dict] = {}
        self._completions: dict[tuple, dict] = {}
        self._recoveries: dict[tuple, int] = {}
        # The start boundary (TURN 97, Codex's SG96): set once, by the first record_start; calls_started is set by the
        # first model_start, so a later start record cannot admit an earlier call.
        self._start: dict | None = None
        self._calls_started = False
        self._authority: "startup.RunAuthority | None" = None
        self._closed = False
        self._void = False
        self._void_published = False
        self._void_drift: list[str] = []
        if self._journal_path.exists():
            for line in self._journal_path.read_text().splitlines():
                rec = json.loads(line)
                key = self._key(rec.get("request_id"), rec.get("fingerprint"), rec.get("meta"))
                kind = rec.get("type", "response")
                if kind == "response":
                    self._responses[key] = rec
                elif kind == "completion":
                    self._completions[key] = rec
                elif kind == "recovery":
                    self._recoveries[key] = self._recoveries.get(key, 0) + 1
        self._event({"kind": "gateway_open", "provider": provider.name, "reportable": provider.reportable,
                     "replay": replay, "limits": self.limits})
        PROTECTED_ROOTS.register(log.path, log.protected)      # this run's own log is an original root (ADM90.ROOT)
        bind = getattr(provider, "bind_anchor", None)
        if callable(bind):
            # An exchange's digests go into this trusted log as soon as the exchange is saved or reused, before its
            # reply is parsed, so the gap before the journal is covered by protected evidence (TURN 85).
            bind(lambda rec: self._event({"kind": "exchange_anchor", **rec}), (log.path, log.protected))

    @staticmethod
    def _key(request_id: Any, fingerprint: Any, meta: Any) -> tuple:
        return (request_id, fingerprint, _canonical(meta) if isinstance(meta, dict) else None)

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    def _event(self, payload: dict) -> None:
        self._seq += 1
        self._log.append({"episode": self.episode, "condition": self.condition, "stream": self.stream,
                          "seq": self._seq, "reportable": self._provider.reportable, **payload})
        if payload.get("kind") == "model_start":
            self._calls_started = True

    # The run's scope is fixed at construction (Codex's SG98.CURRENT): a start bound to one stream cannot be borrowed by
    # another by changing these afterwards.
    @property
    def episode(self) -> str:
        return self._episode

    @property
    def condition(self) -> str:
        return self._condition

    @property
    def stream(self) -> str:
        return self._stream

    def _store_backed(self) -> bool:
        return getattr(self._provider, "_exchanges", None) is not None

    def _start_proof(self) -> list[str]:
        """Re-prove this run's start from its protected log before any usable path (SG98.CURRENT): the start was admitted,
        the log still verifies against its protected heads, its first start record still comes before every model call
        and is still admitted for this log, scope and configuration, the run is not closed, and, if the authority asks,
        the captured code has not drifted. Returns the problems; empty means proven."""
        if not self.started:
            return list((self._start or {}).get("problems", ["no start record"]))
        if self._closed:
            return ["the run is closed"]
        if self._void:
            # Drift once found voids this run's code identity for good; restoring the files does not revalidate it.
            # Only a new run from a new reviewed bootstrap can (Codex's SG100.VOID.PERSIST). A void event that could
            # not be written is retried here; if it still fails, the refusal the caller logs next carries the same
            # disposition, which _code_void also treats as void evidence.
            try:
                self._publish_void()
            except Exception:
                pass
            return ["code identity void"]
        try:
            check = verify(self._log.path, self._log.protected)
            if not check.get("ok"):
                return [f"the run's log does not verify: {check.get('problem')}"]
            event, before = startup.first_start(self._log.path)
        except (OSError, ValueError) as e:
            return [f"the run's log cannot be read: {type(e).__name__}"]
        if event is None or not before:
            return ["the run's start record is missing from its log"]
        problems = startup.decide(event, startup.log_anchor(self._log.protected), self.episode, self.condition,
                                  self.stream, self._authority.config, before_first_call=True)
        if not problems and self._authority.check_drift:
            problems = [f"code drift: {r}" for r in self._authority.drift()]
            if problems:
                # The disposition is set in memory first and never cleared; then it is published. If publishing fails
                # the error propagates now, and every later proof retries it (Codex's VOID102.WRITE).
                self._void = True
                self._void_drift = [p[len("code drift: "):] for p in problems]
                self._publish_void()
        return problems

    def _publish_void(self) -> None:
        if not self._void_published:
            self._event({"kind": "code_identity_void", "code_drift": self._void_drift})
            self._void_published = True

    def close(self) -> list[str]:
        """End the run: check the captured code for drift once more and record the result (SG96.6). Drift found here
        voids the run's code-identity claim; it does not undo or recharge any call. After close no fresh call is made."""
        drift = self._authority.drift() if (self._authority is not None and self._authority.check_drift) else []
        self._void = self._void or bool(drift)
        self._event({"kind": "run_close", "code_drift": drift, "code_identity_void": self._void})
        self._closed = True
        return drift

    @property
    def started(self) -> bool:
        """Whether this run's start was admitted."""
        return bool(self._start and self._start["admitted"])

    def record_start(self, record: dict, authority: "startup.RunAuthority") -> list[str]:
        """Write this run's start record into its own protected log, bound to this log, scope and configuration, and
        decide admission once (Codex's SG96.2). Returns the problems; empty means admitted. Only the first call counts:
        a later record is logged as ignored and cannot admit or retract anything. If the write itself fails, nothing is
        admitted and the error propagates, so an incomplete start never permits provider work (SG96.7)."""
        if self._start is not None:
            self._event({"kind": "start_record_ignored", "reason": "a start record was already written"})
            return ["a start record was already written"]
        if not isinstance(authority, startup.RunAuthority):
            raise GatewayError("a start needs a RunAuthority from the trusted bootstrap")
        anchor = startup.log_anchor(self._log.protected)
        try:
            startup.validate(record)
            clean, digest = record, hashlib.sha256(startup._canonical(record).encode("utf-8")).hexdigest()
        except startup.StartupError:
            clean, digest = None, None
        event = {"kind": "start_record",
                 "binding": startup.binding(anchor, self.episode, self.condition, self.stream,
                                            authority.config.fingerprint),
                 "record": clean, "record_sha256": digest, "code": authority.code}
        problems = startup.decide(event, anchor, self.episode, self.condition, self.stream, authority.config,
                                  before_first_call=not self._calls_started)
        provider_config = getattr(self._provider, "config", None)
        if not isinstance(provider_config, CheckedConfig) or provider_config.fingerprint != authority.config.fingerprint:
            problems.append("the provider's configuration is not the authority's")
        self._event({**event, "admitted": not problems, "problems": problems})
        self._start = {"admitted": not problems, "problems": list(problems)}
        self._authority = authority
        return problems

    def _write(self, rec: dict) -> None:
        with self._journal_path.open("a") as f:
            f.write(_canonical(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _meta(request: ModelRequest) -> dict:
        return request_meta(request)

    def submit(self, role: str, messages: list[tuple[str, str]], max_output_tokens: int, goal_version: int,
               sim_second: int, action_step: int, modelled_latency_s: float, attempt: int = 1) -> dict:
        """Make one model call. Returns the outcome: request_id, status, text, tokens, and the simulated second and
        action step at which the answer becomes available. Over the role's limit, nothing is called or charged and
        the outcome is the public refusal 'over_allowance'."""
        if role not in ROLES:
            raise GatewayError(f"unknown role {role}")
        if (not isinstance(messages, list) or not messages
                or not all(isinstance(m, tuple) and len(m) == 2 and all(isinstance(x, str) for x in m) for m in messages)):
            raise GatewayError("messages must be a non-empty list of (speaker, text) pairs")
        if len(_canonical([list(m) for m in messages]).encode()) > MAX_MESSAGES_BYTES:
            raise GatewayError("messages over the request bound")
        for name, v in (("max_output_tokens", max_output_tokens), ("goal_version", goal_version),
                        ("sim_second", sim_second), ("action_step", action_step), ("attempt", attempt)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise GatewayError(f"{name} must be a non-negative integer")
        if (isinstance(modelled_latency_s, bool) or not isinstance(modelled_latency_s, (int, float))
                or not math.isfinite(modelled_latency_s) or modelled_latency_s < 0):
            raise GatewayError("modelled latency must be finite and non-negative")
        n = self.used[role] + 1
        # The identifier is the JSON encoding of (stream, episode, condition, role, [id_scope,] count), so no two different tuples
        # can share it, whatever characters the names hold (Codex's MG61.IDENTITY and MG65.ID).
        scope = [] if self._id_scope is None else [self._id_scope]
        request_id = _canonical([self.stream, self.episode, self.condition, role, *scope, n])
        request = ModelRequest(request_id, f"attempt{attempt}", self.episode, self.condition, self.stream, role,
                               tuple(messages), max_output_tokens, goal_version, sim_second, action_step,
                               float(modelled_latency_s))
        delay = math.ceil(modelled_latency_s)
        when = {"available_second": sim_second + delay, "available_step": action_step + delay}
        if self.used[role] >= self.limits[role]:
            self._event({"kind": "model_refused", "role": role, "refusal": "over_allowance",
                         "sim_second": sim_second, "action_step": action_step})
            return {"request_id": None, "status": "refused", "refusal": "over_allowance", "text": None}
        meta = self._meta(request)
        key = self._key(request_id, request.fingerprint, meta)
        saved = self._responses.get(key)
        recorded = {c.request_id for c in self._ledger.calls}
        # Refusals before anything is called or charged (Codex's MG67 review):
        if saved is None and request_id in recorded:
            # the same identity was already charged in this ledger for a request with other details (MG67.ID.LEDGER)
            raise GatewayError(f"request {request_id} is already in this ledger with different details")
        proof = self._start_proof() if self._store_backed() else []
        if saved is None and proof:
            # No fresh call for a store-backed provider until this run's start is admitted and still proven from its
            # protected log (SG96.3, SG98.CURRENT): no model_start, no allowance used, no provider contact and no
            # charge. This refusal is not a model attempt.
            self._event({"kind": "startup_refused", "role": role, "sim_second": sim_second, "action_step": action_step,
                         "problems": proof})
            return {"request_id": None, "status": "refused", "refusal": "startup not admitted", "text": None,
                    "reportable": False}
        self.used[role] = n
        scope = "replay" if self._replay else self._purpose
        # The full request is logged before anything else happens, so an interrupted call leaves its start.
        self._event({"kind": "model_start", "request": {**asdict(request), "messages": [list(m) for m in messages]},
                     "fingerprint": request.fingerprint, **when})
        if saved is not None:
            return self._recover(request, key, meta, saved, scope, when)
        started = wall = None
        phase = "start clock"
        response = ModelResponse("failed")
        reportable = bool(self._provider.reportable)
        try:
            try:
                started = self._clock()
            except Exception:
                started = None
            phase = "provider"
            try:
                got = self._provider.complete(request)
                response = got if _response_ok(got) else _salvage(got)
            except Exception:
                response = ModelResponse("failed")
            if getattr(self._provider, "relays", False):
                # a relaying provider's answers are reportable only through a complete origin record
                reportable = response.origin is not None and origin_reportable(response.origin)
            elif response.origin is not None:
                # a producer cannot promote its own output; an origin field can only lower reportability
                reportable = reportable and origin_reportable(response.origin)
            # Journalled before the end clock, so a later run recovers it instead of asking again (MG61.RECEIVED).
            phase = "saving"
            rec = {"type": "response", "request_id": request_id, "fingerprint": request.fingerprint, "meta": meta,
                   "response": asdict(response), "provider": self._provider.name, "reportable": reportable,
                   "wall_s": None, "purpose": scope}
            self._write(rec)
            self._responses[key] = rec
            phase = "end clock"
            reused = isinstance(response.provider_ids, dict) and response.provider_ids.get("exchange_reused") == "true"
            if reused:
                # A reuse took no server time: its time is the original's, verified from the protected log, or unknown
                # (EX84.TIME, COST100.REUSE.TIME); the time spent reading the store is never used.
                reuse_wall = getattr(self._provider, "reuse_wall", None)
                try:
                    wall = reuse_wall(request) if callable(reuse_wall) else None
                except Exception:
                    wall = None
            elif started is not None:
                try:
                    wall = self._clock() - started
                except Exception:
                    wall = None
        except BaseException as exc:
            # The interrupted row carries no tokens; if a response was received, they are charged once, by the first
            # row that recovers it.
            if phase == "provider" and started is not None:
                try:
                    wall = self._clock() - started
                except Exception:
                    wall = None
            self._ledger.add(Call(request_id, role, self.episode, action_step, started, wall, None, None,
                                  "interrupted", scope=scope, condition=self.condition, stream=self.stream,
                                  sim_second=sim_second, answered_step=None))
            self._event({"kind": "model_interrupted", "request_id": request_id, "phase": phase,
                         "error": f"{type(exc).__name__}: {exc}"})
            raise
        # An answer and a failure alike arrive after the modelled latency (DISTURBANCES section 8).
        self._ledger.add(Call(request_id, role, self.episode, action_step, started, wall, response.tokens_in,
                              response.tokens_out, response.status, scope=scope, condition=self.condition,
                              stream=self.stream, sim_second=sim_second, answered_step=when["available_step"]))
        done = {"type": "completion", "request_id": request_id, "fingerprint": request.fingerprint, "meta": meta,
                "charged_row": request_id, "wall_s": wall}
        self._write(done)
        self._completions[key] = done
        self._event({"kind": "model_end", "request_id": request_id, "ledger_row": request_id,
                     "status": response.status, "recovered": False, "response": asdict(response), "wall_s": wall,
                     "reportable": reportable})
        return {"request_id": request_id, "status": response.status, "text": response.text,
                "tokens_in": response.tokens_in, "tokens_out": response.tokens_out, "goal_version": goal_version,
                "reportable": reportable, "recovered": False, **when}

    def _recover(self, request: ModelRequest, key: tuple, meta: dict, saved: dict, scope: str, when: dict) -> dict:
        """Use a saved response without calling the model. Tokens are charged here only if no completion record shows
        an earlier charge; reportability comes from the saved origin and is never promoted.

        The original call's purpose is in its journal record. A gateway of another purpose may not recover it, so a
        learning call is never re-booked as flying or the reverse (Codex's ACC135.PURPOSE); the refusal comes before
        anything is charged. A replay gateway re-flies on purpose and books its rows as 'replay'. A record written
        before the purpose was kept has none; its original purpose is unknown, so a non-replay gateway refuses it too,
        and never books it under its own purpose (Codex's ACC137.LEGACY.PURPOSE)."""
        original = saved.get("purpose")
        if not self._replay and original != scope:
            if original is None:
                raise GatewayError(f"saved response for {request.request_id} predates recorded purposes; its "
                                   f"original purpose is unknown, so it is not recovered as a {scope!r} call")
            raise GatewayError(f"saved response for {request.request_id} was a {original!r} call; this gateway's "
                               f"purpose is {scope!r}")
        response = ModelResponse(**saved["response"])
        verified = None
        admit = getattr(self._provider, "admit_journal", None)
        if callable(admit) and getattr(self._provider, "_exchanges", None) is not None:
            # A provider that keeps exchange evidence checks every journalled response against it before it is used
            # (Codex's ANCH88.JOURNAL, ADM90.REQUIRED); a failure there quarantines the response, with unknown usage.
            try:
                response, verified = admit(request, response)
            except Exception:
                response = replace(response, status="failed", text=None, tokens_in=None, tokens_out=None,
                                   origin={"provider": "admission failed", "reportable": False})
                verified = {"wall_s": None}
        if self._store_backed() and self._start_proof():
            # A recovery in a run whose start is not admitted and proven keeps the original's verified cost but gives no
            # usable text and is not reportable (SG96.3, SG96.4, SG98.CURRENT).
            response = replace(response, status="failed", text=None,
                               provider_ids={**(response.provider_ids or {}), "refused": "startup not admitted"},
                               origin={"provider": "startup not admitted", "reportable": False})
        reportable = (isinstance(saved.get("provider"), str) and saved["provider"] != ""
                      and saved.get("reportable") is True)
        if response.origin is not None:
            reportable = reportable and origin_reportable(response.origin)
        k = self._recoveries.get(key, 0) + 1
        row_id = f"{request.request_id}#recovered{k}"
        completion = self._completions.get(key)
        recorded = {c.request_id for c in self._ledger.calls}
        charged_row = completion.get("charged_row") if completion is not None else None
        # The row carrying this call's charge in this ledger: the journal's charged row if it is here, otherwise an
        # earlier row for the same request here that carried a charge itself (a completed original or a recovery
        # made in this ledger without a link).
        family = [c for c in self._ledger.calls
                  if c.request_id == request.request_id or c.request_id.startswith(request.request_id + "#recovered")]
        here = charged_row if charged_row in recorded else next(
            (c.request_id for c in family if c.recovers is None and c.status != "interrupted"), None)
        if here is not None:
            # The charge is already in this ledger: a linked row with no tokens, its wall time informational.
            tokens_in = tokens_out = None
            wall = completion.get("wall_s") if completion is not None else None
            links_to, charged_here = here, False
        else:
            # This ledger has no charge for the call (a fresh ledger, or an interruption before the charge was
            # booked): this row carries it, so no ledger hides the cost (Codex's MG67.COST.FRESH). It links to the
            # interrupted row only if that row is in this ledger.
            tokens_in, tokens_out = response.tokens_in, response.tokens_out
            wall = completion.get("wall_s") if completion is not None else None
            links_to = request.request_id if (completion is None and request.request_id in recorded) else None
            charged_here = True
        if verified is not None:
            # The journal's completion time is unprotected; only the time the protected original log recorded is used,
            # otherwise it is unknown (COST90.TIME).
            wall = verified.get("wall_s")
        self._ledger.add(Call(row_id, request.role, self.episode, request.action_step, None, wall, tokens_in, tokens_out,
                              response.status, scope=scope, condition=self.condition, stream=self.stream,
                              sim_second=request.sim_second, answered_step=when["available_step"], recovers=links_to))
        self._write({"type": "recovery", "request_id": request.request_id, "fingerprint": request.fingerprint,
                     "meta": meta, "row": row_id})
        self._recoveries[key] = k
        if completion is None:                           # this row carried the charge; later recoveries must not
            done = {"type": "completion", "request_id": request.request_id, "fingerprint": request.fingerprint,
                    "meta": meta, "charged_row": row_id, "wall_s": None}
            self._write(done)
            self._completions[key] = done
        self._event({"kind": "model_end", "request_id": request.request_id, "ledger_row": row_id,
                     "status": response.status, "recovered": True, "charged_here": charged_here,
                     "journal_charged_row": charged_row,
                     "response": asdict(response), "wall_s": wall, "reportable": reportable})
        return {"request_id": request.request_id, "status": response.status, "text": response.text,
                "tokens_in": response.tokens_in, "tokens_out": response.tokens_out,
                "goal_version": request.goal_version, "reportable": reportable, "recovered": True, **when}
