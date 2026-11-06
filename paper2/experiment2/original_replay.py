"""Conditional B8 original-state replay, before acquisition or scored use.

Canonical bytes prevent caller mutation; hashes do not establish protection.
Only the trusted caller's independently retained capture digest can identify
the original. This route neither persists that authority nor admits labels.
Constant and changing-radius motion; finite explicit controls, never a solver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import scipy
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P
from . import original_motion as M
from . import monitor_labels as L; from . import contact_bounds as K
from .monitor_labels import Original

RULE = "B8.original.replay.v2"
QUALIFICATION = "conditional_supplied_original_authority_and_numerical_error"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def _retained(value):
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {k: _retained(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_retained(v) for v in value]
    return value


def sources():
    """Observed file hashes and loaded runtime settings, not host integrity proof."""
    return {"files": {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                      for name, module in (("episode", E), ("physics", P), ("motion", M),
                                           ("labels", L), ("contact", K))} |
                     {"replay": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "python": platform.python_version(), "scipy": scipy.__version__,
            "tie_band": P.TIE_BAND, "numeric_allowance": P.NUMERIC_ALLOWANCE,
            "distance_rate_bound": P.DISTANCE_RATE_BOUND, "horizon": E.HORIZON,
            "join_tolerance": E.JOIN_TOLERANCE, "report_tolerance": E.REPORT_TOLERANCE,
            "report_flag_margin": E.REPORT_FLAG_MARGIN, "inbound_flag_margin": E.INBOUND_FLAG_MARGIN}


@dataclass(frozen=True)
class Identity:
    episode: str
    condition: str
    stream: int
    split: str
    decision: str
    interface: str
    second: int
    action_step: int
    goal_version: int
    ordinal: int
    attempt: int
    rewrite: int | None


@dataclass(frozen=True)
class Capture:
    body: bytes
    sha256: str
    qualification: str = QUALIFICATION
    admissible: bool = False


@dataclass(frozen=True)
class Limits:
    max_steps: int = 3600
    join_evaluations_per_step: int = 4096
    total_distance_queries: int = 32768
    total_tanker_queries: int = 32768
    quadrature_per_query: int = 4096
    total_integrand_evaluations: int = 262144
    input_bytes: int = 1048576
    trace_stop_bytes: int = 8388608

    def validate(self):
        for v, low, high in ((self.max_steps, 1, 3600),
                             (self.join_evaluations_per_step, 3, 200000),
                             (self.total_distance_queries, 1, 32768),
                             (self.total_tanker_queries, 1, 32768),
                             (self.quadrature_per_query, 1, 4096),
                             (self.total_integrand_evaluations, 1, 262144),
                             (self.input_bytes, 1024, 1048576),
                             (self.trace_stop_bytes, 1024, 8388608)):
            if type(v) is not int or not low <= v <= high:
                raise ValueError("invalid replay work limit")


@dataclass(frozen=True)
class Evidence:
    disposition: str
    reason: str
    upper_elapsed_s: int | None
    body: bytes
    sha256: str
    cpu_seconds: float
    qualification: str = QUALIFICATION
    admissible: bool = False


def _state(ep):
    # Keep every mutable world field, including histories under earlier goals.
    return {"t": ep.t, "receiver": asdict(ep.receiver), "goal": asdict(ep.goal),
            "pending_goals": [[s, asdict(g)] for s, g in ep.pending_goals],
            "trigger": None if ep.trigger is None else asdict(ep.trigger),
            "pending_help": [[s, asdict(g), h] for s, g, h in ep.pending_help],
            "queue": ep.queue, "record": asdict(ep.record), "done": ep.done,
            "delivered": ep._delivered, "join_alternatives": ep._join_alternatives}


RECORD_LISTS = ("actions","reports","help_requests","refusals","flags")


def _trace_state(ep):
    """Compact step snapshot; full original/terminal plus deltas retain history.

    Repeating the entire growing action history in every step would make the
    audit quadratic in storage. Counts and digest bind the reconstructed record.
    """
    state=_state(ep)
    record=state.pop("record")
    alternatives=state.pop("join_alternatives")
    state["record_sha256"]=digest(record)
    state["record_counts"]={k:len(record[k]) for k in RECORD_LISTS}
    state["record_status"]={k:v for k,v in record.items() if k not in RECORD_LISTS}
    state["join_alternative_count"]=len(alternatives)
    state["join_alternatives_sha256"]=digest(alternatives)
    return state


def _identity(identity, ep):
    if not isinstance(identity, Identity):
        raise TypeError("Identity required")
    for name in ("episode", "condition", "split"):
        if not isinstance(getattr(identity, name), str) or not getattr(identity, name):
            raise ValueError("missing original identity")
    for name in ("stream", "second", "action_step", "goal_version", "ordinal", "attempt"):
        v = getattr(identity, name)
        if type(v) is not int or v < 0:
            raise ValueError("invalid original integer")
    if identity.rewrite is not None and (type(identity.rewrite) is not int or identity.rewrite < 0):
        raise ValueError("invalid rewrite")
    if identity.decision not in ("heading", "report") or identity.interface != "action":
        raise ValueError("unsupported decision/interface")
    if type(ep.t) is not int or type(ep.goal.version) is not int or type(ep.done) is not bool:
        raise ValueError("invalid original world clock, goal or ending type")
    if (identity.episode != ep.s.episode_id or identity.second != ep.t or
            identity.action_step != len(ep.record.actions) or identity.goal_version != ep.goal.version):
        raise ValueError("original identity conflicts with world")
    if ep.done or ep.t >= E.HORIZON:
        raise ValueError("original episode has ended")
    if ep.receiver.t != ep.t:
        raise ValueError("receiver clock differs from original")


def capture(ep, identity, *, context, input_bundle, issue, actor_pending):
    """Trusted host capture at the decision subphase; detached, no file writes.

    The caller supplies actual actor context/input/issue/pending originals.
    Their content and hashes are retained, not inferred from expected settings.
    No wiring into the live runner or protected registry is supplied here.
    """
    if type(ep) is not E.Episode:
        raise TypeError("exact world Episode required")
    _identity(identity, ep)
    if not isinstance(context, dict) or context.get("t") != ep.t or context.get("goal_version") != ep.goal.version:
        raise ValueError("context clock or goal conflicts")
    if not isinstance(input_bundle, dict) or input_bundle.get("t") != ep.t:
        raise ValueError("input bundle clock conflicts")
    if not isinstance(issue, dict) or not issue or not isinstance(actor_pending, dict):
        raise ValueError("actual issue and pending actor originals required")
    raw = {"rule": RULE, "identity": asdict(identity), "scenario": asdict(ep.s),
           "state": _state(ep), "sources": sources(), "context": context,
           "input_bundle": input_bundle, "issue": issue, "actor_pending": actor_pending,
           "hashes": {"context": digest(context), "input": digest(input_bundle),
                      "issue": digest(issue), "actor_pending": digest(actor_pending),
                      "goal": digest(asdict(ep.goal))}}
    body = encode(raw)
    return Capture(body, hashlib.sha256(body).hexdigest())


def _restore(raw):
    # The retained evidence's original dictionary must not share mutable
    # record/delivery/alternative lists with the resumed Episode.
    raw=json.loads(encode(raw))
    sc = raw["scenario"]
    scenario = E.Scenario(sc["episode_id"], P.ReceiverState(**sc["receiver"]), tuple(sc["wind"]),
                         tuple(E.Tanker(t["callsign"], P.Orbit(**t["orbit"])) for t in sc["tankers"]),
                         tuple((s, E.Goal(**g)) for s, g in sc["goals"]), sc["control_callsign"],
                         sc["delay_s"], tuple(tuple(d) for d in sc["dropouts"]))
    ep = E.Episode(scenario)
    state = raw["state"]
    expected = set(_state(ep))
    if set(state) != expected:
        raise ValueError("missing or unknown world state")
    if set(state["record"]) != set(asdict(E.EpisodeRecord())) or set(sc) != set(asdict(scenario)):
        raise ValueError("missing or unknown scenario or record field")
    ep.t = state["t"]
    ep.receiver = P.ReceiverState(**state["receiver"])
    ep.goal = E.Goal(**state["goal"])
    ep.pending_goals = [(s, E.Goal(**g)) for s, g in state["pending_goals"]]
    ep.trigger = None if state["trigger"] is None else E.Trigger(**state["trigger"])
    ep.pending_help = [(s, E.Goal(**g), h) for s, g, h in state["pending_help"]]
    ep.queue = [(s, r) for s, r in state["queue"]]
    ep.record = E.EpisodeRecord(**state["record"])
    ep.done = state["done"]
    ep._delivered = state["delivered"]
    ep._join_alternatives = state["join_alternatives"]
    _identity(Identity(**raw["identity"]), ep)
    return ep


def _domain(ep):
    """Premises needed by the retained global distance-rate bound."""
    r = ep.receiver
    if not (0 < r.tas <= P.MAX_TAS_KT / 3600 and r.omega > 0 and
            0 <= r.heading < math.tau and 0 <= r.commanded < math.tau):
        return "receiver outside reviewed motion domain"
    if math.hypot(*ep.s.wind) > P.MAX_WIND_KT / 3600:
        return "wind outside reviewed motion domain"
    if len(ep.tankers) != len(ep.s.tankers) or ep.goal.tanker not in ep.tankers:
        return "invalid tanker identities"
    for tanker in ep.s.tankers:
        o = tanker.orbit
        if o.r0 <= 0 or not 0 <= o.v <= P.MAX_TANKER_KT / 3600:
            return "tanker outside reviewed motion domain"
        if o.delta != 0 and (o.r0+o.delta <= 0 or o.duration <= 0 or o.start < 0 or
                            o.start+o.duration > E.HORIZON or abs(o.delta)*math.pi/(2*o.duration) > o.v):
            return "changing motion outside reviewed domain"
    if not 0 < ep.goal.report_at_nm <= 1000:
        return "invalid original report order"
    if ep.goal.deadline is not None and (type(ep.goal.deadline) is not int or ep.goal.deadline <= ep.t):
        return "closed or invalid original deadline"
    return None


class _Cap(Exception):
    pass


def _install_motion(ep, original, raw, limits):
    """Instrument *all* tanker queries: join, report position/velocity, delivery.

    Per-query quadrature budget is narrowed by the remaining combined budget;
    caps and backend failure diagnostics are retained before raising. No
    uninstrumented world quadrature call or cache can bypass these limits.
    """
    identity = original["identity"]
    origin = Original(identity["episode"],identity["goal_version"],identity["second"],
                      original["hashes"]["goal"],original["hashes"]["input"],original["hashes"]["context"],
                      digest(raw["candidate"]),digest(original["sources"]))
    receiver = M.Receiver(ep.t,ep.receiver.x,ep.receiver.y,ep.receiver.heading,ep.receiver.tas,ep.receiver.omega)

    class QueryOrbit:
        def __init__(self, tanker):
            self.orbit, self.callsign = tanker.orbit, tanker.callsign
            o=self.orbit
            mt=M.ChangingTanker(o.cx,o.cy,o.r0,o.v,o.theta0,o.delta,o.start,o.duration)
            self.motion=M.ChangingMotion.bind(origin,receiver,mt,ep.s.wind,math.degrees(ep.receiver.commanded)%360)

        def point(self,t,kind):
            if len(raw["tanker_queries"]) >= limits.total_tanker_queries:
                raise _Cap("total tanker query cap")
            left=limits.total_integrand_evaluations-raw["integrand_evaluations"]
            # Analytic queries consume no integrand budget; the adapter handles
            # a zero remaining budget on an actual changing-radius query.
            motion=replace(self.motion,quadrature_evaluation_cap=min(left,limits.quadrature_per_query))
            q=motion.tanker_query(t)
            raw["tanker_queries"].append({"callsign":self.callsign,"kind":kind,
                                           "motion_fingerprint":self.motion.fingerprint,**asdict(q)})
            raw["integrand_evaluations"]+=q.integrand_evaluations
            if q.state != "ok":
                if q.state == "evaluation_cap": raise _Cap(q.error)
                raise ValueError(q.error)
            return q.point

        def position(self,t):
            return self.point(t,"position")

        def velocity(self,t):
            x,y=self.point(t,"velocity")
            o=self.orbit
            r,rd,ad=o.radius(t),o.radius_rate(t),o._angle_rate(t)
            sn,cs=(x-o.cx)/r,(y-o.cy)/r
            # Differentiate the retained query point under the declared orbit.
            # Floating point normalisation introduces an additional conditional
            # error premise; the quadrature estimate is not a velocity bound.
            return rd*sn+r*ad*cs,rd*cs-r*ad*sn

    tankers=tuple(E.Tanker(k.callsign,QueryOrbit(k)) for k in ep.s.tankers)
    ep.s=replace(ep.s,tankers=tankers)
    ep.tankers={k.callsign:k for k in tankers}


def replay(original, candidate, continuation, *, expected_capture_sha256, limits=Limits()):
    """Replay one candidate second then a fixed finite full-action continuation.

    No policy callback, actor pending reply delivery, auto-completion or seed
    refill. Future controls represent an offline existence witness. All times
    remain absolute; a nominal finite upper remains conditional/inadmissible.
    Exceptions retain the attempted prefix and every completed distance query.
    """
    started = time.process_time()
    raw = {"rule": RULE, "capture_sha256": getattr(original, "sha256", None),
           "candidate": candidate, "continuation": continuation, "limits": asdict(limits),
           "steps": [], "distance_queries": [], "tanker_queries": [], "integrand_evaluations":0,
           "outcome": None}
    ep = None

    def result(disposition, reason, upper=None):
        raw["disposition"], raw["reason"], raw["upper_elapsed_s"] = disposition, reason, upper
        raw["terminal"] = None if ep is None else _state(ep)
        raw["qualification"], raw["admissible"] = QUALIFICATION, False
        body = encode(_retained(raw))
        return Evidence(disposition, reason, upper, body, hashlib.sha256(body).hexdigest(),
                        time.process_time() - started)

    if not isinstance(original, Capture) or type(original.body) is not bytes:
        return result("invalid", "Capture with immutable bytes required")
    if hashlib.sha256(original.body).hexdigest() != original.sha256 or original.sha256 != expected_capture_sha256:
        return result("invalid", "capture differs from independently retained original digest")
    try:
        saved = json.loads(original.body)
    except (ValueError, UnicodeError):
        return result("invalid", "malformed original JSON")
    if not isinstance(saved, dict):
        return result("invalid", "original object required")
    raw["original"] = saved
    if set(saved) != {"rule", "identity", "scenario", "state", "sources", "context", "input_bundle",
                      "issue", "actor_pending", "hashes"}:
        return result("invalid", "missing or unknown original field")
    if saved.get("rule") != RULE or saved.get("sources") != sources():
        return result("invalid", "source or numerical settings conflict")
    # Finite JSON is required before a result can retain this invocation.
    encode(raw)
    limits.validate()
    if len(original.body)+len(encode([candidate,continuation])) > limits.input_bytes:
        return result("invalid","original and plan input byte limit")
    if not isinstance(candidate, dict) or not isinstance(continuation, (list, tuple)):
        return result("invalid", "explicit public candidate and finite control sequence required")
    actions = json.loads(encode([candidate, *continuation]))
    raw["plan_sha256"] = digest(actions)
    try:
        ep = _restore(saved)
        if len(actions) > limits.max_steps or len(actions) > E.HORIZON - ep.t:
            return result("invalid", "plan exceeds original horizon or step limit")
        for name, obj in (("context", saved["context"]), ("input", saved["input_bundle"]),
                          ("issue", saved["issue"]), ("actor_pending", saved["actor_pending"]),
                          ("goal", asdict(ep.goal))):
            if saved["hashes"][name] != digest(obj):
                return result("invalid", "original content fingerprint conflict")
    except (TypeError, ValueError, KeyError):
        return result("invalid", "malformed or conflicting original state")
    why = _domain(ep)
    if why:
        return result("unknown", why)
    if ep.record.flags or ep._join_alternatives:
        return result("unknown", "original numerical ending or report evidence unsettled")
    # A future change outside this tested finite plan is preserved but need not
    # invalidate an earlier ending. A change at its endpoint is relevant.
    if any(s <= ep.t + len(actions) for s, _ in ep.pending_goals):
        return result("unknown", "goal changes within the proposed continuation")
    origin = ep.t
    try:
        _install_motion(ep,saved,raw,limits)
    except (TypeError,ValueError) as error:
        return result("unknown","motion adapter binding: "+str(error)[:200])
    base_range = ep._true_range

    def distance(callsign, second):
        if len(raw["distance_queries"]) >= limits.total_distance_queries:
            raise _Cap("total distance query cap")
        q = {"tanker": callsign, "second": second}
        raw["distance_queries"].append(q)
        q["point"] = list(ep.tankers[callsign].orbit.position(second))
        q["receiver"] = asdict(P.receiver_at(ep.receiver, ep.s.wind, second))
        q["range_nm"] = base_range(callsign, second)
        if not math.isfinite(q["range_nm"]):
            raise ValueError("nonfinite distance query")
        return q["range_nm"]

    ep._true_range = distance
    try:
        for index, action in enumerate(actions):
            if len(encode(_retained(raw))) > limits.trace_stop_bytes:
                raise _Cap("trace stop byte threshold; preceding step retained")
            row = {"second": ep.t, "prefix": index == 0, "action": action, "before": _trace_state(ep)}
            counts={k:len(getattr(ep.record,k)) for k in RECORD_LISTS}
            alternative_count=len(ep._join_alternatives)
            # Detach the pre-step state before the live object appends records.
            row = json.loads(encode(row))
            raw["steps"].append(row)
            error = ep._action_error(action)
            ep._apply_actions(action)
            row["accepted"], row["refusal"] = error is None, error
            ep._check_trigger()  # also runs after complete public refusal
            t0, t1, goal = ep.t, ep.t + 1, ep.goal
            decision, approach = P.join_assessment(lambda u: ep._true_range(goal.tanker, u), float(t0), float(t1),
                                                  max_evaluations=limits.join_evaluations_per_step)
            row["join"] = {"decision": decision, "enclosure": asdict(approach)}
            ep.receiver, ep.t = P.receiver_at(ep.receiver, ep.s.wind, float(t1)), t1
            if decision in ("flagged", "unresolved"):
                kind = "join near tolerance" if decision == "flagged" else "join unresolved at the work cap"
                ep.record.flags.append({"t": t1, "kind": kind, "tanker": goal.tanker, "enclosure": asdict(approach)})
                alternative = ep._score(goal, "join", t1, goal.tanker)
                alternative["assumption"] = "earlier flagged joins are no join; this flagged interval joins"
                alternative["join_assessment"] = asdict(approach)
                ep._join_alternatives.append(alternative)
            if decision == "join":
                ep._end("join", t1, goal.tanker)
            elif goal.deadline is not None and t1 >= goal.deadline:
                ep._end("deadline", t1)
            elif t1 >= E.HORIZON:
                ep._end("horizon", t1)
            else:
                ep._measure_reports(t1)
                row["bundle"] = ep._bundle(goal_changed=False)
            row["record_delta"]={k:getattr(ep.record,k)[counts[k]:] for k in RECORD_LISTS}
            row["join_alternatives_delta"]=ep._join_alternatives[alternative_count:]
            row["after"] = json.loads(encode(_trace_state(ep)))
            raw["outcome"] = ep.outcome()
            if error is not None:
                return result("invalid" if index == 0 else "unknown", "public action refused")
            if ep.record.flags:
                return result("unknown", "numerical join or report evidence unsettled")
            if ep.done:
                if raw["outcome"]["settled"] and raw["outcome"]["goal_met"]:
                    return result("nominal_upper", "settled whole-goal continuation under retained premises", ep.t - origin)
                return result("unknown", "continuation ended without whole-goal success")
        return result("unknown", "finite continuation exhausted; no impossibility claim")
    except Exception as error:
        return result("unknown", ("work cap: " if isinstance(error, _Cap) else "replay error: ") +
                      f"{type(error).__name__}: {str(error)[:200]}")


def reproduce(original, evidence, *, expected_capture_sha256):
    """Reconstruct the retained plan, never trust its supplied success boolean.

    Same numerical backend: deterministic corroboration, not an independent
    error certificate or a store verifier. CPU timing is outside the logical
    fingerprint. A different original/source/plan/result cannot pass.
    """
    if not isinstance(evidence, Evidence) or hashlib.sha256(evidence.body).hexdigest() != evidence.sha256:
        return False
    try:
        saved = json.loads(evidence.body)
        again = replay(original, saved["candidate"], saved["continuation"],
                       expected_capture_sha256=expected_capture_sha256, limits=Limits(**saved["limits"]))
        return (again.body == evidence.body and again.disposition == evidence.disposition and
                again.upper_elapsed_s == evidence.upper_elapsed_s and again.reason == evidence.reason and
                evidence.qualification == QUALIFICATION and evidence.admissible is False)
    except (TypeError, ValueError, KeyError):
        return False
