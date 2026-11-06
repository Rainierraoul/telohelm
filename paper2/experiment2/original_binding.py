"""Typed original-decision binding; conditional on independently held originals.

No registry, containment, model call, admission or live-runner hook. A caller
must retain the original binding digest outside the candidate's authority.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from . import original_replay as R
import telohelm.runtime.context as C; import telohelm.runtime.delivery as D

RULE = "B8.original.binding.v1"
QUALIFICATION = "conditional_independently_retained_original_binding"


@dataclass(frozen=True)
class Issue:
    request_id: str | None
    source: str
    decision: str
    goal_version: int
    issued_second: int
    issued_step: int
    available_second: int
    subphase: str
    context_sha256: str
    source_sha256: str


@dataclass(frozen=True)
class Binding:
    body: bytes
    sha256: str
    qualification: str = QUALIFICATION
    admissible: bool = False


def _sources():
    return {"binding":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "context":hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),
            "delivery":hashlib.sha256(Path(D.__file__).read_bytes()).hexdigest(),
            "replay":R.sources(),"context_history":C.HISTORY}


def bind(original, issue, *, briefing, bundles, issued_context, source_body,
         issued_call=None, pending=(), expected_capture_sha256):
    """Reconstruct original public contexts, then bind typed issue/pending tags.

    The decision clock is the physical action second; the method's earlier
    issue clock remains separate. Neither can be overwritten with replay time.
    A source payload digest is consistency, not provider/source authority.
    Full bundles from reset through the decision second are required here.
    """
    if not isinstance(original,R.Capture) or not isinstance(issue,Issue):
        raise TypeError("Capture and typed Issue required")
    if hashlib.sha256(original.body).hexdigest()!=original.sha256 or original.sha256!=expected_capture_sha256:
        raise ValueError("independently retained capture digest conflict")
    raw=json.loads(original.body)
    ep=R._restore(raw)
    if raw["rule"]!=R.RULE or raw["sources"]!=R.sources():
        raise ValueError("original source or rule conflict")
    for value in (issue.goal_version,issue.issued_second,issue.issued_step,issue.available_second):
        if type(value) is not int or value<0:raise ValueError("invalid issue clock/type")
    if (issue.decision!=raw["identity"]["decision"] or issue.goal_version!=ep.goal.version or
            not issue.issued_second<=ep.t or issue.available_second!=ep.t or
            issue.issued_step>raw["identity"]["action_step"]):
        raise ValueError("issue conflicts with original decision")
    if issue.source not in ("procedure","small_model","reasoning_model","none"):
        raise ValueError("unsupported source")
    model=issue.source in ("small_model","reasoning_model")
    expected_phase="post-issuance" if model and issue.issued_second==ep.t else "reply" if model else "decisions"
    if issue.subphase!=expected_phase:raise ValueError("wrong issue/delivery subphase")
    if not model and (issue.issued_second!=ep.t or issue.issued_step!=raw["identity"]["action_step"]):
        raise ValueError("procedure/hold belongs to this decision step")
    if issue.source_sha256!=R.digest(source_body) or not isinstance(source_body,dict) or not source_body:
        raise ValueError("actual source payload digest conflict")
    if not isinstance(bundles,(list,tuple)) or len(bundles)!=ep.t+1:
        raise ValueError("complete original public bundle prefix required")
    history_digest=R.digest({"briefing":briefing,"bundles":bundles})
    if raw["actor_pending"].get("public_history_sha256")!=history_digest:
        raise ValueError("public input history differs from original capture")
    # Decode once before iteration to detach mutable input aliases.
    prefix=json.loads(R.encode(bundles))
    builder=C.ContextBuilder(json.loads(R.encode(briefing)))
    contexts=[]
    for second,bundle in enumerate(prefix):
        if type(bundle.get("t")) is not int or bundle["t"]!=second:
            raise ValueError("public bundle clocks must be contiguous from reset")
        contexts.append(builder.update(bundle))
    if prefix[-1]!=raw["input_bundle"] or contexts[-1]!=raw["context"]:
        raise ValueError("original input or current context reconstruction conflict")
    if contexts[issue.issued_second]!=issued_context or R.digest(issued_context)!=issue.context_sha256:
        raise ValueError("original issuance context conflict")
    if issued_context["goal_version"]!=issue.goal_version or asdict(issue)!=raw["issue"]:
        raise ValueError("goal or retained issue conflict")
    issued_step=sum(a["t"]<issue.issued_second for a in raw["state"]["record"]["actions"])
    if issue.issued_step!=issued_step:
        raise ValueError("issue action step differs from original event history")
    # Bind visible own-state fields to the physical world snapshot too.
    own=prefix[-1]["own"]
    for key,value in (("x",ep.receiver.x),("y",ep.receiver.y),("tas_nm_s",ep.receiver.tas)):
        if own[key]!=value:raise ValueError("public own state differs from original world")
    if model:
        if not isinstance(issued_call,D.Pending) or issued_call.status!="answered":
            raise ValueError("answered original Pending required for model delivery")
        if not isinstance(issue.request_id,str) or not issue.request_id:
            raise ValueError("model request identity required")
        for key in ("request_id","decision","goal_version","issued_second","issued_step",
                    "available_second","context_sha256"):
            if getattr(issued_call,key)!=getattr(issue,key):raise ValueError("original Pending tags conflict")
        if issued_call.role!=issue.source:raise ValueError("Pending role conflict")
    elif issued_call is not None or issue.request_id is not None:
        raise ValueError("non-model source has no model request")
    if not isinstance(pending,(list,tuple)) or any(not isinstance(p,D.Pending) for p in pending):
        raise ValueError("typed actor Pending sequence required")
    ids=set()
    for p in pending:
        if not isinstance(p.request_id,str) or not p.request_id:
            raise ValueError("pending request identity required")
        if p.request_id in ids or p.request_id==issue.request_id:raise ValueError("conflicting pending request")
        ids.add(p.request_id)
        for v in (p.order,p.goal_version,p.issued_second,p.issued_step,p.available_second):
            if type(v) is not int or v<0:raise ValueError("invalid pending clock/type")
        if (p.issued_second>ep.t or p.available_second<=ep.t or p.issued_step>raw["identity"]["action_step"] or
                p.decision not in ("heading","report") or p.role not in ("small_model","reasoning_model")):
            raise ValueError("pending request does not belong to this original prefix")
        if p.context_sha256!=R.digest(contexts[p.issued_second]) or p.goal_version!=contexts[p.issued_second]["goal_version"]:
            raise ValueError("pending original context or goal conflict")
        if p.issued_step!=sum(a["t"]<p.issued_second for a in raw["state"]["record"]["actions"]):
            raise ValueError("pending step differs from original event history")
    pending_payload={"calls":[asdict(p) for p in pending],"public_history_sha256":history_digest}
    if raw["actor_pending"]!=pending_payload:raise ValueError("original pending snapshot conflict")
    payload={"rule":RULE,"original":json.loads(original.body),"capture_sha256":original.sha256,
             "issue":asdict(issue),"briefing":briefing,"bundles":prefix,"issued_context":issued_context,
             "source_body":source_body,"issued_call":None if issued_call is None else asdict(issued_call),
             "pending":pending_payload,"sources":_sources()}
    body=R.encode(payload)
    return Binding(body,hashlib.sha256(body).hexdigest())


def replay_bound(binding,candidate,continuation,*,expected_binding_sha256,limits=R.Limits()):
    """Check independently held binding, reconstruct its contexts and issue.

    Its output remains conditional. Source/tag integrity cannot establish the
    monitor's principal, trusted head, protected input origin or loaded code.
    """
    if not isinstance(binding,Binding) or hashlib.sha256(binding.body).hexdigest()!=binding.sha256 \
            or binding.sha256!=expected_binding_sha256:
        raise ValueError("independently retained binding digest conflict")
    if binding.qualification!=QUALIFICATION or binding.admissible is not False:
        raise ValueError("conditional binding qualification cannot be upgraded")
    limits.validate()
    if len(binding.body)+len(R.encode([candidate,continuation]))>limits.input_bytes:
        raise ValueError("typed binding and plan input byte limit")
    saved=json.loads(binding.body)
    if saved["rule"]!=RULE or saved["sources"]!=_sources():raise ValueError("binding source or rule conflict")
    body=R.encode(saved["original"])
    original=R.Capture(body,hashlib.sha256(body).hexdigest())
    call=None if saved["issued_call"] is None else D.Pending(**saved["issued_call"])
    rebuilt=bind(original,Issue(**saved["issue"]),briefing=saved["briefing"],bundles=saved["bundles"],
                 issued_context=saved["issued_context"],source_body=saved["source_body"],issued_call=call,
                 pending=tuple(D.Pending(**p) for p in saved["pending"]["calls"]),
                 expected_capture_sha256=saved["capture_sha256"])
    if rebuilt.body!=binding.body:raise ValueError("binding reconstruction conflict")
    evidence=R.replay(original,candidate,continuation,expected_capture_sha256=original.sha256,limits=limits)
    # The checked binding is retained alongside the numerical witness. It is
    # deliberately not hidden in a caller boolean or upgraded to admissible.
    return {"binding_sha256":binding.sha256,"binding":saved,"evidence":json.loads(evidence.body),
            "evidence_sha256":evidence.sha256,"disposition":evidence.disposition,
            "upper_elapsed_s":evidence.upper_elapsed_s,
            "replay_cpu_seconds":evidence.cpu_seconds,"qualification":binding.qualification,"admissible":False}
