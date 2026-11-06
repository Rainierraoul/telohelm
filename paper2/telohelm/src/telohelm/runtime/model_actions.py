"""Public preconditions and refusal feedback for model action proposals; opt-in, outside learned qualification."""
from __future__ import annotations

import json


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class ModelActions:
    def __init__(self, precondition):
        if not callable(precondition):
            raise ValueError("model action control needs a task precondition")
        self.precondition = precondition
        self.latest = {}
        self.events = []
        self.version = None

    def at(self, version):
        if version != self.version:
            self.latest.clear()
            self.version = version

    def refuse(self, decision, context, part, source, request_id):
        """Only a public contradiction is a refusal; None does not certify correctness."""
        self.at(context["goal_version"])
        result = self.precondition(decision, context, part)
        if result is None:
            return None
        if (not isinstance(result, dict) or set(result) != {"reason", "evidence"}
                or not isinstance(result["reason"], str) or not 1 <= len(result["reason"]) <= 200
                or not isinstance(result["evidence"], dict)):
            raise ValueError("task precondition must give a public reason and evidence")
        event = _copy({"decision": decision, "goal_version": context["goal_version"], "t": context["t"],
                       "request_id": request_id, "source": source, "proposed_part": part, **result})
        self.events.append(event)
        self.latest[decision] = event
        return _copy(event)

    def context(self, decision, context):
        self.at(context["goal_version"])
        last = self.latest.get(decision)
        return {**context, "last_action_refusal": _copy(last)} if last is not None else context


def learner_feedback(record):
    """Bound feedback to the latest three self-observed public refusals; all events stay in the episode record."""
    events = record.get("model_action_refusals") or []
    version = ((record.get("outcome") or record).get("goal") or {}).get("version")
    if version is not None:
        events = [e for e in events if e.get("goal_version") == version]
    if not events:
        return {}
    fields = {"decision", "goal_version", "t", "request_id", "source", "proposed_part", "reason", "evidence"}
    return {"model_action_refusals": {"count": len(events),
                                      "latest": [_copy({k: v for k, v in e.items() if k in fields})
                                                 for e in events[-3:]]}}
