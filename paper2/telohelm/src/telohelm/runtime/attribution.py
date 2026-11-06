"""Share of simulated seconds by source of the heading in force (paper2/notes/RUNTIME-PROPOSALS-V1.md section 1;
decision D21 and Codex's TURN 76 and TURN 78 cross-reviews).

Derived after the run from the broker's authenticated log. For each recorded one-second action step t of an episode,
exactly one category z_t is assigned at the final action boundary:
- procedure, small_model or reasoning_model: the world accepted the action, the action carries a heading, and the
  heading decision wrote it; the category is that decision's recorded source;
- holding_pending: otherwise, if a model call for the heading was pending immediately before the action;
- holding_none: otherwise.
A held second records, as a detail only, the command still in force: its source ('initial' before any), method, model
request identifier, the step, second, goal version and context digest of the decision that set it, and for a model
reply the second, step, goal version and context digest it was issued under (Codex's AT78.HELD).
With K steps, N_j = the number of steps with z_t = j and share_j = N_j / K for K > 0; with K = 0 the shares are
undefined (None). Counts sum to K.

The log's signature protects its bytes. It does not show that the records are complete or agree with each other, so
the derivation checks the links each category depends on, and missing or inconsistent provenance raises
AttributionError; it is never guessed as holding or as a model (Codex's AT78.PROV). The checks cover the records the
episode loop writes. They do not show that the loop wrote the right records, and they do not test the sandbox's
containment. A model reply's tags are checked against the second it is used in, but the link from the broker's
delivery record back to the gateway's own log and ledger row is not: the derivation trusts the broker log's record of
the delivery. Joining the two logs is a separate step (paper2/notes/TRANSPORT-CONTRACT.md section 3).

Scope: the whole recorded episode, a recorded ending, or an explicit prefix. A cut at an ending is taken from the
ending's own record in the protected log, so the caller cannot name an arbitrary step count as an ending. An episode
is complete only at a recorded ending; a truncated episode and a prefix are incomplete, and pool() keeps them apart
from completed episodes unless the caller asks for a labelled mixed summary (Codex's AT78.POOL).

These are shares of issuance and holding intervals. They are not causal influence, continuous physical control,
autonomy quality or cost efficiency.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..taskpacks.tanker.execution import TANKER_ROLES
from ..taskpacks.tanker.pack import TANKER_PACK
from .delivery import DECISION_FIELDS

# The decision that flies (slice 2(b)): a second is credited to whichever method wrote its control fields.
FLYING = TANKER_ROLES.flying
CONTROL_FIELDS = frozenset(DECISION_FIELDS[FLYING]) - {"help"}
from .trusted_log import verify

CATEGORIES = ("procedure", "small_model", "reasoning_model", "holding_pending", "holding_none")
SOURCES = ("procedure", "small_model", "reasoning_model")
SUBPHASES = ("reply phase", "post-issuance")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class AttributionError(ValueError):
    """The log cannot support the derivation: unauthenticated, missing records or inconsistent provenance."""


def _events(log_path: Path, heads_path: Path, episode: str) -> list[dict]:
    check = verify(Path(log_path), Path(heads_path))
    if not check.get("ok"):
        raise AttributionError(f"the log does not verify against its protected heads: {check}")
    rows = [json.loads(line)["payload"] for line in Path(log_path).read_text().splitlines()[1:]]
    mine = [e for e in rows if isinstance(e, dict) and e.get("episode") == episode]
    if not mine:
        raise AttributionError(f"the log records no episode {episode!r}")
    return mine


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_step(n: int, bundle: dict, replies: list[dict], decisions: list[dict], action: dict, world: dict,
                *, flying: str, decision_fields) -> None:
    """The links one action step's category depends on. Raises AttributionError on the first that fails."""
    def bad(why: str) -> AttributionError:
        return AttributionError(f"step {n}: {why}")
    for e in [bundle, *replies, *decisions, action, world]:
        if e.get("action_step") != n or e.get("t") != bundle.get("t") or not _int(e.get("t")):
            raise bad(f"the {e.get('kind')} record is not at this step's second and action step")
    if not isinstance(world.get("accepted"), bool) or not isinstance(world.get("ended"), bool):
        raise bad("the world step does not record its acceptance and ending")
    if world.get("action_steps") != n + 1:
        raise bad("the world's step count does not match")
    names = [d.get("decision") for d in decisions]
    if names.count(flying) != 1 or len(set(names)) != len(names) or not set(names) <= set(decision_fields):
        raise bad(f"the decisions recorded are {names}, not one {flying!r} control decision and other known decisions")
    by_name = {d["decision"]: d for d in decisions}
    act, writers, pending = action.get("action"), action.get("field_writers"), action.get("pending")
    if not isinstance(act, dict) or not isinstance(writers, dict) or not isinstance(pending, dict):
        raise bad("the action record lacks its action, field writers or pending flags")
    if not isinstance(pending.get(flying), bool):
        raise bad(f"the action record does not say whether {flying!r} was pending")
    if set(writers) != set(act) - {"goal_version"}:
        raise bad("the field writers do not match the action's fields")
    # Typed before compared: True == 1 and 1.0 == 1 in Python, so a set or an equality alone would accept them
    # (Codex's AT82.GOAL).
    if not all(_int(d.get("goal_version")) and d["goal_version"] >= 0 for d in decisions):
        raise bad("a decision's goal version is not a non-negative whole number")
    versions = {d["goal_version"] for d in decisions}
    contexts = {d.get("context_sha256") for d in decisions}
    if len(versions) != 1:
        raise bad("the decisions do not share one goal version")
    if len(contexts) != 1 or not _HEX64.fullmatch(str(next(iter(contexts)))):
        raise bad("the decisions do not share one context digest")
    if act and (not _int(act.get("goal_version")) or act["goal_version"] not in versions):
        raise bad("the action's goal version is not the decisions' goal version")
    for field, writer in writers.items():
        if field == "help" and writer == "ladder":
            # A help request from the flight ladder (B36) is information for the person, not control: it writes no
            # heading or report and is credited to no decision, so it changes no second's attribution.
            continue
        if writer == "watch" and act[field] is None:
            # The disagreement watch's withdrawal of a standing order a withheld procedure left armed (opt-in,
            # runtime/disagreement.py): a null that clears it, credited to no decision and no method.
            continue
        d = by_name.get(writer)
        if d is None or field not in decision_fields.get(writer, ()):
            raise bad(f"field {field!r} is credited to {writer!r}, which may not write it or was not recorded")
        if not isinstance(d.get("part"), dict) or d["part"].get(field) != act[field]:
            raise bad(f"field {field!r} differs from the part its decision recorded")
    for d in decisions:
        _check_source(bad, d, replies)


def _check_source(bad, d: dict, replies: list[dict]) -> None:
    source, method, request_id, part = d.get("source"), d.get("method"), d.get("request_id"), d.get("part")
    if source not in SOURCES + ("none",):
        raise bad(f"decision {d.get('decision')!r} has source {source!r}")
    if not isinstance(method, str) or not method:
        raise bad(f"decision {d.get('decision')!r} records no method")
    used = isinstance(part, dict) and bool(set(part) - {"goal_version"})
    if source == "none":
        if used or request_id is not None:
            raise bad(f"decision {d.get('decision')!r} wrote fields with no source")
        return
    if source == "procedure":
        if request_id is not None or method.startswith(("model:", "climb:")):
            raise bad(f"decision {d.get('decision')!r} is credited to a procedure but records {method!r}")
        return
    # A model source needs its delivery: a reply in this step for this decision, from this role, usable, not out of
    # date, and proposing every field the decision wrote.
    if not isinstance(request_id, str) or method != f"model:{request_id}":
        raise bad(f"decision {d.get('decision')!r} is credited to {source} without its request identifier")
    reply = [r for r in replies if r.get("request_id") == request_id]
    if len(reply) != 1:
        raise bad(f"decision {d.get('decision')!r} has no single delivery for request {request_id!r}")
    r = reply[0]
    if (r.get("decision") != d.get("decision") or r.get("role") != source or r.get("stale") is not False
            or r.get("refusal") is not None or r.get("status") != "answered" or r.get("subphase") not in SUBPHASES
            or not isinstance(r.get("part"), dict)):
        raise bad(f"the delivery of {request_id!r} does not support a usable {source} part for this decision")
    # The tags the reply was issued under (Codex's AT80.ISSUE). A reply is delivered at its available second, which is
    # this one, and was issued no later; an in-date reply was issued under the goal version in force now; a reply
    # used post-issuance was issued in this very second and step. The digests must be well formed.
    t, n = d.get("t"), d.get("action_step")
    issued_s, issued_n, avail = r.get("issued_second"), r.get("issued_step"), r.get("available_second")
    if not all(_int(v) and v >= 0 for v in (issued_s, issued_n, avail)):
        raise bad(f"the delivery of {request_id!r} lacks whole-number issue and availability tags")
    if avail != t or issued_s > t or issued_n > n or (issued_s == t) != (issued_n == n):
        raise bad(f"the delivery of {request_id!r} has issue or availability tags that do not fit this second")
    if not _int(r.get("goal_version")) or r["goal_version"] != d.get("goal_version"):
        raise bad(f"the delivery of {request_id!r} is marked in date but was issued under another goal version")
    if r["subphase"] == "post-issuance" and (issued_s != t or issued_n != n):
        raise bad(f"the post-issuance delivery of {request_id!r} was issued at another second")
    if not _HEX64.fullmatch(str(r.get("context_sha256"))) or not _HEX64.fullmatch(str(r.get("reply_sha256"))):
        raise bad(f"the delivery of {request_id!r} lacks a well-formed context or reply digest")
    if any(r["part"].get(k) != v for k, v in part.items() if k != "goal_version"):
        raise bad(f"decision {d.get('decision')!r} wrote fields its delivery did not propose")


def _setter(n: int, d: dict, replies: list[dict]) -> dict:
    """The provenance a held second keeps for the command still in force."""
    out = {"source": d["source"], "method": d["method"], "request_id": d.get("request_id"), "step": n, "t": d["t"],
           "goal_version": d["goal_version"], "context_sha256": d["context_sha256"]}
    if d["source"] in ("small_model", "reasoning_model"):
        r = next(r for r in replies if r.get("request_id") == d["request_id"])
        out["issued"] = {k: r.get(k) for k in ("issued_second", "issued_step", "goal_version", "context_sha256",
                                                "reply_sha256", "subphase")}
    return out


def derive(log_path: Path, heads_path: Path, episode: str, ending: str | None = None, prefix: int | None = None,
           labels: dict[str, str] | None = None, *, pack: Any = TANKER_PACK) -> dict[str, Any]:
    """Attribute each action step of one episode.

    ending names a recorded ending ('nominal', 'alternative 0', ...): the cut is that ending's recorded step count,
    and the result is complete. prefix cuts to the first prefix steps, an explicit prefix that is never complete.
    With neither, the whole recorded episode is attributed, and it is complete only if its nominal ending was
    recorded. labels maps categories to explicit comparator labels for an arm whose sources are not Telohelm's (for
    example a hand-written controller). pack supplies the task's control decision and field ownership; saved tanker
    readers may omit it."""
    if labels is not None and (not isinstance(labels, dict) or not set(labels) <= set(CATEGORIES)
                               or not all(isinstance(v, str) and v for v in labels.values())):
        raise AttributionError("labels must map categories to non-empty names")
    if ending is not None and prefix is not None:
        raise AttributionError("give an ending or a prefix, not both")
    flying, decision_fields = pack.roles.flying, pack.public.decision_fields
    if flying not in decision_fields:
        raise AttributionError("the task has no field ownership for its control decision")
    control_fields = frozenset(decision_fields[flying]) - {"help"}
    steps: list[dict] = []
    endings: dict[str, dict] = {}
    held: dict = {"source": "initial"}
    bundle: dict | None = None
    replies: list[dict] = []
    decisions: list[dict] = []
    action: dict | None = None
    last_ended = False
    for e in _events(log_path, heads_path, episode):
        kind = e.get("kind")
        if kind == "bundle":
            if bundle is not None:
                raise AttributionError(f"step {len(steps)}: a second began before the last one's world step")
            bundle, replies, decisions, action = e, [], [], None
        elif kind in ("model_reply", "decision", "action"):
            if bundle is None or action is not None:
                raise AttributionError(f"step {len(steps)}: a {kind} record outside its second")
            if kind == "model_reply":
                replies.append(e)              # in the reply phase, or post-issuance just before its decision
            elif kind == "decision":
                decisions.append(e)
            else:
                action = e
        elif kind == "world_step":
            n = len(steps)
            if bundle is None or action is None:
                raise AttributionError(f"step {n}: no bundle or action before the world step")
            _check_step(n, bundle, replies, decisions, action, e,
                        flying=flying, decision_fields=decision_fields)
            control = next(d for d in decisions if d["decision"] == flying)
            if e["accepted"] and control_fields & set(action["action"]):
                if any(action["field_writers"][field] != flying
                       for field in control_fields & set(action["action"])):
                    raise AttributionError(f"step {n}: control fields were written by another decision")
                if control["source"] not in SOURCES:
                    raise AttributionError(f"step {n}: control was set with source {control['source']!r}")
                held = _setter(n, control, replies)
                steps.append({"step": n, "t": e["t"], "category": control["source"],
                              "request_id": control.get("request_id"), "set": held})
            else:
                z = "holding_pending" if action["pending"][flying] else "holding_none"
                steps.append({"step": n, "t": e["t"], "category": z, "held_by": held["source"], "held": held})
            last_ended = e["ended"]
            bundle, action = None, None
        elif kind == "ending":
            name = e.get("ending")
            if not isinstance(name, str) or name in endings or not _int(e.get("action_steps")):
                raise AttributionError(f"a malformed or repeated ending record: {name!r}")
            endings[name] = e
    if bundle is not None:
        raise AttributionError(f"step {len(steps)}: the log ends inside a second that has no world step")
    recorded = len(steps)
    for name, rec in endings.items():
        # An ending is real only with its reason and time, after at least one action, one second after the last action
        # it counts (the world advances one second per action), and inside the recorded steps (Codex's AT80.CUT).
        k_end, at, reason = rec["action_steps"], rec.get("ended_at"), rec.get("end_reason")
        if not isinstance(reason, str) or not reason or not _int(at):
            raise AttributionError(f"ending {name!r} lacks its reason or time")
        if not 1 <= k_end <= recorded:
            raise AttributionError(f"ending {name!r} is outside the {recorded} recorded steps")
        if at != steps[k_end - 1]["t"] + 1:
            raise AttributionError(f"ending {name!r} at second {at} does not follow its last action")
    if ("nominal" in endings) != last_ended or ("nominal" in endings and endings["nominal"]["action_steps"] != recorded):
        raise AttributionError("the nominal ending record and the last world step disagree")
    if ending is not None:
        if ending not in endings:
            raise AttributionError(f"no recorded ending {ending!r}")
        k, complete, scope = endings[ending]["action_steps"], True, f"ending:{ending}"
    elif prefix is not None:
        if isinstance(prefix, bool) or not isinstance(prefix, int) or not 0 <= prefix <= recorded:
            raise AttributionError(f"prefix {prefix} is outside the {recorded} recorded steps")
        k, complete, scope = prefix, False, "prefix"
    else:
        k, complete = recorded, "nominal" in endings
        scope = "ending:nominal" if complete else "truncated"
    steps = steps[:k]
    K = len(steps)
    counts = {j: sum(1 for s in steps if s["category"] == j) for j in CATEGORIES}
    out = {"episode": episode, "scope": scope, "K": K, "steps_after_cut": recorded - K, "counts": counts,
           "shares": {j: (counts[j] / K if K else None) for j in CATEGORIES},
           "complete": complete, "recorded_endings": sorted(endings), "steps": steps}
    if labels is not None:
        named: dict[str, int] = {}
        for j, n in counts.items():
            name = labels.get(j, j)
            named[name] = named.get(name, 0) + n
        out["counts_by_label"] = named
    return out


def _summary(results: list[dict]) -> dict[str, Any]:
    total = sum(r["K"] for r in results)
    counts = {j: sum(r["counts"][j] for r in results) for j in CATEGORIES}
    usable = [r for r in results if r["K"] > 0]
    return {"episodes": len(results), "K": total, "counts": counts,
            "pooled_shares": {j: (counts[j] / total if total else None) for j in CATEGORIES},
            "equal_episode_mean_shares": {j: (sum(r["shares"][j] for r in usable) / len(usable) if usable else None)
                                          for j in CATEGORIES},
            "episodes_with_no_steps": len(results) - len(usable)}


def pool(results: list[dict], mixed: bool = False) -> dict[str, Any]:
    """Pooled shares over episodes, sum of N_j over sum of K, and, labelled separately, the equal-episode mean of the
    episodes' shares (episodes with K = 0 excluded and counted).

    Completed episodes and incomplete ones (truncated, or explicit prefixes) are different scopes. By default they
    cannot be pooled together. With mixed=True the result has each group's own summary and, under the label
    'mixed_completed_and_incomplete', the summary over both."""
    groups = {"completed": [r for r in results if r["complete"]],
              "incomplete": [r for r in results if not r["complete"]]}
    if not mixed:
        if groups["completed"] and groups["incomplete"]:
            raise AttributionError("completed and incomplete episodes are different scopes; pool them apart or "
                                   "ask for a labelled mixed summary")
        return {"scope": "completed" if groups["completed"] else "incomplete", **_summary(results)}
    return {"scope": "mixed", "completed": _summary(groups["completed"]),
            "incomplete": _summary(groups["incomplete"]),
            "mixed_completed_and_incomplete": _summary(results)}
