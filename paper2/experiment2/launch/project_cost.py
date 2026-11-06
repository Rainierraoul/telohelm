"""M1b's cost records, and the forecast for the full scored study, fixed before M1b runs.

read_run(run, study) builds one inventory of a finished launcher run and its prepared study, read against what each
record's writer promises. Every gateway log must verify against its protected heads (telohelm.runtime.trusted_log).
Each model request it starts must end once, and the end must name a ledger row with the same role, status and time;
every model charge in a ledger must be named by an end. L's learning attempts are expected from its learning ledger
and answer receipts as well as from the attempt folders that exist. Each new request that reached a model server must
match one released hold on that server's slots, counted by role, condition and stream from the slot records' labels.
The run's records must still match the checksums the launcher wrote when it closed, and the launcher must have closed
cleanly. Anything missing, damaged, invalid or unmatched is a defect, kept by kind and place, and a defect never
becomes a known cost.

project(measured, basis) applies the forecast in paper2/notes/M1-MIGRATION-AND-COST-DESIGN-CLAUDE.md with the cost
basis (paper2/reproduction/cost-basis-claude-20261026.json). It can qualify a budget only when the inventory has no
defect and covers the 410 declared positions completely; otherwise every candidate flag is None and the reasons are
listed. It reads records only; it starts nothing and calls no model.

Elapsed spans stay wall time, and overlapping spans are never added. A total with an unknown part has the value None,
with its known subtotal and its unknown parts listed beside it. Nothing unknown is counted as zero.

    python project_cost.py --run RUN --study STUDY --basis paper2/reproduction/cost-basis-claude-20261026.json
        [--scheduler-gpu-hours X]
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean

from telohelm.runtime import trusted_log
from telohelm.runtime.ledger import KINDS, MODEL_KINDS, STATUSES

EPISODE = re.compile(r"s(\d+)-p(\d)-e(\d\d)$")
ATTEMPT = re.compile(r"build/attempt(\d+)$")
TELOHELM = ("L", "C", "F")
COMPARATORS = ("calculator", "adaptive_controller")
SCORED_STREAMS, PER_PHASE, PHASES = 4, 20, (1, 2, 3, 4, 5)
SCORED_POSITIONS = {"L": {p: SCORED_STREAMS * PER_PHASE for p in PHASES},
                    "C": {p: SCORED_STREAMS * 5 for p in (1, 2, 3, 4)},
                    "F": {3: SCORED_STREAMS * 10},
                    "calculator": {p: SCORED_STREAMS * PER_PHASE for p in PHASES},
                    "adaptive_controller": {p: SCORED_STREAMS * PER_PHASE for p in PHASES}}
M1B_POSITIONS = {"L": {1: 40, 2: 40, 3: 40}, "C": {1: 10, 2: 10, 3: 10}, "F": {3: 20},       # 410 in all
                 "calculator": {1: 40, 2: 40, 3: 40}, "adaptive_controller": {1: 40, 2: 40, 3: 40}}
FORK = "p2-e19"                                  # F flies from L's library snapshot after this position
CALL_FIELDS = ("request_id", "kind", "episode", "status", "wall_s", "scope", "condition", "stream", "recovers")
# Refusals telohelm.runtime.models.LocalChatProvider makes before it sends anything. Every other refusal it marks
# follows a reply, so the server did the work and the request's time stays its demand.
PRE_SEND = ("speaker", "configuration changed", "outside the configuration", "exchange mismatch")
AFTER_REPLY = ("choices", "model identity", "text encoding", "evidence not saved")
SLOT_IDENTITY = ("pid", "n", "arm", "stream", "origin", "began", "body_sha256")
LIFECYCLE = (["queued"], ["queued", "acquired"], ["queued", "acquired", "released"])
GPUS = 2


def _seconds(x):
    """A recorded duration if it is a finite, non-negative number, otherwise None."""
    ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
    return float(x) if ok else None


def _attempt(row_id):
    """The learning attempt a learning request's identity names, or None. The identity is the gateway's JSON list
    with the scope 'build/attemptN'; a recovery row adds '#recoveredK'."""
    try:
        parts = json.loads(row_id.split("#recovered")[0])
    except (ValueError, AttributeError):
        return None
    if not isinstance(parts, list):
        return None
    found = [int(m[1]) for m in (ATTEMPT.match(p) for p in parts if isinstance(p, str)) if m]
    return found[0] if len(found) == 1 else None


def _demand(requests):
    """New requests' seconds by role and part, with unknown counts. Recovered, interrupted and refused requests are
    not new demand."""
    out = {}
    for r in requests:
        if r["how"] == "new":
            e = out.setdefault(f"{r['role']}:{r['part']}", {"count": 0, "known_s": 0.0, "unknown": 0})
            e["count"] += 1
            if r["wall_s"] is None:
                e["unknown"] += 1
            else:
                e["known_s"] += r["wall_s"]
    return out


class _Inventory:
    """The records M1b's cost rests on, read against their writers' contracts, with every defect kept by kind and
    place."""

    def __init__(self):
        self.defects, self.reached, self.requests, self.unfinished = [], Counter(), [], Counter()

    def defect(self, kind, where):
        self.defects.append([kind, str(where)])

    def rows(self, path, required=True):
        """(rows, whole). A missing required file is a defect; a missing optional one is whole and empty. A damaged
        line is a defect, and the rows before it are returned."""
        if not Path(path).is_file():
            if required:
                self.defect("missing record", path)
            return [], not required
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            self.defect("unreadable record", f"{path}: {type(exc).__name__}")
            return [], False
        rows = []
        for line in raw.split(b"\n"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    self.defect("damaged record", path)
                    return rows, False
        return rows, True

    def ledger(self, path, required=True):
        """A ledger's rows by row identity, and whether the file was whole and every row had the writer's fields,
        valid values and a unique identity."""
        rows, whole = self.rows(path, required)
        out = {}
        for r in rows:
            ok = (isinstance(r, dict) and all(k in r for k in CALL_FIELDS) and isinstance(r["request_id"], str)
                  and r["kind"] in KINDS and r["status"] in STATUSES
                  and (r["wall_s"] is None or _seconds(r["wall_s"]) is not None)
                  and (r["recovers"] is None or isinstance(r["recovers"], str)) and r["request_id"] not in out)
            if not ok:
                self.defect("invalid or repeated ledger row", path)
                whole = False
                continue
            out[r["request_id"]] = r
        return out, whole

    def gateway(self, folder, charges, label, episodes, attempt=None, required=True):
        """The requests one gateway log records. Each protected start must name itself (see _bound), end once, and
        be charged by a ledger row with the same role, status, time, episode, condition, stream and purpose; every
        model charge must be named by an end. Returns the requests, the episodes of requests that never ended, and
        whether everything joined."""
        log, heads = Path(folder) / "gateway.log", Path(folder) / "gateway.heads"
        if not log.is_file() or not heads.is_file():
            if required or charges:
                self.defect("missing record", log)
                return [], [], False
            return [], [], True
        try:
            check = trusted_log.verify(log, heads)
            events = [json.loads(line)["payload"] for line in log.read_bytes().splitlines()]
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            check = {"ok": False, "problem": f"{type(exc).__name__}: {exc}"}
        if not check.get("ok"):
            self.defect("gateway log fails verification", f"{log}: {check.get('problem')}")
            return [], [], False
        purpose = "running" if attempt is None else "build"
        whole, starts, ends = True, {}, {}
        for e in events:
            kind = e.get("kind") if isinstance(e, dict) else None
            if kind == "model_start":
                request = e.get("request")
                rid = request.get("request_id") if isinstance(request, dict) else None
                if not _bound(request, label, episodes, attempt) or rid in starts:
                    self.defect("request start without its own identity, or repeated", f"{log}: {rid}")
                    self.unfinished[request.get("role") if isinstance(request, dict) else None] += 1
                    whole = False
                    continue
                starts[rid] = request
            elif kind in ("model_end", "model_interrupted"):
                rid = e.get("request_id")
                if rid not in starts or rid in ends:
                    self.defect("request end without its start, or ended twice", f"{log}: {rid}")
                    whole = False
                    continue
                ends[rid] = e
        requests, opened, named = [], [], set()
        for rid, request in starts.items():
            end = ends.get(rid)
            if end is None:
                self.defect("request started and never ended", f"{log}: {rid}")
                opened.append(request["episode"])
                self.unfinished[request["role"]] += 1
                whole = False
                continue
            interrupted = end["kind"] == "model_interrupted"
            response = end.get("response") if isinstance(end.get("response"), dict) else {}
            ids = response.get("provider_ids") if isinstance(response.get("provider_ids"), dict) else {}
            reason = ids.get("refused")
            row_id = rid if interrupted else end.get("ledger_row")
            row = charges.get(row_id) if isinstance(row_id, str) else None
            named.add(row_id)
            status = "interrupted" if interrupted else end.get("status")
            if (row is None or row["kind"] != request["role"] or row["status"] != status or row["scope"] != purpose
                    or [row["episode"], row["condition"], row["stream"]]
                    != [request["episode"], request["condition"], request["stream"]]
                    or not interrupted and _seconds(end.get("wall_s")) != _seconds(row["wall_s"])):
                self.defect("request end without a matching charge", f"{log}: {rid}")
                whole = False
                continue
            if reason is not None and reason not in PRE_SEND + AFTER_REPLY:
                self.defect("refusal of unknown kind", f"{log}: {rid}: {reason}")
                whole = False
            how = ("interrupted" if interrupted else "recovered" if end.get("recovered") is not False else
                   "refused" if reason in PRE_SEND else "new")
            if how in ("interrupted", "recovered"):
                self.defect(f"{how} request", f"{log}: {rid}")
                whole = False
            if how == "new" or interrupted and end.get("phase") != "start clock":
                self.reached[(row["kind"], *label.split("/")[1:])] += 1      # it held one of its server's slots
            part = "learning" if attempt is not None else (
                "watch" if "briefing_disagreement" in json.dumps(request.get("messages")) else "flight")
            requests.append({"role": row["kind"], "part": part, "how": how, "episode": request["episode"],
                             "wall_s": _seconds(row["wall_s"])})
        for row_id, row in charges.items():
            if row["kind"] in MODEL_KINDS and row_id not in named:
                self.defect("model charge without a request end", f"{log}: {row_id}")
                whole = False
        self.requests += requests
        return requests, opened, whole

    def learning(self, root, label, episodes, required):
        """L's learning requests. Every attempt that a learning charge's identity, a sent answer receipt or an attempt
        folder names must have a verified log that joins its charges, for one of L's episodes in this stream. A sent
        answer must repeat its charges exactly, for its own episode and attempt; an unsent one names none. Returns the
        requests, the episodes of requests that never ended, and whether everything joined."""
        charges, whole = self.ledger(root / "learning-ledger.jsonl", required)
        answers, answers_whole = self.rows(root / "learning-answers.jsonl", required or bool(charges))
        whole = whole and answers_whole
        by_attempt = {}
        for row_id, row in charges.items():
            n = _attempt(row_id)
            if n is None:
                self.defect("learning charge outside any attempt", f"{root}: {row_id}")
                whole = False
                continue
            by_attempt.setdefault(n, {})[row_id] = row
        for a in answers:
            try:
                episode, n, _ = json.loads(a["logical"])
                sent, saved = a["sent"], list(a["ledger_rows"])
                bound = all(s == charges.get(s["request_id"]) and s["episode"] == episode
                            and _attempt(s["request_id"]) == n for s in saved)
            except (ValueError, KeyError, TypeError, IndexError):
                self.defect("invalid learning answer receipt", root)
                whole = False
                continue
            if sent is True:
                by_attempt.setdefault(n, {})
            if not bound or sent is not True and saved:
                self.defect("learning answer without its exact charge", f"{root}: attempt {n}")
                whole = False
        for d in root.glob("learning-attempt-*"):
            m = re.fullmatch(r"learning-attempt-(\d+)", d.name)
            if m:
                by_attempt.setdefault(int(m[1]), {})
        requests, opened = [], []
        for n, rows in sorted(by_attempt.items()):
            got, unfinished, ok = self.gateway(root / f"learning-attempt-{n}", rows, label, episodes, attempt=n)
            requests, opened, whole = requests + got, opened + unfinished, whole and ok
        return requests, opened, whole

    def slots(self, run):
        """Each role's slot waits, every one queued, then acquired, then released, once each and with one identity,
        against the new requests that reached its server, by condition and stream. Returns {role: {"held_s": seconds
        or None, "known_held_s", "holds"}}."""
        out = {}
        for role in MODEL_KINDS:
            used = any(k[0] == role for k in self.reached)
            path = run / "slots" / role / "waits.jsonl"
            rows, ok = self.rows(path, used)
            waits = {}
            for r in rows:
                if not isinstance(r, dict) or not isinstance(r.get("wait_id"), str):
                    self.defect("invalid slot record", path)
                    ok = False
                    continue
                waits.setdefault(r["wait_id"], []).append(r)
            held, holds, unreleased = 0.0, Counter(), 0
            for wait_id, events in waits.items():
                order = [r.get("event") for r in events]
                times = [_seconds(events[1].get("wait_s")) if len(events) > 1 else 0.0,
                         _seconds(events[2].get("held_s")) if len(events) > 2 else 0.0]
                same = all(all(r.get(k) == events[0].get(k) for k in SLOT_IDENTITY) for r in events)
                slot = len(events) < 3 or events[1].get("slot") == events[2].get("slot")
                if order not in LIFECYCLE or not same or not slot or None in times:
                    self.defect("invalid slot lifecycle", f"{path}: {wait_id}")
                    ok = False
                    continue
                if len(events) < 3:
                    unreleased += 1
                    continue
                holds[(role, str(events[0].get("arm")), str(events[0].get("stream")))] += 1
                held += times[1]
            if unreleased:
                self.defect("slot wait never released", f"{path}: {unreleased}")
                ok = False
            expected = {k: n for k, n in self.reached.items() if k[0] == role}
            for key in sorted(set(expected) | set(holds)):
                if expected.get(key, 0) != holds.get(key, 0):
                    self.defect("slot holds differ from new requests",
                                f"{role} {key[1]}:{key[2]}: {holds.get(key, 0)} holds, {expected.get(key, 0)} requests")
                    ok = False
            out[role] = {"held_s": held if ok else None, "known_held_s": held,
                         "holds": {f"{a}:{s}": n for (_, a, s), n in sorted(holds.items())}}
        return out

    def launcher(self, run, result):
        """The launcher's own close: the job completed, every cleanup stage finished, and every record in the run
        folder still has the checksum the launcher wrote last."""
        if (result.get("status") != "completed" or result.get("finalisation") != "complete"
                or result.get("host_exit") != 0):
            self.defect("launcher did not close cleanly", f"status {result.get('status')}, finalisation "
                        f"{result.get('finalisation')}, host exit {result.get('host_exit')}")
        try:
            sums = json.loads((run / "checksums.json").read_text())
            files = {str(k): str(v) for k, v in dict(sums["files"]).items()}
            complete = sums.get("complete")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.defect("missing or damaged closing checksums", f"{run}: {type(exc).__name__}")
            return
        if complete is not True:
            self.defect("launcher did not close cleanly", f"checksums: {sums.get('problems')}")
        for rel, digest in files.items():
            path = run / rel
            try:
                matches = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
            except OSError as exc:
                self.defect("unreadable closing record", f"{rel}: {type(exc).__name__}")
                continue
            if not matches:
                self.defect("run record changed after the launcher closed", rel)
        needed = [run / "result.json", run / "host" / "host-receipts.json", *sorted(run.glob("slots/*/waits.jsonl"))]
        for path in needed:
            if str(path.relative_to(run)) not in files:
                self.defect("run record not in the closing checksums", path.relative_to(run))


def _bound(request, label, episodes, attempt):
    """Whether a protected start names itself. Its identity is the gateway's JSON list of stream, episode, condition,
    role and, for learning, the scope 'build/attemptN', then a count; those fields must match the start's own, its
    stream must be this dispatcher's label, its episode one this log may serve, and its role a model role."""
    fields = ("request_id", "stream", "episode", "condition", "role")
    if not isinstance(request, dict) or not all(isinstance(request.get(k), str) for k in fields):
        return False
    try:
        parts = json.loads(request["request_id"])
    except ValueError:
        return False
    scope = [] if attempt is None else [f"build/attempt{attempt}"]
    return (isinstance(parts, list) and len(parts) == 5 + len(scope) and type(parts[-1]) is int
            and parts[:-1] == [request["stream"], request["episode"], request["condition"], request["role"], *scope]
            and request["stream"] == label and request["episode"] in episodes and request["role"] in MODEL_KINDS)


def _spans(inventory, path, timing_rows, episode):
    """Closed seconds by part for one episode, and the number of parts opened but never closed. Each part must be one
    start then one end with a valid time; anything else is a defect and the part has no value."""
    events = {}
    for r in timing_rows:
        if isinstance(r, dict) and r.get("episode") == episode:
            events.setdefault(str(r.get("part")), []).append(r)
    parts, open_parts = {}, 0
    for part, rows in events.items():
        order = [r.get("event") for r in rows]
        if order == ["start"]:
            open_parts += 1
        elif order == ["start", "end"] and _seconds(rows[1].get("seconds")) is not None:
            parts[part] = _seconds(rows[1].get("seconds"))
        else:
            inventory.defect("span without one start, one end and a valid time", f"{path}: {episode} {part}")
    return parts, open_parts


def _declared(study):
    """The positions the prepared study declares for the five conditions, as [stream, episode] pairs."""
    out = {arm: [] for arm in TELOHELM + COMPARATORS}
    for s in json.loads((study / "study.json").read_text())["streams"]:
        for r in json.loads((study / str(s) / "schedule.json").read_text())["schedule"]["rows"]:
            for arm in out:
                if not (arm == "C" and not r["c"] or arm == "F" and not r["f"]):
                    out[arm].append([s, r["episode"]])
    return out


def _receipts(path):
    """The host's receipts, with every dispatcher a known condition and every position a scheduled episode with a
    status; anything else raises."""
    body = json.loads(Path(path).read_text())
    if not isinstance(body.get("host_span"), dict) or not isinstance(body.get("verdict"), dict):
        raise ValueError("no host span or verdict")
    for d in body["dispatch"]:
        if (d["arm"] not in M1B_POSITIONS or type(d["stream"]) is not int or not all(
                isinstance(p, dict) and isinstance(p.get("status"), str) and EPISODE.match(str(p.get("episode")))
                for p in d["positions"])):
            raise ValueError(f"invalid dispatcher receipt {d.get('name')}")
    return body


def _position(inv, item, root, label, shared):
    """One position's spans and new requests, or its comparator work, filled into item. Its records are required
    only if it is complete; anything that does not join leaves its requests unknown."""
    arm, ep, required = item["arm"], item["episode"], item["status"] == "complete"
    if arm in COMPARATORS:
        work, whole = inv.rows(root / ep / "work.jsonl", required)
        times = [_seconds(r.get("wall_s")) if isinstance(r, dict) else None for r in work]
        if None in times:
            inv.defect("invalid time", root / ep / "work.jsonl")
        elif whole and work:
            item["work_s"] = sum(times)
        return
    folder = root / ep if arm == "L" else root / ep / "flight" / ep
    where = root / "timing.jsonl" if arm == "L" else root / ep / "flight" / "timing.jsonl"
    timing = shared["timing"] if arm == "L" else inv.rows(where, required)[0]
    spans, item["open_spans"] = _spans(inv, where, timing, ep)
    item["flight_s"], item["boundary_s"], item["snapshot_s"] = (spans.get(k) for k in ("flight", "after_episode",
                                                                                        "snapshot"))
    charges, ledger_whole = inv.ledger(folder / "ledger.jsonl", required)
    flown, _, gateway_whole = inv.gateway(folder, charges, label, {ep}, required=required)
    mine, learning_ok = flown, True
    if arm == "L":
        mine = flown + [r for r in shared["learned"] if r["episode"] == ep]
        learning_ok = shared["learning_whole"] and ep not in shared["unfinished"] and None not in shared["unfinished"]
    if ledger_whole and gateway_whole and learning_ok:
        item["requests"] = _demand(mine)


def read_run(run, study):
    """The inventory of one finished launcher run and its study: the allocation split into setup, host span and
    teardown, each dispatcher's wall time, every position's spans and new requests, the slots, and every defect."""
    run, study = Path(run), Path(study)
    inv = _Inventory()
    got = {"result": {}, "receipts": {"host_span": {}, "verdict": {}, "dispatch": []},
           "declared": {arm: [] for arm in M1B_POSITIONS}}
    for name, read in (("result", lambda: json.loads((run / "result.json").read_text())),
                       ("receipts", lambda: _receipts(run / "host" / "host-receipts.json")),
                       ("declared", lambda: _declared(study))):
        try:
            got[name] = read()
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            inv.defect("missing or invalid run record", f"{name}: {type(exc).__name__}: {exc}")
    result, receipts, declared = got["result"], got["receipts"], got["declared"]
    if not isinstance(result, dict):
        inv.defect("missing or invalid run record", f"result: {type(result).__name__}")
        result = {}
    span = receipts["host_span"]
    edges = [_seconds(span.get("started_epoch")), _seconds(span.get("finished_epoch")),
             _seconds(result.get("started_epoch")), _seconds(result.get("ended_epoch"))]
    parts = {"setup_s": None, "host_s": _seconds(span.get("wall_s")), "teardown_s": None}
    if None not in edges:
        parts["setup_s"], parts["teardown_s"] = _seconds(edges[0] - edges[2]), _seconds(edges[3] - edges[1])
    for name, value in parts.items():
        if value is None:
            inv.defect("invalid time", f"allocation {name}")
    inv.launcher(run, result)
    measured = {**parts, "host_reason": receipts.get("reason"), "complete": receipts["verdict"].get("complete") is True,
                "declared": declared, "dispatch": [], "positions": []}
    for d in receipts["dispatch"]:
        arm, stream = d["arm"], d["stream"]
        label, root = f"study/{arm}/{stream}", study / str(stream) / "arms" / arm
        measured["dispatch"].append({"name": d.get("name"), "arm": arm, "stream": stream,
                                     "status": d.get("dispatch_status"), "wall_s": d.get("wall_s")})
        done = any(p["status"] == "complete" for p in d["positions"])
        shared = {"timing": [], "learned": [], "unfinished": [None], "learning_whole": False}
        if arm == "L":
            try:
                shared["timing"] = inv.rows(root / "timing.jsonl", done)[0]
                learned = inv.learning(root, label, {p["episode"] for p in d["positions"]}, done)
                shared.update(zip(("learned", "unfinished", "learning_whole"), learned))
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                inv.defect("invalid record", f"{root}: {type(exc).__name__}: {exc}")
        for position in d["positions"]:
            ep = position["episode"]
            item = {"arm": arm, "stream": stream, "phase": int(EPISODE.match(ep).group(2)), "episode": ep,
                    "status": position["status"], "flight_s": None, "boundary_s": None, "snapshot_s": None,
                    "work_s": None, "open_spans": 0, "requests": None}
            try:
                _position(inv, item, root, label, shared)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                inv.defect("invalid record", f"{root / ep}: {type(exc).__name__}: {exc}")
                item.update(flight_s=None, boundary_s=None, snapshot_s=None, work_s=None, requests=None)
            measured["positions"].append(item)
    measured["slots"] = inv.slots(run)
    measured["new_requests"] = {role: _demand([r for r in inv.requests if r["role"] == role]) for role in MODEL_KINDS}
    measured["unfinished_requests"] = {str(role): n for role, n in inv.unfinished.items()}
    measured["defects"] = inv.defects
    return measured


class _Total:
    """A sum whose parts may be unknown. Its value is None unless every part is known; the known subtotal and the
    unknown parts, by name, are kept beside it."""

    def __init__(self, known=0.0, unknown=()):
        self.known, self.unknown = known, list(unknown)

    def add(self, name, x, times=1.0):
        if x is None:
            self.unknown.append(name)
        else:
            self.known += times * x
        return self

    def extend(self, other):
        self.known += other.known
        self.unknown += other.unknown
        return self

    @property
    def value(self):
        return None if self.unknown else self.known

    def report(self, scale=1.0):
        return {"value": None if self.unknown else self.known * scale, "known": self.known * scale,
                "unknown": self.unknown}


def _largest(totals):
    """The largest of several totals: unknown if any of them is, with the largest known subtotal."""
    return _Total(max(t.known for t in totals), [u for t in totals for u in t.unknown])


def _value(p, part):
    """One position's seconds for a part: a span, comparator work, or reasoning-request time by kind, as in
    'reasoning:watch'. None when unknown. A position whose request records are whole and that made no request of a
    kind spent zero on it."""
    if part.startswith("reasoning:"):
        if p["requests"] is None:
            return None
        e = p["requests"].get("reasoning_model:" + part.split(":")[1])
        return 0.0 if e is None else e["known_s"] if not e["unknown"] else None
    return p[part + "_s"]


def _unresolved(measured, done):
    """Why M1b's records cannot qualify a budget. Empty only when they cover M1b's declared design completely."""
    out = []
    declared = {(arm, s, ep) for arm, pairs in measured["declared"].items() for s, ep in pairs}
    counts = {}
    for arm, _, ep in declared:
        phase = int(EPISODE.match(ep).group(2))
        counts.setdefault(arm, {}).setdefault(phase, 0)
        counts[arm][phase] += 1
    if counts != M1B_POSITIONS:
        out.append(f"the study's {len(declared)} declared positions are not M1b's 410 by condition and phase")
    got = {(p["arm"], p["stream"], p["episode"]) for p in done}
    if declared - got:
        out.append(f"{len(declared - got)} of {len(declared)} declared positions not complete")
    if got - declared:
        out.append(f"{len(got - declared)} complete positions the study does not declare")
    if not measured["complete"]:
        out.append(f"the host did not finish complete: {measured['host_reason']}")
    fork_streams = {s for s, _ in measured["declared"]["F"]}
    needed = {"L": ("flight", "boundary"), "C": ("flight",), "F": ("flight",),
              "calculator": ("work",), "adaptive_controller": ("work",)}
    tally = {
        "unknown request durations": sum(e["unknown"] for p in done if p["requests"] for e in p["requests"].values()),
        "Telohelm positions without whole request records": sum(p["arm"] in TELOHELM and p["requests"] is None
                                                                for p in done),
        "unknown spans or work": sum(_value(p, part) is None for p in done for part in needed[p["arm"]])
                                 + sum(p["arm"] == "L" and p["stream"] in fork_streams and p["episode"].endswith(FORK)
                                       and p["snapshot_s"] is None for p in done),
        "open spans": sum(p["open_spans"] for p in measured["positions"])}
    out += [f"{label}: {n}" for label, n in tally.items() if n]
    out += [f"{kind}: {n}" for kind, n in sorted(Counter(k for k, _ in measured["defects"]).items())]
    return out


def project(measured, basis):
    """The forecast G = 2 x (S + T + H + P) / 3600 for each phase 5 learning scenario, with H = max(R, longest chain,
    E). Every assumption is named; the candidate flags are None unless M1b's records qualify."""
    done = [p for p in measured["positions"] if p["status"] == "complete"]
    reasons = _unresolved(measured, done)
    streams = sorted({s for s, _ in measured["declared"]["L"]})
    f_streams = {s for s, _ in measured["declared"]["F"]}
    pp = basis["per_phase"]
    assumptions = ["H = max(R, longest chain, E) is a forecast, not a bound: phases 4 and 5 are projected, and each "
                   "quantity is taken alone, so a path that combines slot queueing with a chain is not covered"]

    def m1b(arm, s, part, phases):
        """M1b's total for one stream's positions of a condition, unknown if any declared position is missing."""
        vals = [_value(p, part) for p in done if p["arm"] == arm and p["stream"] == s and p["phase"] in phases]
        expected = sum(1 for x, ep in measured["declared"][arm]
                       if x == s and int(EPISODE.match(ep).group(2)) in phases)
        t = _Total()
        if len(vals) != expected:
            t.unknown.append(f"{arm}:{s} phases {phases}: {len(vals)} of {expected} positions complete")
        for v in vals:
            t.add(f"{arm}:{s} {part}", v)
        return t

    def ratio(part, proxy_h):
        """L's measured seconds for a part in phases 1 to 3 against the cost basis's serial proxy for one stream: the
        mean of the per-stream ratios, or None if any stream's total is unknown."""
        totals = [m1b("L", s, part, (1, 2, 3)).value for s in streams]
        return None if not totals or None in totals else mean(t / (proxy_h * 3600) for t in totals)

    def scaled(r, hours):
        return None if r is None else r * hours * 3600

    ratios = {"flight": ratio("flight", sum(pp[str(p)]["flight_h"] for p in (1, 2, 3))),
              "boundary": ratio("boundary", sum(pp[str(p)]["learning_request_h"] + pp[str(p)]["residual_share_h"]
                                                for p in (1, 2, 3))),
              "learning": ratio("reasoning:learning", sum(pp[str(p)]["learning_request_h"] for p in (1, 2, 3)))}
    # Seconds per position by condition, phase and part: measured means for phases 1 to 3, projected for 4 and 5.
    parts = {"L": ("flight", "boundary", "reasoning:flight", "reasoning:watch", "reasoning:learning"),
             "C": ("flight", "reasoning:flight", "reasoning:watch"),
             "F": ("flight", "reasoning:flight", "reasoning:watch"),
             "calculator": ("work",), "adaptive_controller": ("work",)}
    rate = {}
    for arm, phases in SCORED_POSITIONS.items():
        for p in [x for x in phases if x <= 3]:
            for part in parts[arm]:
                vals = [_value(x, part) for x in done if x["arm"] == arm and x["phase"] == p]
                rate[(arm, p, part)] = mean(vals) if vals and None not in vals else None
    rate[("L", 4, "flight")] = scaled(ratios["flight"], pp["4"]["flight_h"] / PER_PHASE)
    rate[("L", 5, "flight")] = scaled(ratios["flight"], basis["phase5_episode_h_assumed"])
    rate[("L", 4, "boundary")] = scaled(ratios["boundary"],
                                        (pp["4"]["learning_request_h"] + pp["4"]["residual_share_h"]) / PER_PHASE)
    rate[("L", 4, "reasoning:learning")] = scaled(ratios["learning"], pp["4"]["learning_request_h"] / PER_PHASE)
    assumptions.append("L's phase 4 flights, boundaries and learning requests, and its phase 5 flights, scale the cost "
                       "basis's serial proxies by L's measured-to-proxy ratios for phases 1 to 3, one ratio for each "
                       f"part: {ratios}")
    for arm, p in (("L", 4), ("L", 5), ("C", 4)):
        for kind in ("flight", "watch"):
            rate[(arm, p, "reasoning:" + kind)] = rate[(arm, 3, "reasoning:" + kind)]
    assumptions.append("flight and watch reasoning requests in phases 4 and 5 take phase 3's mean per position; the "
                       "historical phase 4 made no model call, so for L this leans high")
    rate[("C", 4, "flight")] = rate[("C", 3, "flight")]
    for arm in COMPARATORS:
        rate[(arm, 4, "work")] = rate[(arm, 5, "work")] = rate[(arm, 3, "work")]
    assumptions.append("C's phase 4 and the comparators' phases 4 and 5 take their phase 3 means per position")

    # R: the reasoning slot's busy time over every scored position, flight, watch and learning each counted once.
    # Phase 5 learning is P, below, and is not in R. The held-to-request ratio compares the reasoning slot's released
    # holds with every new reasoning request in the run, and is unknown if either side has an unknown part.
    slot, entries = measured["slots"]["reasoning_model"], measured["new_requests"]["reasoning_model"].values()
    unfinished = sum(n for role, n in measured["unfinished_requests"].items() if role != "small_model")
    asked = _Total(sum(e["known_s"] for e in entries),
                   ["reasoning request time"] * (sum(e["unknown"] for e in entries) + unfinished))
    known_q = slot["known_held_s"] / asked.known if asked.known else None
    q = known_q if slot["held_s"] is not None and asked.value is not None else None
    assumptions.append(f"reasoning slot time is request time times M1b's held-to-request ratio {q}"
                       + ("" if q is not None else f"; the known parts give {known_q}"))
    R = _Total()
    for (arm, p, part), seconds in rate.items():
        if part.startswith("reasoning:"):
            R.add(f"{arm} phase {p} {part}", seconds, SCORED_POSITIONS[arm][p] * (1.0 if known_q is None else known_q))
    if q is None:
        R.unknown.append("the held-to-request ratio")

    # Chains: each dispatcher flies its positions in order, and F starts from L's snapshot after p2-e19.
    chains = {}
    for s in streams:
        fork = [_value(p, "snapshot") for p in done if p["arm"] == "L" and p["stream"] == s
                and p["episode"].endswith(FORK)]
        snapshot = fork[0] if len(fork) == 1 else None
        lead = m1b("L", s, "flight", (1, 2)).extend(m1b("L", s, "boundary", (1, 2))).add(f"L:{s} snapshot", snapshot)
        chains[f"L:{s}"] = (_Total().extend(lead).extend(m1b("L", s, "flight", (3,)))
                            .extend(m1b("L", s, "boundary", (3,)))
                            .add("L phase 4 flight", rate[("L", 4, "flight")], PER_PHASE)
                            .add("L phase 4 boundary", rate[("L", 4, "boundary")], PER_PHASE)
                            .add("L phase 5 flight", rate[("L", 5, "flight")], PER_PHASE))
        if s in f_streams:
            chains[f"F:{s}"] = _Total().extend(lead).extend(m1b("F", s, "flight", (3,)))
        chains[f"C:{s}"] = m1b("C", s, "flight", (1, 2, 3)).add("C phase 4 flight", rate[("C", 4, "flight")],
                                                               SCORED_POSITIONS["C"][4] / SCORED_STREAMS)
        for arm in COMPARATORS:
            chains[f"{arm}:{s}"] = m1b(arm, s, "work", (1, 2, 3)).add(
                f"{arm} phases 4 and 5", rate[(arm, 4, "work")], 2 * PER_PHASE)
    longest = _largest(chains.values()) if chains else _Total(unknown=["no chain"])
    assumptions.append("chains sum position spans and leave out time between positions; each scored stream takes the "
                       "longer of M1b's streams' chains")

    # E: M1b's host span scaled by the full schedule's summed position spans.
    snapshots = [_value(p, "snapshot") for p in done if p["arm"] == "L" and p["episode"].endswith(FORK)
                 and p["stream"] in f_streams]
    full = _Total().add("snapshots", mean(snapshots) if snapshots and None not in snapshots else None, SCORED_STREAMS)
    for (arm, p, part), seconds in rate.items():
        if not part.startswith("reasoning:"):
            full.add(f"{arm} phase {p} {part}", seconds, SCORED_POSITIONS[arm][p])
    m1b_sum = _Total()
    for s in streams:
        for arm in TELOHELM + COMPARATORS:
            for part in ("flight", "boundary") if arm == "L" else ("flight",) if arm in TELOHELM else ("work",):
                m1b_sum.extend(m1b(arm, s, part, (1, 2, 3)))
    m1b_sum.add("M1b snapshots", None if None in snapshots else sum(snapshots))
    host_s = measured["host_s"]
    E = _Total(host_s * full.known / m1b_sum.known if m1b_sum.known and host_s is not None else 0.0,
               full.unknown + m1b_sum.unknown)
    if not m1b_sum.known:
        E.unknown.append("M1b has no known position span")
    if host_s is None:
        E.unknown.append("M1b's host span")
    assumptions.append("E assumes the full study overlaps its work as M1b did")
    H = _largest([R, longest, E])

    # P: phase 5 learning, added after the host span. Each call holds the reasoning slot for the configured service
    # limit, which no measured ratio shrinks, plus M1b's boundary time outside learning requests per learning call.
    limit_s = basis["phase5_learning"]["service_per_candidate_gpu_h"] / GPUS * 3600
    calls = sum(p["requests"].get("reasoning_model:learning", {"count": 0})["count"]
                for p in done if p["arm"] == "L" and p["requests"] is not None)
    outside = _Total()
    for p in done:
        if p["arm"] == "L":
            b, used = _value(p, "boundary"), _value(p, "reasoning:learning")
            outside.add(p["episode"], None if b is None or used is None else b - used)
    if calls:
        residual = None if outside.value is None else outside.value / calls
        assumptions.append(f"each phase 5 learning call adds the {limit_s} s service limit, unscaled, and M1b's "
                           f"boundary time outside learning requests per learning call ({residual} s, and nothing if "
                           "negative); a phase 5 boundary without a call adds nothing; P is added after the host span, "
                           "never overlapped")
    else:
        residual = basis["phase5_learning"]["assumed_unclassified_per_candidate_gpu_h"] / GPUS * 3600
        assumptions.append(f"M1b made no learning call, so each phase 5 call adds the {limit_s} s service limit and "
                           f"the cost basis's assumed {residual} s outside it, both unscaled")
    S, T = measured["setup_s"], measured["teardown_s"]
    stresses = sum(s["added_gpu_h"] for s in basis["stress_scenarios"].values())
    assumptions.append("the cost basis's stress scenarios add their GPU-hours unscaled and serially, as there")
    threshold = basis["ceiling"]["freeze_threshold"]
    scenarios = {}
    for name, sc in basis["phase5_learning"]["scenarios"].items():
        outside_s = sc["calls"] * max(0.0, residual) if residual is not None else 0.0 if not sc["calls"] else None
        P = _Total(sc["calls"] * limit_s).add("phase 5 boundary time outside learning calls", outside_s)
        G = _Total().add("setup", S).add("teardown", T).extend(H).extend(P)
        stressed = _Total(G.known + stresses * 3600 / GPUS, G.unknown)
        scenarios[name] = {"P_s": P.report(), "G_gpu_h": G.report(GPUS / 3600),
                           "G_stressed_gpu_h": stressed.report(GPUS / 3600)}
    if not reasons and any(v["G_stressed_gpu_h"]["value"] is None for v in scenarios.values()):
        reasons.append("a forecast total has an unknown part")
    qualifies = not reasons
    return {"qualification": {
                "qualifies": qualifies, "reasons": reasons, "threshold_gpu_h": threshold,
                "candidates": {n: v["G_stressed_gpu_h"]["value"] <= threshold if qualifies else None
                               for n, v in scenarios.items()},
                "note": ("a candidate replaces no other freeze gate; a paused or incomplete M1b qualifies no demand, "
                         "and the arithmetic below is provisional until it qualifies")},
            "provisional": {"S_s": S, "T_s": T, "held_to_request_ratio": q,
                            "held_to_request_ratio_known_parts": known_q, "ratios": ratios,
                            "residual_per_learning_call_s": residual, "R_s": R.report(),
                            "chains_s": {n: t.report() for n, t in chains.items()}, "longest_chain_s": longest.report(),
                            "E_s": E.report(), "H_s": H.report(), "stresses_gpu_h": stresses, "scenarios": scenarios},
            "assumptions": assumptions,
            "not_projected": ["goal-change and help paths: untimed in any development run"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--basis", type=Path, required=True)
    ap.add_argument("--scheduler-gpu-hours", type=float)
    a = ap.parse_args()
    measured = read_run(a.run, a.study)
    out = project(measured, json.loads(a.basis.read_text()))
    out["scheduler_gpu_hours"] = a.scheduler_gpu_hours
    print(json.dumps(out, indent=1, default=str))
