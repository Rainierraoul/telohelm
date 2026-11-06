"""Evidence captured at the start of a run and compared with the checked configuration (paper2/notes/TRANSPORT-CONTRACT.md
section 5). CPU draft for cross-review, TURN 93, revised TURN 95. Nothing in the runtime calls it yet, and no real evidence has been
captured: the installed weights, the launch, the server's model list, its effective defaults and the machine are only
available on the cluster.

A start record has one entry for each required item. Each entry says whether the value was observed or is unknown,
where it came from, when and at what point it was captured, and how the place it came from is protected. An unknown value
is recorded as unknown, never as a default. compare() checks the record against the configuration: a missing item, an
unknown one, or one that differs from the configuration is a problem, and any problem leaves nothing reportable.

The record goes into the gateway's trusted log before the first model call. admit_start() verifies that log, takes the
first start record, and requires it to come before any model_start event; a record written after a call does not admit
that call (the admission cut).

Values are compared by type as well as value (Codex's START94.TYPES): an observed True does not match a pinned 1, and
1024.0 does not match 1024. No normalisation is applied; a capture must record values in the configuration's types.

What a match shows: the declared configuration is the one the capture saw. It does not show which weights the GPU used,
and it does not authenticate the server.

Code identity (Codex's TURN 94 probe): capture_code() hashes the source files on disk at the moment it runs. It does not
show what code a process has already loaded; a file changed after import is hashed in its new form while the old code
keeps running. A claim about executed code therefore needs premises this module only partly checks:
- the runtime is installed somewhere the run cannot write, so the files cannot change during the run;
- the capture is taken before the runtime modules are first imported (a cold load), or the record says it was taken after
  import and is weaker;
- code_drift() is run again at the end of the run, and any file whose hash changed voids the record;
- bytecode caches are disabled or pinned too, since Python may load a cached .pyc instead of the source.
None of these is enforced here.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .manifest import CheckedConfig
from .trusted_log import TrustedLog; from .trusted_log import verify

START_VERSION = 1
STATUSES = ("observed", "unknown")
# Each required item and the part of the configuration it is compared with.
ITEMS = ("installed_code", "prompts", "python", "dependencies", "weights", "launch", "model_list",
         "effective_defaults", "environment")         # prompts added in TURN 95 (Codex's START94.PROMPTS)
ENTRY_FIELDS = {"status", "value", "source", "captured_at", "protection"}


class StartupError(ValueError):
    """A start record that is malformed."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def expected(config: CheckedConfig) -> dict[str, Any]:
    """What each item must be, read from the checked configuration."""
    runtime, generation, env = config.section("runtime"), config.section("generation"), config.section("environment")
    served = config.section("names")["served"]
    return {
        "installed_code": runtime["code_sha256"],
        "prompts": config.section("prompts"),
        "python": runtime["python"],
        "dependencies": runtime["dependencies"],
        "weights": config.section("weights")["files"],
        "launch": config.section("server")["launch_args"],
        "model_list": sorted({name for names in served.values() for name in names}),
        "effective_defaults": {k: generation[k] for k in ("max_tokens", "stop", "seed")},
        "environment": {k: env[k] for k in ("hardware", "precision", "concurrency", "batching")},
    }


def observed(value: Any, source: str, captured_at: str, protection: str) -> dict:
    return {"status": "observed", "value": value, "source": source, "captured_at": captured_at,
            "protection": protection}


def unknown(source: str, captured_at: str, protection: str) -> dict:
    return {"status": "unknown", "value": None, "source": source, "captured_at": captured_at,
            "protection": protection}


def record(config_sha256: str, capture_point: str, items: dict[str, dict]) -> dict:
    rec = {"version": START_VERSION, "config_sha256": config_sha256, "capture_point": capture_point, "items": items}
    validate(rec)
    return rec


def validate(rec: Any) -> None:
    """The record's shape. Missing items are allowed here so that compare() can name them; everything present must be
    well formed."""
    if not isinstance(rec, dict) or set(rec) != {"version", "config_sha256", "capture_point", "items"}:
        raise StartupError("a start record has exactly version, config_sha256, capture_point and items")
    if type(rec["version"]) is not int or rec["version"] != START_VERSION:
        raise StartupError(f"the start record version must be {START_VERSION}")
    if not isinstance(rec["capture_point"], str) or not rec["capture_point"].strip():
        raise StartupError("the capture point must be named")
    if not isinstance(rec["items"], dict) or not set(rec["items"]) <= set(ITEMS):
        raise StartupError(f"items must be a subset of {ITEMS}")
    for name, entry in rec["items"].items():
        if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS or entry["status"] not in STATUSES:
            raise StartupError(f"item {name!r} is not a well-formed entry")
        for field in ("source", "captured_at", "protection"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise StartupError(f"item {name!r} does not say its {field}")
        if entry["status"] == "unknown" and entry["value"] is not None:
            raise StartupError(f"item {name!r} is unknown but carries a value")
    try:
        _canonical(rec)
    except (TypeError, ValueError) as e:
        raise StartupError("the start record is not finite JSON") from e


def compare(rec: dict, config: CheckedConfig) -> list[str]:
    """Problems with the record against the configuration; empty means every required item was observed and matches."""
    validate(rec)
    problems = []
    if rec["config_sha256"] != config.fingerprint:
        problems.append("the start record was made for another configuration")
    want = expected(config)
    for name in ITEMS:
        entry = rec["items"].get(name)
        if entry is None:
            problems.append(f"{name}: missing")
        elif entry["status"] == "unknown":
            problems.append(f"{name}: unknown")
        elif not same(entry["value"], want[name]):
            problems.append(f"{name}: differs from the configuration")
    return problems


def same(a: Any, b: Any) -> bool:
    """Equal in value and in type at every depth: True is not 1, and 1024.0 is not 1024. Lists and objects compare
    element by element; object key order does not matter."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def capture_code(modules: list[ModuleType], base: Path) -> dict[str, str]:
    """SHA-256 of the source file on disk, now, of each module imported, keyed by its path relative to base. This is the
    file's current content, not the code already loaded (see the module notes). A module with no file, or one outside
    base, raises StartupError, because its code could not be pinned."""
    out = {}
    for m in modules:
        f = getattr(m, "__file__", None)
        if not f:
            raise StartupError(f"module {m.__name__} has no source file")
        path = Path(f).resolve()
        try:
            rel = str(path.relative_to(Path(base).resolve()))
        except ValueError as e:
            raise StartupError(f"module {m.__name__} is outside {base}") from e
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def code_drift(captured: dict[str, str], base: Path) -> list[str]:
    """The captured files whose content has changed or gone since capture, to be run again at the end of a run. A
    non-empty result voids the start record's code identity."""
    out = []
    for rel, digest in sorted(captured.items()):
        path = Path(base) / rel
        try:
            now = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            now = None
        if now != digest:
            out.append(rel)
    return out


def imported(prefix: str) -> list[ModuleType]:
    """The modules under a package prefix that this process has actually imported."""
    return [m for name, m in sorted(sys.modules.items()) if (name == prefix or name.startswith(prefix + "."))
            and m is not None and getattr(m, "__file__", None)]


def write(log: TrustedLog, rec: dict) -> None:
    validate(rec)
    log.append({"kind": "start_record", "record": rec,
                "record_sha256": hashlib.sha256(_canonical(rec).encode("utf-8")).hexdigest()})


def admit_start(log_path: Path, heads_path: Path, config: CheckedConfig) -> list[str]:
    """The admission check for a run: its log verifies, and its first start record comes before any model call,
    is intact and matches the configuration. Returns the problems found."""
    check = verify(Path(log_path), Path(heads_path))
    if not check.get("ok"):
        return [f"the log does not verify: {check.get('problem')}"]
    for line in Path(log_path).read_text().splitlines()[1:]:
        e = json.loads(line)["payload"]
        if e.get("kind") == "model_start":
            return ["a model call came before any start record"]
        if e.get("kind") == "start_record":
            rec = e.get("record")
            try:
                validate(rec)
            except StartupError as err:
                return [f"the start record is malformed: {err}"]
            if e.get("record_sha256") != hashlib.sha256(_canonical(rec).encode("utf-8")).hexdigest():
                return ["the start record does not match its digest"]
            return compare(rec, config)
    return ["no start record"]


# The gateway's start boundary (TURN 97, Codex's SG96 contract). CPU code with invented tests; not wired to episodes.

CODE_FIELDS = {"phase", "captured_at", "protection", "base", "files", "bytecode", "loaded", "problems"}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def validate_code(code: Any) -> None:
    """The code evidence's complete typed shape (TURN 99, Codex's SG98.CODE.SCHEMA)."""
    def text(v):
        return isinstance(v, str) and bool(v.strip())
    def digests(v):
        return isinstance(v, dict) and all(isinstance(k, str) and k and isinstance(d, str) and _HEX64.fullmatch(d)
                                           for k, d in v.items())
    if not isinstance(code, dict) or set(code) != CODE_FIELDS:
        raise StartupError(f"code evidence has exactly the fields {sorted(CODE_FIELDS)}")
    if code["phase"] not in ("pre-import", "post-import"):
        raise StartupError("the code evidence phase is unknown")
    for field in ("captured_at", "protection", "base"):
        if not text(code[field]):
            raise StartupError(f"the code evidence does not say its {field}")
    if not digests(code["files"]) or not code["files"] or not digests(code["bytecode"]):
        raise StartupError("the code evidence's hashes are malformed")
    if not isinstance(code["loaded"], list) or not all(
            isinstance(e, dict) and set(e) == {"module", "file", "loader", "cached"} and text(e["module"])
            and text(e["loader"]) and (e["file"] is None or text(e["file"])) and (e["cached"] is None or text(e["cached"]))
            for e in code["loaded"]):
        raise StartupError("the code evidence's loaded modules are malformed")
    if not isinstance(code["problems"], list) or not all(isinstance(p, str) for p in code["problems"]):
        raise StartupError("the code evidence's problems are malformed")


class RunAuthority:
    """What a run's start is checked against, supplied by a trusted bootstrap from independently protected records
    (SG96.1): the configuration made by manifest.checked() from the frozen expected fingerprint, and the code evidence
    from codecapture.evidence(), taken before the runtime was imported. Nothing the run writes (a sidecar, a journal or
    a candidate start record) can supply or change it. Immutable once made."""
    __slots__ = ("_config", "_code", "_check_drift")

    def __init__(self, config: CheckedConfig, code: dict, check_drift: bool = False):
        """check_drift: the gateway re-hashes the captured files before every fresh call and at close (SG96.6). It
        needs evidence from a real catalog, whose files exist under its base."""
        if not isinstance(config, CheckedConfig):
            raise StartupError("a run authority needs a configuration made by manifest.checked()")
        validate_code(code)
        if not isinstance(check_drift, bool):
            raise StartupError("check_drift must be True or False")
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_code", json.loads(_canonical(code)))
        object.__setattr__(self, "_check_drift", check_drift)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a run authority cannot be changed")

    @property
    def config(self) -> CheckedConfig:
        return self._config

    @property
    def code(self) -> dict:
        return json.loads(_canonical(self._code))

    @property
    def check_drift(self) -> bool:
        return self._check_drift

    def drift(self) -> list[str]:
        """Captured source or bytecode files that have changed or gone since capture, and any source or cached bytecode
        file that has appeared under the base since then (Codex's SG100.DRIFT.INVENTORY: a new .pyc beside a pinned
        source could be executed instead of it)."""
        base, out = Path(self._code["base"]), []
        for kind in ("files", "bytecode"):
            for rel, digest in sorted(self._code[kind].items()):
                p = base / rel
                if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                    out.append(rel)
        if base.is_dir():
            for pattern, kind in (("*.py", "files"), ("*.pyc", "bytecode")):
                for p in sorted(base.rglob(pattern)):
                    rel = str(p.relative_to(base))
                    if rel not in self._code[kind]:
                        out.append(f"{rel} (not captured)")
        return out


def log_anchor(heads_path: Path) -> str | None:
    """The anchor of a trusted log, from its protected heads: the identity of that run's log."""
    try:
        first = Path(heads_path).read_text().splitlines()[0]
        return json.loads(first).get("anchor")
    except (OSError, IndexError, ValueError, AttributeError):
        return None


def binding(anchor: str | None, episode: str, condition: str, stream: str, config_sha256: str) -> dict:
    """What a start record is bound to: this run's log, its scope and its configuration (SG96.2)."""
    return {"log_anchor": anchor, "episode": episode, "condition": condition, "stream": stream,
            "config_sha256": config_sha256}


def decide(event: dict, anchor: str | None, episode: str, condition: str, stream: str, config: CheckedConfig,
           before_first_call: bool) -> list[str]:
    """The problems with a start_record event for this run, recomputed from its contents (never from its own admitted
    flag). Empty means admitted: bound to this run, scope and configuration; a complete, typed record that matches the
    configuration; code evidence captured before import, with no problems and covering every pinned code file; and
    written before any model call."""
    problems = []
    if event.get("binding") != binding(anchor, episode, condition, stream, config.fingerprint):
        problems.append("bound to another run, scope or configuration")
    rec = event.get("record")
    try:
        validate(rec)
        if event.get("record_sha256") != hashlib.sha256(_canonical(rec).encode("utf-8")).hexdigest():
            problems.append("the start record does not match its digest")
        problems += compare(rec, config)
    except StartupError as e:
        problems.append(f"malformed start record: {e}")
    code = event.get("code")
    try:
        validate_code(code)
    except StartupError as e:
        problems.append(f"malformed code evidence: {e}")
        code = None
    if code is not None:
        runtime = config.section("runtime")
        if code["phase"] != "pre-import":
            problems.append("the code evidence was not captured before import")
        if code["problems"]:
            problems.append("the code evidence has problems")
        # All and only the pinned source files (SG98.CODE.COVERAGE), and all and only pinned bytecode, by default none,
        # so a cache that could run instead of the source is admitted only if a reviewed build pinned it
        # (SG98.CODE.BYTECODE).
        if code["files"] != runtime["code_sha256"]:
            problems.append("the captured source files are not exactly the pinned ones")
        if code["bytecode"] != runtime.get("bytecode_sha256", {}):
            problems.append("the captured bytecode is not exactly the pinned bytecode")
        for e in code["loaded"]:
            if e["file"] not in code["files"] or (e["cached"] is not None and e["cached"] not in code["bytecode"]) \
                    or e["loader"] != "SourceFileLoader":
                problems.append(f"module {e['module']} was not loaded from pinned source by the source loader")
    if not before_first_call:
        problems.append("written after a model call")
    return problems


def first_start(log_path: Path) -> tuple[dict | None, bool]:
    """The first start_record event in a log and whether it came before every model_start."""
    for line in Path(log_path).read_text().splitlines()[1:]:
        e = json.loads(line)["payload"]
        if e.get("kind") == "model_start":
            return None, False
        if e.get("kind") == "start_record":
            return e, True
    return None, True
