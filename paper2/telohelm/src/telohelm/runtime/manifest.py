"""The pinned configuration manifest and its fingerprint (paper2/notes/TRANSPORT-CONTRACT.md section 2). Draft for
cross-review, TURN 83, revised TURN 85; nothing in the runtime uses it yet, and no section's content is chosen here.

A manifest is a versioned JSON object with one section for each part of the configuration that must be pinned before a
model call can be reportable. Its fingerprint is the SHA-256 of its canonical JSON. The expected fingerprint is frozen
separately, in a file written when the protocol is frozen, and check() compares the two. A missing expected file, a
malformed one, a manifest missing a section, or a mismatch all raise ManifestError, so a reportable configuration
cannot be built without the comparison passing (Codex's TURN 82 requirement: fail closed).

Each section has required fields with checked types (REQUIRED below), so a manifest of empty or unknown fields cannot
pass (Codex's MF84.REQUIRED). checked() returns a CheckedConfig, the only form in which LocalChatProvider accepts a
configuration; it cannot be made from a bare digest (MF84.GATE). The adapter then checks its own arguments against
the configuration's sections. This is a guard against mistakes in trusted code; a hostile caller in the same process
could still forge the object.

What this shows and what it does not: a matching fingerprint shows that the declared configuration is the frozen one.
The manifest's contents are declarations and measurements made at launch; they corroborate which server and weights
were used, but they do not prove which weights the GPU loaded, and they do not authenticate the server.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
# Codex's TURN 80 additions are runtime, roles, generation and environment.
SECTIONS = ("server", "weights", "names", "prompts", "sampling", "transport", "runtime", "roles", "generation",
            "environment")
_HEX64 = re.compile(r"[0-9a-f]{64}")
ROLES = ("small_model", "reasoning_model")      # the model roles; models.ROLES is the same (tested)


def _hex(v: Any) -> bool:
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def _whole(v: Any, low: int = 0) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= low


def _number(v: Any, low: float = 0.0, strict: bool = False) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            and (v > low if strict else v >= low))


def _per_role(check):
    return lambda v: isinstance(v, dict) and set(v) == set(ROLES) and all(check(x) for x in v.values())


def _names(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, str) and x.strip() for x in v)


def _nonempty_text(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


# The fields each section must have, with the check each value must pass. Other fields may be added.
REQUIRED = {
    "server": {"software": _nonempty_text, "version": _nonempty_text, "launch_args": lambda v: isinstance(v, list)},
    "weights": {"repository": _nonempty_text, "revision": _nonempty_text,
                "files": lambda v: isinstance(v, dict) and bool(v) and all(_hex(x) for x in v.values())},
    "names": {"served": _per_role(_names)},
    "prompts": {"messages_builder_sha256": _hex, "chat_template_sha256": _hex, "parser_sha256": _hex},
    "sampling": {"temperature": _number, "max_output_tokens": _per_role(lambda v: _whole(v, 1)),
                 "seed": lambda v: v is None or _whole(v)},
    "transport": {"origin": _nonempty_text, "max_response_bytes": lambda v: _whole(v, 1),
                  "timeout_s": _per_role(lambda v: _number(v, strict=True))},
    "runtime": {"python": _nonempty_text, "dependencies": lambda v: isinstance(v, dict),
                "code_sha256": lambda v: isinstance(v, dict) and bool(v) and all(_hex(x) for x in v.values())},
    "roles": {"model": _per_role(_nonempty_text), "modelled_latency_s": _per_role(_number),
              "reasoning": _per_role(lambda v: isinstance(v, dict))},
    "generation": {"max_tokens": _per_role(lambda v: _whole(v, 1)), "stop": lambda v: isinstance(v, list),
                   "seed": lambda v: v is None or _whole(v)},
    "environment": {"hardware": _nonempty_text, "precision": _nonempty_text, "concurrency": lambda v: _whole(v, 1),
                    "batching": lambda v: isinstance(v, dict)},
}


class ManifestError(ValueError):
    """The manifest or its expected fingerprint is missing, malformed or does not match."""


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _text(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        v.encode("utf-8")
    except UnicodeEncodeError:            # a lone surrogate
        return False
    return True


def _plain(v: Any) -> bool:
    """JSON values only: UTF-8 text, whole or finite numbers, booleans, null, and lists or objects of them with text
    keys."""
    if v is None or isinstance(v, bool) or (isinstance(v, int)):
        return True
    if isinstance(v, str):
        return _text(v)
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, list):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(_text(k) for k in v) and all(_plain(x) for x in v.values())
    return False


def validate(manifest: Any) -> None:
    if not isinstance(manifest, dict) or set(manifest) != {"version", *SECTIONS}:
        raise ManifestError(f"a manifest has exactly the version and the sections {SECTIONS}")
    if type(manifest["version"]) is not int or manifest["version"] != MANIFEST_VERSION:
        raise ManifestError(f"the manifest version must be the whole number {MANIFEST_VERSION}")
    for name in SECTIONS:
        if not isinstance(manifest[name], dict) or not manifest[name]:
            raise ManifestError(f"section {name!r} must be a non-empty object")
        for field, ok in REQUIRED[name].items():
            if field not in manifest[name] or not ok(manifest[name][field]):
                raise ManifestError(f"section {name!r} lacks a valid {field!r}")
    if not _plain(manifest):
        raise ManifestError("the manifest holds a value that is not finite, UTF-8 JSON")


def fingerprint(manifest: dict) -> str:
    validate(manifest)
    return hashlib.sha256(_canonical(manifest)).hexdigest()


def load_expected(path: Path) -> str:
    """The frozen expected fingerprint: one line of 64 lower-case hex digits."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ManifestError(f"no readable expected fingerprint at {path}") from e
    value = text.strip()
    if not _HEX64.fullmatch(value) or text not in (value, value + "\n"):
        raise ManifestError("the expected fingerprint file is not one SHA-256 line")
    return value


def check(manifest: dict, expected_path: Path) -> str:
    """The manifest's fingerprint, if it equals the frozen expected one; otherwise ManifestError."""
    expected = load_expected(expected_path)
    actual = fingerprint(manifest)
    if actual != expected:
        raise ManifestError("the configuration does not match the frozen fingerprint")
    return actual


def file_digests(paths: list[Path]) -> dict[str, str]:
    """SHA-256 of each file, for the runtime and prompts sections. A missing file raises ManifestError."""
    out = {}
    for p in paths:
        try:
            out[str(p)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        except OSError as e:
            raise ManifestError(f"cannot hash {p}") from e
    return out


_FACTORY = object()


class CheckedConfig:
    """A manifest whose fingerprint has been checked against the frozen one. Made only by checked(). It holds the
    manifest as canonical JSON text and hands out copies, and its attributes cannot be set after it is made, so it cannot
    change under its fingerprint (Codex's CFG86.MUTABLE)."""
    __slots__ = ("_json", "_fingerprint")

    def __init__(self, manifest: dict, fingerprint: str, _token: object = None):
        if _token is not _FACTORY:
            raise ManifestError("a checked configuration is made only by manifest.checked()")
        object.__setattr__(self, "_json", _canonical(manifest).decode("utf-8"))
        object.__setattr__(self, "_fingerprint", fingerprint)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a checked configuration cannot be changed")

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def manifest(self) -> dict:
        return json.loads(self._json)

    def section(self, name: str) -> dict:
        return json.loads(self._json)[name]


def checked(manifest: dict, expected_path: Path) -> CheckedConfig:
    """The configuration, if its fingerprint equals the frozen expected one; otherwise ManifestError."""
    return CheckedConfig(manifest, check(manifest, expected_path), _FACTORY)
