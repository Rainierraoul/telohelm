"""Tests for the trusted log and its protected heads (architecture section 10)."""
from __future__ import annotations

import pytest

from telohelm.runtime.trusted_log import TrustedLog; from telohelm.runtime.trusted_log import verify


def make(tmp_path, n=5):
    (tmp_path / "protected").mkdir()
    log = TrustedLog(tmp_path / "runner.log", tmp_path / "protected" / "heads.jsonl", "run-1")
    for i in range(n):
        log.append({"kind": "action", "t": i, "heading_deg": 90 + i})
    return tmp_path / "runner.log", tmp_path / "protected" / "heads.jsonl"


def test_an_untouched_log_verifies(tmp_path):
    log, heads = make(tmp_path)
    assert verify(log, heads) == {"ok": True, "records": 6}


def test_editing_a_record_is_found(tmp_path):
    log, heads = make(tmp_path)
    lines = log.read_text().splitlines()
    lines[3] = lines[3].replace('"heading_deg":92', '"heading_deg":45')
    log.write_text("\n".join(lines) + "\n")
    assert not verify(log, heads)["ok"]


def test_a_change_is_found_while_it_stands_but_an_exact_restore_matches(tmp_path):
    """The limit Codex named (A10.2): heads catch any change still present when checked. A change made and
    exactly undone between checks leaves nothing to find, which is why Telohelm gets no write access."""
    log, heads = make(tmp_path)
    original = log.read_bytes()
    log.write_bytes(original.replace(b'"heading_deg":91', b'"heading_deg":10'))
    assert not verify(log, heads)["ok"]
    log.write_bytes(original)
    assert verify(log, heads)["ok"]


def test_cutting_the_log_short_is_found(tmp_path):
    log, heads = make(tmp_path)
    lines = log.read_text().splitlines()
    log.write_text("\n".join(lines[:-2]) + "\n")
    assert "protected count" in verify(log, heads)["problem"]


def test_a_chain_rebuilt_from_the_start_is_found(tmp_path):
    log, heads = make(tmp_path)
    forged = tmp_path / "forged"
    forged.mkdir()
    fake = TrustedLog(forged / "runner.log", forged / "heads.jsonl", "run-1")
    for i in range(5):
        fake.append({"kind": "action", "t": i, "heading_deg": 0})
    log.write_bytes((forged / "runner.log").read_bytes())         # a consistent chain, but not the anchored one
    assert not verify(log, heads)["ok"]


def test_a_log_is_written_once(tmp_path):
    log, heads = make(tmp_path)
    with pytest.raises(FileExistsError):
        TrustedLog(log, heads, "run-2")
