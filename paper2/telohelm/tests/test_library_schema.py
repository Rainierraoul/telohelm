"""Task-bound storage and shadow trials with authored fixtures, not learned new-task competence."""
from dataclasses import replace
import hashlib

import pytest

import telohelm.learning as G
from telohelm.taskpacks.tanker.public import TANKER
from telohelm.taskpacks.tanker.pack import TANKER_PACK
import telohelm.runtime.library as L
from .test_library import entry as heading_entry

TASK = replace(TANKER, pack_id="storage-fixture", decisions=("move", "stop"), pair_decisions=("move",),
               context_fields={"t", "distance"}, output_fields={"speed", "halt"},
               decision_fields={"move": {"speed"}, "stop": {"halt"}})


def entry(kind="procedure"):
    source = ('def decide(ctx):\n    return {"speed": 1}\n' if kind == "procedure"
              else "def check(ctx, out):\n    return 'accept'\n")
    return {"kind": kind, "decision": "move", "conditions": [{"field": "distance", "op": ">", "value": 0}],
            "source": source, "origin": {"calls": []}}


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_explicit_tanker_contract_preserves_every_legacy_byte(tmp_path):
    for name, kwargs in (("old", {}), ("explicit", {"pack": TANKER})):
        lib = L.Library(tmp_path / name, **kwargs)
        sha = lib.add(heading_entry(1))
        lib.set_status(sha, "adopted", "same")
        lib.recover()
        assert not lib.contract_path.exists()
    assert snapshot(tmp_path / "old") == snapshot(tmp_path / "explicit")


def test_custom_schema_is_durable_and_rejects_legacy_entries(tmp_path):
    lib = L.Library(tmp_path, pack=TASK)
    sha = lib.add(entry())
    lib.set_status(sha, "adopted", "fixture")
    again = L.Library(tmp_path, pack=replace(TASK))
    assert again.entry(sha)["source"] == entry()["source"]
    assert again.adopted("move", "procedure") == [sha]
    assert again.recover()["removed_bytes"] == 0
    assert again.events[0]["contract_sha256"] == hashlib.sha256(lib.contract_path.read_bytes()).hexdigest()
    with pytest.raises(L.LibraryError, match="refused"):
        again.add(heading_entry(1))
    with pytest.raises(L.LibraryError, match="refused"):
        again.add(dict(entry(), source="import os\ndef decide(ctx):\n    return {}\n"))


@pytest.mark.parametrize("change", ({"pack_id": "other"}, {"schema_version": 2}, {"decisions": ("move",)},
                                    {"pair_decisions": ()}, {"context_fields": {"t"}},
                                    {"output_fields": {"speed"}}, {"decision_fields": {"move": {"halt"}}}))
def test_mismatched_reopen_and_rebind_refuse_before_any_write(tmp_path, change):
    lib = L.Library(tmp_path, pack=TASK)
    before = snapshot(tmp_path)
    with pytest.raises(L.LibraryError, match="contract"):
        L.Library(tmp_path, pack=replace(TASK, **change))
    with pytest.raises(L.LibraryError, match="contract"):
        lib.bind_contract(replace(TASK, **change))
    assert snapshot(tmp_path) == before


def test_a_nonlegacy_library_cannot_fall_back_to_an_implicit_legacy_contract(tmp_path):
    lib = L.Library(tmp_path, pack=TASK)
    lib.add(entry())
    with pytest.raises(L.LibraryError, match="contract"):
        L.Library(tmp_path)
    lib.contract_path.unlink()  # lost metadata must not make recorded custom entries legacy
    before = snapshot(tmp_path)
    with pytest.raises(L.LibraryError, match="contract"):
        L.Library(tmp_path)
    with pytest.raises(L.LibraryError, match="populated legacy"):
        L.Library(tmp_path, pack=TASK)
    assert snapshot(tmp_path) == before


def test_empty_legacy_library_can_bind_but_populated_or_orphaned_one_cannot(tmp_path):
    empty = L.Library(tmp_path / "empty")
    empty.bind_contract(TASK)
    assert L.Library(empty.root, pack=TASK).pack == TASK
    for name, crash in (("populated", None), ("orphaned", "after_content")):
        lib = L.Library(tmp_path / name, crash_at=crash)
        if crash:
            with pytest.raises(L.Crash):
                lib.add(heading_entry(1))
        else:
            lib.add(heading_entry(1))
        before = snapshot(lib.root)
        with pytest.raises(L.LibraryError, match="populated legacy"):
            lib.bind_contract(TASK)
        assert snapshot(lib.root) == before


@pytest.mark.parametrize("crash", ("after_content", "mid_event", "after_event"))
def test_existing_process_crash_recovery_also_keeps_the_custom_contract(tmp_path, crash):
    lib = L.Library(tmp_path, crash_at=crash, pack=TASK)
    with pytest.raises(L.Crash):
        lib.add(entry())
    again = L.Library(tmp_path, pack=TASK)
    again.recover()
    sha = again.add(entry())
    again.set_status(sha, "adopted", "after recovery")
    assert L.Library(tmp_path, pack=TASK).adopted("move", "procedure") == [sha]


def test_a_malformed_contract_cannot_be_replaced_by_construction(tmp_path):
    lib = L.Library(tmp_path, pack=TASK)
    lib.contract_path.write_text("{")
    before = snapshot(tmp_path)
    with pytest.raises(L.LibraryError, match="damaged"):
        L.Library(tmp_path, pack=TASK)
    assert snapshot(tmp_path) == before


def test_custom_contract_survives_an_interrupted_atomic_adoption_group(tmp_path):
    lib = L.Library(tmp_path, pack=TASK)
    old = lib.add(entry())
    new = lib.add(dict(entry(), source='def decide(ctx):\n    return {"speed": 2}\n'))
    lib.set_status(old, "adopted", "original")
    plan = [{"entry": new, "status": "adopted", "reason": "replacement"},
            {"entry": old, "status": "suspended", "reason": "replacement"}]
    group = lib.open_group(plan, "replace")
    lib.set_status(new, "adopted", "replacement", group=group)
    again = L.Library(tmp_path, pack=TASK)
    again.recover()
    assert again.adopted("move", "procedure") == [old]
    assert again.complete_open_groups() == [group]
    assert again.complete_open_groups() == []
    assert L.Library(tmp_path, pack=TASK).adopted("move", "procedure") == [new]


def test_recovery_revalidates_nonlegacy_content_under_its_contract(tmp_path):
    lib = L.Library(tmp_path, pack=TASK)
    bad = {"schema": L.SCHEMA_VERSION, **heading_entry(1)}
    body = L.canonical(bad); sha = hashlib.sha256(body).hexdigest()
    (lib.content / f"{sha}.json").write_bytes(body)
    lib._append(lib._event("add", entry=sha, kind_of_entry="procedure", decision="heading", episode=None,
                           contract_sha256=lib._contract_sha))
    with pytest.raises(L.LibraryError, match="stored entry refused"):
        L.Library(tmp_path, pack=TASK).recover()


def test_monitor_shadow_uses_exact_selected_contract_for_procedure_and_check(tmp_path):
    lib = L.Library(tmp_path / "library", pack=TASK)
    task = replace(TANKER_PACK, public=TASK)
    seen = []
    def fly(shadow, completed, root):
        assert shadow.pack is TASK and shadow.contract_path.exists()
        assert len(shadow.adopted("move", "procedure")) == len(shadow.adopted("move", "check")) == 1
        seen.append(shadow.pack)
        return {"goal_met": True, "help": 0}
    co = G.Coordinator(lib, tmp_path / "work", None, None, None, flight_pack=task, fly=fly)
    co.workdir.mkdir()
    procedure, check = lib.add(entry()), lib.add(entry("check"))
    passed, detail = co.monitor.trial(procedure, None, "move", [object()], candidate_check=check)
    assert passed and seen == [TASK] and detail["trial"] == [{"goal_met": True, "help": 0}]
    assert lib.adopted("move", "procedure") == lib.adopted("move", "check") == []


def test_monitor_refuses_a_different_populated_library_without_calls_or_writes(tmp_path):
    lib = L.Library(tmp_path / "library")
    lib.add(heading_entry(1))
    before = snapshot(tmp_path)
    with pytest.raises(L.LibraryError, match="populated legacy"):
        G.Coordinator(lib, tmp_path / "work", None, None, None, flight_pack=replace(TANKER_PACK, public=TASK))
    assert snapshot(tmp_path) == before
