"""Reconstruct saved example contexts with their actual task builder. Invented public records, no model or world."""
from dataclasses import replace
import json

import pytest

import telohelm.examples as X; import telohelm.learning as G
from telohelm.taskpacks.tanker.pack import TANKER_REDUCED_PACK
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog


def builder_type(seen):
    class SensorBuilder:
        def __init__(self, briefing):
            assert briefing == {"offset": 5}
            self.total = briefing["offset"]
        def update(self, bundle):
            assert set(bundle) == {"t", "sensor"}
            seen.append(bundle["t"])
            self.total += bundle["sensor"]
            return {"t": bundle["t"], "sensor_sum": self.total, "derived": self.total * 2, "reports": []}
    return SensorBuilder


def records(tmp_path, hidden=frozenset()):
    root = tmp_path / "saved"
    root.mkdir()
    broker = TrustedLog(root / "broker.log", root / "broker.heads", "invented")
    episode = TrustedLog(root / "episode.log", root / "episode.heads", "invented")
    broker.append({"kind": "briefing", "briefing": {"offset": 5}})
    total = 5
    contexts = {}
    for t in range(3):
        total += t * 2
        full = {"t": t, "sensor_sum": total, "derived": total * 2, "reports": []}
        contexts[t] = {k: v for k, v in full.items() if k not in hidden}
        broker.append({"kind": "bundle", "t": t, "bundle": {"t": t, "sensor": t * 2}})
        broker.append({"kind": "action", "t": t, "action": {}, "field_writers": {}})
        broker.append({"kind": "world_step", "t": t, "accepted": True})
        episode.append({"kind": "decision", "t": t, "decision": "heading",
                        "context_sha256": X._digest(contexts[t])})
    private = {"reports": [], "private_sentinel": "must not reach the builder or example"}
    (root / "summary.json").write_text(json.dumps({"outcome": private}))
    return root, contexts, private


@pytest.mark.parametrize("hidden", (frozenset(), frozenset({"derived"})))
def test_reconstruction_replays_every_bundle_and_matches_the_saved_visible_digest(tmp_path, hidden):
    root, contexts, private = records(tmp_path, hidden)
    seen = []
    (shown, receipt), = X.build(root, "saved", [2], {"goal_met": False}, private, hidden,
                                context_builder=builder_type(seen))
    assert seen == [0, 1, 2] and shown["context"] == contexts[2]
    assert receipt["recorded_context_sha256"] == receipt["full_context_sha256"] == X._digest(contexts[2])
    assert receipt["shown_example_sha256"] == X._digest(shown)
    assert "private_sentinel" not in json.dumps(shown)


def test_a_wrong_builder_is_refused_by_the_existing_saved_context_digest(tmp_path):
    root, _, private = records(tmp_path)
    class WrongBuilder(builder_type([])):
        def update(self, bundle):
            return {**super().update(bundle), "sensor_sum": -1}
    with pytest.raises(X.ExampleError, match="does not match the recorded hash"):
        X.build(root, "saved", [2], {"goal_met": False}, private, context_builder=WrongBuilder)


def test_selection_uses_its_supplied_builder_and_preserves_omission_records(tmp_path):
    root, contexts, _ = records(tmp_path)
    done = [G.Completed(1, "saved", None, False, "join", True)]
    seen = []
    shown, receipts, omitted = X.select(tmp_path, done, [{"episode": "saved", "failed": "join"}],
                                        context_builder=builder_type(seen))
    assert len(shown) == len(receipts) == 1 and shown[0]["context"] == contexts[2]
    assert seen == [0, 1, 2] and receipts[0]["second"] == 2
    assert omitted == [{"episode": "saved", "reason": "no decision at 120 s"}]


def test_corrupt_records_cannot_be_made_into_examples_by_a_custom_builder(tmp_path):
    root, _, private = records(tmp_path)
    with (root / "broker.log").open("a") as f:
        f.write('{"unsigned": true}\n')
    seen = []
    done = [G.Completed(1, "saved", None, False, "join", True)]
    shown, receipts, omitted = X.select(tmp_path, done, [{"episode": "saved", "failed": "join"}],
                                        context_builder=builder_type(seen))
    assert shown == receipts == [] and seen == []
    assert len(omitted) == 2 and omitted[-1]["episode"] == "saved" and omitted[-1]["reason"]


def test_the_monitor_passes_its_selected_builder_hidden_fields_and_goal(tmp_path, monkeypatch):
    selected = builder_type([])
    pack = replace(TANKER_REDUCED_PACK, context_builder=selected)
    calls = []
    def select(workdir, completed, reasons, hidden, goal, *, context_builder, policy):
        assert policy is pack.example_policy
        calls.append((hidden, goal, context_builder))
        return [], [], []
    monkeypatch.setattr(X, "select", select)
    co = G.Coordinator(Library(tmp_path / "library"), tmp_path, None, lambda _: True, None,
                       flight_pack=pack, procedures_only=True)
    assert co.monitor.examples(1, []) == ([], [])
    assert calls == [(pack.hidden, pack.goal, selected)]
