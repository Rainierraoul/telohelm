"""Candidate syntax follows the selected public contract; this does not demonstrate new-task adoption."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

import telohelm.learning as G
from telohelm.taskpacks.tanker.public import TANKER
from telohelm.taskpacks.tanker.pack import TANKER_PACK
import telohelm.runtime.procedures as P
from .test_learning import Scripted; from .test_learning import finish; from .test_learning import reply
import paper2.experiment2.mvp_seed as seed
from telohelm.runtime.library import Library

TASK = replace(TANKER, pack_id="schema-fixture", decisions=("move", "stop"), pair_decisions=("move",),
               context_fields={"t", "distance"}, output_fields={"speed"},
               decision_fields={"move": {"speed"}, "stop": {"halt"}})


def single():
    return {"kind": "procedure", "decision": "move", "conditions": [{"field": "distance", "op": ">", "value": 0}],
            "source": 'def decide(ctx):\n    return {"speed": 1}\n',
            "reason": {"code": "missing_tool", "text": "invented"}}


def pair():
    entry = single()
    return {"kind": "pair", "decision": "move", "procedure": {"conditions": deepcopy(entry["conditions"]),
            "source": entry["source"]}, "check": {"conditions": deepcopy(entry["conditions"]),
            "source": "def check(ctx, out):\n    return 'accept'\n"}, "reason": entry["reason"]}


def test_single_entry_uses_only_the_selected_decisions_and_context_fields():
    assert P.reply_errors(single(), pack=TASK) == []
    assert P.reply_errors(single())
    wrong = single(); wrong["decision"] = "heading"
    assert P.reply_errors(wrong, pack=TASK)
    wrong = single(); wrong["conditions"][0]["field"] = "report_age_s"
    assert P.reply_errors(wrong, pack=TASK)


def test_pairs_and_both_halves_use_the_same_selected_contract():
    assert P.reply_errors(pair(), pack=TASK) == P.pair_errors(pair(), pack=TASK) == []
    assert P.reply_errors(pair())
    wrong = pair(); wrong["decision"] = "stop"
    assert P.reply_errors(wrong, pack=TASK)  # known single decision, but not a permitted pair decision
    for half in ("procedure", "check"):
        wrong = pair(); wrong[half]["conditions"][0]["field"] = "report_age_s"
        assert any(e.startswith(half + ":") for e in P.reply_errors(wrong, pack=TASK))


@pytest.mark.parametrize("fault", ("shape", "decision_type", "condition_type", "nonfinite", "source", "reason"))
def test_selecting_a_task_does_not_relax_shape_source_or_value_safeguards(fault):
    candidate = single()
    if fault == "shape":
        candidate["extra"] = True
    elif fault == "decision_type":
        candidate["decision"] = []
    elif fault == "condition_type":
        candidate["conditions"][0]["op"] = []
    elif fault == "nonfinite":
        candidate["conditions"][0]["value"] = float("nan")
    elif fault == "source":
        candidate["source"] = "import os\ndef decide(ctx):\n    return {}\n"
    else:
        candidate["reason"]["text"] = "x" * (P.MAX_REASON_CHARS + 1)
    assert P.reply_errors(candidate, pack=TASK)


def test_explicit_tanker_selection_preserves_legacy_error_messages():
    valid = json.loads(reply())
    for change in ({}, {"decision": "unknown"}, {"decision": []}, {"conditions": [{"field": "absent", "op": ">", "value": 1}]}):
        candidate = {**deepcopy(valid), **change}
        assert P.reply_errors(candidate) == P.reply_errors(candidate, pack=TANKER)
    bad_pair = pair(); bad_pair["decision"] = "unknown"
    assert P.pair_errors(bad_pair) == P.pair_errors(bad_pair, pack=TANKER)


@pytest.mark.parametrize("candidates", (1, 3))
def test_both_coordinator_paths_pass_the_exact_selected_contract(tmp_path, monkeypatch, candidates):
    chosen = replace(TANKER_PACK, public=replace(TANKER, pack_id="explicit-fixture"))
    seen = []
    original = G.reply_errors
    def check(candidate, *, pack):
        seen.append(pack)
        return original(candidate, pack=pack)
    monkeypatch.setattr(G, "reply_errors", check)
    lib = Library(tmp_path / "library", pack=chosen.public)
    seed.stock(lib)
    co = G.Coordinator(lib, tmp_path, Scripted(reply()), lambda s: True, seed.stand_in_tool,
                       candidates=candidates, flight_pack=chosen,
                       fly=lambda *args: {"goal_met": True, "help": 0})
    finish(co, 5)
    assert seen and all(contract is chosen.public for contract in seen)
