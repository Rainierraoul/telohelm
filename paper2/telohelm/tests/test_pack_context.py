"""Slice 2(c), first half: what Telohelm sees each second comes from the task pack (study2/packs/tanker_context.py).

The builder moved unchanged from runtime/context.py, which re-exports it for the experiment's callers; the episode loop
takes the builder it is given, so another task can bring its own.
"""
import math

from dataclasses import replace

import telohelm.taskpacks.tanker.context as TC
from telohelm.taskpacks.tanker.pack import TANKER_PACK
import telohelm.runtime.context as old
from telohelm.runtime.episode_loop import NoModels; from telohelm.runtime.episode_loop import run_episode
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog
import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.physics as P


def scenario():
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), P.kt(450.0), P.turn_rate(450.0))
    tankers = (E.Tanker("SHELL 1", P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), math.radians(200))),
               E.Tanker("SHELL 2", P.Orbit(40.0, 60.0, 8.0, P.kt(280.0), 0.0)))
    return E.Scenario(episode_id="context-test", receiver=rcv, wind=(0.0, 0.0), tankers=tankers,
                      goals=((0, E.Goal(1, "SHELL 1", 20.0)),))


def test_the_old_import_gives_the_same_builder():
    assert old.ContextBuilder is TC.ContextBuilder and old.HISTORY == TC.HISTORY


def test_the_episode_loop_builds_what_telohelm_sees_with_the_builder_it_is_given(tmp_path):
    seen = []

    class Counting(TC.ContextBuilder):
        def update(self, bundle):
            ctx = super().update(bundle)
            seen.append(ctx["t"])
            return {**ctx, "marker": True}
    (tmp_path / "protected").mkdir()
    log = TrustedLog(tmp_path / "runner.log", tmp_path / "protected" / "heads.jsonl", "context-run")
    run_episode(scenario(), Library(tmp_path / "empty"), NoModels(), log, Ledger(), max_steps=5,
                pack=replace(TANKER_PACK, context_builder=Counting))
    assert seen == [0, 1, 2, 3, 4]
