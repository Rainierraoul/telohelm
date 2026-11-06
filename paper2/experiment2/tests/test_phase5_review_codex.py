"""Independent phase5 bridge controls; conditional certificate limits remain explicit."""
from dataclasses import replace

import telohelm.taskpacks.tanker.world.episode as E; import paper2.experiment2.streams as G; import telohelm.taskpacks.tanker.world.phase5 as phase5


def test_public_histories_do_not_reveal_the_flown_sign_before_help():
    d, _ = G.draw_help(4401, 0)
    a, b = E.Episode(d.world_a), E.Episode(d.world_b)
    assert a.reset() == b.reset()
    for _ in range(240):
        assert a.step({}) == b.step({})
    assert d.world_a.tankers[0].orbit.position(240) != d.world_b.tankers[0].orbit.position(240)


def test_seeded_flown_choice_can_select_either_world_and_does_not_change_the_pair():
    seen = set()
    for i in range(12):
        d, world = G.draw_help(4401, i)
        seen.add(world)
        assert d.seed == G.episode_seed(4401, 5, i, 0)
        a, b = d.world_a, d.world_b
        assert replace(a.tankers[0].orbit, delta=0) == replace(b.tankers[0].orbit, delta=0)
    assert seen == {"a", "b"}


def test_certified_stream_preserves_both_world_confirmations_and_numerical_limit():
    g = G.episode(4401, 5, 0)
    assessment = g.witness
    assert assessment["label"] == "help certified"
    assert assessment["certificate"]["settings"]["numerical_qualification"] == "conditional on empirical budgets, not validated bounds"
    assert all(m > 0 for m in assessment["certificate"]["strict_margin_nm"].values())
    assert all(w["settled"] and w["goal_met"] and w["joined_at"] <= assessment["deadline"]
               for w in assessment["confirmation"].values())
    d, flown = G.draw_help(4401, 0, g.attempts[-1]["attempt"])
    expected = d.world_a if flown == "a" else d.world_b
    assert g.scenario == replace(expected, goals=((0, replace(expected.goals[0][1], deadline=assessment["deadline"])),))
