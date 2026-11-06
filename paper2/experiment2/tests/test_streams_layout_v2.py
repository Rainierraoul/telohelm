"""Layout v2 of the stream generator (paper2/notes/STUDY-DECISIONS-CLAUDE-20261026.md, revisions 2 and 3): phase 3 is
an orbit block then a late-information block of ten, with balanced delays; everything else is unchanged and v1 stays
the default."""
from collections import Counter

import pytest

from paper2.experiment2 import streams as G

STREAMS = (4491, 4492, 4493, 4494)    # development identities outside every reserved block (TURN244)


def _drawn(*args, **kw):
    """The scenario, or the reason a phase 4 draw is unusable, so both layouts can be compared either way."""
    try:
        return G.draw(*args, **kw)
    except G.Unusable as u:
        return ("unusable", u.reason, repr(u.details))


def test_v1_is_the_default_and_v2_leaves_other_phases_alone():
    for phase in (1, 2, 4):
        for index in (0, 7, 13, 19):
            for attempt in (0, 1):
                default = _drawn(4491, phase, index, attempt)
                assert default == _drawn(4491, phase, index, attempt, layout="v1")
                assert default == _drawn(4491, phase, index, attempt, layout="v2")
    assert G.draw(4491, 3, 3) == G.draw(4491, 3, 3, layout="v1")


def test_v2_phase_3_has_an_orbit_block_then_a_delay_block_with_wind():
    for stream in STREAMS:
        for index in range(20):
            s = G.draw(stream, 3, index, layout="v2")
            shell1 = s.tankers[0].orbit
            assert s.wind != (0.0, 0.0) and s.dropouts == ()
            if index < 10:
                assert s.delay_s == 0 and shell1.delta != 0
            else:
                assert s.delay_s in G.DELAYS and shell1.delta == 0


def test_each_half_block_of_delays_has_the_declared_mixture():
    orders = set()
    for stream in STREAMS:
        for half in (2, 3):
            delays = [G.draw(stream, 3, 5 * half + j, layout="v2").delay_s for j in range(5)]
            assert Counter(delays) == Counter(G.V2_DELAY_HALF)
            orders.add(tuple(delays))
    assert len(orders) > 1                     # drawn per stream and half-block, not one fixed order


def test_a_retry_keeps_the_delay():
    for attempt in range(4):
        assert G.draw(4491, 3, 12, attempt=attempt, layout="v2").delay_s == G.v2_delay(4491, 12)


def test_unknown_layout_and_out_of_range_delay_are_refused():
    with pytest.raises(ValueError):
        G.draw(4491, 3, 0, layout="v3")
    with pytest.raises(ValueError):
        G.episode(4491, 3, 0, layout="v3")
    with pytest.raises(ValueError):
        G.v2_delay(4491, 9)


def test_generation_records_name_v2_and_keep_v1_bytes():
    g1 = G.solvable_episode(4491, 3, 10)
    g2 = G.solvable_episode(4491, 3, 10, layout="v2")
    assert "layout" not in G.generation_record(g1)
    assert G.generation_record(g2)["layout"] == "v2"
    assert g2.scenario.delay_s == G.v2_delay(4491, 10) and g2.scenario.dropouts == ()
