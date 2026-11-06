"""Hand-worked tests for the earliest-contact lower bound (note B8, finding B8.16.2)."""
from __future__ import annotations

import math

import pytest

from paper2.experiment2.hidden_set.bounds import earliest_contact_bound
import telohelm.taskpacks.tanker.world.physics as P

S = 0.15        # NM/s, generous aircraft speed bound


def test_stationary_tanker_matches_the_formula():
    # |p − x| = 10, so q(s) = 10 − 0.15·s − 0.25 reaches 0 at s = 65.
    r = earliest_contact_bound((0.0, 0.0), 0.0, S, lambda s: (10.0, 0.0), 0.0, 200.0)
    assert r["status"] == "bound" and 65.0 - 0.01 <= r["lower"] <= 65.0


def test_tanker_moving_away_slower_than_the_bound():
    v = 0.05
    r = earliest_contact_bound((0.0, 0.0), 0.0, S, lambda s: (10.0 + v * s, 0.0), v, 500.0)
    exact = (10.0 - 0.25) / (S - v)                                   # 97.5 s
    # q falls at S − v = 0.1 NM/s while the bound uses V + S = 0.2, so the result can sit up to
    # (V + S)·width/(S − v) + width = 0.03 s early. It must never be late.
    assert r["status"] == "bound" and exact - 0.03 <= r["lower"] <= exact


def test_tanker_moving_away_faster_is_excluded_to_the_horizon():
    v = 0.2
    r = earliest_contact_bound((0.0, 0.0), 0.0, S, lambda s: (10.0 + v * s, 0.0), v, 1000.0)
    assert r == {"status": "excluded", "until": 1000.0, "examined": r["examined"]}


def test_a_deadline_before_any_possible_contact_is_excluded():
    r = earliest_contact_bound((0.0, 0.0), 0.0, S, lambda s: (10.0, 0.0), 0.0, 60.0)
    assert r["status"] == "excluded"


def test_already_in_contact_gives_the_start():
    r = earliest_contact_bound((0.0, 0.0), 30.0, S, lambda s: (0.1, 0.0), 0.0, 100.0)
    assert r["status"] == "bound" and r["lower"] == 30.0


@pytest.mark.parametrize("theta0", [0.0, 1.0, 2.5, 4.0])
def test_circular_orbit_agrees_with_a_fine_grid(theta0):
    orbit = P.Orbit(40.0, 10.0, 8.0, P.kt(280.0), theta0)
    x, t0 = (0.0, 0.0), 0.0

    def q(s):
        px, py = orbit.position(s)
        return math.hypot(px - x[0], py - x[1]) - S * (s - t0) - 0.25

    grid_first = next(k * 0.001 for k in range(2_000_000) if q(k * 0.001) <= 0)
    r = earliest_contact_bound(x, t0, S, orbit.position, orbit.v, 2000.0)
    assert r["status"] == "bound"
    assert r["lower"] <= grid_first + 1e-9                             # never later than the first contact
    # The guarantee at the returned edge: its interval could not be dropped, so q there is at most
    # (V + S)·width plus the allowance. How early it sits depends on how fast q is falling.
    assert q(r["lower"]) <= (orbit.v + S) * 0.01 + 1e-6


def test_work_cap_gives_unresolved():
    r = earliest_contact_bound((0.0, 0.0), 0.0, S, lambda s: (10.0, 0.0), 0.0, 200.0, max_intervals=5)
    assert r["status"] == "unresolved"
