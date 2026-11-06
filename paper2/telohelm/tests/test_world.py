"""Tests for the Experiment 2 physics core against hand-calculated cases. Nothing here is a result."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import telohelm.taskpacks.tanker.world.physics as P

R90 = math.radians(90.0)


def state(heading_deg, commanded_deg, tas_kt=450.0, x=0.0, y=0.0, t=0.0):
    return P.ReceiverState(t, x, y, math.radians(heading_deg), math.radians(commanded_deg),
                           P.kt(tas_kt), P.turn_rate(tas_kt))


def test_turn_rate_matches_hand_calculation():
    w = P.turn_rate(450.0)
    assert math.degrees(w) == pytest.approx(1.401, abs=5e-4)
    assert (math.pi / 2) / w == pytest.approx(64.2, abs=0.05)
    assert P.kt(450.0) / w == pytest.approx(5.111, abs=5e-4)       # turn radius, NM


def test_quarter_turn_without_wind_ends_one_radius_ahead_and_aside():
    s = state(0, 90)
    tau = P.turn_end_time(s)
    end = P.receiver_at(s, (0.0, 0.0), tau)
    radius = P.kt(450.0) / P.turn_rate(450.0)
    assert end.x == pytest.approx(radius, abs=1e-12)
    assert end.y == pytest.approx(radius, abs=1e-12)
    assert end.heading == pytest.approx(R90, abs=1e-12)


def test_wind_adds_its_drift_exactly():
    s = state(0, 90)
    wind = (P.kt(40.0), P.kt(-20.0))
    still = P.receiver_at(s, (0.0, 0.0), 100.0)
    windy = P.receiver_at(s, wind, 100.0)
    assert windy.x - still.x == pytest.approx(wind[0] * 100.0, abs=1e-12)
    assert windy.y - still.y == pytest.approx(wind[1] * 100.0, abs=1e-12)


def test_turn_ending_inside_a_second_then_straight():
    s = state(0, 30)
    tau = P.turn_end_time(s)
    assert 21.0 < tau < 22.0
    a = P.receiver_at(s, (0.0, 0.0), 22.0)
    b = P.receiver_at(P.receiver_at(s, (0.0, 0.0), tau), (0.0, 0.0), 22.0)
    assert a.heading == pytest.approx(math.radians(30), abs=1e-12)
    assert (a.x, a.y) == pytest.approx((b.x, b.y), abs=1e-12)


def test_receiver_in_steps_equals_receiver_in_one_go():
    s = state(10, 250)
    wind = (P.kt(60.0), 0.0)
    one = P.receiver_at(s, wind, 300.0)
    many = s
    for t in range(1, 301):
        many = P.receiver_at(many, wind, float(t))
    assert (many.x, many.y, many.heading) == pytest.approx((one.x, one.y, one.heading), abs=1e-9)


def test_shorter_way_and_tie():
    assert P.turn_direction(math.radians(350), math.radians(10)) == 1
    assert P.turn_direction(math.radians(10), math.radians(350)) == -1
    assert P.turn_direction(0.0, math.pi) == 1
    assert P.turn_direction(1.0, 1.0) == 0


def test_closed_form_turn_matches_an_independent_integrator():
    s = state(20, 200)
    wind = (P.kt(-30.0), P.kt(45.0))
    tau = P.turn_end_time(s)

    def rhs(t, z):
        h = z[2]
        rate = s.omega if t < tau else 0.0
        return [s.tas * math.sin(h) + wind[0], s.tas * math.cos(h) + wind[1], rate]

    sol = solve_ivp(rhs, (0.0, tau), [0.0, 0.0, s.heading], rtol=1e-12, atol=1e-13)
    exact = P.receiver_at(s, wind, tau)
    assert sol.y[0, -1] == pytest.approx(exact.x, abs=1e-8)
    assert sol.y[1, -1] == pytest.approx(exact.y, abs=1e-8)


@pytest.mark.parametrize("delta", [0.0, 2.0, -1.6])
def test_tanker_holds_total_ground_speed_and_radius(delta):
    orbit = P.Orbit(65.0, 47.0, 8.0 if delta != 2.0 else 10.0, P.kt(280.0), math.radians(200), delta)
    for t in (10.0, 75.0, 120.0, 170.0, 400.0):
        h = 1e-4
        (x0, y0), (x1, y1) = orbit.position(t - h), orbit.position(t + h)
        speed = math.hypot(x1 - x0, y1 - y0) / (2 * h)
        assert speed == pytest.approx(orbit.v, rel=1e-6)
        x, y = orbit.position(t)
        assert math.hypot(x - orbit.cx, y - orbit.cy) == pytest.approx(orbit.radius(t), abs=1e-12)
    assert orbit.radius(1000.0) == pytest.approx(orbit.r0 + delta)


def test_tanker_moves_clockwise():
    orbit = P.Orbit(0.0, 0.0, 8.0, P.kt(280.0), 0.0)
    x, y = orbit.position(10.0)
    assert x > 0 and y > 0          # starting due north, clockwise goes east


E = P.NUMERIC_ALLOWANCE


def test_closest_approach_of_two_straight_paths_matches_formula():
    p0, v = (10.0, -3.0), (-0.12, 0.05)
    q0, u = (0.0, 0.0), (0.02, 0.01)
    dp, dv = (p0[0] - q0[0], p0[1] - q0[1]), (v[0] - u[0], v[1] - u[1])
    t_star = -(dp[0] * dv[0] + dp[1] * dv[1]) / (dv[0] ** 2 + dv[1] ** 2)
    exact = math.hypot(dp[0] + dv[0] * t_star, dp[1] + dv[1] * t_star)

    def dist(t):
        return math.hypot(dp[0] + dv[0] * t, dp[1] + dv[1] * t)

    a = P.closest_approach(dist, 0.0, 200.0, rate_bound=math.hypot(*dv) + 1e-9, gap=1e-9)
    assert a.status == "resolved"
    assert a.lower <= exact <= a.upper and a.upper - a.lower <= 1e-9


def test_closest_approach_finds_a_narrow_dip_a_coarse_grid_misses():
    rate = 0.2

    def dist(t):
        return min(5.0, 0.1 + rate * abs(t - 0.73337))   # a sharp dip, slope equal to the bound

    grid = min(dist(k / 10) for k in range(11))
    a = P.closest_approach(dist, 0.0, 1.0, rate_bound=rate, gap=1e-7)
    assert grid > 0.1 + 1e-3
    assert a.lower <= 0.1 <= a.upper <= 0.1 + 1e-7


def test_closest_approach_encloses_brute_force_on_random_cases():
    """A grid's smallest value is an upper bound on the true minimum, so the enclosure must overlap it:
    lower <= grid + e, and upper <= grid + gap + e once resolved (N10.3)."""
    rng = random.Random(4400)
    gap = 1e-6
    for _ in range(20):
        s = state(rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(420, 480),
                  x=rng.uniform(-3, 3), y=rng.uniform(-3, 3))
        wind = (P.kt(rng.uniform(-40, 40)), P.kt(rng.uniform(-40, 40)))
        orbit = P.Orbit(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(6, 10), P.kt(rng.uniform(260, 300)),
                        rng.uniform(0, 2 * math.pi))

        def dist(t):
            r = P.receiver_at(s, wind, t)
            x, y = orbit.position(t)
            return math.hypot(r.x - x, r.y - y)

        a = P.closest_approach(dist, 0.0, 30.0, gap=gap)
        brute = min(dist(k * 0.001) for k in range(30001))
        assert a.status == "resolved"
        assert a.lower <= brute + E
        assert a.upper <= brute + gap + E


def test_closest_approach_across_piece_boundaries():
    """A turn ending and an orbit change starting inside the interval; the bound needs no splitting."""
    s = state(0.0, 10.0, 450.0, x=-1.0, y=2.0, t=59.0)        # a 10-degree turn, about 7 s
    end = P.turn_end_time(s)
    orbit = P.Orbit(0.0, 0.0, 2.2, P.kt(280.0), 0.3, delta=0.4)
    wind = (P.kt(20.0), P.kt(-10.0))

    def dist(t):
        r = P.receiver_at(s, wind, t)
        x, y = orbit.position(t)
        return math.hypot(r.x - x, r.y - y)

    t0, t1 = 59.0, 75.0
    assert t0 < end < t1 and t0 < P.ORBIT_CHANGE_START < t1
    a = P.closest_approach(dist, t0, t1, gap=1e-6)
    brute = min(dist(t0 + k * 0.001) for k in range(16001))
    assert a.status == "resolved" and a.lower <= brute + E and a.upper <= brute + 1e-6 + E


def test_closest_approach_minimum_at_an_endpoint():
    a = P.closest_approach(lambda t: 0.3 + 0.1 * t, 0.0, 1.0, gap=1e-6)
    assert a.t_at_upper == 0.0 and a.lower <= 0.3 <= a.upper
    assert P.join_decision(lambda t: 0.3 - 0.1 * t, 0.0, 1.0) == "join"     # reached only at t = 1


def test_join_decision_classes():
    tol, d = 0.25, 1e-5
    assert P.join_decision(lambda t: 0.2 + 0.1 * abs(t - 0.5), 0.0, 1.0) == "join"
    assert P.join_decision(lambda t: 0.3 + 0.1 * abs(t - 0.5), 0.0, 1.0) == "no join"
    assert P.join_decision(lambda t: 0.250001 + 0.1 * abs(t - 0.5), 0.0, 1.0) == "flagged"
    assert P.join_decision(lambda t: tol - 1.1 * d + 0.1 * abs(t - 0.5), 0.0, 1.0) == "join"
    assert P.join_decision(lambda t: tol + 1.6 * d + 0.1 * abs(t - 0.5), 0.0, 1.0) == "no join"
    for offset in (-0.9, -0.5, 0.0, 0.5, 0.9):       # every minimum within the margin is flagged
        assert P.join_decision(lambda t: tol + offset * d + 0.1 * abs(t - 0.5), 0.0, 1.0) == "flagged"


def test_join_decision_codex_counterexample_is_not_a_join():
    """N10.1: a true distance of 0.250000000025 evaluated as 0.249999999975 must not be called a join."""
    assert P.join_decision(lambda t: 0.249999999975, 0.0, 1.0) == "flagged"


def test_flat_distance_at_the_tolerance_work_and_cap():
    """The worst case for work: a distance flat at the tolerance keeps every interval alive."""
    a = P.closest_approach(lambda t: 0.25, 0.0, 1.0, gap=5e-6, stop_below=0.25 - 1e-5, stop_above=0.25 + 1e-5)
    assert a.status == "resolved" and a.evaluations < P.MAX_EVALUATIONS
    assert P.join_decision(lambda t: 0.25, 0.0, 1.0) == "flagged"
    assert P.join_decision(lambda t: 0.25, 0.0, 1.0, max_evaluations=1000) == "unresolved"


def test_gap_must_exceed_twice_the_allowance():
    with pytest.raises(ValueError):
        P.closest_approach(lambda t: 1.0, 0.0, 1.0, gap=2 * E)


def test_orbit_angle_numerical_qualification():
    """N10.2: an empirical check of the allowance e, not a proof. The integrated angle during an orbit
    change is compared with independent Gauss-Legendre quadrature on 16 sub-intervals of 40 nodes."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    rng = random.Random(4410)
    worst = 0.0
    for _ in range(40):
        r0 = rng.uniform(6.0, 10.0)
        orbit = P.Orbit(0.0, 0.0, r0, P.kt(rng.uniform(250, 300)), rng.uniform(0, 2 * math.pi),
                        delta=rng.choice((-1, 1)) * 0.2 * r0)
        t = rng.uniform(60.0, 180.0)
        edges = np.linspace(60.0, t, 17)
        total = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            mid, half = (a + b) / 2, (b - a) / 2
            total += half * sum(w * orbit._angle_rate(mid + half * x) for x, w in zip(nodes, weights))
        reference = orbit.theta0 + orbit.v * 60.0 / r0 + total
        worst = max(worst, abs(orbit.angle(t) - reference) * max(r0, r0 + orbit.delta))
    assert worst < E / 10        # position error from the angle, in nautical miles


# TURN 14: derivative at form boundaries and during the changing-radius orbit.
def test_orbit_velocity_is_tangent_radial_derivative_and_total_speed():
    for delta in (-1.6, 0, 1.6):
        o = P.Orbit(65, 47, 8, P.kt(280), .4, delta)
        for t in (0, 60, 90, 120, 180, 250):
            vx, vy = o.velocity(t)
            assert math.hypot(vx, vy) == pytest.approx(o.v, abs=1e-14)
            a = o.angle(t)
            assert vx*math.sin(a) + vy*math.cos(a) == pytest.approx(o.radius_rate(t), abs=1e-14)
            # The second derivative jumps at onset/end, so a centred first-difference
            # has O(dt) error there, even though velocity itself is continuous.
            dt = 1e-4
            p0, p1 = o.position(t-dt), o.position(t+dt)
            assert (vx, vy) == pytest.approx(((p1[0]-p0[0])/(2*dt), (p1[1]-p0[1])/(2*dt)), abs=5e-8)


def test_closest_approach_keeps_all_legacy_positional_options():
    distance = lambda t: 2.0 + t
    args = (distance, 0.0, 1.0, 1.0, 1e-6, P.NUMERIC_ALLOWANCE, None, None, 100)
    positional = P.closest_approach(*args)
    keywords = P.closest_approach(distance, 0., 1., rate_bound=1., gap=1e-6,
                                 numeric_allowance=P.NUMERIC_ALLOWANCE, max_evaluations=100)
    assert positional == keywords and positional.lower <= 2.0 <= positional.upper
