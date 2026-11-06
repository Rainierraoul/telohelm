"""Row 8's checks on the exact reference: two independent integrators against the world's closed forms."""
from __future__ import annotations

import math
import random

import pytest

import telohelm.taskpacks.tanker.world.crosscheck as X
import telohelm.taskpacks.tanker.world.physics as P

ACCURACY_NM = 1e-6        # row 8


def flight(seed: int, seconds: int = 300, change: bool = True):
    rng = random.Random(seed)
    tas = P.kt(rng.uniform(420, 480))
    rcv = P.ReceiverState(0.0, rng.uniform(-5, 5), rng.uniform(-5, 5), math.radians(90), math.radians(90),
                          tas, P.turn_rate(tas * 3600))
    wind_from = math.radians(rng.choice((0, 90, 180, 270)))
    speed = P.kt(rng.choice((0, 20, 40, 60)))
    wind = (-speed * math.sin(wind_from), -speed * math.cos(wind_from))
    r0 = rng.uniform(6, 10)
    orbit = P.Orbit(rng.uniform(50, 80), rng.uniform(35, 60), r0, P.kt(rng.uniform(260, 300)),
                    rng.uniform(0, 2 * math.pi), delta=rng.choice((-1, 1)) * 0.2 * r0 if change else 0.0)
    commands = {t: math.radians(rng.uniform(0, 360)) for t in sorted(rng.sample(range(seconds), 12))}
    commands[seconds // 2] = (rcv.heading + math.pi) % P.TWO_PI          # include an exact reversal
    return rcv, wind, orbit, commands, seconds


@pytest.mark.parametrize("seed", [8001, 8002, 8003])
def test_general_integrator_agrees_with_the_exact_method(seed):
    rcv, wind, orbit, commands, seconds = flight(seed)
    exact = X.fly_exact(rcv, wind, orbit, commands, seconds)
    general = X.fly_general(rcv, wind, orbit, commands, seconds)
    receiver, tanker = X.max_difference(exact, general)
    assert receiver < ACCURACY_NM and tanker < ACCURACY_NM, (receiver, tanker)


@pytest.mark.parametrize("seed", [8001, 8002])
def test_fixed_steps_of_0_1_and_0_05_s_agree_with_the_exact_method_and_converge(seed):
    rcv, wind, orbit, commands, seconds = flight(seed)
    exact = X.fly_exact(rcv, wind, orbit, commands, seconds)
    coarse = X.max_difference(exact, X.fly_fixed_step(rcv, wind, orbit, commands, seconds, 0.1))
    fine = X.max_difference(exact, X.fly_fixed_step(rcv, wind, orbit, commands, seconds, 0.05))
    assert max(coarse) < ACCURACY_NM and max(fine) < ACCURACY_NM, (coarse, fine)
    # Fourth-order method: halving the step should cut the tanker's integration error by about 16,
    # unless both are already at the level of rounding.
    if coarse[1] > 1e-11:
        assert fine[1] < coarse[1] / 8, (coarse, fine)


def test_a_single_straight_leg_by_hand():
    """Hand calculation: due east at 450 kt in a 40-kt wind from the north for 100 s."""
    tas = P.kt(450.0)
    rcv = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(90), tas, P.turn_rate(450.0))
    wind = (0.0, -P.kt(40.0))
    orbit = P.Orbit(60.0, 40.0, 8.0, P.kt(280.0), 0.0)
    for fly in (X.fly_exact, X.fly_general, lambda *a: X.fly_fixed_step(*a, 0.1)):
        x, y, _, _ = fly(rcv, wind, orbit, {}, 100)[-1]
        assert x == pytest.approx(450 * 100 / 3600, abs=1e-9) and y == pytest.approx(-40 * 100 / 3600, abs=1e-9)
