"""The real prediction tool (MVP step 4). Invented arguments only; it reads nothing else."""
from __future__ import annotations

import math
import time

import pytest

import telohelm.runtime.broker as B
import telohelm.runtime.prediction as T
import telohelm.taskpacks.tanker.world.physics as P


def args(headings, x=0.0, y=0.0, heading=90.0, tas=0.1, wind=(0.0, 0.0), cx=5.0, cy=1.0, radius=1.0, speed=0.0,
         angle=0.0):
    return {"goal_version": 1, "start": {"x_nm": x, "y_nm": y, "heading_deg": heading, "tas_nm_s": tas},
            "headings_deg": list(headings), "wind_nm_s": list(wind),
            "orbit": {"cx_nm": cx, "cy_nm": cy, "radius_nm": radius, "speed_nm_s": speed, "angle_rad": angle}}


def test_straight_flight_with_wind_is_exact():
    r = T.predict(args([90.0] * 10, wind=(0.0, 0.01)))
    for i, (x, y) in enumerate(r["positions_nm"], start=1):
        assert x == pytest.approx(0.1 * i, abs=1e-12) and y == pytest.approx(0.01 * i, abs=1e-12)


def test_turns_follow_the_worlds_own_turn_model():
    """Holding one command, the per-second steps must land where the world's receiver lands in one go."""
    tas = P.kt(450)
    r = T.predict(args([180.0] * 60, tas=tas, wind=(0.002, -0.001)))
    s0 = P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(180), tas, P.turn_rate(450))
    for i, (x, y) in enumerate(r["positions_nm"], start=1):
        w = P.receiver_at(s0, (0.002, -0.001), float(i))
        assert x == pytest.approx(w.x, abs=1e-9) and y == pytest.approx(w.y, abs=1e-9)


def test_closest_approach_to_a_stationary_tanker():
    """Flying east along y = 0 past a tanker held at (5, 2): the closest approach is 2 NM, at 50 s."""
    r = T.predict(args([90.0] * 100))
    assert r["closest_approach_nm"] == pytest.approx(2.0, abs=T.GAP_NM * 2)
    assert abs(r["closest_at_s"] - 50.0) < 0.5


def test_closest_approach_to_a_moving_tanker_matches_a_fine_scan():
    a = args([30.0] * 300 + [120.0] * 300, tas=P.kt(450), wind=(0.003, 0.004), cx=40.0, cy=20.0, radius=8.0,
             speed=P.kt(280), angle=1.0)
    r = T.predict(a)
    tanker = P.Orbit(40.0, 20.0, 8.0, P.kt(280), 1.0)
    states = [P.ReceiverState(0.0, 0.0, 0.0, math.radians(90), math.radians(30), P.kt(450), P.turn_rate(450))]
    best = math.inf
    for k in range(6001):                                  # every 0.1 s
        t = k / 10
        i = min(int(t), 599)
        while len(states) <= i:
            e = P.receiver_at(states[-1], (0.003, 0.004), float(len(states)))
            states.append(P.ReceiverState(e.t, e.x, e.y, e.heading, math.radians(30 if len(states) < 300 else 120),
                                          e.tas, e.omega))
        p = P.receiver_at(states[i], (0.003, 0.004), t)
        tx, ty = tanker.position(t)
        best = min(best, math.hypot(p.x - tx, p.y - ty))
    # The tool is within GAP_NM of the true minimum, which is at most the scan's best; the scan's best is at most
    # half a step's change in distance (0.05 s at the bound on the rate) above the true minimum.
    rate = P.kt(450) + 0.005 + P.kt(280)
    assert best - rate * 0.05 <= r["closest_approach_nm"] <= best + T.GAP_NM


def test_the_answer_has_the_brokers_form_and_depends_only_on_the_arguments():
    a = args([90.0, 100.0, 110.0] * 20)
    r = T.predict(a)
    assert B._result_ok(r, 60) and T.predict(a) == r


def test_a_stopped_aircraft_drifts_with_the_wind():
    r = T.predict(args([0.0, 180.0, 270.0], tas=0.0, wind=(0.01, 0.02)))
    assert r["positions_nm"] == [[pytest.approx(0.01 * i), pytest.approx(0.02 * i)] for i in (1, 2, 3)]


def test_overflowing_estimates_raise_so_the_broker_reports_tool_failed():
    with pytest.raises(ValueError):
        T.predict(args([90.0] * 5, tas=1e308, wind=(1e308, 0.0)))


def test_the_work_is_bounded_on_a_full_horizon_of_reversals():
    a = args([0.0, 180.0] * 1800, tas=P.kt(480), wind=(P.kt(60), 0.0), cx=60.0, cy=60.0, radius=6.0,
             speed=P.kt(300), angle=2.0)
    started = time.perf_counter()
    r = T.predict(a)
    assert time.perf_counter() - started < 5.0 and len(r["positions_nm"]) == 3600


def test_a_contained_procedure_gets_the_real_tools_answer_through_the_broker(tmp_path):
    import json
    import telohelm.runtime.sandbox as S
    from telohelm.runtime.ledger import Ledger
    from telohelm.runtime.trusted_log import TrustedLog
    if not S.available():
        pytest.skip("no tested sandbox route on this platform")
    a = args([90.0, 120.0, 150.0], heading=45.0, tas=P.kt(450), wind=(0.001, 0.0), speed=P.kt(280))
    a["goal_version"] = 0
    br = B.Broker("invented-episode", TrustedLog(tmp_path / "l", tmp_path / "h", "invented-run"), Ledger(), T.predict,
                  condition="telohelm", stream="invented-stream")
    code = f"import json\ndef decide(ctx):\n    return predict(json.loads({json.dumps(a)!r}))\n"
    r = S.run_with_broker(code, "decide", [{"t": 0}], br, "heading:decide", wall_seconds=5.0, context_args=True)
    assert r.status == "ok" and r.result["ok"] is True
    assert r.result["result"] == json.loads(json.dumps(T.predict(a)))
