"""The tanker task's prediction tool (MVP step 4; architecture section 2, paper2/notes/BROKER-CONTRACT.md).

Moved unchanged from runtime/prediction.py, which re-exports it, on 23 October 2026; served through the broker's
tool table (packs/tanker_tools.py).

A procedure reaches it only through the broker, as predict(args), and the broker has already checked the arguments'
form. The tool flies a receiver forward from estimates the caller supplies, one commanded heading per second, with the
world's own turn model (physics.receiver_at: a level turn at 30 degrees of bank, the shorter way, then straight), in a
steady wind. It flies the tanker round a clockwise circular orbit from the caller's estimate of the centre, radius,
ground speed and current angle, with no orbit change. It returns the receiver's position at the end of each second
and the closest approach between the two over the whole span, measured from the start of the request.

It reads nothing but its arguments: not the scenario, the truth or a clock. So a prediction is only as good as the
caller's estimates, and a tanker that changes its orbit, or a wind that is not the one supplied, is not foreseen.

Its work is bounded, as the broker needs, because a trusted tool is not stopped at the call's deadline. A request has
at most 3,600 seconds (the broker refuses longer ones), each second is one exact step, and the closest-approach
search stops after MAX_EVALUATIONS distance evaluations. The reported closest approach is always a distance actually
reached, at the reported time (physics.closest_approach's upper bound, less its numerical allowance). When the search
settles first it is within GAP_NM of the true minimum; when the cap is reached first it may be further above it.
Estimates so large that the arithmetic overflows raise ValueError, which the broker reports as tool_failed.
"""
from __future__ import annotations

import math
from typing import Any

from . import dynamics as P       # the task's public dynamics, no world state

GAP_NM = 1e-4                    # the closest-approach search stops when it is this sure of the minimum
MAX_EVALUATIONS = 20_000         # the bound on its work; a full 3,600-second request took 0.04 s at the cap on the Mac
STOPPED_TURN_RATE = 1.0          # rad/s, used only when the airspeed is 0, where the turn rate cannot affect position


def predict(args: dict[str, Any]) -> dict[str, Any]:
    start, headings, (wx, wy), orbit = args["start"], args["headings_deg"], args["wind_nm_s"], args["orbit"]
    tas = float(start["tas_nm_s"])
    omega = P.turn_rate(tas / P.NM_PER_S_PER_KNOT) if tas > 0 else STOPPED_TURN_RATE
    wind = (float(wx), float(wy))
    commanded = [math.radians(float(h)) for h in headings]
    seconds = len(commanded)

    # The state at the start of each second, holding that second's commanded heading.
    states = [P.ReceiverState(0.0, float(start["x_nm"]), float(start["y_nm"]),
                              P.wrap(math.radians(float(start["heading_deg"]))), commanded[0], tas, omega)]
    positions = []
    for i in range(seconds):
        end = P.receiver_at(states[i], wind, i + 1.0)
        positions.append([end.x, end.y])
        if i + 1 < seconds:
            states.append(P.ReceiverState(end.t, end.x, end.y, end.heading, commanded[i + 1], tas, omega))

    tanker = P.CircularOrbit(float(orbit["cx_nm"]), float(orbit["cy_nm"]), float(orbit["radius_nm"]),
                             float(orbit["speed_nm_s"]), float(orbit["angle_rad"]))

    def distance(t: float) -> float:
        r = P.receiver_at(states[min(int(t), seconds - 1)], wind, t)
        tx, ty = tanker.position(t)
        return math.hypot(r.x - tx, r.y - ty)

    # How fast the distance can change: the receiver's airspeed and the wind, plus the tanker's ground speed.
    rate = tas + math.hypot(*wind) + float(orbit["speed_nm_s"])
    found = P.closest_approach(distance, 0.0, float(seconds), rate_bound=rate, gap=GAP_NM,
                               max_evaluations=MAX_EVALUATIONS)
    closest = found.upper - P.NUMERIC_ALLOWANCE
    values = [v for p in positions for v in p] + [closest, found.t_at_upper]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("the estimates are too large to predict from")
    return {"positions_nm": positions, "closest_approach_nm": max(0.0, closest), "closest_at_s": found.t_at_upper}
