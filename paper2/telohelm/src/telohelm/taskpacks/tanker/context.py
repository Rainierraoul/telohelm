"""What Telohelm sees each second in the tanker task: the context a procedure, a check or a model receives (slice 2(c)
of the task pack; backlog note B7, 'The context's fields').

Only observed data and fixed formulas on them are used: the briefing and the bundles delivered so far. Moved from
runtime/context.py on 23 October 2026 with its code unchanged; that module now re-exports it for its callers. The
core receives the builder from the composition root (runtime/episode_loop.py, context_builder), so a new task
brings its own.
"""
from __future__ import annotations

import math
from typing import Any

HISTORY = 60


class ContextBuilder:
    def __init__(self, briefing: dict[str, Any]):
        self.tankers = {k["callsign"]: k for k in briefing["tankers"]}
        self.goal = briefing["goal"]
        self.reports: dict[str, list[dict[str, Any]]] = {c: [] for c in self.tankers}
        self.previous_range: dict[int, float] = {}         # observed range at the previous second, by goal version

    def update(self, bundle: dict[str, Any]) -> dict[str, Any]:
        if bundle["goal"] is not None:
            self.goal = bundle["goal"]
        for r in bundle["reports"]:
            held = self.reports[r["callsign"]]
            if not held or r["measured_at"] > held[-1]["measured_at"]:
                held.append({"x": r["x"], "y": r["y"], "measured_at": r["measured_at"]})
                del held[:-HISTORY]
        own, t = bundle["own"], bundle["t"]
        h, trk = math.radians(own["heading_deg"]), math.radians(own["track_deg"])
        wind = (own["ground_speed_nm_s"] * math.sin(trk) - own["tas_nm_s"] * math.sin(h),
                own["ground_speed_nm_s"] * math.cos(trk) - own["tas_nm_s"] * math.cos(h))
        g = self.goal
        tanker = self.tankers[g["tanker"]]
        latest = self.reports[g["tanker"]][-1] if self.reports[g["tanker"]] else None
        ctx: dict[str, Any] = {
            "t": t, "x": own["x"], "y": own["y"], "heading_deg": own["heading_deg"],
            "commanded_heading_deg": own["commanded_heading_deg"], "track_deg": own["track_deg"],
            "tas_nm_s": own["tas_nm_s"], "ground_speed_nm_s": own["ground_speed_nm_s"],
            "wind_x_nm_s": wind[0], "wind_y_nm_s": wind[1],
            "goal_version": g["version"], "goal_tanker": g["tanker"], "report_at_nm": g["report_at_nm"],
            "deadline": g["deadline"],
            "tanker_centre_x": tanker["centre"][0], "tanker_centre_y": tanker["centre"][1],
            "tanker_radius_nm": tanker["radius"], "tanker_speed_nm_s": tanker["speed_nm_s"],
            "report_x": None, "report_y": None, "report_measured_at": None, "report_age_s": None,
            "observed_range_nm": None, "observed_range_rate_nm_s": None,
            "reports": list(self.reports[g["tanker"]]),
        }
        if latest is not None:
            rng = math.hypot(latest["x"] - own["x"], latest["y"] - own["y"])
            prev = self.previous_range.get(g["version"])
            ctx.update(report_x=latest["x"], report_y=latest["y"], report_measured_at=latest["measured_at"],
                       report_age_s=t - latest["measured_at"], observed_range_nm=rng,
                       observed_range_rate_nm_s=None if prev is None else rng - prev)
            self.previous_range = {g["version"]: rng}
        return ctx
