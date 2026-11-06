"""The tanker task's prediction tool, as an entry of the broker's tool table (packs/tools.py).

The request form, the heading-list limit and the reply check moved unchanged from runtime/broker.py; the function is
packs/tanker_prediction.py, moved unchanged from runtime/prediction.py. This module imports nothing from the world,
the monitor, the reference pilot or the prior.
"""
from __future__ import annotations

import math
from typing import Any

from .prediction import predict
from ...contracts.tools import Tool

MAX_PREDICTION_SECONDS = 3600          # headings in one prediction request, one per simulated second


def _is_number(v: Any) -> bool:
    """A finite JSON number; an integer too large for a double is refused, not raised (the broker's rule)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def headings(value: Any) -> bool:
    return isinstance(value, list) and 1 <= len(value) <= MAX_PREDICTION_SECONDS and all(_is_number(v) for v in value)


START = {"x_nm": "number", "y_nm": "number", "heading_deg": "number", "tas_nm_s": "nonneg"}
ORBIT = {"cx_nm": "number", "cy_nm": "number", "radius_nm": "positive", "speed_nm_s": "nonneg", "angle_rad": "number"}
PREDICT = {"goal_version": "int", "start": START, "headings_deg": headings, "wind_nm_s": "pair", "orbit": ORBIT}


def result_ok(result: Any, seconds: int) -> bool:
    """One position per requested second, a non-negative closest approach, and its time within the predicted span.
    Anything else is tool_failed (Codex's BR51.5)."""
    def pair(p):
        return isinstance(p, list) and len(p) == 2 and all(_is_number(v) for v in p)
    return (isinstance(result, dict) and set(result) == {"positions_nm", "closest_approach_nm", "closest_at_s"}
            and isinstance(result["positions_nm"], list) and len(result["positions_nm"]) == seconds
            and all(pair(p) for p in result["positions_nm"])
            and _is_number(result["closest_approach_nm"]) and result["closest_approach_nm"] >= 0
            and _is_number(result["closest_at_s"]) and result["closest_at_s"] >= 0 and result["closest_at_s"] <= seconds)


def predict_tool(run=predict) -> Tool:
    """The prediction tool, with another function in its place for stand-ins and tests."""
    return Tool(run=run, request=PREDICT, reply_ok=lambda result, args: result_ok(result, len(args["headings_deg"])),
                charge="prediction_tool")


TANKER_TOOLS = {"predict": predict_tool()}
