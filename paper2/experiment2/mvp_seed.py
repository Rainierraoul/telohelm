"""The starting library and stand-ins for the minimum viable product (MVP step 1, TURN 112, paper2/plan/MVP-PLAN.md).

The four entries are the hand-written procedures and checks the episode-loop tests use: lead pursuit on the briefed
orbit, a report at the briefed range, and a freshness check for each. The prediction tool and the model here are
stand-ins so the runner can fly whole episodes before the real ones exist (MVP steps 2 and 4). Nothing here is a
result.
"""
from __future__ import annotations

import json
import math
from typing import Any

from telohelm.runtime.library import Library
from telohelm.runtime.models import ModelResponse

AIM = '''import math
def decide(ctx):
    # Lead pursuit on the briefed orbit from the newest report, with wind allowed for.
    a = math.atan2(ctx.report_x - ctx.tanker_centre_x, ctx.report_y - ctx.tanker_centre_y)
    tau = 0.0
    for _ in range(25):
        b = a + ctx.tanker_speed_nm_s * (ctx.t - ctx.report_measured_at + tau) / ctx.tanker_radius_nm
        px = ctx.tanker_centre_x + ctx.tanker_radius_nm * math.sin(b) - ctx.wind_x_nm_s * tau
        py = ctx.tanker_centre_y + ctx.tanker_radius_nm * math.cos(b) - ctx.wind_y_nm_s * tau
        tau = math.hypot(px - ctx.x, py - ctx.y) / ctx.tas_nm_s
    h = math.degrees(math.atan2(px - ctx.x, py - ctx.y)) % 360
    return {"heading_deg": 0.0 if h == 360.0 else h}     # a tiny negative angle wraps to exactly 360.0 (BACKLOG B38)
'''
HEADING_CHECK = "def check(ctx, out):\n    return 'accept' if ctx.report_age_s <= 10 and 0 <= out.heading_deg < 360 else 'reject'\n"
REPORT = '''def decide(ctx):
    rate = ctx.observed_range_rate_nm_s
    if rate is not None and rate < 0 and abs(ctx.observed_range_nm - ctx.report_at_nm) < 0.15:
        return {"report": "Wedgetail, aircraft 2, %g NM from %s" % (ctx.report_at_nm, ctx.goal_tanker)}
    return {}
'''
REPORT_CHECK = "def check(ctx, out):\n    return 'accept' if ctx.report_age_s <= 2 else 'reject'\n"


def stock(library: Library) -> list[str]:
    """Add and adopt the starting procedures and checks; returns their identifiers."""
    shas = []
    for kind, decision, source in (("procedure", "heading", AIM), ("check", "heading", HEADING_CHECK),
                                   ("procedure", "report", REPORT), ("check", "report", REPORT_CHECK)):
        sha = library.add({"kind": kind, "decision": decision, "conditions": [], "source": source,
                           "origin": {"calls": []}})
        library.set_status(sha, "adopted", "MVP starting library")
        shas.append(sha)
    return shas


def stand_in_tool(args: dict) -> dict:
    """Stand-in prediction tool (the real one is MVP step 4): a fixed, harmless answer of the right shape."""
    return {"positions_nm": [[0.0, 0.0]] * len(args.get("headings_deg", [])), "closest_approach_nm": 1.0,
            "closest_at_s": 1.0}


def messages_for(decision: str, ctx: dict, reason: str) -> list:
    """The prompt the model layer sends: the decision, why it climbed, and the current context as JSON."""
    return [("system", "You fly a receiver aircraft to join a tanker. Reply with JSON only."),
            ("user", json.dumps({"decision": decision, "reason": reason, "context": ctx}, default=str, sort_keys=True))]


def stand_in_model(request: Any) -> ModelResponse:
    """Stand-in model (a real one is MVP step 2): for a heading, point at the briefed tanker orbit's centre."""
    try:
        body = json.loads(request.messages[-1][1])
        ctx, decision = body["context"], body["decision"]
        if decision == "heading":
            h = math.degrees(math.atan2(ctx["tanker_centre_x"] - ctx["x"], ctx["tanker_centre_y"] - ctx["y"])) % 360
            text = json.dumps({"heading_deg": round(h, 3)})
        else:
            text = json.dumps({})
    except (KeyError, TypeError, ValueError, IndexError):
        text = json.dumps({})
    return ModelResponse("answered", text, 200, 12)


def parse(decision: str, text: str) -> dict | None:
    part = json.loads(text)
    return part or None
