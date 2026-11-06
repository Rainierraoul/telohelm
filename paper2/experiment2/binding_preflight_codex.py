"""Fixed invented prefix-size characterization, never a stream or label producer.

Runs the unchanged world/binding APIs on two declared geometries. No random
generator, reference pilot, provider, store admission or parameter search.
Timing observations are local characterizations, not a diagnostic cost bound.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time

from . import original_binding as B; from . import original_replay as R
from telohelm.runtime.context import ContextBuilder
import telohelm.taskpacks.tanker.world.episode as E; import telohelm.taskpacks.tanker.world.physics as P

SECONDS = (0, 60, 120, 180, 300, 600, 1200, 1800, 2400, 3000, 3599)
CASES = (("one_constant", 1, False), ("two_changing", 2, True))


def scenario(name, tankers, changing):
    """Far-separated, hand-constructed geometry; no candidate-dependent choice."""
    return E.Scenario(
        "invented-prefix-" + name,
        P.ReceiverState(0., 0., 0., 0., 0., .1, .02), (0., 0.),
        tuple(E.Tanker("SHELL " + str(i + 1),
                      P.Orbit(200. + 30 * i, 100., 8., .075, .3 * i,
                              delta=1. if changing else 0.)) for i in range(tankers)),
        ((0, E.Goal(1, "SHELL 1", 5., None)),))


def bind_prefix(ep, briefing, bundles, context):
    source = {"role": "none", "payload": "fixed invented hold; no provider"}
    issue = B.Issue(None, "none", "heading", 1, ep.t, len(ep.record.actions), ep.t,
                    "decisions", R.digest(context), R.digest(source))
    identity = R.Identity(ep.s.episode_id, "invented-prefix-unit", 4420,
                          "diagnostic", "heading", "action", ep.t,
                          len(ep.record.actions), 1, 0, 0, None)
    capture = R.capture(ep, identity, context=context, input_bundle=bundles[-1],
                        issue=asdict(issue), actor_pending={
                            "calls": [], "public_history_sha256": R.digest({
                                "briefing": briefing, "bundles": bundles})})
    binding = B.bind(capture, issue, briefing=briefing, bundles=bundles,
                     issued_context=context, source_body=source,
                     expected_capture_sha256=capture.sha256)
    return capture, binding


def measure(name, tankers, changing, seconds=SECONDS):
    if (not seconds or any(type(t) is not int or t < 0 or t >= E.HORIZON for t in seconds)
            or tuple(sorted(set(seconds))) != tuple(seconds)):
        raise ValueError("fixed distinct ascending preterminal snapshot seconds required")
    cpu, wall = time.process_time(), time.perf_counter()
    ep = E.Episode(scenario(name, tankers, changing))
    start = ep.reset()
    briefing, bundles = start["briefing"], [start["bundle"]]
    builder = ContextBuilder(briefing)
    context = builder.update(bundles[-1])
    rows = []
    for second in range(seconds[-1] + 1):
        if second in seconds:
            c0, w0 = time.process_time(), time.perf_counter()
            capture, binding = bind_prefix(ep, briefing, bundles, context)
            # Exact original implementation counts canonical [candidate, continuation].
            plan = [{}, []]
            n = len(binding.body) + len(R.encode(plan))
            row = {"second": second, "bundles": len(bundles),
                   "capture_bytes": len(capture.body), "binding_bytes": len(binding.body),
                   "public_prefix_bytes": len(R.encode(bundles)),
                   "binding_plus_empty_plan_bytes": n,
                   "fits_1MiB": n <= R.Limits().input_bytes,
                   "binding_sha256": binding.sha256,
                   "capture_binding_cpu_s": time.process_time() - c0,
                   "capture_binding_wall_s": time.perf_counter() - w0}
            if not row["fits_1MiB"]:
                try:
                    B.replay_bound(binding, {}, [], expected_binding_sha256=binding.sha256)
                except ValueError as exc:
                    row["admission"] = str(exc)
                else:
                    raise AssertionError("over-budget binding admitted")
            rows.append(row)
        if second < seconds[-1]:
            bundle = ep.step({})
            if bundle is None:
                raise AssertionError("fixed prefix geometry ended before the declared snapshots")
            bundles.append(bundle)
            context = builder.update(bundle)
    # Complete this same invented episode; it is never filtered by its outcome.
    while not ep.done:
        ep.step({})
    return {"case": name, "tankers": tankers, "changing": changing,
            "scenario": asdict(ep.s), "snapshots": rows,
            "ending_second": ep.record.ended_at, "end_reason": ep.record.end_reason,
            "total_cpu_s": time.process_time() - cpu,
            "total_wall_s": time.perf_counter() - wall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"rule": "B8.prefix.size.characterization.v1", "admissible": False,
              "sources": B._sources(), "input_limit_bytes": R.Limits().input_bytes,
              "scope": "fixed invented CPU size controls only; no acquisition estimate",
              "cases": [measure(*case) for case in CASES]}
    data = R.encode(result)
    args.output.write_bytes(data + b"\n")
    print(json.dumps({"output": str(args.output), "body_sha256": hashlib.sha256(data).hexdigest(),
                      "cases": [{"case": c["case"], "cpu_s": c["total_cpu_s"],
                                 "sizes": [[r["second"], r["binding_plus_empty_plan_bytes"], r["fits_1MiB"]]
                                           for r in c["snapshots"]]} for c in result["cases"]]}))


if __name__ == "__main__":
    main()
