"""Development identities stay apart from reserved scored ones (Codex's TURN243 review, item 1; Claude's TURN244 repair).

The M1 timing rehearsal and the new development tests once used 4441 to 4444, scored stream 4302's replay sources.
Those are retired and replaced; this control keeps every development identity named here out of every reserved block.
It reads constants only and runs no generator."""
import ast
from pathlib import Path

import paper2.experiment2.outcomecases as O
import telohelm.qualification.store as H

HERE = Path(__file__).resolve().parent
M1 = HERE.parents[1] / "development/m1-timing-rehearsal/positions.py"
TEST_FILES = ("test_streams_layout_v2.py", "test_run_stream_m1.py", "test_study_runner.py", "test_study_host.py",
              "../launch/rehearse_cpu.py")


def _reserved():
    part = O.partition()
    sources = {x for roles in part.values() for block in roles.values() for x in block}
    return sources | set(O.EXPOSED_SOURCES) | set(O.C0_DEVELOPMENT) | set(O.RUN_STREAMS) | set(H.SCORED_STREAMS)


def _m1_streams():
    tree = ast.parse(M1.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "STREAMS" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError("positions.py names no STREAMS")


def _integers(path):
    return {n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, int) and 4400 <= n.value <= 4499}


def test_retired_sources_are_replaced_and_never_used():
    part = O.partition()
    assert part[4302]["replay"] == (4440, 4461, 4462, 4463, 4464)
    used = {x for roles in part.values() for block in roles.values() for x in block}
    assert not used & set(O.EXPOSED_SOURCES)
    assert set(O.EXPOSED_SOURCES.values()) <= used


def test_m1_and_the_new_tests_use_only_unreserved_development_identities():
    reserved = _reserved()
    m1 = _m1_streams()
    assert m1 == {4481, 4482} and not m1 & reserved
    for name in TEST_FILES:
        seeds = _integers(HERE / name)
        assert seeds and not seeds & reserved, (name, sorted(seeds & reserved))
        assert seeds <= set(H.DEVELOPMENT_STREAMS)
