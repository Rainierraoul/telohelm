"""Preflight accounting and immutable-prefix controls on invented geometry."""
import json
import pytest
import paper2.experiment2.binding_preflight_codex as F; import paper2.experiment2.original_replay as R
from telohelm.runtime.context import ContextBuilder
from telohelm.taskpacks.tanker.world.episode import Episode


def test_measured_prefix_is_the_full_typed_binding_not_the_current_bundle():
    ep = Episode(F.scenario("unit", 1, False))
    initial = ep.reset()
    prefix = [initial["bundle"]]
    builder = ContextBuilder(initial["briefing"])
    context = builder.update(prefix[-1])
    for _ in range(3):
        prefix.append(ep.step({}))
        context = builder.update(prefix[-1])
    c, b = F.bind_prefix(ep, initial["briefing"], prefix, context)
    body = json.loads(b.body)
    assert len(body["bundles"]) == 4 and body["original"] == json.loads(c.body)
    assert len(b.body) > len(c.body) > len(R.encode(prefix[-1]))
    assert body["pending"]["public_history_sha256"] == R.digest({
        "briefing": initial["briefing"], "bundles": prefix})
    prefix[0]["own"]["x"] = 100
    assert json.loads(b.body)["bundles"][0]["own"]["x"] == 0


@pytest.mark.parametrize("seconds", [(), (1, 0), (0, 0), (-1,), (3600,), (True,)])
def test_invalid_preterminal_snapshot_schedule_refused(seconds):
    with pytest.raises(ValueError):
        F.measure("unit", 1, False, seconds)
