"""Independent closure controls on the TURN207 counterexamples; design sources only."""
from pathlib import Path
from dataclasses import replace
import hashlib

import pytest

import paper2.experiment2.outcomecases as O; import telohelm.examples as X
import telohelm.runtime.sandbox as S
from paper2.experiment2.planting_runner import FROZEN_C0


@pytest.mark.parametrize('t', (0, 30))
def test_v3_eventful_and_quiet_prefixes_agree_without_modifying_the_original(t):
    sc = O._scenario(4400, 4, 0)
    quiet = replace(sc, goals=tuple(g for g in sc.goals if g[0] <= t), dropouts=())
    p, q = O._prefix_of(sc, t), O._prefix_of(quiet, t)
    assert O._context(p) == O._context(q)
    ctx = O._context(p)
    candidates = [('report', {'report': f"Wedgetail, aircraft 2, {ctx['report_at_nm']:g} NM from {ctx['goal_tanker']}"}),
                  ('heading', {'heading_deg': O._reference_heading(p)})]
    before = sc.goals
    labels = O._labels(sc, t, candidates)
    assert [d for d, _ in labels] == [d for d, _ in O._labels(quiet, t, candidates)]
    assert labels[0][0] == 'wrong'
    assert sc.goals == before and any(s > t for s, _ in sc.goals)


def test_a_refused_replacement_is_no_longer_named_as_the_trigger_that_fired():
    actions = {10: {'action': {'report_trigger_nm': 10}, 'field_writers': {'report_trigger_nm': 'report'}},
               20: {'action': {'report_trigger_nm': 20}, 'field_writers': {'report_trigger_nm': 'report'}}}
    steps = {10: {'accepted': True}, 20: {'accepted': False}}
    sources = {(10, 'report'): 'procedure', (20, 'report'): 'reasoning_model'}
    shown = X.transmission(30, True, actions, sources, steps)
    assert shown == {'by_trigger': True, 'trigger_set_at': 10, 'trigger_nm': 10, 'trigger_by': 'stored procedure'}


@pytest.mark.skipif(not S.available(), reason='native containment unavailable')
def test_frozen_c0_natively_rejects_the_original_wrong_controller_witness():
    root = next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file())
    source = (next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'paper2/development/rq3-reference-checks/report_check.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == FROZEN_C0['report']
    assert 'CHANGE_START' not in source and 'CHANGE_DURATION' not in source
    p = O._prefix(4400, 1, 0, 451)
    ctx = O._context(p)
    right = f"Wedgetail, aircraft 2, {ctx['report_at_nm']:g} NM from {ctx['goal_tanker']}"
    wrong = right.replace('Wedgetail', 'WrongControl')
    for text, wanted in ((right, 'accept'), (wrong, 'reject')):
        result = S.run(source, 'check', [ctx, {'report': text}], context_args=True)
        assert result.status == 'ok' and result.result == wanted
    assert O._labels(O._scenario(4400, 1, 0), 451, [('report', {'report': wrong})])[0][0] == 'wrong'


def test_an_unobserved_change_is_removed_even_if_already_under_way():
    # This is deliberately a quiet counterfactual label, not the actual hidden state.
    sc = O._scenario(4400, 3, 15)     # all-three block, positive radar delay
    t = 62                          # a change has begun but has not reached the radar history
    p = O._prefix_of(sc, t)
    family = O._family(sc, t, O._context(p))
    assert O._context(family[0]) == O._context(p)
    original = next(k for k in sc.tankers if k.orbit.delta)
    normalised = next(k for k in family[0][0].s.tankers if k.callsign == original.callsign)
    assert normalised.orbit.delta == 0 and original.orbit.position(t) != normalised.orbit.position(t)


def test_an_under_way_dropout_ends_after_the_current_measurement_in_quiet_continuation():
    sc = O._scenario(4400, 3, 5)
    callsign, start, end = sc.dropouts[0]
    assert end is None or end > start + 1
    quiet = O._quiet(sc, start)
    assert quiet.dropouts == ((callsign, start, start + 1),)
    assert O._context(O._prefix_of(sc, start)) == O._context(O._prefix_of(quiet, start))
