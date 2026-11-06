"""TURN131 independent invented accounting identities and stand-in endpoint controls."""
import math
from types import SimpleNamespace

import pytest

import telohelm.learning as learning; import paper2.experiment2.mvp_seed as mvp_seed
from telohelm.runtime.ledger import Call; from telohelm.runtime.ledger import Ledger; from telohelm.runtime.ledger import amortised


def test_existing_d1_never_allocates_later_build_cost_to_an_earlier_episode():
    # Four scheduled positions, two builds. Analytic suffix allocations: 12/3 and 8/1.
    running = [1., 2., 3., 4.]
    result = amortised({1: 12., 3: 8.}, running)
    assert result == [1., 6., 7., 16.]
    assert sum(result) == sum(running) + 12. + 8.
    assert result != [x + 20. / 4 for x in running]  # rejected flat retrospective allocation
    # Matched subset {0,2} receives its existing full-stream shares; do not spread the entire build into it again.
    assert sum(result[i] for i in (0, 2)) == 8.


def test_original_issue_before_ending_counts_in_full_and_recovery_does_not_recharge():
    ledger = Ledger()
    ledger.add(Call('before', 'reasoning_model', 'invented', 0, 0., 5., 11, 7, 'answered', answered_step=3))
    ledger.add(Call('unknown_answer', 'reasoning_model', 'invented', 0, None, None, None, 2, 'interrupted'))
    ledger.add(Call('at_cut', 'small_model', 'invented', 1, 5., .2, 3, 4, 'answered', answered_step=1))
    ledger.add(Call('recovered', 'reasoning_model', 'invented', 0, 6., .01, 0, 0, 'answered',
                    answered_step=3, recovers='before'))
    cuts = ledger.ending_calls('invented', 1)
    assert cuts == {'counted': ['before', 'unknown_answer'], 'charge_ids': ['before', 'unknown_answer', 'recovered'],
                    'excluded': ['at_cut'],
                    'straddling': ['before'], 'straddling_unknown': ['unknown_answer']}
    assert ledger.calls[0].wall_s == 5.  # retain full original charge, not the tiny recovery read time
    assert sum(c.tokens_in for c in ledger.calls if c.request_id in cuts['charge_ids'] and c.tokens_in is not None) == 11


@pytest.mark.parametrize('x,y', [(-1e-16, 11.), (1e-12, 11.), (10., 1.), (-10., 1.),
                                (1., -10.), (-1., -10.), (0., 10.)])
def test_stand_in_repair_changes_only_derived_360_and_preserves_direction(x, y):
    namespace = {}; exec(mvp_seed.AIM, namespace)
    # Zero-speed invented orbit makes all iterations' target exactly (x,y).
    ctx = SimpleNamespace(x=0., y=0., t=0, report_measured_at=0, report_x=x, report_y=y,
                          tanker_centre_x=x, tanker_centre_y=y-1., tanker_radius_nm=1., tanker_speed_nm_s=0.,
                          wind_x_nm_s=0., wind_y_nm_s=0., tas_nm_s=.125)
    h = namespace['decide'](ctx)['heading_deg']
    old = math.degrees(math.atan2(x, y)) % 360.
    assert h == (0. if old == 360. else old)
    assert 0 <= h < 360
    assert math.sin(math.radians(h)) == pytest.approx(x/math.hypot(x, y), abs=1e-14)
    assert math.cos(math.radians(h)) == pytest.approx(y/math.hypot(x, y), abs=1e-14)


def test_b38_interface_sentence_is_exact_without_a_new_steering_instruction():
    sentence = ('Each accepted non-null trigger setting re-arms it, even at the same distance; it cannot fire in its '
                'arming second, and omitting this field leaves it armed until it fires, is cleared, or the goal version '
                'changes.')
    assert learning.PROPOSER_RULES.count(sentence) == 1
    assert all(word not in learning.PROPOSER_RULES.lower() for word in ('steer', 'intercept', 'pursuit'))
