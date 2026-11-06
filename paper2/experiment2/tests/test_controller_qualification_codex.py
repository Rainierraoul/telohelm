"""Invented-clock/planner controls only; never execute a controller search or episode."""
from copy import deepcopy
from itertools import count

import pytest

import paper2.experiment2.adaptive_controller.qualification as Q
from paper2.experiment2.adaptive_controller.estimation import EpisodeEstimator


def test_declared_grid_and_disjoint_reserved_samples_are_exact():
    cells = Q.policy_grid()
    assert len(cells) == len({c['id'] for c in cells}) == 24
    assert {(c['search']['horizon_s'], c['search']['heading_spacing_deg'], c['adapter']['report_guard_nm'],
             c['adapter']['stale_after_s']) for c in cells} == {
        (h, l, r, s) for h in (150, 300, 600) for l in (10, 20) for r in (.15, .20) for s in (10, 30)}
    assert Q.SENSITIVITY_SEEDS == tuple(range(4540, 4560))
    assert Q.VERIFICATION_SEEDS == tuple(range(4560, 4600))
    assert not set(Q.SENSITIVITY_SEEDS) & set(Q.VERIFICATION_SEEDS)
    assert all(c['search']['second_leg_at_s'] == 120 and c['search']['sample_s'] == 1 for c in cells)


def test_recipes_cover_declared_disturbances_without_claiming_materialized_inputs():
    recipes = Q.constructed_recipes()
    assert len(recipes) == len({r['id'] for r in recipes})
    assert {tuple(r['wind_kt_from']) for r in recipes if 'wind_kt_from' in r} == {
        (0, 0), *((s, d) for s in (20, 40, 60) for d in (0, 90, 180, 270))}
    assert {r['fractional_radius_change'] for r in recipes if 'fractional_radius_change' in r} == {-.2, 0., .2}
    assert {r['delay_s'] for r in recipes if 'delay_s' in r} == {0, 5, 30, 120}
    assert {'old-report-after-help', 'goal-change-with-help-pending', 'tanker-change-while-turning',
            'report-boundary', 'contact-boundary', 'deadline-boundary', 'different-command-and-actual'} <= {
        r['id'] for r in recipes}
    package = Q.draft_package()
    assert package['stage'] == 'draft-preparation-only' and package['flight_authorized'] is False
    fingerprint = package.pop('draft_sha256')
    assert Q.digest(package) == fingerprint
    assert 'exact sampler/draw order and materialized geometry/input identities' in package['open']


class NoSearch:
    """Validate observations with the actual estimator; issue no heading/report and run no guidance."""
    def __init__(self, search, adapter, allowance):
        self.estimator = EpisodeEstimator(allowance)
    def reset(self, briefing):
        self.estimator.reset(briefing)
    def act(self, bundle):
        self.estimator.update(bundle)
        return {}


def invented_profile(factory=NoSearch):
    cpu, wall = count(step=1_000_000), count(step=2_000_000)
    return Q.profile_search(factory, lambda: next(cpu), lambda: next(wall))


def test_interleaved_profile_is_fresh_inclusive_nonmutating_and_projection_is_only_illustrative():
    before = deepcopy(Q.public_workloads())
    rows = invented_profile()
    assert len(rows) == 144 and Q.public_workloads() == before
    assert [r['repetition'] for r in rows] == [0]*48 + [1]*48 + [2]*48
    assert all(r['error'] is None and r['cpu_ns'] == 1_000_000 and r['wall_ns'] == 2_000_000 for r in rows)
    inputs = {w['id']: Q.digest(w) for w in before}
    assert all(r['input_sha256'] == inputs[r['workload']] for r in rows)
    projection = Q.feasibility_projection(rows)
    # 24 policies ×20 episodes ×3600 seconds ×0.001 processor seconds /3600.
    assert projection['projected_cpu_h'] == pytest.approx(.48)
    assert projection['disposition'] == 'next-review-only; no flight clearance'
    costly = deepcopy(rows)
    for row in costly:
        row['cpu_ns'] = 1_000_000_000
    assert Q.feasibility_projection(costly)['projected_cpu_h'] == 480
    assert Q.feasibility_projection(costly)['disposition'] == 'stop-and-revise'


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'unknown', 'failure'])
def test_incomplete_or_failed_profile_never_gives_flight_clearance(mode):
    rows = invented_profile()
    if mode == 'missing':
        rows.pop()
    elif mode == 'duplicate':
        rows[-1] = deepcopy(rows[0])
    elif mode == 'unknown':
        rows[0]['cpu_ns'] = None
    else:
        rows[0]['error'] = 'invented search failure'
    if mode in ('missing', 'duplicate'):
        with pytest.raises(ValueError, match='complete unique'):
            Q.feasibility_projection(rows)
    else:
        assert Q.feasibility_projection(rows) == {
            'projected_cpu_h': None, 'disposition': 'incomplete-or-unknown; no flight clearance'}


def test_profile_failure_remains_an_explicit_charged_row():
    class Failure(NoSearch):
        def act(self, bundle):
            raise ValueError('invented failed computation')
    rows = invented_profile(Failure)
    assert len(rows) == 144 and all(r['error'] and r['cpu_ns'] == 1_000_000 for r in rows)
    assert Q.feasibility_projection(rows)['projected_cpu_h'] is None
