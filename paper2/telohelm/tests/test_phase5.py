"""Tests for the reference pilot, the help certificate and phase-5 generation. Development seeds only."""
from __future__ import annotations

import math
from dataclasses import replace

import telohelm.taskpacks.tanker.world.episode as E
import telohelm.taskpacks.tanker.world.phase5 as phase5
import telohelm.taskpacks.tanker.world.physics as P


def test_draw_is_deterministic_and_worlds_differ_only_in_change_sign():
    a, b = phase5.draw(4400), phase5.draw(4400)
    assert a == b
    oa, ob = a.world_a.tankers[0].orbit, a.world_b.tankers[0].orbit
    assert oa.delta == -ob.delta and oa.delta > 0
    assert replace(oa, delta=0.0) == replace(ob, delta=0.0)
    assert a.world_a.tankers[1] == a.world_b.tankers[1]
    assert a.world_a.dropouts == a.world_b.dropouts == (("SHELL 1", a.blackout, None),)


def test_second_tanker_offset_is_44_to_56_north_or_south():
    for seed in range(4400, 4420):
        d = phase5.draw(seed)
        c1, c2 = d.world_a.tankers[0].orbit, d.world_a.tankers[1].orbit
        assert 44 <= abs(c2.cy - c1.cy) <= 56
        assert math.hypot(c2.cx - c1.cx, c2.cy - c1.cy) >= 44


def test_identical_worlds_give_no_certificate_window():
    d = phase5.draw(4401)
    same = replace(d, world_b=d.world_a)
    cert = phase5.certificate_limit(same)
    first_reach = cert["earliest_possible_t2"]["ab"]
    witness = phase5.fly(d.world_a, ask_help=True)
    assert first_reach is not None and cert["deadline_limit"] < witness["joined_at"]


def test_pilot_meets_goal_without_disturbance_and_without_help():
    d = phase5.draw(4405)
    calm = replace(d.world_a, dropouts=(), tankers=(replace(d.world_a.tankers[0],
                   orbit=replace(d.world_a.tankers[0].orbit, delta=0.0)), d.world_a.tankers[1]))
    out = phase5.fly(calm, ask_help=False)
    assert out["goal_met"] and out["help_requests"] == []


def test_wind_from_west_blows_east():
    wx, wy = phase5.wind_vector(40, 270)
    assert wx > 0 and abs(wy) < 1e-12


def test_certified_draw_confirms_both_worlds_at_the_deadline():
    r = phase5.assess(4400)
    assert r["label"] in ("help certified", "unknown")
    if r["label"] == "help certified":
        assert r["deadline"] <= r["certificate"]["deadline_limit"]


def test_unsettled_success_is_never_a_certified_witness(monkeypatch):
    monkeypatch.setattr(phase5, 'certificate_limit', lambda d: {'deadline_limit': 100})
    monkeypatch.setattr(phase5, 'fly', lambda *a, **kw: {'goal_met': True, 'joined_at': 50,
                                                       'settled': False, 'flags': [{'kind': 'near'}]})
    r = phase5.assess(4400)
    assert r['label'] == 'unknown' and 'numerical flags' in r['reason']
    assert r['witness']['a']['flags']


def test_nonpositive_certificate_margin_cannot_certify(monkeypatch):
    monkeypatch.setattr(phase5, 'certificate_limit', lambda d: {'deadline_limit': 100,
                                                              'strict_margin_nm': {'ab': 0, 'ba': 1}})
    monkeypatch.setattr(phase5, 'fly', lambda *a, **kw: {'goal_met': True, 'joined_at': 50,
                                                       'settled': True, 'flags': []})
    r = phase5.assess(4400)
    assert r['label'] == 'unknown' and 'strict' in r['reason']


def test_zero_deadline_grid_margin_matches_hand_inequalities():
    d = phase5.draw(4401)
    margins = phase5.strict_grid_margins(d, 0)
    S, h, e = d.generous_speed, phase5.GRID_STEP, E.JOIN_TOLERANCE
    x0 = d.world_a.receiver
    for name, first, second in [('ab', d.world_a, d.world_b), ('ba', d.world_b, d.world_a)]:
        o1, o2 = first.tankers[0].orbit, second.tankers[0].orbit
        a, b = o1.position(0), o2.position(0)
        one = math.hypot(a[0]-x0.x, a[1]-x0.y) - e - (S+o1.v)*h/2 - phase5.NUMERIC_ALLOWANCE
        two = math.hypot(b[0]-a[0], b[1]-a[1]) - 2*e - (2*S+o1.v+o2.v)*h/2 - phase5.NUMERIC_ALLOWANCE
        assert abs(margins[name] - max(one, two)) < 1e-12


def test_total_budget_covers_two_position_errors_plus_arithmetic():
    # Two opposite permitted position errors shorten this distance by 2*e_pos.
    ep, ea = phase5.POSITION_ALLOWANCE, phase5.NORM_ARITHMETIC_ALLOWANCE
    exact = math.hypot(10, 0)
    perturbed = math.hypot((10-ep) - ep, 0) - ea
    assert exact - perturbed <= phase5.NUMERIC_ALLOWANCE + 1e-14


def test_retained_witness_trace_replays_its_actions_and_outcome():
    d = phase5.draw(4405)
    sc = replace(d.world_a, dropouts=(), tankers=(replace(d.world_a.tankers[0],
                 orbit=replace(d.world_a.tankers[0].orbit, delta=0)), d.world_a.tankers[1]))
    original = phase5.fly(sc, ask_help=False, retain_trace=True)
    ep = E.Episode(sc); start = ep.reset(); bundle = start['bundle']
    assert start['briefing'] == original['trace']['briefing']
    for row in original['trace']['steps']:
        assert bundle == row['bundle']
        bundle = ep.step(row['action'])
    assert bundle is None
    replayed = ep.outcome()
    assert all(original[k] == v for k, v in replayed.items())
