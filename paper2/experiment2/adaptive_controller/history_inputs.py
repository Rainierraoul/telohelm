"""Precise synthetic public report histories for cross-review, not timing execution.

Only math/JSON/public workload construction. No estimator/search/world/random/model
runs. The histories are isolated public-input points, not integrated trajectories.
"""
from copy import deepcopy
from itertools import product
import math
from . import qualification as Q


def definitions():
    return [{'id': f't{t}-n{n}-cad{cadence}-{mode}', 'final_second': t,
             'targets': n, 'cadence_s': cadence, 'final_reports': mode}
            for t, n, cadence, mode in product((180, 900, 1800, 3599), (1, 2), (1, 12), ('none', 'fresh'))]


def materialize(spec):
    """Deterministic binary64 public bundles; reports on range(0,t,cadence).

    Zero radius change, theta=(280/3600)/8*measurement. One or two briefed
    targets; no goal change/help/refusal, final delivery either fresh or absent.
    All payload values are inputs only; no outcome labels are computed.
    """
    spec = {k: spec[k] for k in ('id', 'final_second', 'targets', 'cadence_s', 'final_reports')}
    w = Q.public_workloads()[0]
    briefing = deepcopy(w['briefing'])
    if spec['targets'] == 2:
        briefing['tankers'].append(dict(callsign='B', centre=[75., 95.], radius=8.,
                                       speed_nm_s=280/3600, direction='clockwise', position_at_0=[75., 103.]))
    elif spec['targets'] != 1:
        raise ValueError('one or two targets required')
    t, cadence = spec['final_second'], spec['cadence_s']
    if type(t) is not int or t < 1 or type(cadence) is not int or cadence < 1:
        raise ValueError('positive integer time/cadence required')
    if spec['final_reports'] not in ('none', 'fresh'):
        raise ValueError('final report mode required')

    def bundle(second, report):
        b = deepcopy(w['bundle']); b['t'] = second; b['reports'] = []
        if report:
            theta = (280/3600)/8*second
            for item in briefing['tankers']:
                x, y = item['centre']
                b['reports'].append({'callsign': item['callsign'], 'x': x+8*math.sin(theta),
                                     'y': y+8*math.cos(theta), 'measured_at': second, 'received_at': second})
        return b
    return {'definition': deepcopy(spec), 'briefing': briefing,
            'prefix_bundles': [bundle(s, True) for s in range(0, t, cadence)],
            'final_bundle': bundle(t, spec['final_reports']=='fresh')}


def manifest():
    specs = definitions()
    for spec in specs:
        data = materialize(spec)
        spec['prefix_bundle_count'] = len(data['prefix_bundles'])
        spec['prefix_report_count'] = len(data['prefix_bundles'])*spec['targets']
        spec['retained_per_target_after_final'] = len(data['prefix_bundles'])+(spec['final_reports']=='fresh')
        # Hash only the semantic definition plus exact materialized public data,
        # excluding the expected counts/hash added to this manifest entry.
        spec['input_sha256'] = Q.digest(data)
    body = {'stage': 'prospective exact synthetic histories; not executed timing', 'measurement_authorized': False,
            'flight_authorized': False, 'source_rule': 'history_inputs.materialize; zero-change radius8, analytic angle; no random/hidden labels',
            'sequences': specs, 'count': len(specs), 'geometry': 'Q fresh public workload, optional B(75,95), own fixed isolated point',
            'profile_proposal': 'three repeats32histories estimator-only final updates; separate setup, no feasibility clearance; controlled late-act projection requires separate exact loader/code review',
            'limitations': ['isolated public own points, not integrated trajectories or reached policy states',
                           'zero change only; short stateful campaign covers other beliefs',
                           'cold policy/help/report flags must not be presented as actual earlier policy decisions',
                           'no reserved seed or scientific outcome label generated']}
    return body | {'body_sha256': Q.digest(body)}
