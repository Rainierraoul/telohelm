"""Candidate finite original-reference suffix; no pilot, sampler or label producer.

The trusted runner must independently retain both original and reference anchors
before candidate evaluation. Hashes alone do not establish that authority/order.
All replay evidence remains conditional/inadmissible under the existing B8 gates.
"""
from dataclasses import dataclass, asdict
import hashlib
import json

from . import original_binding as B; from . import original_replay as R
from telohelm.runtime.delivery import Pending
from telohelm.taskpacks.tanker.world.episode import HORIZON

RULE = 'B8.original.fixed-reference-suffix.v1'
QUALIFICATION = 'conditional_original_reference_authority_and_numerical_error'


@dataclass(frozen=True)
class Reference:
    body: bytes
    sha256: str
    qualification: str = QUALIFICATION
    admissible: bool = False


@dataclass(frozen=True)
class Plan:
    body: bytes
    sha256: str
    qualification: str = QUALIFICATION
    admissible: bool = False

    @property
    def candidate(self):
        return json.loads(self.body)['candidate']

    @property
    def continuation(self):
        return json.loads(self.body)['continuation']


def _anchored(obj, cls, expected):
    if (type(obj) is not cls or type(obj.body) is not bytes or
            hashlib.sha256(obj.body).hexdigest() != obj.sha256 or obj.sha256 != expected or
            obj.qualification != QUALIFICATION or obj.admissible is not False):
        raise ValueError('independently retained conditional reference/plan required')
    data = json.loads(obj.body)
    if R.encode(data) != obj.body:
        raise ValueError('canonical immutable bytes required')
    return data


def _binding(binding, expected):
    if (type(binding) is not B.Binding or type(binding.body) is not bytes or
            hashlib.sha256(binding.body).hexdigest() != binding.sha256 or binding.sha256 != expected or
            binding.qualification != B.QUALIFICATION or binding.admissible is not False):
        raise ValueError('independently retained conditional binding required')
    saved = json.loads(binding.body)
    if saved['rule'] != B.RULE or saved['sources'] != B._sources():
        raise ValueError('binding source/rule conflict')
    raw = R.encode(saved['original'])
    capture = R.Capture(raw, hashlib.sha256(raw).hexdigest())
    rebuilt = B.bind(capture, B.Issue(**saved['issue']), briefing=saved['briefing'],
                     bundles=saved['bundles'], issued_context=saved['issued_context'],
                     source_body=saved['source_body'],
                     issued_call=None if saved['issued_call'] is None else Pending(**saved['issued_call']),
                     pending=tuple(Pending(**p) for p in saved['pending']['calls']),
                     expected_capture_sha256=saved['capture_sha256'])
    if rebuilt.body != binding.body:
        raise ValueError('original binding reconstruction conflict')
    return saved, R._restore(saved['original'])


def _reference_data(binding, saved, ep, rows, ended_at, source):
    start, version = ep.t, ep.goal.version
    if type(ended_at) is not int or not start < ended_at <= HORIZON:
        raise ValueError('finite absolute reference ending required')
    if ep.goal.deadline is not None and ended_at > ep.goal.deadline:
        raise ValueError('reference extends beyond original deadline')
    if any(t <= ended_at for t, _ in ep.pending_goals):
        raise ValueError('goal changes in reference interval')
    if not isinstance(source, dict) or not source or any(not isinstance(k, str) for k in source):
        raise ValueError('retained reference source provenance required')
    if type(rows) not in (tuple, list) or len(rows) != ended_at - start:
        raise ValueError('exact contiguous original reference action interval required')
    detached = json.loads(R.encode(rows))
    for second, row in enumerate(detached, start):
        if (set(row) != {'t', 'actions', 'accepted'} or type(row['t']) is not int or
                row['t'] != second or row['accepted'] is not True or type(row['actions']) is not dict):
            raise ValueError('reference action clock, fields or acceptance differs')
        action = row['actions']
        if 'help' in action or ep._action_error(action) is not None:
            raise ValueError('reference requires valid fixed-goal actions without new help')
    return {'rule': RULE, 'binding_sha256': binding.sha256, 'capture_sha256': saved['capture_sha256'],
            'scenario_sha256': R.digest(saved['original']['scenario']),
            'episode': ep.s.episode_id, 'goal': asdict(ep.goal),
            'public_history_sha256': saved['pending']['public_history_sha256'],
            'start_second': start, 'ended_at': ended_at, 'rows': detached,
            'reference_source': source, 'qualification': QUALIFICATION, 'admissible': False}


def retain_reference(binding, rows, ended_at, source, *, expected_binding_sha256):
    """Package a caller's already retained original-state reference controls.

    No reference execution or claim that an ending is successful is made here.
    An independent runner must pin the resulting SHA before seeing a candidate.
    Rows from an unrelated initial-state pilot cannot be relabelled as originating
    at a later actor snapshot merely by supplying this constructor's metadata.
    """
    saved, ep = _binding(binding, expected_binding_sha256)
    data = _reference_data(binding, saved, ep, rows, ended_at, source)
    body = R.encode(data)
    return Reference(body, hashlib.sha256(body).hexdigest())


def build(binding, reference, candidate, *, expected_binding_sha256,
          expected_reference_sha256, limits=R.Limits()):
    saved, ep = _binding(binding, expected_binding_sha256)
    ref = _anchored(reference, Reference, expected_reference_sha256)
    expected = _reference_data(binding, saved, ep, ref['rows'], ref['ended_at'], ref['reference_source'])
    if ref != expected:
        raise ValueError('reference original/goal/prefix identity conflict')
    if type(candidate) is not dict or 'help' in candidate or ep._action_error(candidate) is not None:
        raise ValueError('candidate must be a valid fixed-goal public action without new help')
    candidate = json.loads(R.encode(candidate))
    continuation = [row['actions'] for row in ref['rows'][1:]]
    limits.validate()
    if len(ref['rows']) > limits.max_steps:
        raise ValueError('reference suffix exceeds replay step cap')
    plan_bytes = len(R.encode([candidate, continuation]))
    if len(binding.body) + plan_bytes > limits.input_bytes:
        raise ValueError('typed binding and actual plan input byte limit')
    data = {'rule': RULE, 'binding_sha256': binding.sha256, 'reference_sha256': reference.sha256,
            'candidate_second': ep.t, 'continuation_seconds': list(range(ep.t + 1, ref['ended_at'])),
            'candidate': candidate, 'continuation': continuation,
            'binding_plus_plan_bytes': len(binding.body) + plan_bytes,
            'limits': asdict(limits), 'qualification': QUALIFICATION, 'admissible': False}
    body = R.encode(data)
    return Plan(body, hashlib.sha256(body).hexdigest())


def replay(binding, reference, plan, *, expected_binding_sha256,
           expected_reference_sha256, expected_plan_sha256):
    data = _anchored(plan, Plan, expected_plan_sha256)
    rebuilt = build(binding, reference, data['candidate'],
                    expected_binding_sha256=expected_binding_sha256,
                    expected_reference_sha256=expected_reference_sha256,
                    limits=R.Limits(**data['limits']))
    if rebuilt.body != plan.body:
        raise ValueError('finite continuation plan differs from immutable original suffix')
    return B.replay_bound(binding, data['candidate'], data['continuation'],
                          expected_binding_sha256=expected_binding_sha256,
                          limits=R.Limits(**data['limits']))
