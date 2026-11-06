"""Development-only RQ3 eligibility/final pools, separate from replay and learner inputs.

Defaults are a protocol-0.12 proposal, not scored qualification. Each role uses only its
partition source; insufficient yield is a recorded failure, never an extra draw/source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from . import outcomecases as O
import telohelm.qualification.store as H

DEVELOPMENT_RUNS = (4401, 4402, 4403, 4404)


def digest(value: Any) -> str:
    return hashlib.sha256(H.canonical(value)).hexdigest()


def case_key(case: dict) -> str:
    """One public decision/input/output, independent of split or source metadata."""
    return digest([case['decision'], case['context'], case['output']])


def seal(path: Path, record: dict) -> None:
    """A completed immutable record. Interrupted writes remain explicit incomplete files."""
    data = H.canonical({'record': record, 'sha256': digest(record)})
    with path.open('xb') as f:
        f.write(data + b'\n')
        f.flush()
        import os
        os.fsync(f.fileno())


def unseal(path: Path) -> dict:
    value = json.loads(path.read_text())
    if set(value) != {'record', 'sha256'} or digest(value['record']) != value['sha256']:
        raise ValueError(f'record digest does not match: {path}')
    return value['record']


def specification(stream: int, role: str, *, per_phase: int = 5, phases=(1, 2, 3, 4),
                  eligibility_per_decision: int = 100, final_total: int = 200) -> dict:
    if type(stream) is not int or stream not in DEVELOPMENT_RUNS:
        raise ValueError('only development run streams 4401 to 4404; scored sources remain reserved')
    if role not in ('eligibility', 'final'):
        raise ValueError('this builder takes eligibility or final, never replay')
    if type(per_phase) is not int or not 1 <= per_phase <= 20:
        raise ValueError('per_phase must be an integer in 1..20')
    if not phases or len(set(phases)) != len(phases) or any(type(p) is not int or p not in (1, 2, 3, 4) for p in phases):
        raise ValueError('distinct phases in 1..4 are required; no phase-5 competence claim')
    for n in (eligibility_per_decision, final_total):
        if type(n) is not int or n <= 0 or n % 2:
            raise ValueError('positive even pool sizes are required')
    quotas = {d: ({'right': eligibility_per_decision // 2, 'wrong': eligibility_per_decision // 2}
                  if role == 'eligibility' else {'wrong': final_total // 2}) for d in H.DECISIONS}
    return {'schema': 1, 'development': True, 'rule': O.RULE, 'stream': stream, 'role': role,
            'sources': list(O.partition()[stream][role]), 'phases': list(phases), 'per_phase': per_phase,
            'quotas': quotas, 'sampling_seed': f'planting-pool|{O.RULE}|{stream}|{role}'}


def validate_cases(cases: list[dict], stream: int, role: str) -> None:
    """Structural and source-role checks; generation additionally replays every selected label."""
    specification(stream, role)
    seen = set()
    for c in cases:
        errors = H.case_errors(c, development=True)
        if errors:
            raise ValueError(f'case structure: {errors}')
        if c['stream'] != stream or c['split'] != role or c['label']['rule'] != O.RULE:
            raise ValueError('case stream, role or label version mismatch')
        source = c['origin']['seed']
        if source not in O.partition()[stream][role] or c['label']['evidence'].get('source') != source:
            raise ValueError('case source is outside its partition role')
        key = case_key(c)
        if key in seen:
            raise ValueError('duplicate decision/context/output within pool')
        seen.add(key)


def load(path: Path, stream: int, role: str) -> dict:
    record = unseal(path)
    if record.get('status') != 'complete':
        raise ValueError('pool build is not complete')
    spec = record['specification']
    if spec['stream'] != stream or spec['role'] != role or spec['rule'] != O.RULE:
        raise ValueError('pool identity mismatch')
    expected = specification(stream, role, per_phase=spec['per_phase'], phases=spec['phases'],
                             eligibility_per_decision=sum(spec['quotas']['heading'].values()) if role == 'eligibility'
                             else 100, final_total=2 * spec['quotas']['heading']['wrong'] if role == 'final' else 200)
    if spec != expected:
        raise ValueError('pool specification differs from the fixed source partition')
    validate_cases(record['cases'], stream, role)
    for decision, quotas in spec['quotas'].items():
        for label, n in quotas.items():
            if sum(c['decision'] == decision and c['label']['disposition'] == label for c in record['cases']) != n:
                raise ValueError('pool quota is incomplete')
    return record


def disjoint(*roles: list[dict]) -> None:
    if O.duplicates(*roles):
        raise ValueError('context/output overlap between replay, eligibility or final roles')


def build(stream: int, role: str, out: Path, *, per_phase=5, phases=(1, 2, 3, 4), workers=1,
          eligibility_per_decision=100, final_total=200) -> dict:
    spec = specification(stream, role, per_phase=per_phase, phases=phases,
                         eligibility_per_decision=eligibility_per_decision, final_total=final_total)
    if type(workers) is not int or workers < 1:
        raise ValueError('positive integer worker count required')
    out.mkdir(parents=True, exist_ok=False)         # no overwrite/resume/refill after partial work
    seal(out / 'plan.json', spec)                 # written before any draw
    record = {'specification': spec, 'status': 'incomplete', 'labelling': [], 'cases': []}
    try:
        rows = []
        for source in spec['sources']:
            labelled, counts = O.label_all(O.moments(source, per_phase, tuple(phases)), workers)
            record['labelling'].append({'source': source, **counts})
            rows += labelled
        # _keep is applied across the role too; the role currently has one source.
        rows, record['combined_exclusions'] = O._keep(rows)
        unique = {}
        for row in rows:
            key = digest([row['decision'], row['moment']['ctx'], row['output']])
            unique.setdefault(key, row)
        record['duplicate_rows'] = len(rows) - len(unique)
        rng = random.Random(spec['sampling_seed'])
        for decision, quotas in spec['quotas'].items():
            for disposition, n in quotas.items():
                pool = [r for r in unique.values() if r['decision'] == decision and r['disposition'] == disposition]
                if len(pool) < n:
                    raise ValueError(f'insufficient fixed-source yield for {decision}/{disposition}: {len(pool)} < {n}')
                for r in rng.sample(pool, n):
                    case = O._case(stream, role, r, len(record['cases']))
                    if H.case_errors(case, development=True) or not O.verify(case):
                        raise ValueError('selected label failed independent reconstruction')
                    record['cases'].append(case)
        validate_cases(record['cases'], stream, role)
        record['producer_sha256'] = {str(p.relative_to(Path(__file__).parent)):
                                    hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (Path(__file__), Path(O.__file__))}
        record['status'] = 'complete'
        seal(out / 'pool.json', record)
        return record
    except Exception as e:
        record['status'], record['error'] = 'failed', f'{type(e).__name__}: {e}'
        seal(out / 'failure.json', record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=int, required=True)
    parser.add_argument('--role', choices=('eligibility', 'final'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    record = build(args.stream, args.role, args.out, workers=args.workers)
    print(json.dumps({'status': record['status'], 'stream': args.stream, 'role': args.role,
                      'cases': len(record['cases']), 'record': str(args.out / 'pool.json')}))


if __name__ == '__main__':
    main()
