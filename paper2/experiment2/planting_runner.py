"""CPU-only RQ3 option-2 mutation selection, then separately opened final-case analysis.

No flight, live-library mutation, learner feedback or scored acquisition. Each check/case
uses the ordinary fresh-process sandbox. Final cases cannot choose the mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from . import outcomecases as O; from . import plantings as M; from . import planting_pools as P
import telohelm.qualification.store as H
import telohelm.runtime.sandbox as S
from telohelm.runtime.replay import MIN_RIGHT_ACCEPTED

FROZEN_C0 = {'heading': '801ea96e8b3ad9fb2959e5c533fde55c87c317326c11ba26ca7227b02d54719b',
             'report': '6a0e4217083db2ba011fdfad8cd369aaeffc02e108983c28c81f9a8934857054'}


def source_sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def placement_plan(stream: int, seed: int, phases=(1, 2, 3, 4, 5), episodes=20) -> list[dict]:
    """Proposal for cross-review: two placements/phase, one target per decision.

    One shared schedule for both static counterfactuals, not two independent samples.
    It deliberately replaces the old heading-first rule; it never launches a flight.
    """
    P.specification(stream, 'eligibility')
    if type(seed) is not int or type(episodes) is not int or episodes < 2:
        raise ValueError('integer placement seed and at least two episodes required')
    if not phases or len(set(phases)) != len(phases) or any(type(p) is not int or p not in range(1, 6) for p in phases):
        raise ValueError('distinct phases in 1..5 required')
    rng, placements = random.Random(seed), []
    for phase in phases:
        targets = ['heading', 'report']
        rng.shuffle(targets)
        for episode, decision in zip(sorted(rng.sample(range(episodes), 2)), targets):
            ordinal = len(placements) + 1
            placements.append({'stream': stream, 'phase': phase, 'episode_index': episode,
                               'decision': decision, 'ordinal': ordinal, 'counter': (ordinal - 1) % 10 + 1})
    return placements


def evaluate(source, cases, run=S.run):
    """Save actual statuses separately: an execution error is not substantive rejection."""
    verdicts = []
    for c in cases:
        r = run(source, 'check', [c['context'], c['output']], context_args=True)
        status = r.status
        if status == 'ok' and r.result not in ('accept', 'reject'):
            status = 'invalid_verdict'
        verdicts.append({'case': H.case_id(c), 'accepted': status == 'ok' and r.result == 'accept',
                         'status': status})
    return verdicts


def execution_ok(values):
    return all(v['status'] == 'ok' for v in values)


def replay(values_wrong, values_right, old_right=None):
    wrong, right = sum(v['accepted'] for v in values_wrong), sum(v['accepted'] for v in values_right)
    old = None if old_right is None else sum(v['accepted'] for v in old_right)
    passed = (execution_ok(values_wrong + values_right) and wrong == 0 and right >= MIN_RIGHT_ACCEPTED
              and (old is None or right >= old))
    return {'passed': passed, 'wrong_accepted': wrong, 'wrong_cases': len(values_wrong),
            'right_accepted': right, 'right_cases': len(values_right), 'old_right_accepted': old,
            'wrong_verdicts': values_wrong, 'right_verdicts': values_right}


def _replay_cases(cases, stream, decision, rewrite):
    if type(rewrite) is not int or rewrite < 1:
        raise ValueError('each planting requires an explicit positive fresh replay ordinal')
    all_keys, sources = set(), set(O.partition()[stream]['replay'])
    wrong, right = [], []
    for c in cases:
        errors = H.case_errors(c, development=True)
        if errors or c['stream'] != stream or c['label']['rule'] != O.RULE:
            raise ValueError(f'replay case structure/identity mismatch: {errors}')
        if c['split'] not in ('stored_right', 'stored_wrong', 'confirmed_wrong', 'fresh_wrong'):
            raise ValueError('eligibility and final cases cannot be replayed')
        if c['origin']['seed'] not in sources or c['label']['evidence'].get('source') != c['origin']['seed']:
            raise ValueError('replay source is outside the fixed partition')
        if c['decision'] != decision:
            raise ValueError('supply only the selected decision replay snapshot')
        key = P.case_key(c)
        if key in all_keys:
            raise ValueError('duplicate public replay case')
        all_keys.add(key)
        if c['split'] == 'stored_right':
            right.append(c)
        else:
            if c['split'] == 'fresh_wrong' and c['origin'].get('rewrite') != rewrite:
                raise ValueError('fresh replay ordinal mismatch')
            wrong.append(c)
    fresh = sum(c['split'] == 'fresh_wrong' for c in wrong)
    if (len(right) != H.RIGHT_PER_DECISION or fresh != H.FRESH_PER_REWRITE
            or sum(c['split'] == 'stored_wrong' for c in wrong) != O.STORED_WRONG):
        raise ValueError('replay needs 100 stored right, 40 stored wrong and exactly 100 fresh wrong cases')
    return wrong, right


def select(out: Path, placement: dict, c0: str, expected_sha: str, eligibility: Path, replay_cases: list[dict],
           *, rewrite: int, run=S.run) -> dict:
    """Choose by eligibility only, then replay once. Output directory is never reused.

    expected_sha is explicit for authored test fixtures; the CLI always pins FROZEN_C0.
    The caller supplies the exact protected replay snapshot at the placement point.
    """
    placement = json.loads(H.canonical(placement))
    replay_cases = [H.snapshot(c) for c in replay_cases]
    stream, decision, counter = placement['stream'], placement['decision'], placement['counter']
    P.specification(stream, 'eligibility')
    if decision not in H.DECISIONS or type(counter) is not int or counter not in range(1, 11):
        raise ValueError('invalid decision or operation counter')
    if source_sha(c0) != expected_sha:
        raise ValueError('C0 source differs from the declared frozen source')
    if run is S.run and not S.available():
        raise S.SandboxUnavailable('the native sandbox is unavailable; nothing executed')
    pool = P.load(eligibility, stream, 'eligibility')
    eligible = [c for c in pool['cases'] if c['decision'] == decision]
    wrong, right = _replay_cases(replay_cases, stream, decision, rewrite)
    P.disjoint(pool['cases'], replay_cases)
    out.mkdir(parents=True, exist_ok=False)
    bound = {'schema': 1, 'development': True, 'rule': O.RULE, 'placement': placement,
             'rewrite': rewrite, 'baseline_sha256': expected_sha, 'baseline_source': c0,
             'eligibility_sha256': P.digest(pool), 'replay_sha256': P.digest(replay_cases),
             'input_keys': [P.case_key(c) for c in pool['cases'] + replay_cases],
             'eligibility_cases': eligible, 'replay_cases': replay_cases}
    P.seal(out / 'inputs.json', bound)
    record = {**bound, 'status': 'incomplete', 'attempts': [], 'planted': False}
    started = time.perf_counter()
    # Append each completed stage before attempting another; crash leaves the bound inputs and progress.
    def progress(event):
        with (out / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(event, sort_keys=True) + '\n')
            f.flush()
            import os
            os.fsync(f.fileno())
    try:
        base_wrong, base_right = evaluate(c0, wrong, run), evaluate(c0, right, run)
        record['baseline_replay'] = replay(base_wrong, base_right)
        progress({'stage': 'baseline_replay', 'passed': record['baseline_replay']['passed']})
        if not record['baseline_replay']['passed']:
            record['status'] = 'baseline_unqualified'
        else:
            base_elig = evaluate(c0, eligible, run)
            record['baseline_eligibility'] = base_elig
            if not execution_ok(base_elig):
                record['status'] = 'baseline_execution_error'
            else:
                for k in range(10):
                    operation = (counter - 1 + k) % 10 + 1
                    m = M.mutate(c0, operation)
                    attempt = {'operation': operation, 'status': 'inapplicable'}
                    if m is not None:
                        attempt.update(path=list(m.path), before_sha256=m.before_sha256, after_sha256=m.after_sha256,
                                       source=m.source)
                        try:
                            compile(m.source, '<mutation>', 'exec')
                        except (SyntaxError, ValueError) as e:
                            attempt.update(status='compile_error', error=str(e))
                        else:
                            values = evaluate(m.source, eligible, run)
                            attempt['eligibility_verdicts'] = values
                            expands = (all(not b['accepted'] or v['accepted'] for b, v in zip(base_elig, values))
                                       and any(not b['accepted'] and v['accepted'] for b, v in zip(base_elig, values)))
                            attempt['status'] = ('execution_error' if not execution_ok(values)
                                                 else 'eligible' if expands else 'no_expansion')
                    record['attempts'].append(attempt)
                    progress({'stage': 'mutation', **attempt})
                    if attempt['status'] == 'eligible':
                        record.update(planted=True, selected_operation=operation, mutant_source=m.source,
                                      mutant_sha256=m.after_sha256, status='selected')
                        break
                if record['planted']:
                    record['mutant_replay'] = replay(evaluate(record['mutant_source'], wrong, run),
                                                      evaluate(record['mutant_source'], right, run), base_right)
                    record['caught'] = not record['mutant_replay']['passed']
                    record['catch_has_execution_failure'] = not execution_ok(
                        record['mutant_replay']['wrong_verdicts'] + record['mutant_replay']['right_verdicts'])
                    record['status'] = 'complete'
                else:
                    record['status'] = 'no_eligible_mutation'
        record['wall_s'] = time.perf_counter() - started
        P.seal(out / 'selection.json', record)
        return record
    except Exception as e:
        record.update(status='failed', error=f'{type(e).__name__}: {e}', wall_s=time.perf_counter() - started)
        P.seal(out / 'failure.json', record)
        raise


def final(selection: Path, final_pool: Path, out: Path, *, run=S.run) -> dict:
    """Final cases enter only after the selected source and replay result are sealed."""
    record = P.unseal(selection)
    if not record.get('planted') or record['status'] != 'complete':
        return {'status': 'not_evaluated', 'reason': 'no completed planted mutation'}
    if run is S.run and not S.available():
        raise S.SandboxUnavailable('the native sandbox is unavailable; nothing executed')
    stream, decision = record['placement']['stream'], record['placement']['decision']
    for text_key, hash_key in (('baseline_source', 'baseline_sha256'), ('mutant_source', 'mutant_sha256')):
        if source_sha(record[text_key]) != record[hash_key]:
            raise ValueError('selected source digest mismatch')
    pool = P.load(final_pool, stream, 'final')
    if set(record['input_keys']) & {P.case_key(c) for c in pool['cases']}:
        raise ValueError('final cases overlap selection or replay inputs')
    cases = [c for c in pool['cases'] if c['decision'] == decision]
    # Bind final inputs before execution and refuse repeated evaluation/overwriting.
    out.mkdir(parents=True, exist_ok=False)
    P.seal(out / 'inputs.json', {'selection_sha256': P.digest(record), 'final_sha256': P.digest(pool)})
    base, mutant = evaluate(record['baseline_source'], cases, run), evaluate(record['mutant_source'], cases, run)
    known = [b['status'] == m['status'] == 'ok' for b, m in zip(base, mutant)]
    policy = base if record['caught'] else mutant
    result = {'schema': 1, 'status': 'complete' if all(known) else 'execution_incomplete',
              'development': True, 'placement': record['placement'], 'final_sha256': P.digest(pool),
              'selection_sha256': P.digest(record), 'cases': len(cases), 'paired_valid_cases': sum(known),
              'baseline_wrong_accepted': sum(b['accepted'] for b, k in zip(base, known) if k),
              'mutant_wrong_accepted': sum(m['accepted'] for m, k in zip(mutant, known) if k),
              'added_wrong_accepted': sum(not b['accepted'] and m['accepted'] for b, m, k in zip(base, mutant, known) if k),
              'removed_wrong_accepted': sum(b['accepted'] and not m['accepted'] for b, m, k in zip(base, mutant, known) if k),
              'monitored_wrong_accepted': sum(p['accepted'] for p, k in zip(policy, known) if k),
              'unmonitored_wrong_accepted': sum(m['accepted'] for m, k in zip(mutant, known) if k),
              'caught': record['caught'], 'baseline_verdicts': base, 'mutant_verdicts': mutant,
              'interpretation': 'paired static case counterfactual; not flight outcomes or independent arms'}
    P.seal(out / 'final.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    select_parser = subs.add_parser('select')
    for name in ('placement', 'eligibility', 'replay', 'check', 'out'):
        select_parser.add_argument('--' + name, type=Path, required=True)
    select_parser.add_argument('--rewrite', type=int, required=True)
    final_parser = subs.add_parser('final')
    for name in ('selection', 'final-pool', 'out'):
        final_parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'select':
        placement = json.loads(args.placement.read_text())
        store = H.CaseStore(args.replay, verifier=O.verify, verifier_name=O.VERIFIER_NAME, development=True)
        wrong, right, missing = store.replay_cases(placement['stream'], placement['decision'], args.rewrite)
        if missing:
            raise ValueError(f'replay snapshot incomplete: {missing}')
        # replay_cases() intentionally returns only sandbox payloads. Preserve the typed
        # records separately for this harness's source/split/provenance checks.
        full = [c for c in store.cases(placement['stream'], decision=placement['decision'])
                if c['split'] in ('stored_right', 'stored_wrong', 'confirmed_wrong')
                or (c['split'] == 'fresh_wrong' and c['origin'].get('rewrite') == args.rewrite)]
        result = select(args.out, placement, args.check.read_text(), FROZEN_C0[placement['decision']], args.eligibility,
                        full, rewrite=args.rewrite)
    else:
        result = final(args.selection, args.final_pool, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'planted', 'caught') if k in result}))


if __name__ == '__main__':
    main()
