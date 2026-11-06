"""Inspect a learning cycle using saved replies, with no model server or new inference.

Run: python -m telohelm.demo --out .local-runs/learning-demo
The three short, authored episodes repeat one scenario. They are software evidence,
not measurements of learning effectiveness, inference cost or unseen-task performance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

FIXTURE_SHA256 = '3d8893d82b3dcbcada4c08946371892987b1e01e6fafce105b63701f96c2def0'


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _fixture(path=None):
    path = Path(path) if path else Path(__file__).parent / 'demo_data/learning-cycle.json'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != FIXTURE_SHA256:
        raise ValueError('Saved-reply fixture does not match its pinned fingerprint')
    return json.loads(data)


def _scenario(name):
    from .taskpacks.tanker.world import episode as E, physics as P
    return E.Scenario(name, P.ReceiverState(0, 0, -2, math.radians(10), math.radians(10),
                                          0.15, P.turn_rate(540)), (0, 0),
                      (E.Tanker('Shell', P.Orbit(0, -1, 1, 0, 0)),),
                      ((0, E.Goal(1, 'Shell', 1, 60)),))


def _guards(library):
    base = Path(__file__).parent / 'taskpacks/tanker/guards'
    for decision in ('heading', 'report'):
        path = base / f'poc-{decision}-guard.py'
        source = path.read_text()
        identity = library.add({'kind': 'check', 'decision': decision, 'conditions': [],
                                'source': source, 'origin': {f'trusted_{decision}_guard': path.name,
                                'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                'plan': 'Designer-supplied public format guard; not learned by this demo'}})
        library.set_status(identity, 'adopted', 'supplied public format guard')


def run(out):
    """Use the unchanged runtime, gateway, monitor and contained procedure execution."""
    from .learning import Coordinator, GatewayProposer
    from .runtime.broker import Broker
    from .runtime.episode_loop import NoModels, run_episode
    from .runtime.ledger import Ledger
    from .runtime.library import Library
    from .runtime.models import ModelResponse, ScriptedProvider
    from .runtime.trusted_log import TrustedLog, verify
    from .taskpacks.tanker.pack import TANKER_PACK
    from .taskpacks.tanker.prediction import predict

    saved = _fixture()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory; demo records are never overwritten')
    out.mkdir(parents=True)
    started = time.monotonic()
    work = out / 'learning'
    work.mkdir()
    library = Library(out / 'library')
    _guards(library)
    _write(out / 'fixture-provenance.json', {k:v for k,v in saved.items() if k != 'replies'})
    proposer = GatewayProposer('http://127.0.0.1:1', 'Qwen/Qwen3.8-27B', work, 'local-demo',
                               condition='saved-reply-demo', input_cap=16384)
    receipts = []

    def replay(request):
        index = len(receipts)
        text = saved['replies'][index] if index < len(saved['replies']) else ''
        receipts.append({'saved_reply_index': index if index < len(saved['replies']) else None,
                         'request_id': request.request_id, 'fingerprint': request.fingerprint,
                         'response_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'real_model_call': False, 'token_demand': None})
        _write(out / 'replayed-calls.json', receipts)
        return ModelResponse('answered' if text else 'failed', text)

    # Construction makes no request. Every actual call is handled by the marked,
    # unreportable ScriptedProvider through the ordinary gateway and cost ledger.
    proposer.provider = ScriptedProvider(replay)
    coordinator = Coordinator(library, work, proposer, lambda scenario: True, predict,
                              attempt_every=1, trial_set=1, max_attempts=2,
                              procedures_only=True, candidates=3)
    rows = []
    for number in range(1, 4):
        scenario = _scenario(f'demo-{number}')
        folder = out / scenario.episode_id
        folder.mkdir()
        ledger = Ledger()
        log = TrustedLog(folder / 'episode.log', folder / 'episode.heads', 'saved-reply-demo')
        broker_log = TrustedLog(folder / 'broker.log', folder / 'broker.heads', 'saved-reply-demo')
        broker = Broker(scenario.episode_id, broker_log, ledger, predict,
                        condition='saved-reply-demo', stream='local-demo', goal=TANKER_PACK.goal)
        try:
            result = run_episode(scenario, library, NoModels(), log, ledger, max_steps=61,
                                 broker=broker, pack=TANKER_PACK)
        finally:
            log.close()
            broker_log.close()
        outcome = result['outcome']
        outcome['model_calls'] = len(ledger.calls)
        _write(folder / 'outcome.json', outcome)
        _write(folder / 'scenario.json', asdict(scenario))
        payloads = [json.loads(line)['payload'] for line in (folder / 'episode.log').read_text().splitlines()]
        uses = {}
        for event in payloads:
            if event.get('kind') == 'decision' and event.get('source') == 'procedure':
                entry = library.entry(event['method'])
                digest = hashlib.sha256(entry['source'].encode()).hexdigest()
                uses.setdefault(event['decision'], {'entry': event['method'], 'source_sha256': digest, 'seconds': 0})
                uses[event['decision']]['seconds'] += 1
        row = {'episode': scenario.episode_id, 'goal_met': outcome['goal_met'],
               'joined': outcome['joined'], 'report_correct': outcome['report_correct'],
               'settled': outcome['settled'], 'end_reason': outcome['end_reason'],
               'ended_at': outcome['ended_at'], 'procedure_uses': uses, 'live_model_calls': 0}
        rows.append(row)
        _write(out / 'episodes.json', rows)
        coordinator.after_episode(scenario, outcome, complete=outcome['end_reason'] is not None)
        print(f"{scenario.episode_id}: goal={row['goal_met']}, join={row['joined']}, "
              f"report={row['report_correct']}, procedures={','.join(uses) or 'none'}", flush=True)

    trials = [e for e in coordinator.events if e['kind'] == 'attempt_result']
    adopted = {decision: [{'entry': s, 'source_sha256': hashlib.sha256(library.entry(s)['source'].encode()).hexdigest()}
                          for s in library.adopted(decision, 'procedure')]
               for decision in ('heading', 'report')}
    chain_errors = {}
    for heads in out.rglob('*heads'):
        log = heads.with_name(heads.name[:-5] + 'log')
        if log.exists():
            errors = verify(log, heads)
            if not errors.get('ok'):
                chain_errors[str(log.relative_to(out))] = errors
    checks = {'initial_episode_failed': not rows[0]['goal_met'],
              'both_procedures_passed_ordinary_trials': len(trials) == 2 and all(e['passed'] for e in trials),
              'later_heading_use': rows[1]['procedure_uses'].get('heading', {}).get('source_sha256') == saved['programs']['heading'],
              'later_report_use': rows[2]['procedure_uses'].get('report', {}).get('source_sha256') == saved['programs']['report'],
              'final_goal_met': bool(rows[2]['goal_met'] and rows[2]['settled']),
              'protected_chains_valid': not chain_errors}
    summary = {'scope': saved['scope'], 'checks': checks, 'passed': all(checks.values()),
               'episodes': rows, 'adopted': adopted, 'replayed_calls': len(receipts), 'real_model_calls': 0,
               'learning_charges': [asdict(c) for c in proposer.ledger.calls], 'chain_errors': chain_errors,
               'model_token_demand': None, 'cost_note': 'No inference occurred. Recorded charges are unreportable replay calls; token demand is unknown.',
               'elapsed_seconds': time.monotonic() - started,
               'limitations': 'Same short authored scenario repeated. No inference, efficacy, generalisation or in-episode recovery claim.'}
    _write(out / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New directory for all demo records')
    args = parser.parse_args(argv)
    summary = run(args.out)
    print('Software replay checks: ' + ('passed' if summary['passed'] else 'FAILED'))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
