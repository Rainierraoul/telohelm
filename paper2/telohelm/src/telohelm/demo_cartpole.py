"""Exercise the unchanged learning core on Gymnasium, with authored replies by default.

Run: python -m telohelm.demo_cartpole --out NEW_DIRECTORY
An explicit --server-url selects a compatible local server instead. Both are
development examples; neither establishes benchmark performance or weight training.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from .demo import _write
from .taskpacks.cartpole import CARTPOLE_PACK, GUARD, GYMNASIUM_VERSION, Scenario

SOURCES = (
    "def decide(ctx):\n    return {'push': 0}\n",
    "def decide(ctx):\n    return {'push': 1 if ctx.pole_angle + 0.5 * ctx.pole_angular_velocity > 0 else 0}\n")


def run(out, *, server=None):
    if server is not None:
        from .demo_server import ServerConfig
        if not isinstance(server, ServerConfig):
            raise TypeError('server must be a validated ServerConfig')
    from .learning import Coordinator, GatewayProposer
    from .runtime.broker import Broker
    from .runtime.episode_loop import NoModels, run_episode
    from .runtime.ledger import Ledger
    from .runtime.library import Library
    from .runtime.models import ModelResponse, ScriptedProvider
    from .runtime.trusted_log import TrustedLog, verify

    if importlib.metadata.version('gymnasium') != GYMNASIUM_VERSION:
        raise RuntimeError('Install the pinned cartpole extra first')
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory; records are never overwritten')
    out.mkdir(parents=True)
    started = time.monotonic()
    pack = CARTPOLE_PACK
    library = Library(out / 'library', pack=pack.public)
    guard = library.add({'kind': 'check', 'decision': 'control', 'conditions': [], 'source': GUARD,
                         'origin': {'trusted_control_guard': 'designer-supplied integer action schema'}})
    library.set_status(guard, 'adopted', 'supplied action-format guard, not learned')
    work = out / 'learning'
    work.mkdir()
    condition = 'authored-reply-demo' if server is None else 'server-development'
    _write(out / 'run-config.json', {'mode': condition, 'server': server.record() if server else None,
                                    'inference_verified': False})
    receipts = []

    def reply(request):
        number = len(receipts)
        source = SOURCES[number] if number < len(SOURCES) else None
        text = json.dumps({'kind': 'procedure', 'decision': 'control', 'conditions': [], 'source': source,
                           'reason': {'code': 'missing_tool', 'text': 'Authored software control'}}) if source else ''
        receipts.append({'request_id': request.request_id, 'fingerprint': request.fingerprint,
                         'source_sha256': hashlib.sha256(source.encode()).hexdigest() if source else None,
                         'response_sha256': hashlib.sha256(text.encode()).hexdigest(), 'real_model_call': False})
        _write(out / 'authored-calls.json', receipts)
        return ModelResponse('answered' if text else 'failed', text)

    if server is None:
        proposer = GatewayProposer('http://127.0.0.1:1', 'Qwen/Qwen3.8-27B', work, 'cartpole-demo',
                                   condition=condition, input_cap=16384)
        proposer.provider = ScriptedProvider(reply)
    else:
        from .demo_server import proposer as server_proposer
        proposer = server_proposer(work, server)
    coordinator = Coordinator(library, work, proposer, lambda scenario: True, None,
                              attempt_every=1, trial_set=1, max_attempts=1,
                              procedures_only=True, candidates=2 if server is None else 1, flight_pack=pack)
    rows = []
    withdrawal = {}
    scenarios = (('baseline', 11), ('later', 12), ('withdrawn-control', 12)) if server is None else (
        ('baseline', 11), ('later-1', 23), ('later-2', 24))
    for name, seed in scenarios:
        if name == 'withdrawn-control':
            sequence = library.projection()['seq']
            before = library.adopted('control', 'procedure')
            for identity in before:
                library.set_status(identity, 'retired', 'host-requested negative control, not autonomous recovery')
            withdrawal = {'prior_sequence': sequence, 'withdrawn': before,
                          'past_projection': library.projection(sequence), 'current_projection': library.projection()}
            _write(out / 'withdrawal.json', withdrawal)
        scenario = Scenario(name, seed)
        folder = out / name
        folder.mkdir()
        ledger = Ledger()
        log = TrustedLog(folder / 'episode.log', folder / 'episode.heads', condition)
        broker_log = TrustedLog(folder / 'broker.log', folder / 'broker.heads', condition)
        broker = Broker(name, broker_log, ledger, None, condition=condition, stream='cartpole-demo',
                        goal=pack.goal, tools={})
        try:
            result = run_episode(scenario, library, NoModels(), log, ledger, max_steps=128, broker=broker, pack=pack)
        finally:
            log.close()
            broker_log.close()
        outcome = result['outcome']
        outcome['model_calls'] = len(ledger.calls)
        _write(folder / 'outcome.json', outcome)
        _write(folder / 'scenario.json', asdict(scenario))
        decisions = [json.loads(line)['payload'] for line in (folder / 'episode.log').read_text().splitlines()]
        used = [e for e in decisions if e.get('kind') == 'decision' and e.get('source') == 'procedure']
        row = {'episode': name, 'seed': seed, 'goal_met': outcome['goal_met'], 'steps': outcome['action_steps'],
               'end_reason': outcome['end_reason'], 'procedure_steps': len(used),
               'sources_used': sorted({hashlib.sha256(library.entry(e['method'])['source'].encode()).hexdigest() for e in used}),
               'real_model_calls': 0}
        rows.append(row)
        _write(out / 'episodes.json', rows)
        if name == 'baseline':
            coordinator.after_episode(scenario, outcome, complete=outcome['end_reason'] is not None)
        print(f"{name}: goal={row['goal_met']}, steps={row['steps']}, procedure steps={len(used)}", flush=True)

    trials = [e for e in coordinator.events if e['kind'] == 'attempt_result']
    expected_source = hashlib.sha256(SOURCES[1].encode()).hexdigest() if server is None else None
    chains = {}
    for heads in out.rglob('*heads'):
        path = heads.with_name(heads.name[:-5] + 'log')
        if path.exists():
            verdict = verify(path, heads)
            if not verdict.get('ok'):
                chains[str(path.relative_to(out))] = verdict
    historical = withdrawal.get('past_projection', {}).get('entries', {})
    checks = {'baseline_failed_naturally': not rows[0]['goal_met'] and rows[0]['end_reason'] == 'fell',
              'wrong_candidate_failed_trial': len(trials) == 2 and not trials[0]['passed'],
              'controller_passed_ordinary_trial': len(trials) == 2 and trials[1]['passed'],
              'later_exact_source_survived': rows[1]['goal_met'] and rows[1]['procedure_steps'] == 128
                  and rows[1]['sources_used'] == [expected_source],
              'withdrawal_removed_control': bool(withdrawal.get('withdrawn')) and not rows[2]['goal_met']
                  and rows[2]['procedure_steps'] == 0,
              'past_projection_retains_adoption': bool(withdrawal.get('withdrawn')) and
                  all(historical[s]['status'] == 'adopted' for s in withdrawal['withdrawn']),
              'protected_chains_valid': not chains}
    if server is not None:
        adopted = library.adopted('control', 'procedure')
        hashes = [hashlib.sha256(library.entry(s)['source'].encode()).hexdigest() for s in adopted]
        checks = {'baseline_failed_naturally': checks['baseline_failed_naturally'],
                  'candidate_passed_ordinary_trial': len(trials) == 1 and trials[0]['passed'] and len(adopted) == 1,
                  'both_later_episodes_used_adopted_source': len(hashes) == 1 and all(
                      r['goal_met'] and r['procedure_steps'] == 128 and r['sources_used'] == hashes for r in rows[1:]),
                  'protected_chains_valid': not chains}
    summary = {'passed': all(checks.values()), 'checks': checks, 'episodes': rows,
               'gymnasium': GYMNASIUM_VERSION, 'task_contract': pack.public.fingerprint({}),
               'real_model_calls': 0 if server is None else None, 'model_token_demand': None,
               'server_chat_requests': len(proposer.ledger.calls) if server is not None else 0,
               'scope': condition, 'inference_verified': False,
               'learning_charges': [asdict(c) for c in proposer.ledger.calls], 'chain_errors': chains,
               'elapsed_seconds': time.monotonic() - started,
               'limitations': 'Two authored replies and two fixed seeds. No new model learning, benchmark score, '
                   'cost benefit, generalisation, in-episode recovery or real-time latency claim. One tick is one '
                   '0.02-second simulator step. Retirement is an explicit host negative control.' if server is None else
                   'One server request and two fixed later seeds. HTTP delivery does not prove real inference, '
                   'checkpoint, hardware or declared capacity. Development adapter is unreportable; observed usage '
                   'stays in learning_charges, with missing values unknown. No benchmark, cost-benefit, generalisation '
                   'or in-episode model-control claim. One tick is one 0.02-second simulator step.'}
    _write(out / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--server-url', help='Explicit http://numeric-loopback:port; default is offline authored replies')
    parser.add_argument('--context-window', type=int, help='Declared server capacity; required in server mode')
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--max-output-tokens', type=int, default=131072)
    parser.add_argument('--input-cap', type=int, default=16384)
    parser.add_argument('--timeout-s', type=float, default=14400)
    parser.add_argument('--effort', choices=('low', 'medium', 'xhigh'), default='xhigh')
    args = parser.parse_args(argv)
    server = None
    if args.server_url is None and (args.context_window is not None or args.model != 'Qwen/Qwen3.8-27B'
            or args.max_output_tokens != 131072 or args.input_cap != 16384
            or args.timeout_s != 14400 or args.effort != 'xhigh'):
        parser.error('Server settings require an explicit --server-url')
    if args.server_url is not None:
        if args.context_window is None:
            parser.error('--context-window is required with --server-url')
        from .demo_server import ServerConfig
        try:
            server = ServerConfig(args.server_url, args.context_window, model=args.model,
                                  max_output_tokens=args.max_output_tokens, input_cap=args.input_cap,
                                  timeout_s=args.timeout_s, effort=args.effort)
        except ValueError as exc:
            parser.error(str(exc))
    summary = run(args.out, server=server)
    print('Second-task development checks: ' + ('passed' if summary['passed'] else 'FAILED'))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
