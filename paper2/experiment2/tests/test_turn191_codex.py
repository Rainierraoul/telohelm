from pathlib import Path
"""Failed-pair feedback, store-rule admission and model/tool accounting.

Invented components only; no model server, generated code or new world flight.
"""

import telohelm.learning_monitor as GM  # replay qualification moved to the monitor
import json
import subprocess
from types import SimpleNamespace

import pytest

import telohelm.learning as G; import paper2.experiment2.outcomecases as O; import paper2.experiment2.run_stream as R
import telohelm.qualification.store as H
from telohelm.runtime.ledger import Call
from telohelm.runtime.library import Library
from .test_option_c import CHECK; from .test_option_c import HEADING; from .test_option_c import REPORT; from .test_option_c import guarded


@pytest.mark.parametrize('trial_passed', [False, True])
def test_no_replay_pair_keeps_trial_verdict_and_feedback(tmp_path, monkeypatch, trial_passed):
    lib, co, hg, rg = guarded(tmp_path, no_replay=True)
    pair = {'kind': 'pair', 'decision': 'heading',
            'procedure': {'conditions': [], 'source': 'def decide(ctx):\n    return {"heading_deg": 10}\n'},
            'check': {'conditions': [], 'source': CHECK},
            'reason': {'code': 'missing_tool', 'text': 'invented'}}
    co.propose = lambda *_: json.dumps(pair)
    co.fly = lambda *_: {'episode': 'invented', 'goal_met': trial_passed, 'help': 0}
    monkeypatch.setattr(GM, 'replay_test', lambda *a, **k: pytest.fail('ablation ran replay'))
    for i in range(5):
        co.after_episode(SimpleNamespace(episode_id=f'e{i}'),
                         {'goal_met': False, 'joined': False, 'report_correct': False}, True)
    result = next(e for e in co.events if e['kind'] == 'attempt_result')
    assert result['passed'] is trial_passed
    assert result['detail']['check']['replay']['skipped'] is True
    if trial_passed:
        assert co.history[-1]['adopted'] is True
        assert result['candidate'] in lib.adopted('heading', 'procedure')
        # A later wrong answer is recorded, but does not run replay or retire.
        co.confirm_wrong({'context': {}, 'output': {'heading_deg': 10}})
        assert result['paired_check'] in lib.adopted('heading', 'check')
    else:
        assert co.history[-1]['refused_at'] == 'trial'
        assert result['candidate'] not in lib.adopted('heading', 'procedure')
        assert result['paired_check'] not in lib.adopted('heading', 'check')
    assert hg in lib.adopted('heading', 'check') and rg in lib.adopted('report', 'check')


@pytest.mark.parametrize('rules', [['dev.reference.v1'], [O.RULE, 'dev.reference.v1']])
def test_legacy_and_mixed_rule_stores_stop_before_flight(tmp_path, monkeypatch, rules):
    class ExistingStore:
        def cases(self, *args):
            return [{'label': {'rule': r}} for r in rules]
    monkeypatch.setattr(H, 'CaseStore', lambda *a, **k: ExistingStore())
    monkeypatch.setattr(R, 'run_episode', lambda *a, **k: pytest.fail('wrong-rule flight ran'))
    with pytest.raises(ValueError, match='labelled only by'):
        R.run_stream(1, 1, tmp_path / 'run', stream_seed=4401, empty_library=True,
                     trusted_heading_guard=HEADING, trusted_report_guard=REPORT,
                     dev_cases=tmp_path / 'store', learn_checks=True,
                     learn={'base_url': 'http://127.0.0.1:1', 'model': 'unused'})


@pytest.mark.parametrize('model_counts', [[], [(12, 3)], [(12, 3), (None, 4)]])
def test_summary_counts_models_and_preserves_tools_in_ledger(tmp_path, monkeypatch, model_counts):
    def loop(scenario, library, models, log, ledger, **kw):
        ledger.add(Call('tool', 'prediction_tool', scenario.episode_id, 0, 0, .1, None, None, 'answered'))
        for i, (tin, tout) in enumerate(model_counts):
            ledger.add(Call(f'model{i}', 'small_model' if i == 0 else 'reasoning_model',
                            scenario.episode_id, 0, 0, .1, tin, tout, 'answered'))
        return {'outcome': {'goal_met': False, 'joined': False, 'report_correct': False,
                           'end_reason': 'horizon', 'ended_at': 1, 'action_steps': 1}, 'climbs': 0}
    monkeypatch.setattr(R.L, 'run_episode', loop)
    monkeypatch.setattr(R.A, 'derive', lambda *a, **kw: {'shares': {}, 'K': 1, 'complete': True})
    result = R.run_episode(SimpleNamespace(episode_id='invented'), Library(tmp_path / 'library'),
                           tmp_path / 'run', 'invented', 1, {})['record']
    assert result['model_calls'] == len(model_counts) and result['tool_calls'] == 1
    unknown = any(a is None or b is None for a, b in model_counts)
    assert result['calls_with_unknown_usage'] == int(unknown)
    assert result['tokens_in'] == (None if unknown else sum(a for a, _ in model_counts))
    assert result['tokens_out'] == sum(b for _, b in model_counts)
    rows = [json.loads(l) for l in (tmp_path / 'run/invented/ledger.jsonl').read_text().splitlines()]
    assert len(rows) == len(model_counts) + 1 and rows[0]['kind'] == 'prediction_tool'


@pytest.mark.parametrize('script', ['poc-step3-stream.sbatch', 'poc-step4.sbatch'])
@pytest.mark.parametrize('intended', [0, 1])
def test_cleanup_failed_kill_does_not_overwrite_intended_exit(tmp_path, script, intended):
    source = ((next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / "dcarr-runtime-0.1/cluster") / script).read_text()
    start = source.index('cleanup()')
    body = source[start:source.index('\ntrap cleanup EXIT', start)]
    for role in ('t1', 't2'):
        (tmp_path / f'{role}-server.pid').write_text('123\n')
    # Override kill before running the exact cleanup function: no signal is sent.
    probe = 'LOGS=$1\nkill() { return 1; }\n' + body + '\nrc=$2\ncleanup\nexit "$rc"\n'
    result = subprocess.run(['bash', '-c', probe, 'probe', str(tmp_path), str(intended)], capture_output=True)
    assert result.returncode == intended
