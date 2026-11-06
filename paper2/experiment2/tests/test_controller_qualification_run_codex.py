"""Invented short worlds/toy pilots and clocks, never reserved controller draws."""
from copy import deepcopy
import json
import pytest
import paper2.experiment2.adaptive_controller.qualification_run as R; import paper2.experiment2.adaptive_controller.qualification as Q
import paper2.experiment2.binding_preflight_codex as B
import telohelm.taskpacks.tanker.world.episode as E
from dataclasses import replace


class Clock:
    cpu = 0
    wall = 1
    def cpu_now(self): return self.cpu
    def wall_now(self): return self.wall


class Toy:
    def __init__(self, clock):
        self.clock, self.cpu_ns, self.trace = clock, 0, []
    def reset(self, briefing):
        self.clock.cpu += 2
    def act(self, bundle):
        self.clock.cpu += 3
        action = {'goal_version': 1, 'heading_deg': 0}
        self.trace.append({'t': bundle['t'], 'actions': action})
        return action


def cases():
    scenario = B.scenario('invented-budget', 1, False)
    scenario = replace(scenario, goals=((0, E.Goal(1, 'SHELL 1', 5., 2)),))
    return [('invented-A', scenario), ('invented-B', replace(scenario, episode_id='invented-B'))]


def execute(tmp_path, *, clock=None, budget=100, stop=1000, factory=None):
    clock = clock or Clock()
    data = cases()
    plan = R.make_plan(data, cells=Q.policy_grid()[:1], budget_cpu_ns=budget, stop_at_unix_ns=stop)
    result = R.run(plan, data, tmp_path / 'run', expected_plan_sha256=plan['body_sha256'],
                   factory=factory or (lambda *args: Toy(clock)),
                   cpu_clock=clock.cpu_now, wall_clock=clock.wall_now)
    return result


def test_complete_cases_use_shared_runner_and_keep_full_records_and_trace(tmp_path):
    result = execute(tmp_path)
    assert result['all_completed'] is True and result['unattempted'] == []
    assert result['budget']['cpu_ns'] == 16  # inclusive fake clock, not nested counters
    for item in result['records']:
        folder = tmp_path / 'run' / item['records_directory']
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['outcome']['actions'] and summary['complete'] is True
        assert len(json.loads((folder / 'controller-trace.json').read_text())) == 2
        assert item['summary_sha256'] and item['controller_trace_sha256']
    schedule = [json.loads(line) for line in (tmp_path / 'run/schedule.jsonl').read_text().splitlines()]
    assert [r['kind'] for r in schedule] == ['plan', 'case-start', 'case-result', 'case-start', 'case-result', 'campaign-end']
    assert result['scientific_admission'] is False


def test_budget_stops_inside_episode_without_applying_over_budget_action(tmp_path):
    result = execute(tmp_path, budget=7)
    assert result['all_completed'] is False and len(result['unattempted']) == 1
    item = result['records'][0]
    assert item['status'] == 'censored' and item['censor_reason'] == 'CPU-ceiling'
    assert result['budget']['cpu_ns'] == 8 and result['budget']['overshoot_ns'] == 1
    raw = json.loads((tmp_path / 'run/case-00000/summary.json').read_text())
    assert raw['status'] == 'policy_failed' and raw['action_steps'] == 1
    assert raw['outcome'] is None and 'QualificationStop' in item['raw_runner_error']
    assert len(json.loads((tmp_path / 'run/case-00000/controller-trace.json').read_text())) == 2


def test_exact_deadline_prevents_any_policy_or_episode(tmp_path):
    result = execute(tmp_path, stop=1, factory=lambda *a: pytest.fail('must not start'))
    assert result['records'] == [] and len(result['unattempted']) == 2
    assert result['budget']['stop_reason'] == 'absolute-deadline'


def test_clock_unknown_or_regression_stops_and_retains_known_lower_subtotal():
    clock = Clock()
    budget = R.Budget(100, 1000, clock.cpu_now, clock.wall_now)
    budget.check(); clock.cpu = 8; budget.check(); clock.cpu = 7
    with pytest.raises(R.QualificationStop): budget.check()
    assert budget.record()['cpu_ns'] is None and budget.record()['known_cpu_lower_ns'] == 8
    for bad in [None, True, -1, 1.5]:
        b = R.Budget(100, 1000, lambda: bad, lambda: 1)
        with pytest.raises(R.QualificationStop): b.check()
        assert b.record()['cpu_ns'] is None
    b = R.Budget(100, 1000, lambda: 1, lambda: 3)
    b.check(); b.wall_clock = lambda: 2
    with pytest.raises(R.QualificationStop): b.check()


def test_changed_anchor_source_scenario_or_plan_is_rejected_before_work(tmp_path):
    data = cases()
    plan = R.make_plan(data, cells=Q.policy_grid()[:1], stop_at_unix_ns=1000)
    with pytest.raises(ValueError, match='fingerprint'):
        R.run(plan, data, tmp_path / 'wrong', expected_plan_sha256='0' * 64)
    changed = deepcopy(plan); changed['entries'][0]['scenario_sha256'] = '0' * 64
    with pytest.raises(ValueError): R.run(changed, data, tmp_path / 'changed', expected_plan_sha256=plan['body_sha256'])
    changed_cases = [(data[0][0], replace(data[0][1], episode_id='changed')), data[1]]
    with pytest.raises(ValueError, match='frozen plan'):
        R.run(plan, changed_cases, tmp_path / 'scenario', expected_plan_sha256=plan['body_sha256'])
    assert not list(tmp_path.iterdir())


def test_refuse_reuse_or_overwrite_and_distinguish_real_policy_failure(tmp_path):
    clock = Clock(); obj = Toy(clock)
    result = execute(tmp_path, clock=clock, factory=lambda *a: obj)
    assert [r['status'] for r in result['records']] == ['completed', 'failed']
    assert 'fresh controller' in result['records'][1]['raw_runner_error']
    with pytest.raises(FileExistsError): execute(tmp_path)


def test_real_failure_is_not_reclassified_when_later_clock_sample_hits_budget(tmp_path):
    clock = Clock()
    class Broken(Toy):
        def act(self, bundle):
            self.clock.cpu = 100
            raise ValueError('invented policy bug')
    result = execute(tmp_path, clock=clock, budget=100, factory=lambda *a: Broken(clock))
    assert result['records'][0]['status'] == 'failed'
    assert 'invented policy bug' in result['records'][0]['raw_runner_error']
    assert result['budget']['stop_reason'] == 'CPU-ceiling'
