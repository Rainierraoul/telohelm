"""D8 boundary controls on invented public inputs and stand-ins, without flights."""
from copy import deepcopy
from types import SimpleNamespace
import json
import math

import pytest

import telohelm.examples as examples; import telohelm.learning as learning
from telohelm.runtime.library import Library
from paper2.experiment2.adaptive_controller import EpisodeEstimator
from .test_adaptive_controller_codex import briefing; from .test_adaptive_controller_codex import bundle
import telohelm.taskpacks.tanker.world.phase5 as phase5
from telohelm.taskpacks.tanker.world.pilot import OrbitModel


def test_scored_examples_profile_aliases_feedback_and_previous_trial_identifiers(tmp_path):
    co = learning.Coordinator(Library(tmp_path / 'library'), tmp_path, None, lambda _: True, None, examples=True)
    episode = 's4301-p5-e00'
    co.completed = [learning.Completed(1, episode, None, False, 'report', True)]
    co.history = [{'attempt': 1, 'why': [f'procedure failed in trial episode {episode}']}]
    co._attempt_examples = ([], [])
    messages = co.messages([{'episode': episode, 'failed': 'report'}])
    user = json.loads(messages[1][1])
    assert episode not in messages[1][1]
    assert user['feedback'][0]['episode'] == examples.alias(episode)
    assert examples.alias(episode) in user['previous_attempts'][0]['why'][0]


def test_controller_keeps_change_and_no_change_without_using_hidden_extras():
    class Hidden:
        def __bool__(self):
            raise AssertionError('hidden event was used')

        def __float__(self):
            raise AssertionError('hidden event was used')

    b = briefing()
    b.update(phase=Hidden(), seed=Hidden(), scenario=Hidden(), future=Hidden())
    estimator = EpisodeEstimator()
    estimator.reset(b)
    observation = bundle(0)
    observation.update(phase=Hidden(), seed=Hidden(), scenario=Hidden(), future=Hidden())
    estimator.update(observation)
    assert {round(h['change_nm'], 6) for h in estimator.forecast('A', 180)} == {-1.6, 0.0, 1.6}


def test_reference_forecast_is_a_point_model_not_a_retained_hypothesis_set():
    # A diagnostic of the current model, not clearance of D8's retain-both clause.
    model = OrbitModel(deepcopy(briefing()['tankers'][0]))
    assert model.inferred_delta() == 0.0
    x, y = model.position(180.0)
    assert math.hypot(x - 65, y - 47) == pytest.approx(8.0)


def test_witness_wrapper_passes_only_public_inputs_to_its_policy(monkeypatch):
    hidden_scenario = object()
    public_brief = {'only': 'public briefing'}
    public_bundle = {'only': 'public bundle'}
    calls = []

    class World:
        def __init__(self, scenario):
            assert scenario is hidden_scenario
            self.record = SimpleNamespace(joined_at=1)

        def reset(self):
            return {'briefing': public_brief, 'bundle': public_bundle}

        def step(self, action):
            assert action == {'public action': True}
            return None

        def outcome(self):
            return {'goal_met': True}

    class Policy:
        def __init__(self, brief, **kwargs):
            assert brief is public_brief
            calls.append('briefing')

        def act(self, observed):
            assert observed is public_bundle
            calls.append('bundle')
            return {'public action': True}

    monkeypatch.setattr(phase5.E, 'Episode', World)
    monkeypatch.setattr(phase5, 'ReferencePilot', Policy)
    result = phase5.fly(hidden_scenario, ask_help=True, max_steps=1)
    assert result['goal_met'] and calls == ['briefing', 'bundle']
