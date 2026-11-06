"""A headless Gymnasium task connection. No tanker state or derived control features.

One runtime tick is one simulator step (0.02 simulated seconds). This example has
no delayed flight model and does not validate the runtime's model-latency units.
"""
from dataclasses import dataclass
import math
from types import SimpleNamespace

from ..contracts import PublicContract, TaskPack
from ..contracts.decisions import DecisionRoles, HelpEvidence
from ..contracts.goal_parts import GoalPart, GoalParts
from ..runtime.environment import Environment

GYMNASIUM_VERSION = '1.4.0'
TARGET_STEPS = 128
READINGS = ('cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity')
GUARD = """def check(ctx, out):
    try:
        push = out.push
    except Exception:
        return 'reject'
    if isinstance(push, bool) or not isinstance(push, int) or push not in (0, 1):
        return 'reject'
    return 'accept'
"""


def part_error(part):
    push = part.get('push')
    return None if type(push) is int and push in (0, 1) else 'push must be integer 0 or 1'


@dataclass(frozen=True)
class Scenario:
    episode_id: str
    seed: int


class ContextBuilder:
    def __init__(self, briefing):
        self.goal = dict(briefing['goal'])

    def update(self, bundle):
        # Copy the four native readings without selecting or deriving features.
        return {**bundle, 'goal_version': 1, 'goal': dict(self.goal)}


class Episode:
    """Thin adapter: native termination is failure, the fixed time limit is success."""
    def __init__(self, scenario):
        import gymnasium as gym
        if gym.__version__ != GYMNASIUM_VERSION:
            raise RuntimeError(f'Install the cartpole extra with Gymnasium {GYMNASIUM_VERSION}')
        self.scenario = scenario
        self.env = gym.make('CartPole-v1', max_episode_steps=TARGET_STEPS, render_mode=None)
        self.t, self.push = 0, 0  # declared constant-left fallback when no procedure acts
        self.terminated = self.truncated = False
        self.record = SimpleNamespace(actions=[])
        self.refusals = []
        self.native_steps = []

    def _bundle(self, observation):
        values = [float(v) for v in observation]
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            raise ValueError('CartPole returned invalid observations')
        return {'t': self.t, **dict(zip(READINGS, values)), 'observation_age_s': 0,
                'step_duration_s': 0.02}

    def reset(self):
        if self.t or self.record.actions:
            raise RuntimeError('Create another episode rather than overwriting this record')
        observation, _ = self.env.reset(seed=self.scenario.seed)
        return {'briefing': {'goal': {'version': 1, 'survive_steps': TARGET_STEPS},
                            'fallback_push': 0, 'step_duration_s': 0.02},
                'bundle': self._bundle(observation)}

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError('The episode has ended')
        accepted = (not action or (set(action) <= {'push', 'goal_version'}
                    and part_error(action) is None and action.get('goal_version', 1) == 1))
        if accepted and 'push' in action:
            self.push = action['push']
        elif not accepted:
            self.refusals.append({'t': self.t, 'reason': 'invalid control action'})
        observation, reward, self.terminated, self.truncated, _ = self.env.step(self.push)
        self.record.actions.append({'actions': dict(action), 'accepted': accepted})
        self.native_steps.append({'t': self.t, 'push_applied': self.push,
                                  'observation_after': [float(v) for v in observation],
                                  'reward': float(reward), 'terminated': bool(self.terminated),
                                  'truncated': bool(self.truncated)})
        self.t += 1
        if self.terminated or self.truncated:
            self.env.close()
            return None
        return self._bundle(observation)

    def outcome(self):
        balanced = bool(self.truncated and not self.terminated and self.t == TARGET_STEPS)
        ended = self.terminated or self.truncated
        return {'goal_met': balanced, 'balanced': balanced, 'settled': True,
                'end_reason': ('fell' if self.terminated else 'survived') if ended else None,
                'ended_at': self.t, 'action_steps': len(self.record.actions), 'alternatives': [],
                'goal': {'version': 1, 'survive_steps': TARGET_STEPS}, 'help_requests': [],
                'refusals': list(self.refusals), 'terminated': bool(self.terminated),
                'truncated': bool(self.truncated), 'native_steps': list(self.native_steps)}


def component_trial(decision, sha, outcome, records):
    refused = {r['t'] for r in outcome.get('refusals', [])}
    used = [r for r in records if r.get('decision') == decision == 'control'
            and r.get('method') == sha and r.get('goal_version') == 1
            and part_error(r.get('part') or {}) is None and r['t'] not in refused]
    passed = bool(sha and {r['t'] for r in used} == set(range(TARGET_STEPS))
                  and outcome.get('goal_met') is True
                  and outcome.get('settled') is True and outcome.get('end_reason') == 'survived'
                  and not outcome.get('terminated') and outcome.get('truncated') is True
                  and outcome.get('action_steps') == TARGET_STEPS
                  and not outcome.get('refusals') and not outcome.get('help_requests'))
    return {'passed': passed, 'candidate': sha, 'used_steps': len(used),
            'reason': None if passed else 'did not survive with accepted candidate control'}


def tools_text(limits):
    return 'No tools are available. Read the named observations directly.'


PUBLIC = PublicContract(
    pack_id='cartpole-demo', schema_version=1, decisions=('control',),
    decision_fields={'control': frozenset({'push'})}, pair_decisions=(),
    context_fields=frozenset({'t', 'goal_version', 'goal', 'observation_age_s', 'step_duration_s', *READINGS}),
    output_fields=frozenset({'push'}),
    learner_purpose='balance a pole by pushing its cart left or right',
    learner_actions='control returns {"push": 0} for left or {"push": 1} for right; integers only.',
    learner_goal='Survive 128 steps without native CartPole termination. Each step is 0.02 simulated seconds.',
    learner_tools=tools_text,
    learner_trial_rule='A control procedure must actually act and survive the fixed trial without refusals or help.',
    learner_attribute_example='ctx.pole_angle and ctx.pole_angular_velocity in a procedure, and out.push in a check',
    learner_decision_choices='"control"',
    flight_system='Choose a left or right push from the four native CartPole observations.',
    flight_ask={'control': 'Return {"push": 0} or {"push": 1} to balance the pole.'})

CARTPOLE_PACK = TaskPack(
    public=PUBLIC, goal=GoalParts((GoalPart('balanced', 'fell', {'control'}),), 'not_balanced'),
    roles=DecisionRoles(('control',), 'control', (), {'control': READINGS}),
    help=HelpEvidence('observation_age_s', 0.1, 'The simulator observation is stale.'),
    part_error=part_error, context_builder=ContextBuilder,
    environment=Environment(PUBLIC.pack_id, PUBLIC.schema_version, Episode, component_trial, TARGET_STEPS, tools={}))
