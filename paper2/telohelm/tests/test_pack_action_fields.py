"""Selected-task field ownership, with invented replies and a short software fixture only."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from telohelm.contracts.decisions import DecisionRoles; from telohelm.contracts.decisions import HelpEvidence
from telohelm.taskpacks.tanker.public import TANKER
from telohelm.taskpacks.tanker.pack import TANKER_PACK
from telohelm.runtime.delivery import DelayedModels; from telohelm.runtime.delivery import Pending
from telohelm.runtime.episode_loop import run_episode
from telohelm.runtime.escalation import FlightLadder; from telohelm.runtime.escalation import own_part
from telohelm.runtime.ledger import Ledger
from telohelm.runtime.library import Library
from telohelm.runtime.trusted_log import TrustedLog
from .test_pack_decisions import scenario


def pending(text, version=1):
    return Pending(1, 'drive', 'small_model', 'invented', version, 0, 0, 2,
                   '0' * 64, 'answered', text, False)


def scheduler():
    return DelayedModels(None, lambda *a: 'small_model', lambda *a: [], {},
                         lambda decision, text: json.loads(text))


def test_delivery_accepts_the_selected_tasks_fields_and_keeps_legacy_default():
    p = pending('{"throttle": 0.5}')
    m = scheduler()
    assert m.part_for(p, 1) == (None, False, 'invalid_part')
    assert m.part_for(p, 1, action_keys=frozenset({'throttle'})) == ({'throttle': 0.5}, False, None)
    heading = pending('{"heading_deg": 45}')
    assert m.part_for(heading, 1) == ({'heading_deg': 45}, False, None)
    assert m.part_for(heading, 1, action_keys=frozenset({'throttle'})) == (None, False, 'invalid_part')


@pytest.mark.parametrize('reply', ['{"throttle": 1}', '{"heading_deg": 1}'])
def test_an_empty_task_field_set_does_not_fall_back_to_the_tanker(reply):
    assert scheduler().part_for(pending(reply), 1, action_keys=frozenset()) == (None, False, 'invalid_part')


def test_a_custom_task_keeps_goal_version_and_finite_json_refusals():
    m = scheduler(); keys = frozenset({'throttle'})
    assert m.part_for(pending('{"throttle": 1}', version=1), 2, action_keys=keys) == (None, True, 'out_of_date')
    assert m.part_for(pending('{"throttle": NaN}'), 1, action_keys=keys) == (None, False, 'invalid_part')


def test_a_decision_cannot_write_another_decisions_fields():
    table = {'drive': frozenset({'throttle', 'help'}), 'signal': frozenset({'lamp'})}
    part = {'throttle': 0.5, 'lamp': True, 'heading_deg': 1, 'help': {'reason': 'x'}}
    assert own_part('drive', part, fields=table) == {'throttle': 0.5, 'help': {'reason': 'x'}}
    assert own_part('heading', {'heading_deg': 1, 'report': 'x'}) == {'heading_deg': 1}


@pytest.mark.parametrize('latency', [0, 2])
def test_the_loop_delivers_and_masks_a_renamed_decision_through_the_ladder(tmp_path, latency):
    # Same short tanker world to isolate field routing, not a second-domain demonstration.
    ledger = Ledger(); sc = scenario()
    class Gateway:
        episode, condition, stream = sc.episode_id, 'invented', 'invented'
        def __init__(self):
            self.ledger, self.calls = ledger, 0
        def submit(self, role, messages, limit, version, t, step, delay):
            self.calls += 1
            return {'status': 'answered', 'request_id': f'invented-{self.calls}',
                    'available_second': t + int(delay), 'reportable': False,
                    'text': '{"heading_deg": 37, "report": "must not act"}'}
    gateway = Gateway()
    models = DelayedModels(gateway, lambda *a: 'small_model', lambda *a: [],
                           {'small_model': latency}, lambda decision, text: json.loads(text))
    fields = {'navigate': frozenset({'heading_deg', 'help'}),
              'callout': frozenset({'report', 'report_trigger_nm'})}
    public = replace(TANKER, decisions=('navigate', 'callout'), pair_decisions=('navigate', 'callout'),
                     decision_fields=fields)
    roles = DecisionRoles(order=('navigate',), flying='navigate', once_per_goal=(), needs={'navigate': ()})
    pack = replace(TANKER_PACK, public=public, roles=roles)
    log = TrustedLog(tmp_path / 'episode.log', tmp_path / 'episode.heads', 'invented')
    ladder = FlightLadder(gateway, help_evidence=HelpEvidence('t', 9999, 'invented'))
    result = run_episode(sc, Library(tmp_path / 'library'), models, log, ledger, max_steps=4,
                         ladder=ladder, pack=pack)
    decisions = [json.loads(line)['payload'] for line in (tmp_path / 'episode.log').read_text().splitlines()
                 if json.loads(line)['payload']['kind'] == 'decision']
    acted = [d for d in decisions if d.get('part')]
    assert acted and all(d['decision'] == 'navigate' for d in acted)
    assert all(d['source'] == 'small_model' and d['part'] == {'heading_deg': 37} for d in acted)
    assert min(d['t'] for d in acted) == latency
    assert not result['outcome']['reports']
