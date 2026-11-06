"""The packaged replay must exercise real qualification and source use, without a server."""
import json
from pathlib import Path
import socket

import pytest

from telohelm import demo


def test_replay_qualifies_before_later_source_use_without_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('the saved-reply demonstration must not contact a model server')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    out = tmp_path / 'replay'
    result = demo.run(out)
    assert result['passed'], result['checks']
    assert len(result['learning_charges']) == result['replayed_calls'] == 3
    assert result['episodes'][0]['procedure_uses'] == {}
    assert result['episodes'][1]['joined'] and not result['episodes'][1]['report_correct']
    assert result['episodes'][2]['goal_met']
    calls = json.loads((out / 'replayed-calls.json').read_text())
    assert len({r['request_id'] for r in calls}) == 3
    events = [json.loads(line)['event'] for line in (out / 'learning/learning-events.jsonl').read_text().splitlines()]
    assert any(e['kind'] == 'candidate_refused' and e['stage'] == 'format' for e in events)
    passed = [e for e in events if e['kind'] == 'attempt_result' and e['passed']]
    assert [e['decision'] for e in passed] == ['heading', 'report']
    for event in passed:
        assert event['detail']['trial'] and all(t['component']['passed'] for t in event['detail']['trial'])
        for trial in event['detail']['trial']:
            assert trial['component']['used_seconds'] > 0
            assert Path(trial['evidence_dir'], 'result.json').is_file()
    assert all(c['tokens_in'] is None and c['tokens_out'] is None for c in result['learning_charges'])


def test_existing_output_is_refused_without_overwriting_any_record(tmp_path):
    sentinel = tmp_path / 'sentinel'
    sentinel.write_text('retain')
    with pytest.raises(FileExistsError):
        demo.run(tmp_path)
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == 'retain'


def test_changed_saved_reply_is_refused_before_candidate_execution(tmp_path):
    path = tmp_path / 'changed.json'
    path.write_text('{"replies": ["a different program"]}')
    with pytest.raises(ValueError, match='fingerprint'):
        demo._fixture(path)
