"""Read terminal adviser timing without changing a flight record.

Control flow: index slot events by (stream, exact request-body hash); join each
ledger request once; retain gateway wall time beside slot wait and held time;
report missing records explicitly and reject ambiguous identities. The held
interval measures ownership of the serving slot, including HTTP overhead.
Synthetic controls cover a body-hash join, missing release and duplicate event.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_events(events):
    indexed = defaultdict(dict)
    for event in events:
        key = (str(event['stream']), event['body_sha256'])
        kind = event['event']
        if kind in indexed[key]:
            raise ValueError('ambiguous slot event: ' + repr((key, kind)))
        indexed[key][kind] = event
    return indexed


def join_call(call, events):
    key = (str(call['stream']).rsplit('/', 1)[-1], call['body_sha256'])
    records = events.get(key, {})
    missing = sorted({'queued', 'acquired', 'released'} - records.keys())
    result = {k: call.get(k) for k in ('request_id', 'episode', 'stream', 'status', 'started_s', 'wall_s')}
    result.update({'join_key': list(key), 'missing_events': missing, 'wait_s': None, 'held_s': None})
    if not missing:
        queued, acquired, released = (records[k] for k in ('queued', 'acquired', 'released'))
        ids = {r['wait_id'] for r in (queued, acquired, released)}
        if len(ids) != 1:
            raise ValueError('slot wait identities differ')
        wait, held = acquired['wait_s'], released['held_s']
        if wait < 0 or held < 0:
            raise ValueError('negative recorded duration')
        result.update({'wait_id': queued['wait_id'], 'wait_s': wait, 'held_s': held,
                       'slot_start_minus_ledger_start_s': queued['began_monotonic_s'] - call['started_s'],
                       'wall_minus_wait_and_held_s': call['wall_s'] - wait - held,
                       'released_outcome': released['outcome']})
    return result


def self_test():
    call = {'stream': 'study/adviser/4301', 'started_s': 9.99997, 'wall_s': 9.005, 'body_sha256': 'fixture-body'}
    events = [{'stream': '4301', 'began_monotonic_s': 10.0, 'body_sha256': 'fixture-body', 'wait_id': 'fixture', 'event': kind,
               **data} for kind, data in [('queued', {}), ('acquired', {'wait_s': 7.0}),
                                          ('released', {'held_s': 2.0, 'outcome': 'returned'})]]
    row = join_call(call, index_events(events))
    assert row['wait_s'] == 7 and row['held_s'] == 2
    assert abs(row['wall_minus_wait_and_held_s'] - .005) < 1e-9
    row = join_call(call, index_events(events[:-1]))
    assert row['missing_events'] == ['released'] and row['held_s'] is None
    try:
        index_events(events + [events[0]])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate event accepted')
    assert join_call({**call, 'body_sha256': 'missing-body'}, index_events(events))['missing_events']
    print('4 synthetic timing controls passed')


def stats(values):
    return {'count': len(values), 'total_s': sum(values), 'mean_s': statistics.mean(values),
            'median_s': statistics.median(values), 'min_s': min(values), 'max_s': max(values)} if values else None


def analyse(study, run):
    log = run / 'slots/reasoning_model/waits.jsonl'
    events = index_events(json.loads(line) for line in log.read_text().splitlines() if line.strip())
    rows, seen, inputs = [], set(), {str(log): sha(log)}
    for path in sorted(study.glob('*/arms/adviser/*/ledger.jsonl')):
        inputs[str(path)] = sha(path)
        gateway = path.with_name('gateway.log')
        inputs[str(gateway)] = sha(gateway)
        requests = {}
        for line in gateway.read_text().splitlines():
            event = json.loads(line).get('payload', {})
            if event.get('kind') == 'model_start':
                request = event['request']
                if request['request_id'] in requests:
                    raise ValueError('duplicate gateway request identity')
                # Exact frozen LocalChatProvider.body and local_model.SETTINGS recipe.
                body = {'model': 'Qwen/Qwen3.8-27B',
                        'messages': [{'role': s, 'content': t} for s, t in request['messages']],
                        'max_tokens': request['max_output_tokens'], 'temperature': 1.0,
                        'top_p': .95, 'top_k': 20,
                        'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}}
                requests[request['request_id']] = hashlib.sha256(
                    json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        for line in path.read_text().splitlines():
            call = json.loads(line)
            call['body_sha256'] = requests[call['request_id']]
            row = join_call(call, events)
            key = tuple(row['join_key'])
            if key in seen:
                raise ValueError('multiple calls map to the same slot identity')
            seen.add(key)
            row['call_position'] = json.loads(call['request_id'])[-1]
            row['phase'] = int(call['episode'].split('-')[1][1:])
            rows.append(row)
    unmatched = [list(k) for k in events.keys() - seen]
    complete = [r for r in rows if not r['missing_events']]
    differences = [abs(r['wall_minus_wait_and_held_s']) for r in complete]
    return {'reader_sha256': sha(Path(__file__)), 'inputs_sha256': inputs,
            'definition': 'held_s is recorded ownership of the model slot, including HTTP overhead; wall_s also includes waiting',
            'join': 'exact stream and request-body SHA-256 reconstructed from saved gateway request and frozen provider settings; no nearest-match or imputation',
            'calls': len(rows), 'matched_complete_calls': len(complete), 'unmatched_slot_identities': unmatched,
            'missing_call_event_records': sum(bool(r['missing_events']) for r in rows),
            'max_abs_wall_minus_wait_and_held_s': max(differences, default=None),
            'all_wall_decompositions_within_0_02_seconds': all(d < .02 for d in differences),
            'call_statuses': dict(Counter(r['status'] for r in rows)),
            'by_stream': {s: {field: stats([r[field] for r in complete if r['stream'].endswith('/' + s)])
                               for field in ('wall_s', 'wait_s', 'held_s')}
                          for s in ('4301', '4302', '4303', '4304')},
            'totals': {field: stats([r[field] for r in complete]) for field in ('wall_s', 'wait_s', 'held_s')},
            'per_call': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--study', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        output = analyse(args.study, args.run)
        args.out.write_text(json.dumps(output, indent=2) + '\n')
        print(json.dumps({k: output[k] for k in ('calls', 'matched_complete_calls', 'missing_call_event_records',
                                                'max_abs_wall_minus_wait_and_held_s',
                                                'call_statuses', 'totals')}, indent=2))
