"""TURN149 estimator diagnostic harness for code cross-review before real timing.

Explicit APIs only; importing this module performs no construction or measurement.
No policy, GuidanceSearch, world, sampler or model calls. Full binary64 payloads
are exported once and verified against the accepted manifest before measurement.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from . import history_inputs as H; from . import qualification as Q
from .estimation import EpisodeEstimator

MANIFEST_DIGEST = '65d7ce286d42db36dc4dde3a83797c26209e1fae4fa214078bf5ecf0ec3153f8'
REPETITIONS = 3
COORDINATE_ALLOWANCE_NM = 1e-6


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_manifest(path):
    body = json.loads(Path(path).read_bytes())
    saved = body.pop('body_sha256')
    if saved != MANIFEST_DIGEST or Q.digest(body) != saved:
        raise ValueError('accepted long-history manifest required')
    expected = H.definitions()
    fields = tuple(expected[0])
    if [{k: s[k] for k in fields} for s in body['sequences']] != expected or body['count'] != 32:
        raise ValueError('exact ordered 32 definitions required')
    return body


def freeze_payloads(manifest_path, destination):
    """Preparation only. Refuse overwrites; hash actual bytes before later timing.

    A partial directory after an I/O failure is kept for diagnosis, never admitted
    without a complete index. No estimator is created here.
    """
    manifest = read_manifest(manifest_path)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for spec in manifest['sequences']:
        data = H.materialize(spec)
        raw = canonical(data)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != spec['input_sha256']:
            raise ValueError('native materialization differs; do not silently repin')
        name = spec['id'] + '.json'
        with (root / name).open('xb') as f:
            f.write(raw)
        entries.append({'id': spec['id'], 'file': name, 'bytes': len(raw), 'input_sha256': sha})
    body = {'manifest_sha256': MANIFEST_DIGEST, 'entries': entries,
            'scope': 'frozen public numeric inputs only; no timing or flight'}
    package = body | {'body_sha256': Q.digest(body)}
    with (root / 'index.json').open('xb') as f:
        f.write(canonical(package))
    return package


def read_payloads(manifest_path, root):
    """All input validation/load is outside measurement intervals."""
    manifest = read_manifest(manifest_path)
    root = Path(root)
    index = json.loads((root / 'index.json').read_bytes())
    saved = index.pop('body_sha256')
    if Q.digest(index) != saved or index['manifest_sha256'] != MANIFEST_DIGEST:
        raise ValueError('frozen package index differs')
    if [e['id'] for e in index['entries']] != [s['id'] for s in manifest['sequences']]:
        raise ValueError('complete ordered payload index required')
    payloads = []
    for entry, spec in zip(index['entries'], manifest['sequences']):
        if entry['file'] != spec['id'] + '.json' or entry['input_sha256'] != spec['input_sha256']:
            raise ValueError('payload path/hash differs')
        raw = (root / entry['file']).read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != spec['input_sha256']:
            raise ValueError('payload bytes differ')
        data = json.loads(raw)
        if canonical(data) != raw:
            raise ValueError('canonical payload required')
        payloads.append(data)
    return manifest, payloads, saved


def _clock(clock):
    try:
        return clock()
    except Exception:
        return None


def _delta(start, end):
    return end - start if type(start) is int and type(end) is int and end >= start else None


def process_peak():
    """Cumulative process high-water RSS, not a per-interval allocation peak."""
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return {'value': value, 'unit': 'bytes', 'scope': 'cumulative process high-water',
                    'unit_basis': 'declared modern XNU assumption; installed kernel not independently pinned'}
        if sys.platform.startswith('linux'):
            return {'value': value, 'unit': 'KiB', 'scope': 'cumulative process high-water',
                    'unit_basis': 'Linux getrusage documentation, since 2.6.32'}
        return {'value': value, 'unit': 'platform-native-unknown', 'scope': 'cumulative process high-water'}
    except Exception as exc:
        return {'value': None, 'unit': None, 'error': type(exc).__name__}


def state(estimator):
    """Public estimator evidence; export work remains outside timed intervals."""
    return {'t': estimator.t, 'goal': vars(estimator.goal) if estimator.goal else None,
            'targets': {name: {'retained': len(b.observations), 'latest': vars(b.latest),
                               'change_low': b.change_low, 'change_high': b.change_high}
                        for name, b in sorted(estimator.targets.items())}}


def validate_history(audit, spec, final=False):
    """Verify retained counts/latest measurements; no truth or outcome labels."""
    t, cadence = spec['final_second'], spec['cadence_s']
    latest = t if final and spec['final_reports'] == 'fresh' else ((t - 1) // cadence) * cadence
    retained = spec['retained_per_target_after_final'] if final else spec['prefix_bundle_count']
    names = ('A', 'B') if spec['targets'] == 2 else ('A',)
    if set(audit['targets']) != set(names) or audit['t'] != (t if final else latest):
        raise ValueError('history targets/time differ')
    if audit['goal']['version'] != 1 or audit['goal']['tanker'] != 'A':
        raise ValueError('history goal differs')
    if any(b['retained'] != retained or b['latest']['measured_at'] != latest or
           b['latest']['received_at'] != latest for b in audit['targets'].values()):
        raise ValueError('retained history count/latest differs')


def profile_final(manifest, payloads, factory=EpisodeEstimator,
                  cpu_clock=time.process_time_ns, wall_clock=time.perf_counter_ns,
                  memory_reader=process_peak):
    """96 intervals: repeat outermost, fixed manifest order, fresh estimator each.

    Factory/reset/copy/all prefix updates are setup. Only copied final update is
    timed. Audits/memory/payload loading are outside both intervals. Errors and
    unknown clocks retained; no scientific outcome or feasibility projection.
    Caller uses read_payloads() before invoking this explicit measurement API.
    """
    if len(payloads) != 32 or len(manifest['sequences']) != 32:
        raise ValueError('32 frozen payloads required')
    for spec, data in zip(manifest['sequences'], payloads):
        if Q.digest(data) != spec['input_sha256']:
            raise ValueError('pre-measurement payload differs')
    rows = []
    for repetition in range(REPETITIONS):
        for spec, data in zip(manifest['sequences'], payloads):
            est, before, after, error, stage = None, None, None, None, 'setup'
            memory_before = memory_reader()
            c0, w0 = _clock(cpu_clock), _clock(wall_clock)
            try:
                est = factory(COORDINATE_ALLOWANCE_NM)
                est.reset(deepcopy(data['briefing']))
                for bundle in data['prefix_bundles']:
                    est.update(deepcopy(bundle))
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            setup_cpu, setup_wall = _delta(c0, _clock(cpu_clock)), _delta(w0, _clock(wall_clock))
            cpu = wall = None
            if error is None:
                stage = 'before-audit'
                try:
                    before = state(est)
                    validate_history(before, spec)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
            if error is None:
                stage = 'final'
                c0, w0 = _clock(cpu_clock), _clock(wall_clock)
                try:
                    est.update(deepcopy(data['final_bundle']))
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                cpu, wall = _delta(c0, _clock(cpu_clock)), _delta(w0, _clock(wall_clock))
            if est is not None:
                try:
                    after = state(est)
                    if error is None:
                        validate_history(after, spec, final=True)
                except Exception as exc:
                    if error is None:
                        stage, error = 'after-audit', f'{type(exc).__name__}: {exc}'
            rows.append({'input': spec['id'], 'input_sha256': spec['input_sha256'], 'repetition': repetition,
                         'definition': deepcopy(data['definition']), 'prefix_bundle_count': len(data['prefix_bundles']),
                         'prefix_report_count': spec['prefix_report_count'], 'before': before, 'after': after,
                         'setup_cpu_ns': setup_cpu, 'setup_wall_ns': setup_wall, 'cpu_ns': cpu, 'wall_ns': wall,
                         'error': error, 'error_stage': stage if error else None,
                         'memory_before': memory_before, 'memory_after': memory_reader()})
    return rows


def summarize(rows, manifest):
    """Complete observed diagnostic only; no observed maximum becomes a bound."""
    expected = {(s['id'], r): s for r in range(REPETITIONS) for s in manifest['sequences']}
    keys = [(r['input'], r['repetition']) for r in rows]
    if len(keys) != 96 or len(set(keys)) != 96 or set(keys) != set(expected):
        raise ValueError('complete unique 96 intervals required')
    for key, row in zip(keys, rows):
        spec = expected[key]
        if row['input_sha256'] != spec['input_sha256']:
            raise ValueError('timed input hash differs')
        if row['definition'] != {k: spec[k] for k in H.definitions()[0]} or \
                row['prefix_bundle_count'] != spec['prefix_bundle_count'] or \
                row['prefix_report_count'] != spec['prefix_report_count']:
            raise ValueError('timed input definition/counts differ')
    known = lambda v: type(v) is int and v >= 0
    available = all(not r['error'] and all(known(r[k]) for k in
                    ('setup_cpu_ns', 'setup_wall_ns', 'cpu_ns', 'wall_ns')) for r in rows)
    return {'intervals': 96, 'complete_known_timing': available,
            'final_cpu_ns': sum(r['cpu_ns'] for r in rows) if available else None,
            'setup_cpu_ns': sum(r['setup_cpu_ns'] for r in rows) if available else None,
            'by_input': {s['id']: {'definition': {k: s[k] for k in H.definitions()[0]},
                         'final_cpu_ns': [r['cpu_ns'] for r in rows if r['input'] == s['id']],
                         'setup_cpu_ns': [r['setup_cpu_ns'] for r in rows if r['input'] == s['id']],
                         'history_after': [r['after'] for r in rows if r['input'] == s['id']]}
                         for s in manifest['sequences']},
            'disposition': 'estimator-only diagnostic; no feasibility, global bound or flight clearance',
            'non_additive': 'update is nested in act; never add these times to old inclusive act estimates'}


def campaign(manifest_path, frozen_root, snapshot, source_commit, code_review_commit, output_path):
    """Explicit future recording wrapper, requiring independently reviewed source.

    String commit arguments record provenance, not authorization. Before calling,
    verify the pushed exact source/review and freeze payload bytes. This wrapper
    is prospective and must itself be cross-reviewed before actual measurement.
    """
    import numpy
    import scipy
    snapshot, output_path = Path(snapshot).resolve(), Path(output_path)
    if output_path.exists():
        raise FileExistsError('preserve earlier raw diagnostic')
    if not source_commit or not code_review_commit or not Path(__file__).resolve().is_relative_to(snapshot):
        raise ValueError('explicit source/review pins and archived execution path required')
    manifest, payloads, package_sha = read_payloads(manifest_path, frozen_root)
    paths = sorted((snapshot / 'dcarr-runtime-0.1/study2/adaptive_controller').glob('*.py'))
    paths.append(Path(manifest_path).resolve())
    pins = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def load():
        try:
            return os.getloadavg()
        except (AttributeError, OSError):
            return None
    provenance = {'source_commit': source_commit, 'code_review_commit': code_review_commit,
                  'source_sha256': pins, 'manifest_sha256': MANIFEST_DIGEST, 'payload_package_sha256': package_sha,
                  'python': platform.python_version(), 'executable': sys.executable,
                  'platform': platform.platform(), 'machine': platform.machine(), 'logical_cpus': os.cpu_count(),
                  'numpy': numpy.__version__, 'scipy': scipy.__version__,
                  'dependency_entry_sha256': {str(Path(m.__file__).resolve()):
                      hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (numpy, scipy)},
                  'cpu_clock': vars(time.get_clock_info('process_time')),
                  'wall_clock': vars(time.get_clock_info('perf_counter')), 'load_start': load(),
                  'coordinate_allowance_nm': COORDINATE_ALLOWANCE_NM,
                  'scope': 'one sequential process, shared load; no affinity or isolation claim',
                  'limitations': ['source/dependency entry pins are not full binary or loaded-code closure',
                                  'input load/hash/import outside campaign; audit outside final/setup intervals',
                                  'RSS cumulative process high-water, not isolated per-update memory',
                                  'zero-change public points; no reached policy state or outcome label',
                                  'finite observed costs, not global bound; no additive act projection']}
    c0, w0 = _clock(time.process_time_ns), _clock(time.perf_counter_ns)
    rows = profile_final(manifest, payloads)
    summary = summarize(rows, manifest)
    body = {'stage': 'estimator-only diagnostic; no flight', 'rows': rows, 'summary': summary,
            'provenance': provenance,
            'inclusive_campaign_cpu_ns': _delta(c0, _clock(time.process_time_ns)),
            'inclusive_campaign_wall_ns': _delta(w0, _clock(time.perf_counter_ns)), 'load_end': load(),
            'sources_unchanged_at_export': all(hashlib.sha256(p.read_bytes()).hexdigest() ==
                pins[str(p.relative_to(snapshot))] for p in paths)}
    body['diagnostic_valid'] = summary['complete_known_timing'] and body['sources_unchanged_at_export']
    result = body | {'body_sha256': Q.digest(body)}
    with output_path.open('xb') as f:
        f.write(canonical(result))
    return result
