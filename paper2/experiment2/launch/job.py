"""One study job: the payload, an isolated installation, native containment, two owned model servers and the study
host. No submission, retry or extension happens here.

Adapted from paper2/development/m1-timing-rehearsal/job.py (accepted at df0e7ab), reusing its native stages and the
independently reviewed native primitives (native_common.py from 81a5af3) unchanged. What differs: the prepared study is
bound by a manifest checked before anything starts; the plan's profile is a template whose server addresses the job
fills in; the host's startup check runs isolated in the installed environment before any server starts; the plan's load
limits (processors, memory, sandbox limits, slots) must match the allocation and the installed product; the plan's
serving settings, profile models and host duration are checked before anything starts; and close-out runs every stage on
its own, so a failure in one never skips the next.

Ownership is shown by a fresh random marker, TELOHELM_JOB_TOKEN, that the job sets in its own environment before it
starts anything, so every process it starts inherits it: the host and its dispatchers and workers, both servers and their
engine processes, the sampler, and every short command. Close-out stops each resource on its own (the host, each server,
the sampler), then ends every process group that still holds a marked process, renewing that evidence before every
forced signal, and confirms that nothing in those groups is left; then it closes each log handle and writes checksums. A
process whose environment cannot be read, or a listing that fails or is incomplete, leaves the sweep unknown, and the
checksum record then says incomplete.

    python -I -B -c BOOT PAYLOAD PAYLOAD/job.py --payload PAYLOAD --study STUDY --out OUT --hf HF --vllm VLLM

--cpu-rehearsal runs every stage that needs no graphics processor or model server, with stand-in models and the
plan's rehearsal profile, and records each skipped stage by name. It is a software rehearsal and measures nothing about
real models."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error

import native_common as N
from paper2.experiment2.spawn import spawn

SAMPLE_S = 60
MAX_HOST_HOURS = 240                    # the longest queue allows ten days
HOST_GRACE_S = 150                      # the host's dispatcher grace (60 s), the worker grace (10 s) and its receipts
SWEEP_GRACE_S = 30                      # marked groups' time to stop after SIGTERM before SIGKILL
SERVER_GRACE_S = 60                     # a model server's time to stop after SIGTERM before SIGKILL
JOB_TOKEN = 'TELOHELM_JOB_TOKEN'
ROLES = ('small_model', 'reasoning_model')
BOOT = ("import sys,runpy;sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_module(sys.argv.pop(1),run_name='__main__',alter_sys=True)")
NATIVE_STAGES = ("allocated-linux", "gpus", "cache", "native-containment-31", "owned-model-servers", "token-parity",
                 "gpu-sampler")


def module_command(python, module, args, payload):
    """The same isolated form as study_runner.module_command, which the launcher cannot import before installation."""
    return [str(python), '-I', '-B', '-c', BOOT, str(Path(payload).resolve()), module, *map(str, args)]


def study_manifest(study):
    """Every file of the prepared study, by relative path, with its SHA-256. A link is refused."""
    files = {}
    for path in sorted(Path(study).rglob('*')):
        N.require(not path.is_symlink(), 'The study holds no links')
        if path.is_file():
            files[str(path.relative_to(study))] = N.sha(path)
    return files


def manifest_sha(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def adviser_args(arms, urls, plan):
    """The study host's address for the reasoning model alone, when the plan flies it. No other plan changes."""
    if 'adviser' not in arms:
        return []
    return ['--adviser-url', urls['reasoning_model'], '--adviser-model', plan['models']['reasoning_model']['id']]


def fill_profile(template, urls):
    """The plan's profile with the owned servers' addresses in place of {small_model} and {reasoning_model}."""
    profile = json.loads(json.dumps(template))
    for key, role in (('local', 'small_model'), ('learn', 'reasoning_model')):
        if isinstance(profile.get(key), dict):
            N.require(profile[key].get('base_url') == '{' + role + '}', 'Profile server address is not a template')
            profile[key]['base_url'] = urls[role]
    return profile


def pinned_cache(hf):
    """The pinned offline cache. HF_HOME and --hf must name the same existing folder (a file is refused); they are compared as files, not
    as text, since on this cluster /scratch/USER and /vast/USER are two names for one place. The hub must be offline."""
    home = os.environ.get('HF_HOME')
    try:
        same = home is not None and Path(hf).is_dir() and os.path.samefile(home, hf)
    except OSError:
        same = False
    N.require(same and os.environ.get('HF_HUB_OFFLINE') == '1', 'Pinned offline cache required')
    return {'hf_home': home, 'hf': str(hf), 'resolved': str(Path(hf).resolve())}


def server_command(binary, model, port, server, role):
    """The serve command for one role, from the plan's declared serving settings (validate_plan checks them)."""
    args = [str(binary), 'serve', model['id'], '--revision', model['revision'],
            '--tokenizer-revision', model['revision'], '--dtype', model['dtype'], '--host', '127.0.0.1',
            '--port', str(port), '--tensor-parallel-size', str(server['tensor_parallel']),
            '--max-model-len', str(model['context']), '--gpu-memory-utilization', str(model['gpu_memory_utilization']),
            '--max-num-seqs', str(server['max_num_seqs'][role])]
    if server['prefix_caching']:
        args.append('--enable-prefix-caching')
    if role == 'reasoning_model':
        args += ['--reasoning-parser', 'qwen3']
    return args


def validate_plan(plan, rehearsal):
    """The plan's declared configuration, checked before anything starts: a finite positive host duration; serving
    settings that one GPU per server can apply (tensor parallelism 1, a positive whole number of sequences per role,
    prefix caching declared); slots equal to those sequence counts; and profile model names equal to the admitted
    roles' models."""
    hours = plan.get('host_hours')
    N.require(type(hours) in (int, float) and math.isfinite(hours) and 0 < hours <= MAX_HOST_HOURS
              and math.isfinite(time.time() + hours * 3600), f'Host duration must be finite, positive and at most {MAX_HOST_HOURS} hours')
    N.require(type(plan.get('gpus')) is int and plan['gpus'] == len(ROLES), 'The job runs one server per GPU on exactly two GPUs')
    server, models = plan['server'], plan['models']
    N.require(all(type(x) is int and x == 1 for x in (server.get('tensor_parallel'),
                                                       *(models[r].get('tensor_parallel') for r in ROLES))),
              'Each server has one GPU, so tensor parallelism must be the whole number 1')
    seqs = server.get('max_num_seqs', {})
    N.require(set(seqs) == set(ROLES) and all(type(seqs[r]) is int and seqs[r] > 0 for r in ROLES),
              'Sequences per role must be positive whole numbers for both roles')
    N.require(type(server.get('prefix_caching')) is bool, 'Prefix caching must be declared')
    N.require(plan['slots'] == seqs, 'Slots must equal the servers\' sequences per role')
    limits = plan['limits']
    N.require(type(limits.get('dispatchers')) is int and limits['dispatchers'] > 0, 'Dispatcher count must be declared')
    if not rehearsal:
        profile = plan['profile']
        N.require(profile['local']['model'] == models['small_model']['id']
                  and profile['learn']['model'] == models['reasoning_model']['id'],
                  'Profile models differ from the admitted roles')


TOKEN_PARITY = r'''
import json, sys, urllib.request
from telohelm import examples as X, learning as G, local_model as LM, token_count as T
from telohelm.runtime.procedures import CONTEXT_FIELDS
import paper2.experiment2.run_stream as RS
url, effort = sys.argv[1], sys.argv[2]
kw = {**LM.SETTINGS["reasoning_model"]["chat_template_kwargs"], "reasoning_effort": effort}
system = G.PROPOSER_RULES + "\n" + RS.POC_RULES_BOTH + "\n" + X.RULES
fresh = [("system", system), ("user", json.dumps({"feedback": [{"episode": "check", "failed": "both"}], "library": {},
                                                 "context_fields": sorted(CONTEXT_FIELDS)}, sort_keys=True))]
body = {"model": "Qwen/Qwen3.8-27B", "messages": [{"role": r, "content": c} for r, c in fresh],
        "add_generation_prompt": True, "chat_template_kwargs": kw}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
req = urllib.request.Request(url + "/tokenize", json.dumps(body).encode(), {"Content-Type": "application/json"})
server, local = json.load(opener.open(req, timeout=60))["count"], T.count(fresh, kw)
print(json.dumps({"local": local, "server": server}))
sys.exit(0 if local == server else "input token counts differ from the server")
'''

SAMPLER = r'''
import json, subprocess, sys, time
path, every, uuids = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
with open(path, "x") as f:
    while True:
        try:
            rows = subprocess.check_output(["nvidia-smi", "--id=" + ",".join(uuids),
                                            "--query-gpu=index,uuid,utilization.gpu,memory.used",
                                            "--format=csv,noheader,nounits"], text=True, timeout=30)
            f.write(json.dumps({"t": time.time(), "gpus": rows.strip().splitlines()}) + "\n")
        except (subprocess.SubprocessError, OSError) as exc:
            f.write(json.dumps({"t": time.time(), "error": type(exc).__name__}) + "\n")
        f.flush()
        time.sleep(every)
'''

LIMITS = r'''
import json, os
import telohelm.runtime.sandbox as X
print(json.dumps({"cpu_seconds": X.CPU_SECONDS, "wall_seconds": X.WALL_SECONDS, "memory_limit_mb": X.MEMORY_LIMIT_MB,
                  "max_open_files": X.MAX_OPEN_FILES, "linux_address_space_mb": X.LINUX_ADDRESS_SPACE_MB}))
'''


def stop_sampler(process, grace=10):
    """Stop the sampler before anything is checksummed: SIGTERM, then SIGKILL. Returns how it ended."""
    if process is None:
        return 'not started'
    if process.poll() is not None:
        return 'stopped'
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace)
        return 'stopped'
    except ProcessLookupError:
        return 'stopped'
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
            return 'killed'
        except subprocess.TimeoutExpired:
            return 'still running'


def stop_host(host):
    """SIGTERM to the host's group, then time for the host to stop its dispatchers and write its receipts."""
    if host is None or host.poll() is not None:
        return 'not running'
    try:
        os.killpg(host.pid, signal.SIGTERM)
        host.wait(timeout=HOST_GRACE_S)
        return 'stopped'
    except ProcessLookupError:
        return 'stopped'
    except subprocess.TimeoutExpired:
        try:
            os.killpg(host.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            host.wait(timeout=10)
            return 'killed'
        except subprocess.TimeoutExpired:
            return 'still running'


class Unobservable(Exception):
    """A process listing failed or was incomplete, so nothing can be concluded from it."""


UNREADABLE = None          # a process of this user whose environment could not be read


def _owner(entry):
    return entry.stat().st_uid


def _listing(environment=False, proc=Path('/proc')):
    """Every running process's group (with its environment, if asked) from one listing that must include this process.
    Linux reads /proc; elsewhere ps. A process that disappears during the listing is left out. A process of another user
    is foreign: its environment is not read and it is never this job's. A process of this user whose environment cannot
    be read is kept with UNREADABLE, so the caller cannot mistake it for an unmarked one."""
    me, uid, rows = os.getpid(), os.getuid(), {}
    if (proc / str(me)).exists():
        try:
            entries = list(proc.iterdir())
        except OSError as exc:
            raise Unobservable(f'/proc: {exc}') from None
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                foreign = _owner(entry) != uid
            except (FileNotFoundError, ProcessLookupError):
                continue                                   # it has gone
            except (OSError, IndexError) as exc:
                raise Unobservable(f'{entry}: {type(exc).__name__}') from None
            if stat[0] == 'Z':
                continue
            env = []
            if environment and not foreign:
                try:
                    env = [e.decode(errors='replace') for e in (entry / 'environ').read_bytes().split(b'\0')]
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except OSError:
                    env = UNREADABLE
            rows[int(entry.name)] = (int(stat[2]), env)
    else:
        try:
            done = subprocess.run(['ps', *(['eww'] if environment else []), '-A', '-o', 'pid=,pgid=,uid=,stat=,command='],
                                  capture_output=True, text=True, timeout=30)
            for line in done.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 4 and 'Z' not in parts[3]:
                    rows[int(parts[0])] = (int(parts[1]), parts[4:] if int(parts[2]) == uid else [])
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise Unobservable(f'ps: {type(exc).__name__}: {exc}') from None
        if done.returncode != 0:
            raise Unobservable(f'ps exited {done.returncode}')
    if me not in rows:
        raise Unobservable('the listing lacks this process')
    return rows


def marked(token, listing=None):
    """What now carries this job's marker: process groups (other than the job's own) that hold a marked process, and
    marked processes inside the job's own group, which are signalled one by one so the job never signals itself.
    Raises Unobservable if a process of this user cannot be read, since it might be one of the job's."""
    rows = _listing(environment=True) if listing is None else listing
    unreadable = sorted(pid for pid, (_, env) in rows.items() if env is UNREADABLE)
    if unreadable:
        raise Unobservable(f'environments unreadable for processes {unreadable[:10]}')
    me, own = os.getpid(), os.getpgid(0)
    hits = [(pid, pgid) for pid, (pgid, env) in rows.items() if pid != me and f'{JOB_TOKEN}={token}' in env]
    return sorted({g for _, g in hits if g != own}), sorted(p for p, g in hits if g == own)


def sweep_marked(token, grace=SWEEP_GRACE_S):
    """End everything that carries this job's marker: SIGTERM, then SIGKILL after grace, to each marked group and to
    each marked process in the job's own group. Ownership is renewed before every signal, so nothing is signalled
    unless it carries the marker at that moment. Afterwards nothing marked may remain, and any process still in a group
    that held a marked one is reported, since its ownership can no longer be shown. Returns closed, left_running or
    unknown; it never raises for a failed listing."""
    report = {'groups': None, 'strays': None, 'terminated': [], 'killed': []}

    def send(targets, sig):
        groups, strays = targets
        for group in groups:
            _signal(lambda: os.killpg(group, sig))
        for pid in strays:
            _signal(lambda: os.kill(pid, sig))
    try:
        first = marked(token)
        report['groups'], report['strays'] = first
        send(first, signal.SIGTERM)
        report['terminated'] = first
        current, end = first, time.monotonic() + grace
        while any(current) and time.monotonic() < end:
            time.sleep(0.25)
            current = marked(token)
        current = marked(token) if any(first) else ([], [])   # renewed just before any forced signal
        send(current, signal.SIGKILL)
        report['killed'] = current
        end = time.monotonic() + 10
        while any(current) and time.monotonic() < end:
            time.sleep(0.1)
            current = marked(token)
        rows = _listing()
        unmarked = sorted(pid for pid, (pgid, _) in rows.items() if pgid in first[0] and pid != os.getpid())
        report['remaining'] = current
        report['unmarked_in_former_groups'] = unmarked
        report['status'] = 'left_running' if any(current) else ('unknown' if unmarked else 'closed')
    except Unobservable as exc:
        report.update(status='unknown', error=str(exc))
    return report


def _signal(send):
    try:
        send()
    except (ProcessLookupError, PermissionError):
        pass


def stop_server(process):
    """SIGTERM to a server's group, then SIGKILL after SERVER_GRACE_S; every wait bounded."""
    if process is None or process.poll() is not None:
        return 'not running'
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=SERVER_GRACE_S)
        return 'stopped'
    except ProcessLookupError:
        return 'stopped'
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
            return 'killed'
        except subprocess.TimeoutExpired:
            return 'still running'


def finalise(out, result):
    """Write the result and the checksum manifest last. If a cleanup stage failed, or anything may still be running or
    could not be seen, the manifest says it is incomplete and why, and makes no claim to be final."""
    problems = []
    cleanup = result.get('cleanup', {})
    for stage, entry in cleanup.items():
        if entry.get('status') != 'done':
            problems.append(f"cleanup {stage}: {entry.get('error')}")
    acceptable = {'host': ('not running', 'stopped', 'killed'), 'sampler': ('stopped', 'killed', 'not started')}
    for stage, ok in acceptable.items():
        if stage in cleanup and cleanup[stage].get('status') == 'done' and cleanup[stage].get('value') not in ok:
            problems.append(f"{stage} {cleanup[stage].get('value')}")
    sweep = cleanup.get('sweep', {})
    if sweep.get('status') == 'done' and sweep.get('value', {}).get('status') != 'closed':
        value = sweep['value']
        problems.append(f"sweep {value.get('status')}: {value.get('error') or value.get('remaining') or value.get('unmarked_in_former_groups')}")
    for stage, entry in cleanup.items():
        if stage.startswith('server ') and entry.get('status') == 'done' and entry.get('value') not in ('not running', 'stopped', 'killed'):
            problems.append(f"{stage} {entry.get('value')}")
    result['finalisation'] = 'complete' if not problems else 'incomplete: ' + '; '.join(problems)
    N.save(out / 'result.json', result)
    files = {str(p.relative_to(out)): N.sha(p) for p in out.rglob('*')
             if p.is_file() and not p.is_relative_to(out / 'venv') and not p.is_relative_to(out / 'cache')}
    N.save(out / 'checksums.json', {'complete': not problems, 'problems': problems, 'files': files})


def check_limits(plan, python, rehearsal):
    """The plan's load limits against the allocation and the installed product's sandbox constants."""
    limits = plan['limits']
    if not rehearsal:
        N.require(os.environ.get('SLURM_CPUS_PER_TASK') == str(limits['cpus_per_task']), 'Processor allocation differs')
        N.require(os.environ.get('SLURM_MEM_PER_NODE') == str(limits['mem_mb']), 'Memory allocation differs')
    sandbox = json.loads(subprocess.check_output([str(python), '-I', '-B', '-c', LIMITS], text=True))
    N.require(sandbox == limits['sandbox'], 'Installed sandbox limits differ from the plan')
    N.require(plan['slots'] == limits['slots'], 'Plan slots differ from its limits')
    return {'sandbox': sandbox, 'slots': plan['slots'], 'cpus_per_task': limits['cpus_per_task'],
            'mem_mb': limits['mem_mb'], 'dispatchers': limits['dispatchers']}


def check_dispatch(planned, plan):
    """The study host's planned dispatchers against the plan: their number must equal the declared limit, and each
    must be a declared condition."""
    N.require(len(planned) == plan['limits']['dispatchers'], 'The study plans a different number of dispatchers')
    N.require(all(d['name'].split(':')[0] in plan['arms'] for d in planned), 'The study plans an undeclared condition')
    return [d['name'] for d in planned]


def job(payload, study, out, hf, vllm, rehearsal=False):
    started = time.time()
    payload, study, out = (Path(p).resolve() for p in (payload, study, out))
    N.require(JOB_TOKEN not in os.environ, 'An inherited job marker is present; this job could not tell its own processes apart')
    if not rehearsal:
        N.require(sys.platform == 'linux' and os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Allocated Linux job required')
    N.require(not out.exists(), 'A new output directory is required')
    manifest = N.verify_payload(payload)
    out.mkdir(parents=True)
    plan = json.loads((payload / 'plan.json').read_text())
    validate_plan(plan, rehearsal)
    token = os.urandom(16).hex()
    os.environ[JOB_TOKEN] = token          # every process the job starts from here on inherits it
    result = {'scope': plan['scope'], 'mode': 'cpu rehearsal with stand-in models' if rehearsal else 'native',
              'status': 'started', 'job_id': os.environ.get('SLURM_JOB_ID'), 'payload_manifest_sha256': manifest,
              'started_epoch': started, 'stages': [], 'skipped': list(NATIVE_STAGES) if rehearsal else [],
              'parent_allocation': {'job_id': os.environ.get('SLURM_JOB_ID'), 'gpus': None,
                                    'gpu_hours': None, 'reconciled_by': 'terminal scheduler audit'}}
    servers, handles, procs = {}, [], {'host': None, 'sampler': None}
    python = None
    try:
        freeze = plan.get('freeze')
        if freeze:
            protocol = payload / freeze['protocol_path']
            N.require(protocol.is_file() and N.sha(protocol) == freeze['protocol_sha256'],
                      'The payload does not carry the frozen protocol')
        N.save(out / 'plan.json', plan)
        N.save(out / 'source.json', json.loads((payload / 'source.json').read_text()))
        files = study_manifest(study)
        N.require(manifest_sha(files) == plan['study_manifest_sha256'], 'The prepared study differs from its manifest')
        N.save(out / 'study-manifest.json', files)
        result['stages'].append('study-manifest')
        uuids = []
        if not rehearsal:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
            N.require(len(visible) == len(set(visible)) == plan['gpus'] and all(visible), 'GPU count differs')
            allocation = [subprocess.check_output(['nvidia-smi', '-i', v, '--query-gpu=name,uuid',
                           '--format=csv,noheader'], text=True).strip() for v in visible]
            N.require(all(len(v.splitlines()) == 1 and plan['gpu_type'] in v for v in allocation), 'GPU type differs')
            uuids = [record.split(',')[-1].strip() for record in allocation]
            N.require(all(u.startswith('GPU-') for u in uuids), 'Allocated GPU identifiers unreadable')
            result['parent_allocation']['gpus'] = len(visible)
            N.save(out / 'allocation.json', {'visible_devices': visible, 'gpu_records': allocation})
            result['cache_path'] = pinned_cache(hf)
        python, inherited = N.make_environment(out / 'venv')
        wheel, = payload.glob('*.whl')
        with (out / 'install.log').open('x') as log:
            subprocess.run([str(python), '-I', '-B', '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--ignore-installed', str(wheel)], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=300)
        raw = subprocess.check_output([str(python), '-I', '-B', '-c',
            "import importlib.metadata as M,json,pathlib,sys,telohelm; p=pathlib.Path(telohelm.__file__).resolve(); "
            "assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()); print(json.dumps({'product':str(p),"
            "'dependencies':{n:M.version(n) for n in ['numpy','scipy','jinja2','tokenizers','pytest']}}))"])
        installed = json.loads(raw)
        N.require(installed['dependencies'] == json.loads((payload / 'source.json').read_text())['base_dependencies'],
                  'Declared dependencies differ')
        N.save(out / 'installed.json', {**installed, 'base_dependencies': inherited})
        result['stages'].append('installed')
        result['limits'] = check_limits(plan, python, rehearsal)
        result['stages'].append('limits')
        if not rehearsal:
            cache = {}
            for role in ('small_model', 'reasoning_model'):
                model = plan['models'][role]
                check = N.snapshot_record(Path(hf), {**model, 'context_window': model['context']},
                           model['cache_metadata_sha256']['tokenizer.json'], model['template_sha256'])
                N.require(check['metadata_sha256'] == model['cache_metadata_sha256'], 'Cached metadata differs')
                cache[role] = check
            N.save(out / 'cache.json', cache)
            result['stages'].append('cache')
            plugin = out / 'gate_plugin.py'
            plugin.write_text("import json\nfrom pathlib import Path\nreports=[]\ndef pytest_runtest_logreport(report):\n    reports.append({'when':report.when,'outcome':report.outcome,'nodeid':report.nodeid})\ndef pytest_sessionfinish(session,exitstatus):\n    Path(__file__).with_name('native-tests.json').write_text(json.dumps({'exitstatus':exitstatus,'collected':session.testscollected,'reports':reports}))\n")
            with (out / 'native-tests.log').open('x') as log:
                subprocess.run([str(python), '-I', '-B', '-c',
                    'import sys;sys.path.insert(0,sys.argv.pop(1));import pytest;raise SystemExit(pytest.main(sys.argv[1:]))',
                    str(out), '-p', 'gate_plugin', '-c', '/dev/null', '--rootdir', str(payload / 'native_tests'),
                    '-p', 'no:cacheprovider', '-q', '-rs',
                    str(payload / 'native_tests')], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            gate = json.loads((out / 'native-tests.json').read_text())
            N.require(gate['exitstatus'] == 0 and gate['collected'] == 31 and
                      sum(r['when'] == 'call' and r['outcome'] == 'passed' for r in gate['reports']) == 31 and
                      all(r['outcome'] == 'passed' for r in gate['reports']), 'Native controls incomplete or failed')
            result['stages'].append('native-containment-31')
        arms = plan['arms']
        # A scored launch names the one declared stream it flies (protocol amendment 7, item 4); without the field the
        # host flies every declared stream, as before.
        streams = ['--streams', *map(str, plan['streams'])] if plan.get('streams') else []
        template = plan['rehearsal_profile'] if rehearsal else plan['profile']
        # The isolated startup check, in the installed environment, before any server starts. The template's server
        # addresses are placeholders here; the check needs only the profile's shape.
        check_profile = out / 'check-profile.json'
        N.save(check_profile, template if rehearsal else fill_profile(template, {
            'small_model': 'http://127.0.0.1:1', 'reasoning_model': 'http://127.0.0.1:2'}))
        check = subprocess.run(module_command(python, 'paper2.experiment2.study_host',
                               ['check', '--study', study, '--profile', check_profile, '--arms', *arms, *streams,
                                *(['--max-steps', plan['rehearsal_max_steps']] if rehearsal else [])], payload),
                               capture_output=True, text=True, timeout=600, cwd=out)
        N.save(out / 'study-check.json', {'exit': check.returncode, 'stdout': check.stdout, 'stderr': check.stderr})
        N.require(check.returncode == 0 and json.loads(check.stdout)['ok'] is True, 'The study host refused the study')
        result['dispatch'] = check_dispatch(json.loads(check.stdout)['dispatch'], plan)
        result['stages'].append('study-check')
        if rehearsal:
            urls, env = {}, {k: v for k, v in os.environ.items() if k != 'TELOHELM_MODEL_SLOTS'}
            profile = template
        else:
            roles = ('small_model', 'reasoning_model')
            base_port = 32000 + 2 * (int(result['job_id']) % 10000)
            admissions = {}
            for index, role in enumerate(roles):
                model, port = plan['models'][role], base_port + index
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', port))
                command = server_command(vllm, model, port, plan['server'], role)
                server_env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[index],
                                  VLLM_CACHE_ROOT=str(out / 'cache' / role), VLLM_NO_USAGE_STATS='1',
                                  HF_HUB_DISABLE_TELEMETRY='1')
                server_env.pop('TELOHELM_MODEL_SLOTS', None)
                log = (out / (role + '-server.log')).open('x')
                handles.append(log)
                process = spawn(command, lambda p, role=role, port=port: servers.__setitem__(role, (p, port)),
                                stdout=log, stderr=subprocess.STDOUT, env=server_env)
                N.save(out / (role + '-command.json'), {'argv': command, 'visible_device': visible[index],
                       'vllm_entrypoint_sha256': N.sha(vllm), 'pid': process.pid})
            deadline = time.monotonic() + 1800
            while len(admissions) != 2:
                for role in roles:
                    process, port = servers[role]
                    N.require(process.poll() is None, 'An owned server exited at startup')
                    if role in admissions:
                        continue
                    try:
                        owner = N.owned_listener(process.pid, port)
                        model = plan['models'][role]
                        listing = N.served_model(port, {'id': model['id'], 'context_window': model['context']})
                        admissions[role] = {'ownership': owner, 'models': listing}
                    except (ValueError, urllib.error.URLError, TimeoutError):
                        pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Owned model servers did not become ready in thirty minutes')
                if len(admissions) != 2:
                    time.sleep(5)
            N.save(out / 'server-admission.json', admissions)
            result['stages'].append('owned-model-servers')
            urls = {role: f'http://127.0.0.1:{servers[role][1]}' for role in roles}
            parity = subprocess.run([str(python), '-I', '-B', '-c', "import sys;sys.path.insert(0,sys.argv.pop(1));"
                                     "exec(sys.argv.pop(1))", str(payload), TOKEN_PARITY, urls['reasoning_model'],
                                     template['learn']['effort']], capture_output=True, text=True, timeout=120)
            N.save(out / 'token-parity.json', {'exit': parity.returncode, 'stdout': parity.stdout,
                                               'stderr': parity.stderr})
            N.require(parity.returncode == 0, 'Offline and server input token counts differ')
            result['stages'].append('token-parity')
            slots = {urls[role]: {'dir': str(out / 'slots' / role), 'n': plan['slots'][role]} for role in roles}
            env = {k: v for k, v in os.environ.items() if k != 'TELOHELM_MODEL_SLOTS'}
            env['TELOHELM_MODEL_SLOTS'] = json.dumps(slots, sort_keys=True)
            profile = fill_profile(template, urls)
            procs['sampler'] = spawn([sys.executable, '-I', '-B', '-c', SAMPLER, str(out / 'gpu-samples.jsonl'),
                                      str(SAMPLE_S), *uuids], lambda p: procs.__setitem__('sampler', p),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result['stages'].append('gpu-sampler')
        N.save(out / 'servers.json', urls)
        N.save(out / 'profile.json', profile)
        host_deadline = started + plan['host_hours'] * 3600
        command = module_command(python, 'paper2.experiment2.study_host',
                                 ['run', '--study', study, '--profile', out / 'profile.json', '--out', out / 'host',
                                  '--deadline-epoch', host_deadline, '--arms', *arms, *streams, '--payload', payload,
                                  *adviser_args(arms, urls, plan),
                                  '--python', python, '--gpus', 0 if rehearsal else plan['gpus'],
                                  *(['--max-steps', plan['rehearsal_max_steps']] if rehearsal else [])], payload)
        N.save(out / 'host-command.json', {'argv': command, 'host_deadline_epoch': host_deadline,
                                          'slots': None if rehearsal else json.loads(env['TELOHELM_MODEL_SLOTS'])})
        with (out / 'host.log').open('x') as log:
            host = spawn(command, lambda p: procs.__setitem__('host', p), stdout=log, stderr=subprocess.STDOUT,
                         env=env, cwd=out)
            while host.poll() is None:
                for process, port in servers.values():
                    N.require(process.poll() is None, 'Owned server died during the study')
                    N.owned_listener(process.pid, port)
                time.sleep(2)
            result['host_exit'] = host.returncode
        result['stages'].append('host-terminal')
        result['status'] = 'completed' if result['host_exit'] == 0 else 'incomplete'
        return 0
    except BaseException as exc:
        result.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                      error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        close_out(out, result, procs, servers, handles, token)


def close_out(out, result, procs, servers, handles, token):
    """Stop each resource on its own, in order: the host, each server, the sampler; then the marked sweep, which also
    ends anything those left behind; then each log handle; then the checksums. A further stop request is only
    recorded, every stage's error is kept and the next stage still runs, and every wait is bounded."""
    later = []
    previous = {signum: signal.signal(signum, lambda s, f: later.append(s)) for signum in (signal.SIGTERM, signal.SIGINT)}
    cleanup = result['cleanup'] = {}

    def stage(name, fn):
        try:
            cleanup[name] = {'status': 'done', 'value': fn()}
        except BaseException as exc:
            cleanup[name] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    try:
        stage('host', lambda: stop_host(procs['host']))
        for role, (process, port) in reversed(list(servers.items())):
            stage('server ' + role, lambda process=process: stop_server(process))
        stage('sampler', lambda: stop_sampler(procs['sampler']))
        stage('sweep', lambda: sweep_marked(token, SWEEP_GRACE_S))
        for index, handle in enumerate(handles):
            stage(f'handle {index}', handle.close)
        result['signals_during_cleanup'] = later
        result['ended_epoch'] = time.time()
        result['parent_allocation'].update(started_epoch=result['started_epoch'], ended_epoch=result['ended_epoch'])
        finalise(out, result)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if os.environ.get(JOB_TOKEN) == token:
            del os.environ[JOB_TOKEN]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('payload', 'study', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hf', type=Path)
    parser.add_argument('--vllm', type=Path)
    parser.add_argument('--cpu-rehearsal', action='store_true')
    args = parser.parse_args()
    if not args.cpu_rehearsal and (args.hf is None or args.vllm is None):
        parser.error('a native job needs --hf and --vllm')
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt('signal ' + str(s))))
    raise SystemExit(job(args.payload, args.study, args.out, args.hf, args.vllm, args.cpu_rehearsal))
