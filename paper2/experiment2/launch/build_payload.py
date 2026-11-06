"""Minimal committed study payload: the launcher, the exact product wheel, the experiment modules the study host and
runner import, and the native containment tests.

Adapted from paper2/development/m1-timing-rehearsal/build_payload.py. Every file comes from one commit by git show; the
wheel must equal that commit's product source file for file. The plan is a committed file in this folder, copied in as
plan.json; its base dependency versions are the ones the launcher checks after installation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

COMMON = '81a5af30969d00e6145457a6d7c09ae713c064f3'     # the independently reviewed native primitives
LAUNCH = 'paper2/experiment2/launch/'
EXPERIMENT = ('__init__.py', 'study_host.py', 'study_runner.py', 'spawn.py', 'run_stream.py', 'mvp_seed.py',
              'streams.py', 'comparators.py', 'comparison.py', 'adviser.py', 'outcomecases.py', 'devcases.py')
PACKAGES = ('adaptive_controller', 'baselines')         # every tracked file under these experiment packages


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze_files(show, commit, plan_body):
    """For a plan that declares a freeze: the frozen protocol, at the path the freeze record binds, from the commit.
    The study runner verifies the freeze against that file under the payload root, so it must be carried exactly."""
    freeze = plan_body.get('freeze')
    if not freeze:
        return {}
    path = freeze['protocol_path']
    if path.startswith('/') or '..' in Path(path).parts:
        raise ValueError('The protocol path must stay inside the payload')
    raw = show(commit + ':' + path)
    if sha(raw) != freeze['protocol_sha256']:
        raise ValueError('The committed protocol does not have the frozen digest')
    return {path: raw}


def build(repo, wheel, plan, out):
    repo, wheel, out = map(lambda p: Path(p).resolve(), (repo, wheel, out))
    if out.exists():
        raise FileExistsError(out)

    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    commit = git('rev-parse', 'HEAD').decode().strip()
    bindings = {}
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel member')
        for name in names:
            if name.startswith('telohelm/') and not name.endswith('/'):
                source = 'paper2/telohelm/src/' + name
                raw = git('show', commit + ':' + source)
                if raw != archive.read(name):
                    raise ValueError('Product wheel differs: ' + name)
                bindings[source] = sha(raw)
            elif not ('.dist-info/' in name and '/' not in name.split('.dist-info/')[0]):
                raise ValueError('Unexpected wheel member')
    tracked = git('ls-tree', '-r', '--name-only', commit, 'paper2/telohelm/src/').decode().splitlines()
    if set(tracked) != set(bindings):
        raise ValueError('Wheel omits committed product content')
    plan_raw = git('show', commit + ':' + LAUNCH + plan)
    plan_body = json.loads(plan_raw)
    files = {wheel.name: wheel.read_bytes(), 'plan.json': plan_raw}
    for name in ('job.py', 'run.sbatch'):
        files[name] = git('show', commit + ':' + LAUNCH + name)
    files['native_common.py'] = git('show', COMMON + ':paper2/development/cartpole-model-test/job.py')
    files['paper2/__init__.py'] = git('show', commit + ':paper2/__init__.py')
    for name in EXPERIMENT:
        files['paper2/experiment2/' + name] = git('show', commit + ':paper2/experiment2/' + name)
    for package in PACKAGES:
        listed = git('ls-tree', '-r', '--name-only', commit, 'paper2/experiment2/' + package + '/').decode().split()
        for path in (p for p in listed if p.endswith('.py')):
            files[path] = git('show', commit + ':' + path)
    files.update(freeze_files(lambda ref: git('show', ref), commit, plan_body))
    for name in ('test_sandbox_linux.py', 'test_broker_channel.py'):
        files['native_tests/' + name] = git('show', commit + ':paper2/telohelm/tests/' + name)
    files['WORKSPACE.md'] = b'# Minimal native test fixture\nNo repository backup or model weights.\n'
    files['coordination/TURN.json'] = b'{"private_containment_sentinel":true}\n'
    files['source.json'] = (json.dumps({'commit': commit, 'product_sha256': bindings,
        'wheel_sha256': sha(wheel.read_bytes()), 'native_common_commit': COMMON,
        'native_common_sha256': sha(files['native_common.py']), 'plan': plan, 'plan_sha256': sha(plan_raw),
        'base_dependencies': plan_body['base_dependencies'], 'scope': plan_body['scope']},
        indent=2, sort_keys=True) + '\n').encode()
    out.mkdir(parents=True)
    for name, raw in files.items():
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    (out / 'manifest.json').write_text(json.dumps({n: sha(r) for n, r in sorted(files.items())},
                                                indent=2, sort_keys=True) + '\n')
    return {'source': commit, 'plan': plan, 'files': len(files) + 1, 'bytes': sum(map(len, files.values())),
            'manifest_sha256': sha((out / 'manifest.json').read_bytes())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('repo', 'wheel', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan', required=True, help='a committed plan file in paper2/experiment2/launch/')
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.wheel, args.plan, args.out), sort_keys=True))
