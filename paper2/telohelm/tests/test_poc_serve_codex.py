"""Native Linux dummy HTTP controls for the job serving helper; no models/GPUs."""
import contextlib
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

SOURCE = Path(os.environ.get('POC_SERVE_SOURCE', next(p for p in Path(__file__).resolve().parents if (p / "WORKSPACE.md").is_file()) / 'dcarr-runtime-0.1/cluster/poc-serve.sh'))
pytestmark = pytest.mark.skipif(sys.platform != 'linux' or not shutil.which('ss'), reason='Linux ss/proc required')

SERVER = '''#!/usr/bin/env python3
import http.server, sys
port = int(sys.argv[sys.argv.index('--port') + 1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data": []}')
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''


@pytest.fixture
def setup(tmp_path):
    job = os.getpid() % 8000 + 1000
    for offset in range(100):
        candidate = (job + offset) % 10000
        ports = (20000 + 2*candidate, 20001 + 2*candidate)
        try:
            with contextlib.ExitStack() as stack:
                for port in ports:
                    s = stack.enter_context(socket.socket())
                    s.bind(('127.0.0.1', port))
        except OSError:
            continue
        job = candidate
        break
    else:
        pytest.fail('no free pair for dummy test')
    logs = tmp_path / 'logs'; logs.mkdir()
    bindir = tmp_path / 'bin'; bindir.mkdir()
    smi = bindir / 'nvidia-smi'; smi.write_text('#!/bin/sh\nprintf "GPU-dummy-%s\\n" "$2"\n'); smi.chmod(0o755)
    vllm = tmp_path / 'envs/vllm/bin/vllm'; vllm.parent.mkdir(parents=True)
    vllm.write_text(SERVER); vllm.chmod(0o755)
    env = dict(os.environ, DCARR_ROOT=str(tmp_path), LOGS=str(logs), SLURM_JOB_ID=str(job),
               CUDA_VISIBLE_DEVICES='0,1', PATH=str(bindir)+':'+os.environ['PATH'],
               TEST_PYTHON=sys.executable)
    return tmp_path, logs, vllm, env, ports


def run(setup, body, timeout=8):
    _, _, _, env, _ = setup
    script = 'set -euo pipefail\nsource "$1"\nsleep() { command sleep .05; }\n' + body
    p = subprocess.Popen(['bash', '-c', script, 'probe', str(SOURCE)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)


@contextlib.contextmanager
def foreign_server(setup):
    root, _, _, _, ports = setup
    script = root / 'foreign.py'; script.write_text(SERVER)
    p = subprocess.Popen([sys.executable, str(script), '--port', str(ports[1])],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    try:
        for _ in range(100):
            assert p.poll() is None, 'dummy foreign server failed'
            try:
                with socket.create_connection(('127.0.0.1', ports[1]), timeout=.1): break
            except OSError: time.sleep(.01)
        else: pytest.fail('dummy foreign server did not start')
        yield p
    finally:
        with contextlib.suppress(ProcessLookupError): os.killpg(p.pid, signal.SIGTERM)
        p.communicate(timeout=2)


def test_owned_listener_is_accepted(setup):
    rc, out, err = run(setup, 'poc_serve\n')
    assert rc == 0, (out, err)
    assert out.count('owns port') == 2


def test_owned_descendant_listener_is_accepted(setup):
    root, _, vllm, env, _ = setup
    server = root / 'descendant.py'; server.write_text(SERVER)
    vllm.write_text('#!/bin/bash\n"$TEST_PYTHON" "$DUMMY_HTTP" "$@" &\nwait $!\n')
    env['DUMMY_HTTP'] = str(server)
    rc, out, err = run(setup, 'poc_serve\n')
    assert rc == 0 and out.count('owns port') == 2, (out, err)


def test_occupied_port_is_refused_before_launch(setup):
    _, logs, _, _, _ = setup
    with foreign_server(setup):
        rc, out, _ = run(setup, 'poc_serve || exit 17\n')
    assert rc == 17 and 'already in use' in out
    assert not list(logs.glob('*.pid'))


@pytest.mark.parametrize('owned_alive', [False, True])
def test_foreign_listener_cannot_pass_ownership_after_port_race(setup, owned_alive):
    _, _, vllm, _, _ = setup
    vllm.write_text('#!/bin/sh\n' + ('exec sleep 60\n' if owned_alive else 'exit 1\n'))
    with foreign_server(setup):
        rc, out, _ = run(setup, 'poc_port_free() { return 0; }\npoc_serve || exit 17\n')
    assert rc == 17 and ('not this job' in out or 'server exited' in out), out


def test_watch_stops_owned_dummy_stream_on_server_death(setup):
    _, logs, _, _, _ = setup
    for role in ('t1', 't2'): (logs/f'{role}-server.pid').write_text('999999999\n')
    rc, out, _ = run(setup, 'command sleep 60 &\nstream=$!\npoc_watch "$stream" &\nwatch=$!\n'
                    'wait "$stream" || stream_rc=$?\nwait "$watch" || watch_rc=$?\n'
                    'test "${stream_rc:-0}" = 143\ntest "${watch_rc:-0}" = 1\n')
    assert rc == 0 and 'stopping the streams' in out
