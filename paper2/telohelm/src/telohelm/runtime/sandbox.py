"""Running model-written code inside a contained process (architecture section 6, macOS route).

One call runs one function from untrusted source in a fresh process. The boundary is:
- a macOS sandbox profile that denies everything by default. It allows only executing the Python
  interpreter, reading the interpreter's own installation and the system libraries, and reading the
  root directory's listing, which Python needs at start-up. No other file can be read, listed, stat-ed
  or written, and network, process creation, fork and reading system values (sysctl) are denied. Denying
  sysctl closes a gap found in testing: with it allowed, sandboxed code could list every process and read
  another process's command line;
- an empty environment, no inherited file descriptors besides the stdin and stdout pipes, and the
  root directory as the working directory;
- limits on processor time, wall-clock time, open files, file size and the size of the reply;
- a memory limit enforced from outside: macOS refuses to lower a process's memory limits, so the parent
  reads the child's resident memory every 50 ms, through libproc, and kills it above MEMORY_LIMIT_MB. An allocation can
  overshoot by what the child touches between two readings.

The code receives one JSON input and returns one JSON value. It never sees a file path, the world, the
monitor or the logs. Whether the boundary holds is shown by the containment tests in
tests/test_sandbox.py, not by this description.

The Linux route (TURN 117, for the cluster; tests in tests/test_sandbox_linux.py) builds the same boundary from
unprivileged kernel features, with no change to the machine and no special rights:
- each call starts in new user, network, process-ID, mount, IPC and host-name namespaces (unshare). The network
  namespace has only a loopback that is down, so there is no network;
- inside them a trusted setup script mounts an empty in-memory root, binds into it, read-only, only the Python
  installation, the system libraries and /dev/null and /dev/urandom, and then makes the root read-only too. No other
  file of the machine exists there; /proc is not mounted, so no other process can be seen;
- the process then changes into that root and starts Python, whose first lines (LINUX_DROP, trusted) drop every
  privilege before the runner reads any model-written code: an empty capability bounding set, no capabilities of any
  kind, root in the namespace given none back on exec (the noroot securebits, locked), and no new privileges ever. So
  it cannot undo the mounts. If any step fails, or a capability is still held afterwards, the process exits before
  the runner starts. setpriv cannot do this inside the root, because its library reads /proc, which is not there;
- limits on processor time, open files and file size as on macOS; at most one process (RLIMIT_NPROC, set in the same
  first lines), so it cannot start another; and an address-space limit that makes an oversized allocation fail
  inside the process, reported as status 'memory'. macOS refuses to set that limit, which is why it polls memory
  from outside.
The route is available only where unshare and chroot exist and unprivileged user namespaces are enabled; otherwise
SandboxUnavailable is raised, as before.
"""
from __future__ import annotations

import ctypes
import json
import os
import resource
import select
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CPU_SECONDS = 2
WALL_SECONDS = 5.0
MAX_REPLY_BYTES = 256_000
MAX_OPEN_FILES = 32
MEMORY_LIMIT_MB = 512
MEMORY_POLL_S = 0.05

RUNNER = r'''
import json, sys

class Ctx:
    """Read-only record with attribute access only (backlog note B7)."""
    __slots__ = ("_d",)
    def __init__(self, d):
        object.__setattr__(self, "_d", d)
    def __getattr__(self, k):
        d = object.__getattribute__(self, "_d")
        if k in d:
            return wrap(d[k])
        raise AttributeError(k)
    def __setattr__(self, k, v):
        raise AttributeError("context is read-only")

def wrap(v):
    if isinstance(v, dict):
        return Ctx(v)
    if isinstance(v, list):
        return tuple(wrap(x) for x in v)
    return v

msg = json.loads(sys.stdin.read())
ns = {"__name__": "procedure"}
try:
    exec(compile(msg["code"], "<procedure>", "exec"), ns)
    fn = ns[msg["entry"]]
    args = [wrap(a) for a in msg["input"]] if msg["wrap"] else [msg["input"]]
    out = {"ok": True, "result": fn(*args)}
except BaseException as e:
    out = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:500]}
try:
    text = json.dumps(out, allow_nan=False)
except (TypeError, ValueError) as e:
    text = json.dumps({"ok": False, "error": "result is not JSON: " + str(e)[:200]})
sys.stdout.write(text)
'''


# The broker channel (TURN 56; paper2/notes/BROKER-CONTRACT.md). Every message in either direction is one frame:
# a kind byte, ten ASCII digits giving the body's length, then the body. The parent sends M (the call) and A
# (each broker reply); the child sends R (each broker request) and F (its final result). predict(args) is the
# only name the procedure gets for it. Anything else on the pipe is a protocol violation and ends the call.
FRAME_HEAD = 11
RUNNER_BROKER = r"""
import json, sys
_in, _out = sys.stdin.buffer, sys.stdout.buffer

def _frame(kind):
    head = _in.read(11)
    if len(head) != 11 or head[:1] != kind or not head[1:].isdigit():
        raise SystemExit(3)
    n = int(head[1:])
    body = _in.read(n)
    if len(body) != n:
        raise SystemExit(3)
    return body

def _send(kind, body):
    _out.write(kind + b"%010d" % len(body) + body)
    _out.flush()

class Ctx:
    __slots__ = ("_d",)
    def __init__(self, d):
        object.__setattr__(self, "_d", d)
    def __getattr__(self, k):
        d = object.__getattribute__(self, "_d")
        if k in d:
            return wrap(d[k])
        raise AttributeError(k)
    def __setattr__(self, k, v):
        raise AttributeError("context is read-only")

def wrap(v):
    if isinstance(v, dict):
        return Ctx(v)
    if isinstance(v, list):
        return tuple(wrap(x) for x in v)
    return v

def predict(args):
    _send(b"R", json.dumps({"capability": "predict", "args": args}, allow_nan=False).encode())
    return json.loads(_frame(b"A"))

msg = json.loads(_frame(b"M"))
ns = {"__name__": "procedure", "predict": predict}
try:
    exec(compile(msg["code"], "<procedure>", "exec"), ns)
    fn = ns[msg["entry"]]
    args = [wrap(a) for a in msg["input"]] if msg["wrap"] else [msg["input"]]
    out = {"ok": True, "result": fn(*args)}
except BaseException as e:
    out = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:500]}
try:
    text = json.dumps(out, allow_nan=False)
except (TypeError, ValueError) as e:
    text = json.dumps({"ok": False, "error": "result is not JSON: " + str(e)[:200]})
_send(b"F", text.encode())
"""


class SandboxUnavailable(RuntimeError):
    """No tested containment route exists on this platform. Model-written code must not run."""


@dataclass(frozen=True)
class SandboxResult:
    status: str               # 'ok', 'error' (the code raised or returned non-JSON), 'timeout', 'memory', 'crashed', 'bad_reply'
    result: Any = None
    error: str | None = None
    wall_seconds: float = 0.0
    cpu_seconds: float | None = None   # None when not measured, never an unmeasured zero
    reply_bytes: int = 0
    broker_requests: int = 0           # requests the child sent over the broker channel


def interpreter() -> Path:
    """The framework Python binary itself; bin/python3.12 re-launches it, which the profile would refuse."""
    base = Path(os.path.realpath(sys.base_prefix))
    inner = base / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    return inner if inner.exists() else Path(os.path.realpath(sys.executable))


def _ancestors(path: Path) -> list[str]:
    return [str(p) for p in reversed(path.parents)]


def profile(python: Path) -> str:
    """The sandbox policy, deny by default."""
    base = Path(os.path.realpath(sys.base_prefix))
    install = base.parents[3] if len(base.parents) > 3 else base    # e.g. .../Cellar/python@3.12
    reads = [str(install), "/usr/lib", "/System"]
    if str(install).startswith("/opt/homebrew/"):
        reads.append("/opt/homebrew/opt")                           # Homebrew's links to shared libraries
    devices = ["/dev/null", "/dev/urandom", "/dev/random"]
    meta = sorted({a for r in reads for a in _ancestors(Path(r))} | {"/dev", "/private", "/var", "/etc", "/tmp"})
    q = lambda s: json.dumps(s)
    return "\n".join([
        "(version 1)",
        "(deny default)",
        f"(allow process-exec (literal {q(str(python))}))",
        "(allow file-read* (literal \"/\") " + " ".join(f"(subpath {q(r)})" for r in reads)
        + " (subpath \"/private/var/db/dyld\") " + " ".join(f"(literal {q(d)})" for d in devices) + ")",
        "(allow file-read-metadata " + " ".join(f"(literal {q(m)})" for m in meta)
        + " " + " ".join(f"(subpath {q(r)})" for r in reads) + " (subpath \"/private/var/db/dyld\") "
        + " ".join(f"(literal {q(d)})" for d in devices) + ")",
    ])


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (MAX_OPEN_FILES, MAX_OPEN_FILES))
    if sys.platform == "linux":            # Linux can limit address space; macOS refuses (see _watch_memory)
        limit = LINUX_ADDRESS_SPACE_MB * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16), ("ri_user_time", ctypes.c_uint64),
                ("ri_system_time", ctypes.c_uint64), ("ri_pkg_idle_wkups", ctypes.c_uint64),
                ("ri_interrupt_wkups", ctypes.c_uint64), ("ri_pageins", ctypes.c_uint64),
                ("ri_wired_size", ctypes.c_uint64), ("ri_resident_size", ctypes.c_uint64),
                ("ri_phys_footprint", ctypes.c_uint64), ("ri_proc_start_abstime", ctypes.c_uint64),
                ("ri_proc_exit_abstime", ctypes.c_uint64)]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None


def resident_mb(pid: int) -> float | None:
    """The process's resident memory in megabytes, None once it has gone. On macOS this reads libproc's
    proc_pid_rusage in the parent (about 1.6 µs; launching ps took about 4.7 ms and slowed every call)."""
    if _LIBPROC is not None:
        info = _RusageInfoV0()
        if _LIBPROC.proc_pid_rusage(pid, 0, ctypes.byref(info)) != 0:
            return None
        return info.ri_resident_size / 1024**2
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else None


def _watch_memory(proc: subprocess.Popen, limit_mb: float, flag: dict) -> None:
    while proc.poll() is None:
        mb = resident_mb(proc.pid)
        if mb is not None:
            flag["peak_mb"] = max(flag.get("peak_mb", 0.0), mb)
            if mb > limit_mb:
                flag["exceeded"] = True
                proc.kill()
                return
        time.sleep(MEMORY_POLL_S)


LINUX_TOOLS = ("unshare", "chroot")
LINUX_ADDRESS_SPACE_MB = 1024          # the address-space limit; resident memory stays well below it
LINUX_ROOT = "/mnt"                    # where the empty root is mounted, inside the private mount namespace


def _linux_available() -> bool:
    if sys.platform != "linux" or not all(shutil.which(t) for t in LINUX_TOOLS):
        return False
    flag = Path("/proc/sys/kernel/unprivileged_userns_clone")
    return not flag.exists() or flag.read_text().strip() != "0"


def available() -> bool:
    if sys.platform == "darwin":
        return shutil.which("sandbox-exec") is not None
    return _linux_available()


# The trusted setup run inside the new namespaces, before any model-written code. $1 is the interpreter, $2 the
# privilege drop, $3 the runner source, $4 the interpreter's installation. Every bind is made read-only, then the
# root itself; the process changes root and starts Python with the drop, which then runs the runner.
LINUX_SETUP = r"""set -eu
R=__ROOT__
PY="$1"; DROP="$2"; RUN="$3"; INSTALL="$4"
mount -t tmpfs -o size=1m,mode=0755 telohelm-sandbox "$R"
mount --bind "$R" "$R"
mkdir -p "$R/dev" "$R$INSTALL"
for d in /usr/lib /usr/lib64 "$INSTALL"; do
  [ -d "$d" ] || continue
  mkdir -p "$R$d"
  mount --bind "$d" "$R$d"
  mount -o remount,bind,ro,nosuid,nodev "$R$d"
done
[ -e "$R/lib" ] || ln -s usr/lib "$R/lib"
[ -e "$R/lib64" ] || ln -s usr/lib64 "$R/lib64"
for f in /dev/null /dev/urandom; do
  : > "$R$f"
  mount --bind "$f" "$R$f"
  mount -o remount,bind,ro "$R$f"
done
mount -o remount,bind,ro,nosuid,nodev "$R"
exec chroot "$R" "$PY" -I -S -c "$DROP" "$RUN"
""".replace("__ROOT__", LINUX_ROOT)

# The first lines Python runs inside the root, while it is still root in the namespace. The order matters: dropping
# from the bounding set and setting securebits need CAP_SETPCAP, so the capabilities themselves go last. Then the
# capabilities are read back and must all be zero. The runner's source is the one argument; it runs as __main__.
LINUX_DROP = r"""
import ctypes, resource, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
def check(ok, what):
    if not ok:
        sys.stderr.write("privilege drop failed: %s (errno %d)\n" % (what, ctypes.get_errno()))
        sys.stderr.flush()
        import os; os._exit(97)
cap = 0
while libc.prctl(24, cap, 0, 0, 0) == 0:                 # PR_CAPBSET_DROP, until the number is past the last one
    cap += 1
check(ctypes.get_errno() == 22 and cap >= 38, "bounding set")
check(libc.prctl(47, 4, 0, 0, 0) == 0, "ambient set")     # PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL
check(libc.prctl(28, 0xEF, 0, 0, 0) == 0, "securebits")   # PR_SET_SECUREBITS: noroot, no setuid fixup, keep caps off,
                                                           # no ambient raise, each locked
class Header(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
Data = (ctypes.c_uint32 * 3) * 2                           # effective, permitted, inheritable; two 32-bit halves
check(libc.capset(ctypes.byref(Header(0x20080522, 0)), ctypes.byref(Data())) == 0, "capset")
held = Data()
check(libc.capget(ctypes.byref(Header(0x20080522, 0)), ctypes.byref(held)) == 0, "capget")
check(not any(v for half in held for v in half), "capabilities still held")
check(libc.prctl(38, 1, 0, 0, 0) == 0, "no new privileges")   # PR_SET_NO_NEW_PRIVS
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
source = sys.argv.pop(1)
del libc, check, cap, held, Header, Data
exec(compile(source, "<runner>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
"""


def _command(python: Path, runner: str) -> list[str]:
    """The command that runs the runner in the contained process on this platform."""
    if sys.platform == "darwin":
        return ["sandbox-exec", "-p", profile(python), str(python), "-I", "-S", "-c", runner]
    install = str(python.parents[1])           # .../cpython-3.12.x/ holds bin/ and lib/
    return ["unshare", "--user", "--map-root-user", "--net", "--pid", "--fork", "--kill-child", "--mount",
            "--ipc", "--uts", "sh", "-c", LINUX_SETUP, "sh", str(python), LINUX_DROP, runner, install]


_ONE_CHILD = threading.Lock()


def run(code: str, entry: str, payload: Any, wall_seconds: float = WALL_SECONDS,
        context_args: bool = False) -> SandboxResult:
    """Calls through this function are serialised. The CPU figure is the difference in the process's completed-
    children usage around one call, so it belongs to that call's child only if no other child process of this
    parent finishes meanwhile. That exclusive-child condition is assumed, not enforced, for any other subprocess
    (Codex's TURN 26 note). The recorded wall time starts after any wait for the lock."""
    with _ONE_CHILD:
        return _run(code, entry, payload, wall_seconds, context_args)


def _run(code: str, entry: str, payload: Any, wall_seconds: float, context_args: bool) -> SandboxResult:
    """Run entry(payload) from code in a contained process and return its JSON result.

    With context_args, payload is a list of records, each passed as a read-only attribute-only context
    object (backlog note B7), so that decide(ctx) or check(ctx, out) can be called."""
    if not available():
        raise SandboxUnavailable(f"no tested containment route on {sys.platform}")
    python = interpreter()
    # Every call, live or replay, sends a message of exactly this form, so the running code cannot tell a
    # test from real work by anything the runner holds (Codex's R22.2 read the runner's frame; batch mode,
    # which a replay used, is removed for that reason).
    message = json.dumps({"code": code, "entry": entry, "input": payload, "wrap": context_args}, allow_nan=False)
    command = _command(python, RUNNER)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    deadline = start + wall_seconds
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            env={}, cwd="/", close_fds=True, preexec_fn=_limits)
    memory: dict = {}
    if sys.platform == "darwin":           # on Linux the address-space limit holds memory inside the process
        watcher = threading.Thread(target=_watch_memory, args=(proc, MEMORY_LIMIT_MB, memory), daemon=True)
        watcher.start()

    def cpu_used() -> float:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        return (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)

    def stop(status: str, error: str) -> SandboxResult:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        return SandboxResult(status, error=error, wall_seconds=time.perf_counter() - start, cpu_seconds=cpu_used())

    # Write the message and read the reply with non-blocking pipes under one deadline, in bounded chunks, so that
    # neither a slow child, a blocked write nor a large or slow reply can hold the parent past the deadline or fill
    # its memory (Codex's R22.4, R22.8 and R24.IO: a blocking write before the read loop could overrun).
    pending = message.encode()
    out_fd, in_fd = proc.stdout.fileno(), proc.stdin.fileno()
    os.set_blocking(in_fd, False)
    os.set_blocking(out_fd, False)
    writing = True
    chunks, size = [], 0
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return stop("timeout", "wall-clock limit")
        readable, writable, _ = select.select([out_fd], [in_fd] if writing else [], [], remaining)
        if writable:
            try:
                sent = os.write(in_fd, pending[:65536])
                pending = pending[sent:]
            except BlockingIOError:
                pass
            except BrokenPipeError:
                writing = False
                proc.stdin.close()
            if writing and not pending:
                writing = False
                proc.stdin.close()
        if readable:
            try:
                chunk = os.read(out_fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_REPLY_BYTES:
                return stop("bad_reply", "reply too large")
            chunks.append(chunk)
    try:
        proc.wait(timeout=max(0.0, deadline - time.perf_counter()))
    except subprocess.TimeoutExpired:
        return stop("timeout", "wall-clock limit after output")
    out = b"".join(chunks)
    wall = time.perf_counter() - start
    cpu = cpu_used()
    # No join: the watcher sleeps between readings, and waiting for it added about 25 ms to every call.
    # It sets the flag before it kills the child, so the flag is already set if memory was the cause.
    if memory.get("exceeded"):
        return SandboxResult("memory", error=f"resident memory above {MEMORY_LIMIT_MB} MB", wall_seconds=wall,
                             cpu_seconds=cpu)
    if proc.returncode != 0:
        status = "timeout" if proc.returncode in (-24, -9) else "crashed"   # SIGXCPU or kill at the CPU limit
        return SandboxResult(status, error=f"exit {proc.returncode}", wall_seconds=wall, cpu_seconds=cpu)
    if len(out) > MAX_REPLY_BYTES:
        return SandboxResult("bad_reply", error="reply too large", wall_seconds=wall, cpu_seconds=cpu,
                             reply_bytes=len(out))
    try:
        reply = json.loads(out)
    except ValueError:
        return SandboxResult("bad_reply", error="reply is not JSON", wall_seconds=wall, cpu_seconds=cpu,
                             reply_bytes=len(out))
    if reply.get("ok"):
        return SandboxResult("ok", result=reply["result"], wall_seconds=wall, cpu_seconds=cpu, reply_bytes=len(out))
    if _out_of_memory(reply.get("error")):
        return SandboxResult("memory", error=reply.get("error"), wall_seconds=wall, cpu_seconds=cpu, reply_bytes=len(out))
    return SandboxResult("error", error=reply.get("error"), wall_seconds=wall, cpu_seconds=cpu, reply_bytes=len(out))


def _out_of_memory(error: Any) -> bool:
    """On Linux the address-space limit turns an oversized allocation into MemoryError inside the process."""
    return sys.platform == "linux" and isinstance(error, str) and error.startswith("MemoryError")


def _final_envelope(body: bytes) -> dict | None:
    """The final frame must be exactly {"ok": true, "result": <finite JSON>} or {"ok": false, "error": <text>}, with
    no other keys, no repeated keys and no NaN or infinity; anything else is refused (Codex's BR57.FINAL)."""
    def no_constant(name):
        raise ValueError(name)

    def unique(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("duplicate key")
        return dict(pairs)
    try:
        reply = json.loads(body, parse_constant=no_constant, object_pairs_hook=unique)
    except (ValueError, RecursionError):
        return None
    if not isinstance(reply, dict) or not isinstance(reply.get("ok"), bool):
        return None
    if reply["ok"] is True and set(reply) == {"ok", "result"}:
        try:
            json.dumps(reply["result"], allow_nan=False)
        except (ValueError, OverflowError):
            return None
        return reply
    if reply["ok"] is False and set(reply) == {"ok", "error"} and isinstance(reply["error"], str):
        return reply
    return None


def run_with_broker(code: str, entry: str, payload: Any, broker: Any, purpose: str,
                    wall_seconds: float = WALL_SECONDS, context_args: bool = False) -> SandboxResult:
    """Like run(), with a broker channel: the procedure may call predict(args), which reaches the broker as one
    request and returns its reply. The broker call is opened and closed here, with the sandbox's outcome as its
    status. One deadline covers the whole exchange: writing the call, every request and reply, the broker's own
    work and the child's exit. The trusted tool is not pre-empted; if it overruns, the deadline is found when it
    returns. Calls are serialised with run()."""
    with _ONE_CHILD:
        call = broker.open_call(purpose)
        try:
            result = _run_channel(code, entry, payload, call, wall_seconds, context_args)
        except BaseException as exc:
            broker.close_call(call, "interrupted", f"{type(exc).__name__}: {exc}")
            raise
        broker.close_call(call, {"ok": "completed"}.get(result.status, result.status), result.error)
        return result


def _run_channel(code: str, entry: str, payload: Any, call: Any, wall_seconds: float,
                 context_args: bool) -> SandboxResult:
    from .broker import MAX_MESSAGE_BYTES; from .broker import MAX_REPLY_BYTES as BROKER_REPLY_BYTES
    if not available():
        raise SandboxUnavailable(f"no tested containment route on {sys.platform}")
    python = interpreter()
    message = json.dumps({"code": code, "entry": entry, "input": payload, "wrap": context_args}, allow_nan=False)
    command = _command(python, RUNNER_BROKER)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    deadline = start + wall_seconds
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            env={}, cwd="/", close_fds=True, preexec_fn=_limits)
    memory: dict = {}
    if sys.platform == "darwin":           # on Linux the address-space limit holds memory inside the process
        watcher = threading.Thread(target=_watch_memory, args=(proc, MEMORY_LIMIT_MB, memory), daemon=True)
        watcher.start()
    requests = 0

    def frame(kind: bytes, body: bytes) -> bytes:
        return kind + b"%010d" % len(body) + body

    def cpu_used() -> float:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        return (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)

    def stop(status: str, error: str) -> SandboxResult:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        return SandboxResult(status, error=error, wall_seconds=time.perf_counter() - start, cpu_seconds=cpu_used(),
                             broker_requests=requests)

    try:
        pending = frame(b"M", message.encode())
        out_fd, in_fd = proc.stdout.fileno(), proc.stdin.fileno()
        os.set_blocking(in_fd, False)
        os.set_blocking(out_fd, False)
        buf = b""
        final = None
        while final is None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return stop("timeout", "wall-clock limit")
            readable, writable, _ = select.select([out_fd], [in_fd] if pending else [], [], remaining)
            if writable and pending:
                try:
                    pending = pending[os.write(in_fd, pending[:65536]):]
                except BlockingIOError:
                    pass
                except BrokenPipeError:
                    pending = b""
            if readable:
                try:
                    chunk = os.read(out_fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                buf += chunk
            while len(buf) >= FRAME_HEAD and final is None:
                kind, digits = buf[:1], buf[1:FRAME_HEAD]
                if kind not in (b"R", b"F") or not digits.isdigit():
                    return stop("bad_reply", "protocol violation on the broker channel")
                n = int(digits)
                if n > (MAX_MESSAGE_BYTES if kind == b"R" else MAX_REPLY_BYTES):
                    return stop("bad_reply", "frame too large")
                if len(buf) < FRAME_HEAD + n:
                    break
                body, buf = buf[FRAME_HEAD:FRAME_HEAD + n], buf[FRAME_HEAD + n:]
                if kind == b"F":
                    final = body
                    break
                requests += 1
                reply = call.request(body.decode("utf-8", "surrogateescape")).encode()
                if len(reply) > BROKER_REPLY_BYTES:          # the broker refuses first; this is a second guard
                    return stop("bad_reply", "broker reply over the channel bound")
                pending += frame(b"A", reply)
                if time.perf_counter() > deadline:
                    return stop("timeout", "wall-clock limit during a broker request")
        if final is None:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.perf_counter()))
                except subprocess.TimeoutExpired:      # output closed but the child lives on (Codex's BR57.EOF)
                    return stop("timeout", "output closed without a final frame; wall-clock limit")
            wall, cpu = time.perf_counter() - start, cpu_used()
            if memory.get("exceeded"):
                return SandboxResult("memory", error=f"resident memory above {MEMORY_LIMIT_MB} MB", wall_seconds=wall,
                                     cpu_seconds=cpu, broker_requests=requests)
            status = "timeout" if proc.returncode in (-24, -9) else "crashed"
            return SandboxResult(status, error=f"no final frame; exit {proc.returncode}", wall_seconds=wall,
                                 cpu_seconds=cpu, broker_requests=requests)
        if buf:
            return stop("bad_reply", "bytes after the final frame")
        # Read on to end of file under the same deadline, so bytes written after the final frame, even late, are
        # found (Codex's BR57.DRAIN).
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return stop("timeout", "wall-clock limit after the final frame")
            readable, _, _ = select.select([out_fd], [], [], remaining)
            if not readable:
                continue
            try:
                chunk = os.read(out_fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                break
            return stop("bad_reply", "bytes after the final frame")
        try:
            proc.wait(timeout=max(0.0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            return stop("timeout", "wall-clock limit after output")
        wall, cpu = time.perf_counter() - start, cpu_used()
        if memory.get("exceeded"):
            return SandboxResult("memory", error=f"resident memory above {MEMORY_LIMIT_MB} MB", wall_seconds=wall,
                                 cpu_seconds=cpu, broker_requests=requests)
        if proc.returncode != 0:
            return SandboxResult("crashed", error=f"exit {proc.returncode}", wall_seconds=wall, cpu_seconds=cpu,
                                 broker_requests=requests)
        reply = _final_envelope(final)
        if reply is None:
            return SandboxResult("bad_reply", error="final frame is not a valid envelope", wall_seconds=wall,
                                 cpu_seconds=cpu, reply_bytes=len(final), broker_requests=requests)
        if reply["ok"]:
            return SandboxResult("ok", result=reply["result"], wall_seconds=wall, cpu_seconds=cpu,
                                 reply_bytes=len(final), broker_requests=requests)
        status = "memory" if _out_of_memory(reply["error"]) else "error"
        return SandboxResult(status, error=reply["error"], wall_seconds=wall, cpu_seconds=cpu,
                             reply_bytes=len(final), broker_requests=requests)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        raise

