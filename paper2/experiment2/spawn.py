"""Create a process and put it on record with SIGTERM and SIGINT held, so no interruption falls between the two.

This is the M1 rehearsal's spawn helper (paper2/development/m1-timing-rehearsal/spawn.py), accepted by Codex at
df0e7ab after its TURN245 to TURN251 reviews, copied here for the study host; tests/test_study_host.py keeps the two
function bodies identical. spawn() blocks the two signals in the calling thread for the creation and the registration,
then restores the mask, so a held signal is delivered only once the process is on record. If registration fails, the
new process group is killed before the error propagates. The caller must use a handler that only records the request.

A child inherits its parent's blocked signals and keeps them across exec, so every process starts through a short
trampoline that unblocks the two signals and then executes the real command in place: the process number, and so the
record, stays the same, and the worker can be stopped by SIGTERM as usual.
"""
import os
import signal
import subprocess
import sys

HELD = {signal.SIGTERM, signal.SIGINT}
UNMASK_EXEC = ("import os,signal,sys;signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT});"
               "os.execvp(sys.argv[1],sys.argv[1:])")


def spawn(argv, register, popen=None, **kwargs):
    """Start argv in its own session and call register(process) before any held signal can be delivered."""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, HELD)
    try:
        process = (popen or subprocess.Popen)([sys.executable, "-I", "-B", "-c", UNMASK_EXEC, *map(str, argv)],
                                              start_new_session=True, **kwargs)
        try:
            register(process)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        return process
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)
