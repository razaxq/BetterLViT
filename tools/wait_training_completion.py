"""Wait for status-file events, without polling logs, models or GPU state.

Used before the sole final inspection. A successful training exit allows at
most 20 minutes for automatic evaluation; failures wake the inspection at once.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import re
import time


def wait_for_completion(run, grace_seconds=1200):
    if not 0 <= grace_seconds <= 1200:
        raise ValueError('Evaluation grace must stay within 20 minutes')
    run = Path(run).resolve(strict=True)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    pid_fd = None
    try:
        if libc.inotify_add_watch(fd, os.fsencode(run), 0x8 | 0x80) < 0:
            raise OSError(ctypes.get_errno(), 'inotify_add_watch')
        runtime = json.loads((run / 'runtime.json').read_text())
        try:
            if hasattr(os, 'pidfd_open'):
                pid_fd = os.pidfd_open(int(runtime['pid']))
            else:
                # The server's Conda Python omits this wrapper. Use the syscall
                # number declared by its installed Linux x86_64 headers.
                if os.uname().machine != 'x86_64':
                    raise RuntimeError('pidfd fallback supports the verified x86_64 server only')
                header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
                number = int(re.search(r'#define __NR_pidfd_open (\d+)', header).group(1))
                pid_fd = libc.syscall(ctypes.c_long(number), ctypes.c_int(runtime['pid']), ctypes.c_uint(0))
                if pid_fd < 0:
                    pid_fd = None
                    raise OSError(ctypes.get_errno(), 'pidfd_open')
        except ProcessLookupError:
            pass
        while True:
            status_path = run / 'training.status'
            trained_at = status_path.stat().st_mtime if status_path.exists() else None
            code = status_path.read_text().strip() if trained_at is not None else None
            chain = (run / 'chain.status').read_text().strip()
            now = time.time()
            process_dead = pid_fd is None or bool(select.select([pid_fd], [], [], 0)[0])
            reason = None
            if chain == 'complete':
                reason = 'training_and_evaluation_complete'
            elif chain == 'failed' or (code is not None and code not in ('', '0')):
                reason = 'run_failed'
            elif process_dead:
                reason = 'runner_exited'
            elif trained_at is not None and now >= trained_at + grace_seconds:
                reason = 'evaluation_grace_expired'
            if reason:
                return {'ready_for_final_inspection': True, 'reason': reason,
                    'chain_status': chain, 'training_exit_code': code,
                    'training_finished_unix': trained_at, 'signal_received_unix': now,
                    'seconds_since_training_finished': None if trained_at is None else now - trained_at}
            # Sleep in the kernel until a status file closes, the runner exits,
            # or the post-training grace expires. No periodic training queries.
            timeout = None if trained_at is None else max(0, trained_at + grace_seconds - now)
            readers = [fd] + ([] if pid_fd is None else [pid_fd])
            ready, _, _ = select.select(readers, [], [], timeout)
            if fd in ready:
                os.read(fd, 65536)
    finally:
        os.close(fd)
        if pid_fd is not None:
            os.close(pid_fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--grace-seconds', type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(wait_for_completion(args.run, args.grace_seconds)), flush=True)
