"""Stream a subprocess, optionally enforcing an explicitly requested timeout."""
import subprocess
import threading
import time


def run_bounded(command, *, cwd, env, log_path, seconds):
    if seconds is not None and seconds <= 0:
        raise ValueError("Runtime limit must be positive")
    start = time.monotonic()
    expired = threading.Event()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)

    def stop():
        if process.poll() is None:
            expired.set()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    timer = None
    if seconds is not None:
        timer = threading.Timer(seconds, stop)
        timer.daemon = True
        timer.start()
    try:
        with open(log_path, "w") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        code = process.wait()
    finally:
        if timer is not None:
            timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        if timer is not None:
            timer.join(timeout=6)
    return {"exit_code": code, "timed_out": expired.is_set(),
            "seconds": time.monotonic() - start}
