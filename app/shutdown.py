# app/shutdown.py
import threading
import os

_idle_timer: threading.Timer | None = None
_idle_timeout: int = 0
_lock = threading.Lock()


def start_idle_timer(timeout_seconds: int):
    global _idle_timeout
    _idle_timeout = timeout_seconds
    _reset_timer()


def reset_idle_timer():
    if _idle_timeout > 0:
        _reset_timer()


def _reset_timer():
    global _idle_timer
    with _lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
        t = threading.Timer(_idle_timeout, _do_shutdown)
        t.daemon = True
        _idle_timer = t
        t.start()


def _do_shutdown():
    print("[Server] Idle timeout reached — shutting down", flush=True)
    os._exit(0)
