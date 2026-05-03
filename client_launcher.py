"""
RhythmFallServer launcher — include this file in your game client project.

Typical usage in your client:

    from client_launcher import RhythmFallServerLauncher

    launcher = RhythmFallServerLauncher()

    # On "Generate Notes" button click:
    try:
        launcher.ensure_running()        # starts server if not running
        result = launcher.generate_notes("path/to/song.mp3", {
            "original_filename": "song.mp3",
            "lanes": 4,
            "generation_mode": "enhanced",
        })
        notes = result["notes"]
    except Exception as e:
        print("Error:", e)
    # Server shuts itself down automatically after idle_timeout seconds.
    # Or call launcher.shutdown() to close it immediately after the request.
"""

import json
import os
import subprocess
import sys
import time

import requests

SERVER_PORT = 5000
STARTUP_TIMEOUT = 90  # seconds; ML models are slow to load the first time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_server_exe(exe_path: str | None = None) -> str | None:
    if exe_path and os.path.isfile(exe_path):
        return exe_path

    base = os.path.dirname(os.path.abspath(__file__))
    exe_base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else base

    candidates = [
        os.path.join(base, "RhythmFallServer", "RhythmFallServer.exe"),
        os.path.join(base, "RhythmFallServer.exe"),
        os.path.join(exe_base, "RhythmFallServer", "RhythmFallServer.exe"),
        os.path.join(exe_base, "RhythmFallServer.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def is_server_running(port: int = SERVER_PORT) -> bool:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Launcher class
# ---------------------------------------------------------------------------

class RhythmFallServerLauncher:
    """
    Manages the lifecycle of RhythmFallServer.exe for the game client.

    Parameters
    ----------
    port : int
        Port the server listens on (default 5000).
    idle_timeout : int
        Seconds of inactivity before the server shuts itself down.
        0 = never (not recommended for embedded use).
    exe_path : str | None
        Explicit path to RhythmFallServer.exe.
        If None, searches next to this script and next to sys.executable.
    """

    def __init__(
        self,
        port: int = SERVER_PORT,
        idle_timeout: int = 120,
        exe_path: str | None = None,
    ):
        self.port = port
        self.idle_timeout = idle_timeout
        self.exe_path = _find_server_exe(exe_path)
        self._process: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_running(self) -> None:
        """Start the server if it is not already running.

        Blocks until the server responds on /health or raises on failure.
        """
        if is_server_running(self.port):
            return

        if not self.exe_path:
            raise FileNotFoundError(
                "RhythmFallServer.exe not found. "
                "Place the RhythmFallServer/ folder next to your game executable."
            )

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        self._process = subprocess.Popen(
            [
                self.exe_path,
                "--port", str(self.port),
                "--idle-timeout", str(self.idle_timeout),
            ],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if is_server_running(self.port):
                return
            if self._process.poll() is not None:
                raise RuntimeError(
                    "RhythmFallServer.exe exited unexpectedly during startup. "
                    "Check that all Python dependencies are bundled correctly."
                )
            time.sleep(0.5)

        self._process.terminate()
        raise TimeoutError(
            f"RhythmFallServer did not become ready within {STARTUP_TIMEOUT} s. "
            "The ML models may be loading — try increasing STARTUP_TIMEOUT."
        )

    def shutdown(self) -> None:
        """Ask the server to shut down gracefully."""
        try:
            requests.post(f"http://127.0.0.1:{self.port}/shutdown", timeout=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def analyze_bpm(self, audio_path: str) -> dict:
        """Analyze BPM of an audio file. Returns {'bpm': float, ...}."""
        self.ensure_running()
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"http://127.0.0.1:{self.port}/analyze_bpm",
                files={"audio_file": (os.path.basename(audio_path), f)},
                timeout=120,
            )
        response.raise_for_status()
        return response.json()

    def generate_notes(self, audio_path: str, metadata: dict) -> dict:
        """
        Generate drum notes for an audio file.

        Parameters
        ----------
        audio_path : str
            Path to the audio file (.mp3, .wav, .ogg, .flac).
        metadata : dict
            Generation options, e.g.:
            {
                "original_filename": "song.mp3",
                "lanes": 4,                        # 1-8
                "generation_mode": "enhanced",     # minimal/basic/enhanced/natural/custom
                "bpm": 128.0,                      # optional; auto-detected if omitted
                "use_stems": True,
            }

        Returns
        -------
        dict with keys: notes, notes_variants, bpm, lanes, statistics, track_info, status
        """
        self.ensure_running()

        if "original_filename" not in metadata:
            metadata = dict(metadata)
            metadata["original_filename"] = os.path.basename(audio_path)

        with open(audio_path, "rb") as f:
            response = requests.post(
                f"http://127.0.0.1:{self.port}/generate_drums",
                files={"audio_file": (os.path.basename(audio_path), f)},
                data={"metadata": json.dumps(metadata)},
                timeout=600,
            )
        response.raise_for_status()
        return response.json()
