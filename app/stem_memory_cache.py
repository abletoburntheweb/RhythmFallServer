"""Short-lived in-memory stem cache (no persistent temp_uploads reuse on disk)."""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_DIR = Path(os.environ.get("TEMP", "/tmp")) / "RhythmFall" / "stems"


def _ttl_seconds() -> int:
    raw = os.environ.get("RFALL_STEM_CACHE_TTL", "900").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 900


def is_enabled() -> bool:
    return _ttl_seconds() > 0


def ttl_seconds() -> int:
    return _ttl_seconds()

def _purge_expired(now: Optional[float] = None) -> None:
    now = now or time.time()
    expired = [key for key, (_, exp) in _CACHE.items() if exp <= now]
    for key in expired:
        path, _ = _CACHE.pop(key, ("", 0.0))
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


def _audio_key(audio_path: Path) -> str:
    digest = hashlib.sha256()
    with open(audio_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_id(audio_path: str, stem_type: str = "drums") -> Optional[str]:
    path = Path(audio_path)
    if not path.is_file():
        return None
    kind = str(stem_type or "drums").strip().lower() or "drums"
    return f"{_audio_key(path)}:{kind}"


def get_cached_stem(audio_path: str, stem_type: str = "drums") -> Optional[str]:
    if not is_enabled():
        return None
    cache_id = _cache_id(audio_path, stem_type)
    if not cache_id:
        return None
    now = time.time()
    with _LOCK:
        _purge_expired(now)
        entry = _CACHE.get(cache_id)
        if not entry:
            return None
        cached_path, expires_at = entry
        if expires_at <= now or not Path(cached_path).is_file():
            _CACHE.pop(cache_id, None)
            return None
        return cached_path


def store_cached_stem(audio_path: str, stem_path: str, stem_type: str = "drums") -> Optional[str]:
    if not is_enabled():
        return None
    src = Path(stem_path)
    audio = Path(audio_path)
    if not src.is_file() or not audio.is_file():
        return None
    cache_id = _cache_id(audio_path, stem_type)
    if not cache_id:
        return None
    kind = str(stem_type or "drums").strip().lower() or "drums"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / f"{cache_id.split(':', 1)[0][:16]}_{kind}.wav"
    try:
        shutil.copy2(src, dest)
    except Exception:
        return None
    expires_at = time.time() + _ttl_seconds()
    with _LOCK:
        _purge_expired()
        old = _CACHE.get(cache_id)
        if old and old[0] != str(dest):
            try:
                Path(old[0]).unlink(missing_ok=True)
            except Exception:
                pass
        _CACHE[cache_id] = (str(dest), expires_at)
    return str(dest)
