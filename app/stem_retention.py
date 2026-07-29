"""Disk retention for stem wav files under temp_uploads/*/splitter/."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.Lock()
_REGISTRY_PATH = Path("temp_uploads") / ".stem_retention.json"
_VALID_MODES = frozenset({"after_job", "ttl", "keep_recent", "keep_all"})


def _default_ttl_seconds() -> int:
    raw = os.environ.get("RFALL_STEM_CACHE_TTL", "900").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 900


def _legacy_keep_all_env() -> bool:
    return os.getenv("RFALL_KEEP_TEMP_UPLOADS", "1").strip().lower() in ("1", "true", "yes", "on")


def normalize_policy(metadata: Optional[dict] = None) -> dict[str, Any]:
    meta = metadata if isinstance(metadata, dict) else {}
    keep_all = meta.get("stem_keep_all")
    if keep_all is None:
        keep_all = _legacy_keep_all_env()
    else:
        keep_all = bool(keep_all)

    mode = str(meta.get("stem_retention_mode", "after_job")).strip().lower()
    if mode not in _VALID_MODES:
        mode = "after_job"
    if keep_all:
        mode = "keep_all"

    keep_count = meta.get("stem_keep_count", 10)
    try:
        keep_count = max(1, min(100, int(keep_count)))
    except (TypeError, ValueError):
        keep_count = 10

    ttl_seconds = meta.get("stem_ttl_seconds", _default_ttl_seconds())
    try:
        ttl_seconds = max(0, int(ttl_seconds))
    except (TypeError, ValueError):
        ttl_seconds = _default_ttl_seconds()

    return {
        "mode": mode,
        "keep_all": keep_all,
        "keep_count": keep_count,
        "ttl_seconds": ttl_seconds,
    }


def _load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.is_file():
        return {"recent": [], "scheduled": []}
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"recent": [], "scheduled": []}
    if not isinstance(data, dict):
        return {"recent": [], "scheduled": []}
    recent = data.get("recent", [])
    scheduled = data.get("scheduled", [])
    if not isinstance(recent, list):
        recent = []
    if not isinstance(scheduled, list):
        scheduled = []
    return {"recent": recent, "scheduled": scheduled}


def _save_registry(data: dict[str, Any]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_REGISTRY_PATH)


def _norm_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def find_stem_wavs(song_folder: str | Path) -> list[Path]:
    folder = Path(song_folder)
    splitter = folder / "splitter"
    if not splitter.is_dir():
        return []
    out: list[Path] = []
    for wav in splitter.glob("*.wav"):
        if wav.is_file():
            out.append(wav)
    return out


def _delete_stem_files(paths: list[Path]) -> int:
    removed = 0
    for path in paths:
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
                print(f"[StemRetention] Удалён стем: {path}")
        except Exception as exc:
            print(f"[StemRetention] Не удалось удалить {path}: {exc}")
    return removed


def _remove_from_registry(path_key: str) -> None:
    with _LOCK:
        data = _load_registry()
        data["recent"] = [p for p in data["recent"] if p != path_key]
        data["scheduled"] = [e for e in data["scheduled"] if e.get("path") != path_key]
        _save_registry(data)


def _register_recent(path_key: str, keep_count: int) -> None:
    with _LOCK:
        data = _load_registry()
        recent: list[str] = [p for p in data["recent"] if isinstance(p, str) and p != path_key]
        recent.insert(0, path_key)
        data["recent"] = recent[:keep_count]
        pinned = set(data["recent"])
        _save_registry(data)
    _enforce_keep_recent(pinned, keep_count)


def _schedule_ttl(path_key: str, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return
    expires_at = time.time() + ttl_seconds
    with _LOCK:
        data = _load_registry()
        scheduled = [e for e in data["scheduled"] if e.get("path") != path_key]
        scheduled.append({"path": path_key, "expires_at": expires_at})
        data["scheduled"] = scheduled
        _save_registry(data)
    print(f"[StemRetention] TTL {ttl_seconds}s для {path_key}")


def _enforce_keep_recent(pinned: set[str], keep_count: int) -> None:
    with _LOCK:
        data = _load_registry()
        recent = [p for p in data["recent"] if isinstance(p, str)][:keep_count]
        pinned = set(recent) | pinned
        data["recent"] = list(recent)
        _save_registry(data)

    to_delete: list[Path] = []
    uploads = Path("temp_uploads")
    if not uploads.is_dir():
        return
    for splitter in uploads.glob("*/splitter"):
        if not splitter.is_dir():
            continue
        for wav in splitter.glob("*.wav"):
            key = _norm_path(wav)
            if key not in pinned:
                to_delete.append(wav)
    if to_delete:
        removed = _delete_stem_files(to_delete)
        print(f"[StemRetention] keep_recent: удалено {removed} старых стемов (лимит {keep_count})")


def purge_expired(now: Optional[float] = None) -> int:
    now = now or time.time()
    expired_paths: list[str] = []
    with _LOCK:
        data = _load_registry()
        kept = []
        for entry in data.get("scheduled", []):
            if not isinstance(entry, dict):
                continue
            path_key = str(entry.get("path", ""))
            expires_at = float(entry.get("expires_at", 0.0))
            if path_key and expires_at <= now:
                expired_paths.append(path_key)
            else:
                kept.append(entry)
        data["scheduled"] = kept
        data["recent"] = [p for p in data.get("recent", []) if p not in expired_paths]
        _save_registry(data)

    removed = 0
    for path_key in expired_paths:
        path = Path(path_key)
        if path.is_file():
            removed += _delete_stem_files([path])
        _remove_from_registry(path_key)
    if expired_paths:
        print(f"[StemRetention] TTL истёк: удалено {removed} стемов")
    return removed


def apply_post_job(
    song_folder: str | Path,
    metadata: Optional[dict] = None,
    *,
    success: bool = True,
    use_stems: bool = True,
) -> None:
    purge_expired()
    if not success or not use_stems:
        return

    policy = normalize_policy(metadata)
    mode = policy["mode"]
    stems = find_stem_wavs(song_folder)
    if not stems:
        return

    if mode == "keep_all":
        audio_hint = ""
        folder = Path(song_folder)
        if folder.is_dir():
            audio_files = [
                p.name
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
            ]
            if audio_files:
                audio_hint = f", аудио: {', '.join(audio_files)}"
        print(
            f"[StemRetention] keep_all — {len(stems)} стем(ов) сохранены в {song_folder}{audio_hint}"
        )
        return

    for stem in stems:
        key = _norm_path(stem)
        if mode == "after_job":
            _delete_stem_files([stem])
            _remove_from_registry(key)
        elif mode == "ttl":
            _schedule_ttl(key, policy["ttl_seconds"])
        elif mode == "keep_recent":
            _register_recent(key, policy["keep_count"])

    if mode == "after_job":
        print(f"[StemRetention] after_job — стемы удалены для {song_folder}")
    elif mode == "ttl":
        print(f"[StemRetention] ttl — стемы оставлены на {policy['ttl_seconds']}s")
    elif mode == "keep_recent":
        print(f"[StemRetention] keep_recent — последние {policy['keep_count']} стемов на диске")


def purge_all_stem_wavs() -> dict:
    """Delete ALL stem wav files under temp_uploads/*/splitter/.*.

    This is intended for an interactive "storage reclaim" UI where the user
    explicitly asks to reclaim disk space. Notes (.rf/.rfd) are not affected.
    """
    uploads = Path("temp_uploads")
    if not uploads.is_dir():
        return {"deleted_files": 0, "freed_bytes": 0, "errors": 0}

    stem_wavs: list[Path] = []
    try:
        for splitter in uploads.glob("*/splitter"):
            if splitter.is_dir():
                for wav in splitter.glob("*.wav"):
                    if wav.is_file():
                        stem_wavs.append(wav)
    except Exception:
        # Best-effort: fall back to empty.
        stem_wavs = []

    freed_bytes: int = 0
    for p in stem_wavs:
        try:
            freed_bytes += int(p.stat().st_size)
        except OSError:
            pass

    deleted_files = 0
    errors = 0
    for p in stem_wavs:
        try:
            if p.is_file():
                p.unlink(missing_ok=True)
                deleted_files += 1
        except Exception:
            errors += 1

    # Reset retention registry so it won't keep references to deleted stems.
    try:
        _REGISTRY_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    return {"deleted_files": deleted_files, "freed_bytes": freed_bytes, "errors": errors}
