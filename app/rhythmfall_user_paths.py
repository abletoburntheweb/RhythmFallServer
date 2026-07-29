"""Resolve RhythmFall user:// folder on disk (Windows play build)."""

from __future__ import annotations

import os
from pathlib import Path


def rhythmfall_user_data_dir() -> Path:
    """Godot user data when config/use_custom_user_dir + custom_user_dir_name=RhythmFall."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        custom = Path(appdata) / "RhythmFall"
        legacy = Path(appdata) / "Godot" / "app_userdata" / "RhythmFall"
        if custom.is_dir():
            return custom
        if legacy.is_dir():
            return legacy
        return custom
    return Path.home() / ".local" / "share" / "RhythmFall"


def generation_quality_reports_dir() -> Path:
    return rhythmfall_user_data_dir() / "generation_quality_reports"
