# app/bass_utils.py — bass chart I/O and lane mapping.
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.drum_utils import CANONICAL_MAX_LANES, chart_variant_suffix, _resolve_track_labels_from_song_path


def robust_midi_range(
    midis: Sequence[float],
    *,
    lo_pct: float = 8.0,
    hi_pct: float = 92.0,
    min_span: float = 5.0,
) -> Tuple[float, float]:
    """Percentile range so a few outliers don't crush the bass into lanes 0–1."""
    if not midis:
        return 40.0, 52.0
    arr = np.asarray([float(m) for m in midis], dtype=np.float64)
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi - lo < min_span:
        mid = float(np.median(arr))
        half = min_span * 0.5
        lo = mid - half
        hi = mid + half
    return lo, hi


def pitch_to_lane(midi: float, midi_min: float, midi_max: float, lanes: int = 5) -> int:
    """Map MIDI pitch to lane.

    Uses a mild mid-expansion curve so typical bass lines (clustered mid-range)
    spread across more playable lanes instead of stacking on 0–1.
    """
    lanes = max(1, min(int(lanes), CANONICAL_MAX_LANES))
    midi_q = float(midi)
    lo = float(midi_min)
    hi = float(midi_max)
    if hi <= lo:
        return lanes // 2
    span = hi - lo
    # Clamp outliers to the robust range edges (still use edge lanes).
    t = (midi_q - lo) / span
    t = max(0.0, min(1.0, t))
    # Ease-in-out around center → more occupancy on inner lanes without ignoring highs/lows.
    t_spread = 0.5 - 0.5 * np.cos(np.pi * t)
    return int(max(0, min(lanes - 1, round(float(t_spread) * (lanes - 1)))))


def save_bass_notes(
    notes_data: List[Dict[str, Any]],
    song_path: str,
    *,
    chart_intent: Optional[str] = None,
    chart_stem: Optional[str] = None,
    lanes: int = 5,
    artist: str = "",
    title: str = "",
    chart_id: str = "",
) -> bool:
    from app.rfc_chart_codec import write_bass_file
    from app import song_storage

    cid = str(chart_id or "").strip() or song_storage.chart_id_from_song_path(song_path)
    if cid:
        song_folder = song_storage.song_dir(cid)
    else:
        song_folder = Path("temp_uploads") / Path(song_path).stem
    notes_folder = song_folder / "notes"
    notes_folder.mkdir(parents=True, exist_ok=True)

    stem_key = str(chart_stem or chart_intent or "original").strip().lower() or "original"
    variant_suffix = chart_variant_suffix()
    chart_variant = variant_suffix[1:] if variant_suffix.startswith("_") else variant_suffix
    chart_lanes = max(int(lanes), CANONICAL_MAX_LANES)
    notes_path = notes_folder / song_storage.chart_notes_filename("bass", stem_key, chart_lanes, chart_variant)

    def convert_types(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert_types(i) for i in obj]
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj

    serializable = convert_types(notes_data)
    track_artist, track_title = _resolve_track_labels_from_song_path(song_path, artist, title)
    write_bass_file(
        notes_path,
        serializable,
        intent=stem_key,
        lanes=chart_lanes,
        artist=track_artist,
        title=track_title,
    )
    print(f"[BassUtils] Chart saved: {notes_path}")
    return True
