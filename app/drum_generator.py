import numpy as np
from typing import List, Dict, Optional

from .audio_analysis import analyze_audio
from .drum_utils import (
    apply_temporal_filter,
    apply_groove_pattern,
    assign_lanes_to_notes,
    detect_drum_section_start,
    save_drums_notes,
)
from .note_types import NoteType
from .genre_detector import get_genre_config

MODE_PRESETS: Dict[str, Dict[str, int]] = {
    "minimal":  {"fill": 0,  "groove": 20, "density": 30, "accent_strong_beats": 1, "genre_template_strength": 45},
    "basic":    {"fill": 0,  "groove": 50, "density": 50, "accent_strong_beats": 1, "genre_template_strength": 60},
    "enhanced": {"fill": 75, "groove": 55, "density": 70, "accent_strong_beats": 0, "genre_template_strength": 80},
    "natural":  {"fill": 0,  "groove": 50, "density": 50, "accent_strong_beats": 0, "genre_template_strength": 20},
}

_HARD_CAPS: Dict[str, Dict] = {
    "pop":           {"min": 3, "max": 6,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.30},
    "hyperpop":      {"min": 3, "max": 6,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.30},
    "k-pop":         {"min": 3, "max": 6,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.30},
    "j-pop":         {"min": 3, "max": 6,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.30},
    "electronic":    {"min": 5, "max": 8,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
    "house":         {"min": 5, "max": 8,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
    "techno":        {"min": 5, "max": 8,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
    "trance":        {"min": 5, "max": 8,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
    "drum and bass": {"min": 6, "max": 10, "per_measure": 3, "per_measure_break": 5, "cap_ratio": 0.28},
    "rap":           {"min": 3, "max": 6,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.22},
    "r&b":           {"min": 3, "max": 6,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.22},
    "rock":          {"min": 3, "max": 7,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.20},
    "metal":         {"min": 3, "max": 7,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.20},
    "hardcore":      {"min": 3, "max": 7,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.20},
    "default":       {"min": 4, "max": 7,  "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
}

_GENRE_PATTERN_POSITIONS: Dict[str, List[float]] = {
    "electronic":    [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
    "house":         [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
    "techno":        [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
    "trance":        [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
    "drum and bass": [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.0, 1.25, 1.75, 2.0, 2.25, 2.75, 3.0, 3.25, 3.75],
    "funk":          [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.0, 1.25, 1.75, 2.0, 2.25, 2.75, 3.0, 3.25, 3.75],
    "rap":           [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.75, 1.25, 2.75, 3.25],
    "r&b":           [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.75, 1.25, 2.75, 3.25],
    "default":       [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
}


def _presets_for_mode(
    generation_mode: str,
    fill: Optional[int],
    groove: Optional[int],
    density: Optional[int],
    accent_strong_beats: Optional[bool],
    genre_template_strength: Optional[int],
):
    mode = (generation_mode or "basic").lower()
    if mode != "custom":
        preset = MODE_PRESETS.get(mode, MODE_PRESETS["basic"])
        return (
            int(preset["fill"]),
            int(preset["groove"]),
            int(preset["density"]),
            bool(preset.get("accent_strong_beats", 0)),
            int(preset.get("genre_template_strength", 60)),
        )
    base = MODE_PRESETS["basic"]
    return (
        int(fill if fill is not None else base["fill"]),
        int(groove if groove is not None else base["groove"]),
        int(density if density is not None else base["density"]),
        bool(accent_strong_beats if accent_strong_beats is not None else bool(base.get("accent_strong_beats", 0))),
        int(genre_template_strength if genre_template_strength is not None else int(base.get("genre_template_strength", 60))),
    )


def _density_to_min_distance(base_distance: float, density: int) -> float:
    d = max(0, min(100, int(density)))
    scale = (d - 50) / 50.0
    factor = 1.0 - 0.4 * scale
    value = base_distance * factor
    return max(0.035, min(0.22, value))


def _mode_distance_multiplier(generation_mode: str) -> float:
    mode = (generation_mode or "basic").lower()
    if mode == "minimal":
        return 1.45
    if mode == "basic":
        return 1.0
    if mode == "enhanced":
        return 0.95
    return 1.2


def _has_near(t: float, existing: List[float], tol: float) -> bool:
    return any(abs(t - x) <= tol for x in existing)


def _count_in_window(times: List[float], start: float, end: float) -> int:
    return sum(1 for t in times if start <= t < end)


def _measure_bounds(start_time: float, end_time: float, beat_interval: float) -> List:
    measure_duration = beat_interval * 4
    bounds, current = [], start_time
    while current <= end_time:
        bounds.append((current, current + measure_duration))
        current += measure_duration
    return bounds


def _timing_flags_from_genre(genre_params: Dict, sync_tolerance: float):
    pattern_style = genre_params.get("pattern_style", "groove")
    use_grid_sync = bool(genre_params.get("sync_to_beats", False))
    apply_groove = bool(genre_params.get("apply_groove_pattern", False))
    adjusted_tolerance = float(sync_tolerance) * float(genre_params.get("sync_tolerance_multiplier", 1.0))
    return pattern_style, use_grid_sync, apply_groove, adjusted_tolerance


def _pull_to_grid(events: List[float], beats: np.ndarray, tolerance: float, strength: int) -> List[float]:
    if not events or beats is None or len(beats) == 0:
        return events
    if strength <= 0:
        return events
    alpha = max(0.0, min(1.0, float(strength) / 100.0))
    tol = max(0.0, float(tolerance))
    pulled: List[float] = []
    for t in events:
        distances = np.abs(beats - t)
        idx = int(np.argmin(distances))
        nearest = float(beats[idx])
        dist = float(distances[idx])
        if dist <= tol:
            pulled.append(float(t + (nearest - t) * alpha))
        else:
            pulled.append(float(t))
    return pulled


def _accent_to_strong_beats(events: List[float], beats: np.ndarray, tolerance: float, strength: int = 70) -> List[float]:
    if not events or beats is None or len(beats) < 2:
        return events
    strong_beats = beats[::2]
    if len(strong_beats) == 0:
        return events
    alpha = max(0.0, min(1.0, float(strength) / 100.0))
    tol = max(0.0, float(tolerance)) * 1.5
    accented: List[float] = []
    for t in events:
        distances = np.abs(strong_beats - t)
        idx = int(np.argmin(distances))
        nearest = float(strong_beats[idx])
        dist = float(distances[idx])
        if dist <= tol:
            accented.append(float(t + (nearest - t) * alpha))
        else:
            accented.append(float(t))
    return accented


def _merge_pattern_and_onsets(pattern: List[float], onset: List[float], strength: int) -> List[float]:
    s = max(0, min(100, int(strength)))
    if s >= 70:
        return pattern + onset
    if s <= 30:
        return onset + pattern
    merged: List[float] = []
    i, j = 0, 0
    p_take = max(1, int(round(1 + (s - 50) / 25.0)))
    o_take = 1
    while i < len(pattern) or j < len(onset):
        for _ in range(p_take):
            if i < len(pattern):
                merged.append(pattern[i])
                i += 1
        for _ in range(o_take):
            if j < len(onset):
                merged.append(onset[j])
                j += 1
        if i >= len(pattern) and j >= len(onset):
            break
    return merged


def _sparsify_by_beats(events: List[float], beats: np.ndarray, bpm: float, max_per_beat: int = 1) -> List[float]:
    if not events:
        return events
    max_per_beat = max(1, int(max_per_beat))
    if beats is not None and len(beats) >= 2:
        sparse: List[float] = []
        i = 0
        ev = sorted(events)
        for b in range(len(beats) - 1):
            start = float(beats[b])
            end = float(beats[b + 1])
            taken = 0
            while i < len(ev) and ev[i] < start:
                i += 1
            j = i
            while j < len(ev) and ev[j] < end:
                if taken < max_per_beat:
                    sparse.append(ev[j])
                    taken += 1
                j += 1
            i = j
        while i < len(ev):
            sparse.append(ev[i])
            i += 1
        return sparse
    beat_interval = 60.0 / max(1.0, bpm)
    return apply_temporal_filter(sorted(events), beat_interval * 0.95)


def _get_pattern_positions(genre_label: str, is_break: bool) -> List[float]:
    base = _GENRE_PATTERN_POSITIONS.get(genre_label)
    if base is None:
        if genre_label in ("pop", "hyperpop", "k-pop", "j-pop"):
            base = [0.0, 2.0]
            if is_break:
                base += [0.5, 1.5, 2.5, 3.5]
            base += [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
        else:
            base = _GENRE_PATTERN_POSITIONS["default"]
    return base


def _augment_notes(
    base_times: List[float],
    kick_times: List[float],
    snare_times: List[float],
    dominant_onsets: List[float],
    bpm: float,
    fill: int,
    genre_label: str,
    genre_template_strength: int,
    verbose: bool,
) -> List[float]:
    if fill <= 0 or not base_times:
        return []

    beat_interval = 60.0 / max(1.0, bpm)
    bounds = _measure_bounds(min(base_times), max(base_times), beat_interval)
    caps = _HARD_CAPS.get(genre_label, _HARD_CAPS["default"])
    scale = 0.4 + (fill / 100.0) * 0.6
    total_cap = int(len(base_times) * caps["cap_ratio"] * scale)
    tol = 0.03
    added: List[float] = []
    added_total = 0

    for (m_start, m_end) in bounds:
        base_in_measure = [t for t in base_times if m_start <= t < m_end]
        energy = (
            _count_in_window(kick_times, m_start, m_end)
            + _count_in_window(snare_times, m_start, m_end)
            + _count_in_window(dominant_onsets, m_start, m_end)
        )
        is_break = energy >= 6
        target_min = max(1, int(caps["min"] * scale))
        target_max = max(target_min, int(caps["max"] * scale))

        need_fill = energy > 0 and (
            len(base_in_measure) < target_min
            or (is_break and len(base_in_measure) < target_max)
        )
        if not need_fill:
            continue

        pattern_candidates = [
            m_start + pos * beat_interval
            for pos in _get_pattern_positions(genre_label, is_break)
            if not _has_near(m_start + pos * beat_interval, base_in_measure, tol)
            and not _has_near(m_start + pos * beat_interval, added, tol)
        ]
        onset_candidates = [
            t for t in dominant_onsets
            if m_start <= t < m_end
            and not _has_near(t, base_in_measure, tol)
            and not _has_near(t, added, tol)
        ]
        proposed = _merge_pattern_and_onsets(pattern_candidates, onset_candidates, genre_template_strength)
        if not proposed or (total_cap and added_total >= total_cap):
            continue

        per_measure_limit = caps["per_measure_break"] if is_break else caps["per_measure"]
        needed_to_min = max(0, target_min - len(base_in_measure))
        room_to_max = max(0, target_max - len(base_in_measure))
        limit = min(per_measure_limit, room_to_max)
        if needed_to_min > 0:
            limit = min(limit, needed_to_min)
        if total_cap:
            limit = min(limit, max(0, total_cap - added_total))
        if limit <= 0:
            continue

        keep = proposed[:limit]
        added.extend(keep)
        added_total += len(keep)

    if verbose and added:
        print(f"[DrumGen] Добавлено нот: +{len(added)} (fill={fill}, genre={genre_label})")
    return added


def generate_drums_notes(
    song_path: str,
    bpm: float,
    lanes: int = 4,
    sync_tolerance: float = 0.2,
    use_madmom_beats: bool = True,
    use_stems: bool = True,
    generation_mode: str = "basic",
    fill: Optional[int] = None,
    groove: Optional[int] = None,
    density: Optional[int] = None,
    grid_snap_strength: Optional[int] = None,
    accent_strong_beats: Optional[bool] = None,
    genre_template_strength: Optional[int] = None,
    track_info: Optional[Dict] = None,
    auto_identify_track: bool = False,
    use_filename_for_genres: bool = False,
    provided_genres: Optional[List[str]] = None,
    provided_primary_genre: Optional[str] = None,
    verbose: bool = True,
    status_cb=None,
    cancel_cb=None,
) -> Optional[List[Dict]]:
    mode = generation_mode.lower()
    fill, groove, density, accent_strong_beats, genre_template_strength = _presets_for_mode(
        mode,
        fill,
        groove,
        density,
        accent_strong_beats,
        genre_template_strength,
    )
    genre_template_strength = int(max(0, min(100, int(genre_template_strength))))
    if grid_snap_strength is None:
        if mode == "minimal":
            grid_snap_strength = 85
        elif mode == "basic":
            grid_snap_strength = 60
        elif mode == "enhanced":
            grid_snap_strength = 35
        elif mode == "natural":
            grid_snap_strength = 0
        else:
            grid_snap_strength = 35
    grid_snap_strength = int(max(0, min(100, int(grid_snap_strength))))
    grid_snap_enabled = grid_snap_strength > 0

    if verbose:
        print(f"[DrumGen] режим={mode} fill={fill} groove={groove} density={density} grid_snap_strength={grid_snap_strength} accent_strong_beats={accent_strong_beats} genre_template_strength={genre_template_strength} bpm={bpm} lanes={lanes}")

    if cancel_cb:
        cancel_cb()
    if status_cb:
        status_cb("Разделение на стемы...")

    analysis = analyze_audio(
        song_path=song_path,
        bpm=bpm,
        use_stems=use_stems,
        auto_identify_track=auto_identify_track,
        use_filename_for_genres=use_filename_for_genres,
        track_info=track_info,
        stem_type="drums",
        cancel_cb=cancel_cb,
    )
    if cancel_cb:
        cancel_cb()

    bpm = analysis.get("bpm", bpm)
    beats = np.array(analysis.get("beats", []))
    kick_times: List[float] = analysis.get("kick_times", [])
    snare_times: List[float] = analysis.get("snare_times", [])
    dominant_onsets: List[float] = analysis.get("dominant_onsets", [])
    unique_genres: List[str] = analysis.get("genres", [])
    track_info = analysis.get("track_info") or track_info or {}

    if provided_genres:
        pg = [g.strip() for g in provided_genres if isinstance(g, str) and g.strip()]
        if pg:
            unique_genres = list({*unique_genres, *pg})
    if isinstance(provided_primary_genre, str) and provided_primary_genre.strip():
        track_info["primary_genre"] = provided_primary_genre.strip()

    primary_genre = track_info.get("primary_genre", "") if isinstance(track_info, dict) else ""
    if not primary_genre and unique_genres:
        primary_genre = unique_genres[0]
    genre_label = primary_genre.strip().lower() if primary_genre else "groove"
    genre_params = get_genre_config(genre_label)

    if verbose:
        print(f"[DrumGen] Жанр: {genre_label} | уникальные: {unique_genres}")

    if status_cb:
        status_cb("Детекция ударных...")
    if cancel_cb:
        cancel_cb()

    if verbose:
        print(
            f"[DrumGen][этап] beats={len(beats)} kick={len(kick_times)} "
            f"snare={len(snare_times)} dominant={len(dominant_onsets)}"
        )

    raw_events = sorted(set(dominant_onsets)) if dominant_onsets else sorted(set(kick_times + snare_times))
    if not raw_events:
        return None

    if "sync_tolerance_multiplier" in genre_params:
        sync_tolerance = float(sync_tolerance) * float(genre_params.get("sync_tolerance_multiplier", 1.0))

    drum_start_window = float(genre_params.get("drum_start_window", 4.0))
    drum_density_threshold = float(genre_params.get("drum_density_threshold", 0.5))
    drum_section_start = detect_drum_section_start(raw_events, drum_start_window, drum_density_threshold)
    filtered_events = [t for t in raw_events if t >= drum_section_start]

    min_note_distance = float(genre_params.get("min_note_distance", 0.05))
    if mode == "custom":
        min_note_distance = _density_to_min_distance(min_note_distance, density)
    elif mode == "minimal":
        min_note_distance = min(0.22, max(0.06, min_note_distance * 1.35))

    pattern_style = genre_params.get("pattern_style", "groove")
    apply_groove = bool(genre_params.get("apply_groove_pattern", False))
    use_grid_sync = bool(genre_params.get("sync_to_beats", False))
    if mode == "natural":
        apply_groove = False
        use_grid_sync = False

    if verbose:
        print(
            f"[DrumGen][этап] mode={mode} raw={len(raw_events)} after_start={len(filtered_events)} "
            f"start={drum_section_start:.3f} min_dist={min_note_distance:.3f} "
            f"sync={use_grid_sync} grid_strength={grid_snap_strength} groove={apply_groove} style={pattern_style} tol={sync_tolerance:.3f}"
        )

    events = apply_temporal_filter(sorted(filtered_events), min_note_distance)
    if mode == "custom":
        if groove <= 40:
            use_grid_sync = True
            apply_groove = False
        elif groove >= 60:
            apply_groove = True
        if groove >= 80:
            use_grid_sync = False
    use_grid_sync = bool(use_grid_sync and grid_snap_enabled)

    grooved_events = apply_groove_pattern(events, pattern_style, bpm) if apply_groove else events
    synced_events = _pull_to_grid(grooved_events, beats, sync_tolerance, grid_snap_strength) if use_grid_sync else grooved_events
    events_after_timing = synced_events
    if accent_strong_beats:
        events_after_timing = _accent_to_strong_beats(events_after_timing, beats, sync_tolerance, 70)
    if mode == "minimal":
        before_sparse = len(events_after_timing)
        events_after_timing = _sparsify_by_beats(events_after_timing, beats, bpm, max_per_beat=1)
        if verbose:
            print(f"[DrumGen][этап] minimal_sparsify={before_sparse}->{len(events_after_timing)}")

    if verbose:
        print(
            f"[DrumGen][этап] after_filter={len(events)} after_groove={len(grooved_events)} "
            f"after_sync={len(synced_events)}"
        )

    if cancel_cb:
        cancel_cb()

    base_times = list(events_after_timing)
    if mode == "basic":
        fill = 0
    if mode == "minimal":
        fill = 0
    if mode == "natural":
        fill = 0

    added_times: List[float] = []
    if fill > 0:
        added_times = _augment_notes(
            base_times=base_times,
            kick_times=kick_times,
            snare_times=snare_times,
            dominant_onsets=dominant_onsets,
            bpm=bpm,
            fill=fill,
            genre_label=genre_label,
            genre_template_strength=genre_template_strength,
            verbose=verbose,
        )

    all_times = sorted(set(base_times + added_times))
    if mode == "enhanced" and base_times:
        min_target = int(len(base_times) * 1.15)
        if len(all_times) < min_target:
            extra = _augment_notes(
                base_times=all_times,
                kick_times=kick_times,
                snare_times=snare_times,
                dominant_onsets=dominant_onsets,
                bpm=bpm,
                fill=100,
                genre_label=genre_label,
                genre_template_strength=genre_template_strength,
                verbose=False,
            )
            all_times = sorted(set(all_times + extra))

    if status_cb:
        status_cb("Назначение линий...")
    if cancel_cb:
        cancel_cb()

    all_events = [{"type": NoteType.DRUM, "time": t} for t in all_times]
    notes = assign_lanes_to_notes(all_events, lanes=lanes, song_offset=0.0)

    if verbose:
        print(
            f"[DrumGen] Итого: {len(notes)} (обнаружено={len(base_times)}, "
            f"добавлено={len(all_times) - len(base_times)})"
        )

    return notes if notes else None
