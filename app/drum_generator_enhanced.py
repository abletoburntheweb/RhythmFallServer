# app/drum_generator_enhanced.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from .audio_analysis import analyze_audio
from .drum_utils import (
    assign_lanes_to_notes,
    save_drums_notes,
    load_drums_notes
)
from .note_types import NoteType
from . import drum_generator_basic

def _coefficient_of_variation(values: List[float]) -> float:
    if not values:
        return 0.0
    mean_val = float(np.mean(values))
    if mean_val == 0:
        return 0.0
    return float(np.std(values) / mean_val)


    if not events or bpm <= 0:
        return 0.0
    beat_interval = 60.0 / bpm
    duration = max(beat_interval, events[-1]["time"] - events[0]["time"])
    total_beats = duration / beat_interval
    return float(len(events) / max(1.0, total_beats))


def _measure_bounds(start_time: float, end_time: float, beat_interval: float) -> List[Tuple[float, float]]:
    measure_duration = beat_interval * 4
    bounds: List[Tuple[float, float]] = []
    current = start_time
    while current <= end_time:
        bounds.append((current, current + measure_duration))
        current += measure_duration
    return bounds

def _count_in_window(times: List[float], start: float, end: float) -> int:
    return sum(1 for t in times if start <= t < end)

def _has_near(t: float, existing: List[float], tol: float) -> bool:
    return any(abs(t - x) <= tol for x in existing)


def generate_drums_notes(
    song_path: str,
    bpm: float,
    lanes: int = 4,
    sync_tolerance: float = 0.2,
    use_madmom_beats: bool = True,
    use_stems: bool = True,
    track_info: Optional[Dict] = None,
    auto_identify_track: bool = False,
    use_filename_for_genres: bool = True,
    provided_genres: Optional[List[str]] = None,
    provided_primary_genre: Optional[str] = None,
    status_cb=None,
    cancel_cb=None
) -> Optional[List[Dict]]:
    print(f"🎮 Генерация барабанных нот (enhanced) для: {song_path} (BPM: {bpm})")

    if cancel_cb:
        cancel_cb()

    analysis = analyze_audio(
        song_path=song_path,
        bpm=bpm,
        use_stems=use_stems,
        auto_identify_track=auto_identify_track,
        use_filename_for_genres=use_filename_for_genres,
        track_info=track_info,
        stem_type="drums",
        cancel_cb=cancel_cb
    )
    if cancel_cb:
        cancel_cb()

    if not analysis or "bpm" not in analysis:
        print("[DrumGen-Enhanced] Fallback: отсутствуют данные анализа")
        return drum_generator_basic.generate_drums_notes(
            song_path,
            bpm,
            lanes=lanes,
            sync_tolerance=sync_tolerance,
            use_madmom_beats=use_madmom_beats,
            use_stems=use_stems,
            track_info=track_info,
            auto_identify_track=auto_identify_track,
            use_filename_for_genres=use_filename_for_genres,
            provided_genres=provided_genres,
            provided_primary_genre=provided_primary_genre,
            status_cb=status_cb,
            cancel_cb=cancel_cb
        )

    bpm = analysis.get("bpm", bpm)
    beats = np.array(analysis.get("beats", []))
    genre_params = analysis.get("genre_params", {})
    unique_genres = analysis.get("genres", [])
    track_info = analysis.get("track_info") or track_info or {}
    analysis_path = analysis.get("analysis_path", song_path)

    if provided_genres is not None:
        pg = [g.strip() for g in provided_genres if isinstance(g, str) and g.strip()]
        if pg:
            unique_genres = list({*unique_genres, *pg})
    if isinstance(provided_primary_genre, str) and provided_primary_genre.strip():
        track_info["primary_genre"] = provided_primary_genre.strip()

    basic_notes = drum_generator_basic.generate_drums_notes(
        song_path,
        bpm,
        lanes=lanes,
        sync_tolerance=sync_tolerance,
        use_madmom_beats=use_madmom_beats,
        use_stems=use_stems,
        track_info=track_info,
        auto_identify_track=auto_identify_track,
        use_filename_for_genres=use_filename_for_genres,
        provided_genres=provided_genres,
        provided_primary_genre=provided_primary_genre,
        status_cb=status_cb,
        cancel_cb=cancel_cb
    )
    base_times = [n["time"] for n in (basic_notes or []) if isinstance(n, dict) and "time" in n]
    if beats.size == 0:
        beats = np.arange(0.0, (max(base_times) if base_times else 180.0) + (60.0 / max(1.0, bpm)), 60.0 / max(1.0, bpm))
    beat_start = float(beats[0]) if beats.size else 0.0
    beat_interval = 60.0 / max(1.0, bpm)

    start_time = min(base_times) if base_times else beat_start
    end_time = max(base_times) if base_times else (beat_start + beat_interval * 64)
    bounds = _measure_bounds(start_time, end_time, beat_interval)

    genre_label = None
    if isinstance(track_info, dict) and isinstance(track_info.get("primary_genre"), str) and track_info.get("primary_genre").strip():
        genre_label = track_info.get("primary_genre").strip().lower()
    elif isinstance(provided_primary_genre, str) and provided_primary_genre.strip():
        genre_label = provided_primary_genre.strip().lower()
    elif unique_genres:
        genre_label = str(unique_genres[0]).strip().lower()
    else:
        genre_label = "default"
    print(f"[DrumGen-Enhanced] Жанр для дополнений: {genre_label}")

    kick_times = analysis.get("kick_times", [])
    snare_times = analysis.get("snare_times", [])
    dominant_onsets = analysis.get("dominant_onsets", [])

    added: List[float] = []
    tol = 0.03
    hard_caps = {
        "pop": {"min": 3, "max": 6, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.30},
        "hyperpop": {"min": 3, "max": 6, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.30},
        "k-pop": {"min": 3, "max": 6, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.30},
        "j-pop": {"min": 3, "max": 6, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.30},
        "electronic": {"min": 5, "max": 8, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
        "house": {"min": 5, "max": 8, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
        "techno": {"min": 5, "max": 8, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
        "trance": {"min": 5, "max": 8, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
        "drum and bass": {"min": 6, "max": 10, "per_measure": 3, "per_measure_break": 5, "cap_ratio": 0.28},
        "rap": {"min": 3, "max": 6, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.22},
        "r&b": {"min": 3, "max": 6, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.22},
        "rock": {"min": 3, "max": 7, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.20},
        "metal": {"min": 3, "max": 7, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.20},
        "hardcore": {"min": 3, "max": 7, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.20},
        "default": {"min": 4, "max": 7, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
    }
    caps = hard_caps.get(genre_label, hard_caps["default"])
    total_cap = int(len(base_times) * caps["cap_ratio"]) if base_times else 0
    added_total = 0
    for (m_start, m_end) in bounds:
        base_in_measure = [t for t in base_times if m_start <= t < m_end]
        energy = (
            _count_in_window(kick_times, m_start, m_end)
            + _count_in_window(snare_times, m_start, m_end)
            + _count_in_window(dominant_onsets, m_start, m_end)
        )
        intervals = [base_in_measure[i] - base_in_measure[i - 1] for i in range(1, len(base_in_measure))]
        cv = _coefficient_of_variation(intervals) if intervals else 0.0
        density = float(len(base_in_measure) / max(1.0, (m_end - m_start) / beat_interval))

        is_break = energy >= 6
        target_min = caps["min"]
        target_max = caps["max"]

        need_fill = (energy > 0) and (len(base_in_measure) < target_min or (is_break and len(base_in_measure) < target_max) or (density < 0.3 and cv > 1.0))
        if not need_fill:
            continue

        pattern_positions = None
        if genre_label in {"electronic", "house", "techno", "trance"}:
            pattern_positions = [0.0, 2.0]
            pattern_positions += [0.5, 1.5, 2.5, 3.5]
            pattern_positions += [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
        elif genre_label in {"drum and bass", "funk"}:
            pattern_positions = [0.0, 2.0]
            pattern_positions += [0.5, 1.5, 2.5, 3.5]
            pattern_positions += [0.25, 0.75, 1.0, 1.25, 1.75, 2.0, 2.25, 2.75, 3.0, 3.25, 3.75]
        elif genre_label in {"pop", "hyperpop", "k-pop", "j-pop"}:
            pattern_positions = [0.0, 2.0]
            if is_break or len(base_in_measure) < target_min:
                pattern_positions += [0.5, 1.5, 2.5, 3.5]
            pattern_positions += [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
        elif genre_label in {"rap", "r&b"}:
            pattern_positions = [0.0, 2.0]
            pattern_positions += [0.5, 1.5, 2.5, 3.5]
            pattern_positions += [0.75, 1.25, 2.75, 3.25]
        else:
            pattern_positions = [0.0, 2.0]
            pattern_positions += [0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]

        proposed = []
        for pos in pattern_positions:
            tt = m_start + pos * beat_interval
            if not _has_near(tt, base_in_measure, tol) and not _has_near(tt, added, tol):
                proposed.append(tt)

        if proposed:
            if total_cap and added_total >= total_cap:
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

    all_times = sorted(set(base_times + added))
    all_events = [{"type": NoteType.DRUM, "time": t, "source": ("enhanced" if t in added else "basic")} for t in all_times]
    if status_cb:
        status_cb("Назначение линий...")
    if cancel_cb:
        cancel_cb()
    notes = assign_lanes_to_notes(all_events, lanes=lanes, song_offset=0.0)

    drum_count = len(notes)
    print(f"✅ Сгенерировано {drum_count} барабанных нот (enhanced-augment)")
    print(f"   - Дополнено: {len(added)} | Базовых: {len(base_times)}")
    print(f"   - Жанры: {unique_genres if unique_genres else 'не определены'}")
    print(f"   - BPM: {bpm}")
    return notes
