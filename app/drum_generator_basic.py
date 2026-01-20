# app/drum_generator_basic.py

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from .audio_analysis import analyze_audio
from .drum_utils import (
    apply_temporal_filter,
    apply_groove_pattern,
    sync_to_beats,
    assign_lanes_to_notes,
    save_drums_notes,
    load_drums_notes,
    detect_drum_section_start
)
from .note_types import NoteType


def generate_drums_notes(
        song_path: str,
        bpm: float,
        lanes: int = 4,
        sync_tolerance: float = 0.2,
        use_madmom_beats: bool = True,
        use_stems: bool = True,
        track_info: Optional[Dict] = None,
        auto_identify_track: bool = False,
        use_filename_for_genres: bool = True
) -> Optional[List[Dict]]:
    print(f"🎧 Генерация барабанных нот (basic) для: {song_path} (BPM: {bpm})")

    analysis = analyze_audio(
        song_path=song_path,
        bpm=bpm,
        use_stems=use_stems,
        auto_identify_track=auto_identify_track,
        use_filename_for_genres=use_filename_for_genres,
        track_info=track_info,
        stem_type="drums"
    )

    bpm = analysis["bpm"]
    beats = np.array(analysis["beats"])
    kick_times = analysis["kick_times"]
    snare_times = analysis["snare_times"]
    genre_params = analysis["genre_params"]
    unique_genres = analysis["genres"]
    track_info = analysis["track_info"]

    if 'sync_tolerance_multiplier' in genre_params:
        sync_tolerance *= genre_params['sync_tolerance_multiplier']
        print(f"[DrumGen-Basic] Sync tolerance изменён: {sync_tolerance:.2f}")

    drum_start_window = genre_params.get('drum_start_window', 4.0)
    drum_density_threshold = genre_params.get('drum_density_threshold', 0.5)

    all_raw_events = sorted(set(kick_times + snare_times))

    drum_section_start = detect_drum_section_start(
        all_raw_events,
        drum_start_window,
        drum_density_threshold
    )

    # Фильтруем события до начала секции
    filtered_events = [t for t in all_raw_events if t >= drum_section_start]

    min_note_distance = genre_params.get('min_note_distance', 0.05)
    pattern_style = genre_params.get('pattern_style', 'groove')

    final_events = apply_temporal_filter(sorted(filtered_events), min_note_distance)

    grooved_events = apply_groove_pattern(final_events, pattern_style, bpm)
    synced_events = sync_to_beats(grooved_events, beats, sync_tolerance)

    if len(synced_events) == 0:
        print("[DrumGen-Basic] Нет нот после синхронизации — используем грув-паттерн")
        synced_events = grooved_events

    all_events = [{"type": NoteType.DRUM, "time": t} for t in synced_events]

    notes = assign_lanes_to_notes(all_events, lanes=lanes, song_offset=0.0)

    drum_count = len(notes)
    print(f"✅ Сгенерировано {drum_count} барабанных нот (basic)")
    print(f"   - Жанры: {unique_genres if unique_genres else 'не определены'}")
    print(f"   - BPM: {bpm}, Style: {pattern_style}")

    if drum_count == 0:
        print("[DrumGen-Basic] ВНИМАНИЕ: Сгенерировано 0 нот!")

    return notes