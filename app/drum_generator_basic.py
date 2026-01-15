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
    remove_kick_snare_collisions,
    detect_drum_section_start
)


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

    all_raw_events = sorted(kick_times + snare_times)
    drum_section_start = detect_drum_section_start(all_raw_events, drum_start_window, drum_density_threshold)

    filtered_kicks = [t for t in kick_times if t >= drum_section_start]
    filtered_snares = [t for t in snare_times if t >= drum_section_start]

    min_note_distance = genre_params.get('min_note_distance', 0.05)
    pattern_style = genre_params.get('pattern_style', 'groove')
    kick_priority = genre_params.get('kick_priority', False)

    final_kicks = apply_temporal_filter(sorted(filtered_kicks), min_note_distance)
    final_snares = apply_temporal_filter(sorted(filtered_snares), min_note_distance)

    grooved_kicks = apply_groove_pattern(final_kicks, pattern_style, bpm)
    grooved_snares = apply_groove_pattern(final_snares, pattern_style, bpm)

    synced_kicks = sync_to_beats(grooved_kicks, beats, sync_tolerance)
    synced_snares = sync_to_beats(grooved_snares, beats, sync_tolerance)

    if len(synced_kicks) + len(synced_snares) == 0:
        print("[DrumGen-Basic] Нет нот после синхронизации — используем грув-паттерн")
        synced_kicks = grooved_kicks
        synced_snares = grooved_snares

    tolerance = min_note_distance
    synced_kicks, synced_snares = remove_kick_snare_collisions(
        synced_kicks, synced_snares, tolerance, kick_priority
    )
    print(f"[DrumGen-Basic] После удаления коллизий: Kick={len(synced_kicks)}, Snare={len(synced_snares)}")

    all_events = []
    for t in synced_kicks:
        all_events.append({"type": "KickNote", "time": t})
    for t in synced_snares:
        all_events.append({"type": "SnareNote", "time": t})

    notes = assign_lanes_to_notes(all_events, lanes=lanes, song_offset=0.0)

    kicks_count = len([n for n in notes if n["type"] == "KickNote"])
    snares_count = len([n for n in notes if n["type"] == "SnareNote"])

    print(f"✅ Сгенерировано {len(notes)} барабанных нот (basic)")
    print(f"   - Kick: {kicks_count} | Snare: {snares_count}")
    print(f"   - Жанры: {unique_genres if unique_genres else 'не определены'}")
    print(f"   - BPM: {bpm}, Style: {pattern_style}")

    if track_info and track_info.get('success'):
        notes.append({
            "type": "TrackInfo",
            "title": track_info['title'],
            "artist": track_info['artist'],
            "genres": unique_genres,
            "album": track_info['album'],
            "year": track_info['year'],
            "time": -1
        })

    if len(notes) == 0:
        print("[DrumGen-Basic] ВНИМАНИЕ: Сгенерировано 0 нот!")

    return notes
