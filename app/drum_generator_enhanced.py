# app/drum_generator_enhanced.py
import numpy as np
from typing import List, Dict, Optional
from .audio_analysis import analyze_audio
from .drum_utils import (
    apply_temporal_filter,
    apply_groove_pattern,
    sync_to_beats,
    assign_lanes_to_notes,
    detect_drum_section_start
)


def analyze_rhythm_pattern(
    events: List[float],
    bpm: float,
    beats: np.ndarray,
    density_threshold: float = 0.6
) -> Dict:
    if not events or len(beats) < 4:
        return {"patterns": [], "density": {}, "accents": []}

    beat_interval = 60.0 / bpm
    measure_duration = beat_interval * 4
    start_time = min(events)
    end_time = max(beats)

    patterns = []
    density_map = {}
    accents = set()

    current_time = start_time

    while current_time < end_time:
        measure_start = current_time
        measure_end = current_time + measure_duration
        measure_events = [e for e in events if measure_start <= e < measure_end]

        if not measure_events:
            current_time += measure_duration
            continue

        normalized_positions = [(e - measure_start) / beat_interval for e in measure_events]
        rounded_positions = [round(pos, 2) for pos in normalized_positions if 0 <= pos < 4]

        pattern_tuple = tuple(sorted(set(rounded_positions)))
        if pattern_tuple:
            patterns.append(pattern_tuple)

        density = len(measure_events) / measure_duration
        density_map[(measure_start, measure_end)] = density

        for pos in rounded_positions:
            if round(pos % 1.0, 2) == 0.0:  # на доле
                closest_beat = min(beats, key=lambda b: abs(b - (measure_start + pos * beat_interval)))
                accents.add(closest_beat)

        current_time += measure_duration

    return {
        "patterns": patterns,
        "density": density_map,
        "accents": sorted(list(accents))
    }


def enhance_rhythm_events(
    base_events: List[float],
    beats: np.ndarray,
    bpm: float,
    genre_params: dict,
    event_type: str = "kick"
) -> List[float]:
    if not base_events or len(beats) < 4:
        return base_events

    rhythm_analysis = analyze_rhythm_pattern(base_events, bpm, beats)

    min_distance = genre_params.get('min_note_distance', 0.05)
    max_hits_per_second = genre_params.get('max_hits_per_second', 4)
    beat_interval = 60.0 / bpm
    measure_duration = beat_interval * 4

    enhanced = list(base_events)
    start_time = min(base_events)
    end_time = max(beats)

    current_time = start_time
    while current_time < end_time:
        measure_start = current_time
        measure_end = current_time + measure_duration
        measure_events = [e for e in enhanced if measure_start <= e < measure_end]
        measure_beats = [b for b in beats if measure_start <= b < measure_end]

        if not measure_beats:
            current_time += measure_duration
            continue

        density = len(measure_events) / measure_duration
        target_density = max_hits_per_second * 0.6  # Цель: не перегружать

        if density < target_density * 0.6:
            for beat in measure_beats:
                if all(abs(beat - e) >= min_distance for e in enhanced):
                    import random
                    if random.random() < 0.35:  # 35% шанс добавить
                        enhanced.append(beat)

        common_positions = [0.0, 2.0]  # для kick
        if event_type == "snare":
            common_positions = [1.0, 3.0]

        for pos in common_positions:
            aligned_time = measure_start + pos * beat_interval
            if aligned_time > measure_end:
                continue
            closest_beat = min(measure_beats, key=lambda b: abs(b - aligned_time))
            if all(abs(closest_beat - e) >= min_distance for e in enhanced):
                enhanced.append(closest_beat)

        current_time += measure_duration

    return sorted(set(round(t, 3) for t in enhanced))


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
    print(f"🎮 Генерация барабанных нот (enhanced) для: {song_path} (BPM: {bpm})")

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

    drum_start_window = genre_params.get('drum_start_window', 4.0)
    drum_density_threshold = genre_params.get('drum_density_threshold', 0.5)
    all_raw_events = sorted(kick_times + snare_times)
    drum_section_start = detect_drum_section_start(all_raw_events, drum_start_window, drum_density_threshold)

    filtered_kicks = [t for t in kick_times if t >= drum_section_start]
    filtered_snares = [t for t in snare_times if t >= drum_section_start]

    min_note_distance = genre_params.get('min_note_distance', 0.05)
    pattern_style = genre_params.get('pattern_style', 'groove')

    final_kicks = apply_temporal_filter(sorted(filtered_kicks), min_note_distance)
    final_snares = apply_temporal_filter(sorted(filtered_snares), min_note_distance)

    grooved_kicks = apply_groove_pattern(final_kicks, pattern_style, bpm)
    grooved_snares = apply_groove_pattern(final_snares, pattern_style, bpm)

    enhanced_kicks = enhance_rhythm_events(grooved_kicks, beats, bpm, genre_params, "kick")
    enhanced_snares = enhance_rhythm_events(grooved_snares, beats, bpm, genre_params, "snare")

    synced_kicks = sync_to_beats(enhanced_kicks, beats, sync_tolerance)
    synced_snares = sync_to_beats(enhanced_snares, beats, sync_tolerance)

    if len(synced_kicks) + len(synced_snares) == 0:
        synced_kicks = grooved_kicks
        synced_snares = grooved_snares

    all_events = []
    for t in synced_kicks:
        all_events.append({"type": "KickNote", "time": t})
    for t in synced_snares:
        all_events.append({"type": "SnareNote", "time": t})

    notes = assign_lanes_to_notes(all_events, lanes=lanes, song_offset=0.0)

    kicks_count = len([n for n in notes if n["type"] == "KickNote"])
    snares_count = len([n for n in notes if n["type"] == "SnareNote"])

    print(f"✅ Сгенерировано {len(notes)} барабанных нот (enhanced)")
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
        print("[DrumGen-Enhanced] ВНИМАНИЕ: Сгенерировано 0 нот!")

    return notes