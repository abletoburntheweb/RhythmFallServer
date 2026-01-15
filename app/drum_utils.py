# app/drum_utils.py
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def apply_temporal_filter(events: List[float], min_distance: float = 0.05) -> List[float]:
    if not events:
        return events
    filtered = [events[0]]
    for event in events[1:]:
        if event - filtered[-1] >= min_distance:
            filtered.append(event)
    return filtered


def apply_groove_pattern(events: List[float], pattern_style: str = "groove", bpm: float = 120.0) -> List[float]:
    if not events:
        return events

    if pattern_style == "precise":
        return events
    elif pattern_style == "sparse":
        return events[::2]
    else:
        grid_step = 60.0 / bpm
        grooved_events = []
        for event in events:
            grid_position = round(event / (grid_step / 2)) * (grid_step / 2)
            groove_amount = 0.02
            offset = random.uniform(-groove_amount, groove_amount) * grid_step
            grooved_time = max(0.0, grid_position + offset)
            grooved_events.append(grooved_time)
        return sorted(grooved_events)


def sync_to_beats(hit_times: List[float], beats: np.ndarray, sync_tolerance: float = 0.2) -> List[float]:
    if len(beats) == 0 or not hit_times:
        return hit_times

    synced = []
    for t in hit_times:
        distances = np.abs(beats - t)
        min_dist = np.min(distances)
        if min_dist <= sync_tolerance:
            synced.append(float(beats[np.argmin(distances)]))

    unique = []
    for t in sorted(synced):
        if not unique or abs(t - unique[-1]) > 0.01:
            unique.append(t)
    return unique


def detect_drum_section_start(times: List[float], window_duration: float = 2.0, threshold: float = 0.5) -> float:
    if len(times) < 2:
        return 0.0

    times = np.array(times)
    start_time = 0.0
    end_time = max(times)
    step = window_duration / 2
    current_time = start_time

    while current_time < end_time:
        window_start = current_time
        window_end = current_time + window_duration
        hits_in_window = sum(1 for t in times if window_start <= t < window_end)
        density = hits_in_window / window_duration
        if density >= threshold:
            return window_start
        current_time += step

    return 0.0


def assign_lanes_to_notes(notes: List[Dict], lanes: int = 4, song_offset: float = 0.0) -> List[Dict]:
    notes = [n for n in notes if n["time"] + song_offset > 0]
    notes.sort(key=lambda x: x["time"])

    last_lane_usage = {}
    result = []

    for note in notes:
        adjusted_time = note["time"] + song_offset
        available_lanes = [
            lane for lane in range(lanes)
            if last_lane_usage.get(lane, -999) < adjusted_time
        ]
        if available_lanes:
            lane = random.choice(available_lanes)
        else:
            lane = min(range(lanes), key=lambda l: last_lane_usage.get(l, -999))

        last_lane_usage[lane] = adjusted_time
        result.append({
            "type": note["type"],
            "lane": lane,
            "time": float(adjusted_time)
        })

    return sorted(result, key=lambda x: x["time"])


def load_genre_configs(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "genre_configs.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации жанров не найден: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_genre_aliases(alias_path: Optional[Path] = None) -> dict:
    if alias_path is None:
        alias_path = Path(__file__).parent / "genre_aliases.json"
    if not alias_path.exists():
        print("[GenreAliases] Файл genre_aliases.json не найден — используем пустой маппинг")
        return {}

    with open(alias_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    alias_map = {}
    for config_key, aliases in raw.items():
        for alias in aliases:
            alias_norm = alias.strip().lower()
            alias_map[alias_norm] = config_key
    return alias_map


def get_genre_params(genres: List[str], genre_configs: dict, genre_alias_map: dict) -> dict:
    if not genres:
        return genre_configs.get("default", {})

    for raw_genre in genres:
        if not raw_genre or raw_genre.lower() == "unknown":
            continue

        key = raw_genre.strip().lower()

        if key in genre_configs:
            return genre_configs[key]

        if key in genre_alias_map:
            target_key = genre_alias_map[key]
            if target_key in genre_configs:
                return genre_configs[target_key]

    return genre_configs.get("default", {})


def save_drums_notes(notes_data: List[Dict], song_path: str, mode: str = "basic") -> bool:
    if not notes_data:
        print(f"[DrumUtils] Нет данных нот для сохранения (mode: {mode}).")
        return False

    base_name = Path(song_path).stem
    song_folder = Path("temp_uploads") / base_name
    notes_folder = song_folder / "notes"
    notes_folder.mkdir(parents=True, exist_ok=True)

    notes_path = notes_folder / f"{base_name}_drums_{mode}.json"

    try:
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            return obj

        serializable = convert_types(notes_data)
        filtered_notes = [note for note in serializable if note.get('type') != 'TrackInfo']

        temp_path = notes_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_notes, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(notes_path)
        print(f"[DrumUtils] Ноты сохранены в: {notes_path}")
        return True
    except Exception as e:
        print(f"[DrumUtils] Ошибка сохранения нот: {e}")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return False


def load_drums_notes(song_path: str, mode: str = "basic") -> Optional[List[Dict]]:
    base_name = Path(song_path).stem
    notes_path = Path("temp_uploads") / base_name / "notes" / f"{base_name}_drums_{mode}.json"

    if not notes_path.exists():
        print(f"[DrumUtils] Файл нот не найден: {notes_path}")
        return None

    try:
        with open(notes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[DrumUtils] Ноты загружены из: {notes_path}")
        return data
    except Exception as e:
        print(f"[DrumUtils] Ошибка загрузки нот: {e}")
        return None