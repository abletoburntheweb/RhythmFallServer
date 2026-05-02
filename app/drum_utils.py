# app/drum_utils.py
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Deque
import numpy as np

LANE_TIME_BUCKET_EPS = 1e-5

LANE_REPEAT_BIAS_GAP_S = 0.085
LANE_REPEAT_WEIGHT_POWER = 1.35
LANE_REPEAT_MIN_WEIGHT = 0.06
LANE_SAME_AS_PREV_MULT = 0.11
LANE_RECENT_HISTORY_LEN = 10
LANE_HISTORY_OVERUSE_PENALTY = 0.55

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dedupe_notes_same_lane_same_time(notes: List[Dict], eps: float = LANE_TIME_BUCKET_EPS) -> List[Dict]:
    if not notes:
        return notes
    ordered = sorted(notes, key=lambda x: (float(x["time"]), int(x.get("lane", 0))))
    out: List[Dict] = []
    last_t_by_lane: Dict[int, float] = {}
    for n in ordered:
        lane = int(n.get("lane", 0))
        t = float(n["time"])
        prev = last_t_by_lane.get(lane)
        if prev is not None and abs(t - prev) <= eps:
            continue
        last_t_by_lane[lane] = t
        out.append(n)
    return out


def _pick_lane_avoid_spam(
    available_lanes: List[int],
    adjusted_time: float,
    last_lane_usage: Dict[int, float],
    prev_lane: Optional[int],
    recent_lanes: Deque[int],
) -> int:
    if len(available_lanes) == 1:
        return available_lanes[0]

    bias_gap = LANE_REPEAT_BIAS_GAP_S
    weights: List[float] = []
    floor_w = LANE_REPEAT_MIN_WEIGHT

    for lane in available_lanes:
        last_t = last_lane_usage.get(lane, -1e9)
        dt = max(adjusted_time - last_t, LANE_TIME_BUCKET_EPS)
        w = min(1.0, (dt / bias_gap) ** LANE_REPEAT_WEIGHT_POWER)
        w = max(w, floor_w)
        if prev_lane is not None and lane == prev_lane:
            w *= LANE_SAME_AS_PREV_MULT
        cnt = sum(1 for x in recent_lanes if x == lane)
        w /= 1.0 + LANE_HISTORY_OVERUSE_PENALTY * cnt
        w = max(w, floor_w * 0.2)
        weights.append(w)

    return random.choices(available_lanes, weights=weights, k=1)[0]


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

    last_lane_usage: Dict[int, float] = {}
    result: List[Dict] = []
    recent_lanes: Deque[int] = deque(maxlen=LANE_RECENT_HISTORY_LEN)
    prev_lane: Optional[int] = None

    eps = LANE_TIME_BUCKET_EPS
    i = 0
    while i < len(notes):
        t_anchor = float(notes[i]["time"]) + song_offset
        bucket_end = i
        while bucket_end < len(notes):
            t_here = float(notes[bucket_end]["time"]) + song_offset
            if abs(t_here - t_anchor) > eps:
                break
            bucket_end += 1

        used_this_instant: set[int] = set()
        for note in notes[i:bucket_end]:
            adjusted_time = float(note["time"]) + song_offset
            available_lanes = [
                lane for lane in range(lanes)
                if last_lane_usage.get(lane, -999.0) < adjusted_time - eps
                and lane not in used_this_instant
            ]
            if not available_lanes:
                continue

            lane = _pick_lane_avoid_spam(
                available_lanes,
                adjusted_time,
                last_lane_usage,
                prev_lane,
                recent_lanes,
            )
            used_this_instant.add(lane)
            last_lane_usage[lane] = adjusted_time
            recent_lanes.append(lane)
            prev_lane = lane
            payload = dict(note)
            payload["lane"] = lane
            payload["time"] = adjusted_time
            result.append(payload)

        i = bucket_end

    result = sorted(result, key=lambda x: x["time"])
    return dedupe_notes_same_lane_same_time(result, eps=eps)


def remove_kick_snare_collisions(
        kicks: List[float],
        snares: List[float],
        tolerance: float = 0.03,
        kick_priority: bool = False
) -> tuple[List[float], List[float]]:
    all_times = sorted(set(kicks + snares))
    final_kicks: List[float] = []
    final_snares: List[float] = []

    for t in all_times:
        has_kick = any(abs(t - k) < tolerance for k in kicks)
        has_snare = any(abs(t - s) < tolerance for s in snares)

        if has_kick and has_snare:
            if kick_priority:
                final_kicks.append(t)
            else:
                final_snares.append(t)
        elif has_kick:
            final_kicks.append(t)
        elif has_snare:
            final_snares.append(t)

    collisions_removed = len(kicks) + len(snares) - len(final_kicks) - len(final_snares)
    if collisions_removed > 0:
        print(f"[CollisionFix] Убрано {collisions_removed} дубликатов")

    return final_kicks, final_snares


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


def save_drums_notes(notes_data: List[Dict], song_path: str, mode: str = "basic", lanes: int = 4) -> bool:
    base_name = Path(song_path).stem
    song_folder = Path("temp_uploads") / base_name
    notes_folder = song_folder / "notes"
    notes_folder.mkdir(parents=True, exist_ok=True)

    notes_path = notes_folder / f"{base_name}_drums_{mode}_lanes{lanes}.json"

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

        env_flag = os.environ.get("RHYTHMFALL_NOTE_TIMING_LOG", "").strip().lower()
        if env_flag in ("1", "true", "yes", "on"):
            times = []
            for n in filtered_notes:
                if isinstance(n, dict) and "time" in n:
                    try:
                        times.append(float(n["time"]))
                    except (TypeError, ValueError):
                        pass
            t_min = min(times) if times else 0.0
            t_max = max(times) if times else 0.0
            log_line = (
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t"
                f"song_path={song_path}\tmode={mode}\tlanes={lanes}\t"
                f"count={len(filtered_notes)}\tt_min={t_min:.6f}\tt_max={t_max:.6f}\t"
                f"notes_json={notes_path}\n"
            )
            log_path = PROJECT_ROOT / "temp_uploads" / "note_generation_timing.log"
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(log_line)
                print(f"[DrumUtils] timing log → {log_path}")
            except OSError as e:
                print(f"[DrumUtils] timing log failed: {e}")

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
