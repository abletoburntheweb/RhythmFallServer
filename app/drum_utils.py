# app/drum_utils.py
import json
import os
import random
import re
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

CANONICAL_MAX_LANES = 5


def chart_variant_suffix() -> str:
    raw = os.environ.get("RFALL_CHART_VARIANT", "").strip().lower()
    if not raw or raw in ("default", "prod", "production", "main"):
        return ""
    safe = re.sub(r"[^a-z0-9_]", "", raw)
    return f"_{safe}" if safe else ""


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


def detect_drum_section_start(times: List[float], window_duration: float = 2.0, threshold: float = 0.5) -> float:
    """First dense drum window start as the earliest hit in that window.

    Older code returned ``window_start`` (step grid). That often sits ~0.5–2s
    *before* the first real onset in the window, so the trim
    ``t >= section_start`` looked fine while recovery still had to fight a
    short grace window — classic "kick heard, first note half a second late".
    """
    if len(times) < 2:
        return 0.0

    times_arr = np.asarray(times, dtype=float)
    end_time = float(np.max(times_arr))
    step = window_duration / 2.0
    current_time = 0.0

    while current_time < end_time:
        window_start = current_time
        window_end = current_time + window_duration
        in_window = times_arr[(times_arr >= window_start) & (times_arr < window_end)]
        density = float(in_window.size) / window_duration
        if density >= threshold:
            return float(np.min(in_window))
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


DEFAULT_DRUM_LANE_MAP = {
    "kick": 0,
    "snare": 1,
    "hat": 2,
    "tom": 3,
    "perc": 3,
    "cymbal": 4,  # used when lanes >= 5; otherwise clamped to perc/tom
}


# --- Arcade Ergonomic Router (arcade_mode.md pass 6) -------------------------
# Assigns lanes with a cost model instead of weighted-random. Each drum class
# has a preferred lane "zone" (kick low, snare mid, hats/cymbals high) so the
# backbone stays readable, while the router smooths transitions (small hand
# travel), avoids jackhammering one lane on fast repeats, and rewards
# continuing a melodic direction (staircase flow). Zones are fraction-based so
# they scale to 3/4/5 lanes.
ARC_JACK_DT = 0.15            # s: a same-lane repeat faster than this is a "jack"
ARC_JACK_PENALTY = 10.0
ARC_MOVE_WEIGHT = 1.0         # cost per lane of travel from the class's last lane
ARC_FLOW_DT = 0.40           # s: within this window reward continuing direction
ARC_INERTIA_BONUS = 1.0
ARC_STAY_PENALTY = 2.5        # nudge loose classes to move (avoid one-lane collapse)
ARC_GLOBAL_SMOOTH = 0.1       # light cross-class travel smoothing
ARC_ZONE_WEIGHT_STRICT = 3.0
ARC_ZONE_WEIGHT_LOOSE = 0.8

# class -> anchor/allowed as fractions of (lanes - 1); strict = stays put.
ARC_CLASS_ZONES = {
    "kick":   {"anchor": 0.0, "lo": 0.0, "hi": 0.25, "strict": True},
    "snare":  {"anchor": 0.5, "lo": 0.25, "hi": 0.75, "strict": True},
    "hat":    {"anchor": 1.0, "lo": 0.5, "hi": 1.0, "strict": False},
    "cymbal": {"anchor": 1.0, "lo": 0.75, "hi": 1.0, "strict": False},
    "tom":    {"anchor": 0.75, "lo": 0.5, "hi": 1.0, "strict": False},
    "perc":   {"anchor": 0.5, "lo": 0.0, "hi": 1.0, "strict": False},
}


def _arc_zone_for_class(drum: str, lanes: int):
    z = ARC_CLASS_ZONES.get(str(drum or "perc").lower(), ARC_CLASS_ZONES["perc"])
    maxl = max(0, int(lanes) - 1)
    anchor = int(round(z["anchor"] * maxl))
    lo = int(round(z["lo"] * maxl))
    hi = int(round(z["hi"] * maxl))
    lo, hi = max(0, min(lo, hi)), min(maxl, max(lo, hi))
    allowed = list(range(lo, hi + 1)) or [anchor]
    return anchor, allowed, bool(z["strict"])


def assign_lanes_ergonomic(
    notes: List[Dict],
    classified_hits: List[Dict],
    lanes: int = 5,
    song_offset: float = 0.0,
    bpm: float = 120.0,
    tolerance: float = 0.06,
) -> List[Dict]:
    """Cost-based, class-aware lane assignment for arcade charts.

    Greedy left-to-right within a per-class + ergonomic cost model. Strict
    classes (kick/snare) claim their anchor first each instant; loose classes
    (hat/tom/cymbal) flow across their zone. Clearly beats random assignment
    for readability; a full global DP is a possible later refinement.
    """
    from .drum_hit_detector import resolve_drum_at_time

    lanes = max(1, int(lanes))
    notes = [n for n in notes if float(n["time"]) + song_offset > 0]
    notes.sort(key=lambda x: x["time"])

    result: List[Dict] = []
    class_last_lane: Dict[str, int] = {}
    class_last_time: Dict[str, float] = {}
    class_last_dir: Dict[str, int] = {}
    prev_lane_global: Optional[int] = None

    eps = LANE_TIME_BUCKET_EPS
    i = 0
    while i < len(notes):
        t_anchor = float(notes[i]["time"]) + song_offset
        j = i
        while j < len(notes) and abs(float(notes[j]["time"]) + song_offset - t_anchor) <= eps:
            j += 1

        enriched = []
        for n in notes[i:j]:
            t = float(n["time"]) + song_offset
            drum = resolve_drum_at_time(t, classified_hits, tolerance=tolerance) or "perc"
            enriched.append((n, t, drum))
        # strict classes pick first so kick/snare keep their anchors
        order = sorted(
            range(len(enriched)),
            key=lambda k: 0 if ARC_CLASS_ZONES.get(
                enriched[k][2], ARC_CLASS_ZONES["perc"]
            )["strict"] else 1,
        )

        used: set[int] = set()
        for k in order:
            n, t, drum = enriched[k]
            anchor, allowed, strict = _arc_zone_for_class(drum, lanes)
            cands = [L for L in allowed if L not in used]
            if not cands:
                cands = [L for L in range(lanes) if L not in used]
            if not cands:
                continue

            cl_lane = class_last_lane.get(drum)
            cl_time = class_last_time.get(drum, -1e9)
            cl_dir = class_last_dir.get(drum, 0)
            zone_w = ARC_ZONE_WEIGHT_STRICT if strict else ARC_ZONE_WEIGHT_LOOSE

            best_lane: Optional[int] = None
            best_cost: Optional[float] = None
            for L in cands:
                cost = zone_w * abs(L - anchor)
                if cl_lane is not None:
                    dt_c = t - cl_time
                    cost += ARC_MOVE_WEIGHT * abs(L - cl_lane)
                    if L == cl_lane:
                        if dt_c < ARC_JACK_DT:
                            cost += ARC_JACK_PENALTY
                        elif not strict:
                            cost += ARC_STAY_PENALTY
                    else:
                        move_dir = 1 if L > cl_lane else -1
                        if dt_c < ARC_FLOW_DT and cl_dir != 0 and move_dir == cl_dir:
                            cost -= ARC_INERTIA_BONUS
                if prev_lane_global is not None:
                    cost += ARC_GLOBAL_SMOOTH * abs(L - prev_lane_global)
                if best_cost is None or cost < best_cost:
                    best_cost, best_lane = cost, L

            L = int(best_lane if best_lane is not None else cands[0])
            used.add(L)
            if cl_lane is not None and L != cl_lane:
                class_last_dir[drum] = 1 if L > cl_lane else -1
            class_last_lane[drum] = L
            class_last_time[drum] = t
            prev_lane_global = L

            payload = dict(n)
            payload["lane"] = L
            payload["time"] = t
            payload["drum"] = drum
            result.append(payload)

        i = j

    result.sort(key=lambda x: x["time"])
    return dedupe_notes_same_lane_same_time(result, eps=eps)


def _lane_for_drum_class(drum: str, lanes: int, mapping: Dict[str, int]) -> int:
    """Keep real tom/cymbal as percussion lanes; never fold cymbal onto kick when lanes=4."""
    key = str(drum or "perc").lower()
    if key == "cymbal" and int(lanes) < 5:
        key = "tom"
    base = int(mapping.get(key, mapping.get("perc", 3)))
    return base % max(1, int(lanes))


def assign_lanes_by_drum_class(
    notes: List[Dict],
    classified_hits: List[Dict],
    lanes: int = 4,
    song_offset: float = 0.0,
    lane_map: Optional[Dict[str, int]] = None,
    tolerance: float = 0.06,
) -> List[Dict]:
    """Map kick/snare/hat/tom/cymbal to fixed lanes; fallback for unknowns."""
    from .drum_hit_detector import resolve_drum_at_time

    mapping = dict(lane_map or DEFAULT_DRUM_LANE_MAP)
    notes = [n for n in notes if n["time"] + song_offset > 0]
    notes.sort(key=lambda x: x["time"])

    result: List[Dict] = []
    fallback: List[Dict] = []
    used_at_time: Dict[float, set] = {}

    for note in notes:
        t = float(note["time"]) + song_offset
        drum = resolve_drum_at_time(t, classified_hits, tolerance=tolerance) or "perc"
        lane = _lane_for_drum_class(drum, lanes, mapping)
        bucket = round(t, 4)
        used = used_at_time.setdefault(bucket, set())
        while lane in used and len(used) < lanes:
            lane = (lane + 1) % lanes
        if lane in used:
            fallback.append(note)
            continue
        used.add(lane)
        payload = dict(note)
        payload["lane"] = lane
        payload["time"] = t
        payload["drum"] = drum
        result.append(payload)

    if fallback:
        result.extend(assign_lanes_to_notes(fallback, lanes=lanes, song_offset=song_offset))

    result = sorted(result, key=lambda x: x["time"])
    return dedupe_notes_same_lane_same_time(result, eps=LANE_TIME_BUCKET_EPS)


def load_genre_configs(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "drum_genre_profiles.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации жанров не найден: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_drum_augment_profiles(config_path: Optional[Path] = None) -> dict:
    """Per-genre note-count budgets and pattern grids used to augment/fill charts.

    Non-critical polish data: missing/broken file falls back to an empty dict,
    callers apply their own tiny "default" fallback rather than crashing the server.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "drum_augment_profiles.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[DrumUtils] Не удалось загрузить drum_augment_profiles.json: {e}")
        return {}


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


def _resolve_track_labels_from_song_path(song_path: str, artist: str = "", title: str = "") -> tuple[str, str]:
    a = str(artist or "").strip()
    t = str(title or "").strip()
    if a.lower() in ("", "unknown", "неизвестен"):
        a = ""
    if t.lower() in ("", "unknown", "н/д", "без названия"):
        t = ""
    if a and t:
        return a, t
    stem = Path(song_path).stem
    for sep in (" — ", " - ", " – "):
        if sep in stem:
            parts = stem.split(sep, 1)
            if len(parts) == 2:
                if not a:
                    a = parts[0].strip()
                if not t:
                    t = parts[1].strip()
            break
    if not t:
        t = stem
    return a, t


def save_drums_notes(
    notes_data: List[Dict],
    song_path: str,
    mode: str = "basic",
    chart_intent: Optional[str] = None,
    chart_stem: Optional[str] = None,
    lanes: int = 4,
    artist: str = "",
    title: str = "",
    rhythm_dna: Optional[Dict] = None,
    chart_id: str = "",
) -> bool:
    from app.rfc_chart_codec import notes_to_spawn_array, write_file
    from app.rhythm_dna import save_rhythm_dna_sidecar, format_rhythm_dna_log
    from app import song_storage

    cid = str(chart_id or "").strip() or song_storage.chart_id_from_song_path(song_path)
    if cid:
        song_folder = song_storage.song_dir(cid)
    else:
        base_name = Path(song_path).stem
        song_folder = Path("temp_uploads") / base_name
    notes_folder = song_folder / "notes"
    notes_folder.mkdir(parents=True, exist_ok=True)

    stem_key = str(chart_stem or chart_intent or mode or "groove").strip().lower() or "groove"
    variant_suffix = chart_variant_suffix()
    chart_variant = variant_suffix[1:] if variant_suffix.startswith("_") else variant_suffix
    chart_lanes = max(int(lanes), CANONICAL_MAX_LANES)
    notes_path = notes_folder / song_storage.chart_notes_filename("drums", stem_key, chart_lanes, chart_variant)

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
        filtered_notes = notes_to_spawn_array(serializable)
        track_artist, track_title = _resolve_track_labels_from_song_path(song_path, artist, title)
        write_file(
            notes_path,
            filtered_notes,
            instrument="drums",
            intent=stem_key,
            lanes=chart_lanes,
            artist=track_artist,
            title=track_title,
        )
        print(f"[DrumUtils] Ноты сохранены в: {notes_path}")
        sidecar_ok = save_rhythm_dna_sidecar(notes_path, rhythm_dna)
        if not sidecar_ok:
            print(format_rhythm_dna_log(rhythm_dna, context=f"temp sidecar not written for {notes_path.name}"))

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
                f"song_path={song_path}\tstem={stem_key}\tlanes={chart_lanes}\t"
                f"count={len(filtered_notes)}\tt_min={t_min:.6f}\tt_max={t_max:.6f}\t"
                f"notes_rfc={notes_path}\n"
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
        return False


