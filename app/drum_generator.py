# app/drum_generator.py
import os
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
from .generation_presets import resolve_generation_preset

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
    preset: Optional[Dict] = None,
):
    preset = preset or resolve_generation_preset(None, generation_mode)
    mode = str(preset.get("mode", generation_mode or "basic")).lower()
    if not bool(preset.get("allow_client_overrides", False)):
        return (
            int(preset["fill"]),
            int(preset["groove"]),
            int(preset["density"]),
            bool(preset.get("accent_strong_beats", 0)),
            int(preset.get("genre_template_strength", 60)),
        )
    base = preset
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


def _select_raw_events(
    kick_times: List[float],
    snare_times: List[float],
    dominant_onsets: List[float],
    policy: str,
) -> List[float]:
    drum_hits = sorted(set(kick_times + snare_times))
    dominant = sorted(set(dominant_onsets))

    if policy == "drum_hits_with_dominant_fallback":
        return drum_hits or dominant
    if policy == "drum_hits_only":
        return drum_hits
    return dominant or drum_hits


def _cap_events_per_second(events: List[float], max_hits_per_second: int) -> List[float]:
    if not events or max_hits_per_second <= 0:
        return events
    kept: List[float] = []
    for event in sorted(events):
        window_start = event - 1.0
        recent = [t for t in kept if t > window_start]
        if len(recent) < max_hits_per_second:
            kept.append(event)
    return kept


def _effective_max_hits_per_second(preset: Dict, bpm: float) -> int:
    configured = int(preset.get("max_hits_per_second", 0) or 0)
    if configured <= 0:
        return 0
    if str(preset.get("mode", "")) != "basic":
        return configured
    sixteenth_rate = (4.0 * max(1.0, bpm)) / 60.0
    adaptive_floor = int(round(sixteenth_rate * 0.85))
    return max(configured, adaptive_floor)


def _effective_max_notes_per_measure(preset: Dict, bpm: float) -> int:
    configured = int(preset.get("max_notes_per_measure", 0) or 0)
    if configured <= 0:
        return 0
    if str(preset.get("mode", "")) != "basic":
        return configured
    if bpm >= 170:
        return max(configured, 10)
    return configured


def _cap_events_per_measure(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    max_notes_per_measure: int,
) -> List[float]:
    if not events or max_notes_per_measure <= 0:
        return events

    beat_interval = 60.0 / max(1.0, bpm)
    measure_duration = beat_interval * 4
    if beats is not None and len(beats) >= 2:
        first_measure_start = float(beats[0])
    else:
        first_measure_start = min(events)

    buckets: Dict[int, List[float]] = {}
    for event in sorted(events):
        idx = int(max(0, np.floor((event - first_measure_start) / measure_duration)))
        bucket = buckets.setdefault(idx, [])
        if len(bucket) < max_notes_per_measure:
            bucket.append(event)

    capped: List[float] = []
    for idx in sorted(buckets):
        capped.extend(buckets[idx])
    return capped


def _cluster_hit_events(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
) -> List[float]:
    if not events:
        return events
    cluster_window = float(preset.get("hit_cluster_window", 0.0) or 0.0)
    if cluster_window <= 0:
        return sorted(events)

    beat_interval = 60.0 / max(1.0, bpm)
    max_musical_window = beat_interval * 0.16
    window = min(cluster_window, max_musical_window)

    clusters: List[List[float]] = []
    current: List[float] = []
    for event in sorted(events):
        if not current:
            current = [event]
            continue
        if event - current[-1] <= window:
            current.append(event)
        else:
            clusters.append(current)
            current = [event]
    if current:
        clusters.append(current)

    clustered: List[float] = []
    for cluster in clusters:
        if len(cluster) == 1:
            clustered.append(cluster[0])
            continue
        if beats is not None and len(beats) > 0:
            best = min(cluster, key=lambda t: float(np.min(np.abs(beats - t))))
            clustered.append(best)
        else:
            clustered.append(cluster[0])
    return sorted(clustered)


def _apply_density_guardrails(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
) -> List[float]:
    guarded = _cap_events_per_second(events, _effective_max_hits_per_second(preset, bpm))
    guarded = _cap_events_per_measure(guarded, beats, bpm, _effective_max_notes_per_measure(preset, bpm))
    return guarded


def _split_core_and_extra_events(
    events: List[float],
    core_sources: List[float],
    preset: Dict,
) -> tuple[List[float], List[float]]:
    if not events:
        return [], []
    if not bool(preset.get("preserve_core_hits", False)) or not core_sources:
        return [], sorted(events)

    tol = float(preset.get("core_hit_tolerance", 0.08) or 0.08)
    core_sorted = sorted(core_sources)
    core_events: List[float] = []
    extra_events: List[float] = []
    for event in sorted(events):
        if _has_near(event, core_sorted, tol):
            core_events.append(event)
        else:
            extra_events.append(event)
    return core_events, extra_events


def _apply_density_guardrails_preserving_core(
    events: List[float],
    core_sources: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
) -> List[float]:
    core_events, extra_events = _split_core_and_extra_events(events, core_sources, preset)
    if not core_events:
        return _apply_density_guardrails(events, beats, bpm, preset)

    kept = sorted(set(core_events))
    for event in sorted(set(extra_events)):
        if event in kept:
            continue
        candidate = sorted(kept + [event])
        guarded = _apply_density_guardrails(candidate, beats, bpm, preset)
        if event in guarded:
            kept = candidate
    return kept


def _measure_position_buckets(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
) -> tuple[Dict[int, set], float, float]:
    beat_interval = float(np.median(np.diff(beats))) if beats is not None and len(beats) >= 2 else 60.0 / max(1.0, bpm)
    first_measure_start = float(beats[0]) if beats is not None and len(beats) > 0 else min(events)
    measure_duration = beat_interval * 4
    buckets: Dict[int, set] = {}
    for event in sorted(events):
        measure_idx = int(np.floor((event - first_measure_start) / measure_duration))
        if measure_idx < 0:
            continue
        rel_beats = (event - (first_measure_start + measure_idx * measure_duration)) / beat_interval
        if rel_beats < -0.1 or rel_beats >= 4.1:
            continue
        quantized = round(max(0.0, min(3.75, rel_beats)) * 4.0) / 4.0
        buckets.setdefault(measure_idx, set()).add(quantized)
    return buckets, first_measure_start, beat_interval


def _complete_groove_from_neighbors(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    verbose: bool,
) -> List[float]:
    if not bool(preset.get("groove_completion", False)) or not events:
        return events
    if beats is None or len(beats) < 8:
        return events

    radius = int(preset.get("groove_completion_radius", 4) or 4)
    min_support = int(preset.get("groove_completion_min_support", 3) or 3)
    max_add_per_measure = int(preset.get("groove_completion_max_add_per_measure", 1) or 1)
    max_notes_per_measure = int(preset.get("max_notes_per_measure", 0) or 0)
    if radius <= 0 or min_support <= 0 or max_add_per_measure <= 0:
        return events

    buckets, first_measure_start, beat_interval = _measure_position_buckets(events, beats, bpm)
    if not buckets:
        return events

    measure_duration = beat_interval * 4
    completed = sorted(set(events))
    added: List[float] = []
    cluster_window = float(preset.get("hit_cluster_window", 0.0) or 0.0)
    existing_tol = max(min(0.04, beat_interval * 0.12), min(cluster_window, beat_interval * 0.28))

    for measure_idx in sorted(buckets.keys()):
        current_positions = set(buckets.get(measure_idx, set()))
        neighbor_counts: Dict[float, int] = {}
        neighbor_sizes: List[int] = []
        for other_idx in range(measure_idx - radius, measure_idx + radius + 1):
            if other_idx == measure_idx or other_idx not in buckets:
                continue
            positions = buckets[other_idx]
            neighbor_sizes.append(len(positions))
            for pos in positions:
                neighbor_counts[pos] = neighbor_counts.get(pos, 0) + 1
        if not neighbor_counts or not neighbor_sizes:
            continue

        target_count = int(round(float(np.median(neighbor_sizes))))
        if max_notes_per_measure > 0:
            target_count = min(target_count, max_notes_per_measure)
        room = max(0, target_count - len(current_positions))
        if room <= 0:
            continue
        room = min(room, max_add_per_measure)

        candidates = [
            (pos, count)
            for pos, count in neighbor_counts.items()
            if count >= min_support and pos not in current_positions
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        measure_start = first_measure_start + measure_idx * measure_duration
        added_here = 0
        for pos, _count in candidates:
            if added_here >= room:
                break
            candidate_time = measure_start + pos * beat_interval
            if candidate_time < 0:
                continue
            if _has_near(candidate_time, completed, existing_tol):
                continue
            completed.append(candidate_time)
            added.append(candidate_time)
            current_positions.add(pos)
            added_here += 1

    if verbose and added:
        print(f"[DrumGen][этап] groove_completion=+{len(added)}")
    return sorted(completed)


def _apply_expected_groove_grid(
    events: List[float],
    candidate_events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    verbose: bool,
) -> List[float]:
    expected_groove = str(preset.get("expected_groove", "") or "")
    if expected_groove != "halfbeat_drive" or not events:
        return events
    if beats is None or len(beats) < 8:
        return events

    buckets, first_measure_start, beat_interval = _measure_position_buckets(events, beats, bpm)
    if not buckets:
        return events

    expected_positions = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    radius = int(preset.get("expected_groove_radius", 4) or 4)
    min_support = int(preset.get("expected_groove_min_support", 4) or 4)
    max_add_per_measure = int(preset.get("expected_groove_max_add_per_measure", 2) or 2)
    max_notes_per_measure = int(preset.get("max_notes_per_measure", 0) or 0)
    seek_window = 0.09
    allow_grid_fallback = False
    if radius <= 0 or min_support <= 0 or max_add_per_measure <= 0:
        return events
    if seek_window <= 0:
        return events

    cluster_window = float(preset.get("hit_cluster_window", 0.0) or 0.0)
    existing_tol = max(min(0.04, beat_interval * 0.12), min(cluster_window, beat_interval * 0.28))
    measure_duration = beat_interval * 4
    completed = sorted(set(events))
    candidate_pool = sorted(set(candidate_events or []))
    added: List[float] = []
    attempted = 0

    for measure_idx in sorted(buckets.keys()):
        current_positions = set(buckets.get(measure_idx, set()))
        if max_notes_per_measure > 0 and len(current_positions) >= max_notes_per_measure:
            continue
        support: Dict[float, int] = {}
        for other_idx in range(measure_idx - radius, measure_idx + radius + 1):
            if other_idx == measure_idx or other_idx not in buckets:
                continue
            positions = buckets[other_idx]
            for pos in expected_positions:
                if pos in positions:
                    support[pos] = support.get(pos, 0) + 1
        supported_positions = [pos for pos in expected_positions if support.get(pos, 0) >= min_support]
        if not supported_positions:
            continue
        missing = [pos for pos in supported_positions if pos not in current_positions]
        if not missing:
            continue
        missing.sort(key=lambda pos: (-support.get(pos, 0), pos))

        measure_start = first_measure_start + measure_idx * measure_duration
        added_here = 0
        for pos in missing:
            if added_here >= max_add_per_measure:
                break
            if max_notes_per_measure > 0 and len(current_positions) >= max_notes_per_measure:
                break
            target_time = measure_start + pos * beat_interval
            if target_time < 0:
                continue
            attempted += 1

            selected_time = None
            if candidate_pool:
                local_candidates = [
                    t for t in candidate_pool
                    if abs(t - target_time) <= seek_window and not _has_near(t, completed, existing_tol)
                ]
                if local_candidates:
                    selected_time = min(local_candidates, key=lambda t: abs(t - target_time))
            if selected_time is None and allow_grid_fallback:
                selected_time = target_time

            if selected_time is None:
                continue
            if _has_near(selected_time, completed, existing_tol):
                continue
            completed.append(selected_time)
            added.append(selected_time)
            current_positions.add(pos)
            added_here += 1

    if verbose:
        print(f"[DrumGen][этап] expected_groove=+{len(added)} attempts={attempted} seek_window={seek_window:.3f}")
    return sorted(completed)


def _detect_fill_candidate_measures(
    events: List[float],
    candidate_events: List[float],
    beats: np.ndarray,
    bpm: float,
) -> set:
    if not events or beats is None or len(beats) < 8:
        return set()

    buckets, _first_measure_start, _beat_interval = _measure_position_buckets(events, beats, bpm)
    if not buckets:
        return set()

    if len(buckets) < 6:
        return set()

    counts = [len(positions) for positions in buckets.values()]
    median_count = float(np.median(counts)) if counts else 0.0
    dense_cutoff = max(6, int(np.ceil(median_count + 2)))

    candidate_buckets, _cand_start, _cand_beat = _measure_position_buckets(candidate_events, beats, bpm)
    candidate_counts = [len(positions) for positions in candidate_buckets.values()]
    candidate_median = float(np.median(candidate_counts)) if candidate_counts else 0.0
    candidate_dense_cutoff = max(6, int(np.ceil(candidate_median + 2)))

    fill_indices: set = set()
    for measure_idx in sorted(buckets.keys()):
        current_count = len(buckets.get(measure_idx, set()))
        prev_count = len(buckets.get(measure_idx - 1, set())) if measure_idx - 1 in buckets else current_count
        next_count = len(buckets.get(measure_idx + 1, set())) if measure_idx + 1 in buckets else current_count
        local_jump = abs(current_count - prev_count) >= 3 or abs(current_count - next_count) >= 3

        cand_count = len(candidate_buckets.get(measure_idx, set()))
        is_dense = current_count >= dense_cutoff or cand_count >= candidate_dense_cutoff
        if is_dense and local_jump:
            fill_indices.add(measure_idx)
            fill_indices.add(measure_idx - 1)
            fill_indices.add(measure_idx + 1)

    return fill_indices


def _reinforce_repeating_measure_hits(
    events: List[float],
    candidate_events: List[float],
    beats: np.ndarray,
    bpm: float,
    mode: str,
    preset: Dict,
    verbose: bool,
) -> List[float]:
    if not events or not bool(preset.get("loop_reinforce", False)):
        return events
    if beats is None or len(beats) < 8:
        return events

    buckets, first_measure_start, beat_interval = _measure_position_buckets(events, beats, bpm)
    if not buckets:
        return events

    measure_duration = beat_interval * 4.0
    existing_tol = max(min(0.04, beat_interval * 0.12), min(float(preset.get("hit_cluster_window", 0.0) or 0.0), beat_interval * 0.28))
    seek_window = 0.09
    max_add_per_measure = 2
    max_notes_per_measure = int(preset.get("max_notes_per_measure", 0) or 0)

    completed = sorted(set(events))
    candidate_pool = sorted(set(candidate_events or []))
    if not candidate_pool:
        return completed

    added: List[float] = []
    attempts = 0
    fill_indices = _detect_fill_candidate_measures(events, candidate_pool, beats, bpm)
    all_indices = sorted(buckets.keys())
    for measure_idx in all_indices:
        if measure_idx in fill_indices:
            continue
        if measure_idx - 1 not in buckets:
            continue
        prev_positions = set(buckets.get(measure_idx - 1, set()))
        if measure_idx - 2 in buckets:
            prev_positions &= set(buckets.get(measure_idx - 2, set()))
        if not prev_positions:
            continue

        current_positions = set(buckets.get(measure_idx, set()))
        if max_notes_per_measure > 0 and len(current_positions) >= max_notes_per_measure:
            continue
        missing = sorted(pos for pos in prev_positions if pos not in current_positions)
        if not missing:
            continue

        added_here = 0
        measure_start = first_measure_start + measure_idx * measure_duration
        for pos in missing:
            if added_here >= max_add_per_measure:
                break
            if max_notes_per_measure > 0 and len(current_positions) >= max_notes_per_measure:
                break
            target_time = measure_start + pos * beat_interval
            attempts += 1
            local_candidates = [
                t for t in candidate_pool
                if abs(t - target_time) <= seek_window and not _has_near(t, completed, existing_tol)
            ]
            if not local_candidates:
                continue
            selected_time = min(local_candidates, key=lambda t: abs(t - target_time))
            if _has_near(selected_time, completed, existing_tol):
                continue
            completed.append(selected_time)
            added.append(selected_time)
            current_positions.add(pos)
            added_here += 1

    if verbose:
        print(
            f"[DrumGen][этап] loop_reinforce=+{len(added)} attempts={attempts} "
            f"seek_window={seek_window:.3f} fill_skip={len(fill_indices)}"
        )
    return sorted(completed)


def _reinforce_four_bar_loop_hits(
    events: List[float],
    candidate_events: List[float],
    beats: np.ndarray,
    bpm: float,
    mode: str,
    preset: Dict,
    verbose: bool,
) -> List[float]:
    if not events or not bool(preset.get("loop4_reinforce", False)):
        return events
    if beats is None or len(beats) < 16:
        return events

    buckets, first_measure_start, beat_interval = _measure_position_buckets(events, beats, bpm)
    if not buckets:
        return events

    measure_duration = beat_interval * 4.0
    cluster_window = float(preset.get("hit_cluster_window", 0.0) or 0.0)
    existing_tol = max(min(0.04, beat_interval * 0.12), min(cluster_window, beat_interval * 0.28))
    seek_window = 0.09
    max_add_per_measure = 2
    max_notes_per_measure = int(preset.get("max_notes_per_measure", 0) or 0)

    completed = sorted(set(events))
    candidate_pool = sorted(set(candidate_events or []))
    if not candidate_pool:
        return completed

    fill_indices = _detect_fill_candidate_measures(events, candidate_pool, beats, bpm)
    added: List[float] = []
    attempts = 0
    for measure_idx in sorted(buckets.keys()):
        if measure_idx in fill_indices:
            continue
        reference_positions: set = set()
        for ref_idx in (measure_idx - 4, measure_idx + 4):
            if ref_idx in buckets:
                reference_positions.update(set(buckets.get(ref_idx, set())))
        if not reference_positions:
            continue

        current_positions = set(buckets.get(measure_idx, set()))
        if max_notes_per_measure > 0 and len(current_positions) >= max_notes_per_measure:
            continue
        missing = sorted(pos for pos in reference_positions if pos not in current_positions)
        if not missing:
            continue

        added_here = 0
        measure_start = first_measure_start + measure_idx * measure_duration
        for pos in missing:
            if added_here >= max_add_per_measure:
                break
            if max_notes_per_measure > 0 and len(current_positions) >= max_notes_per_measure:
                break
            target_time = measure_start + pos * beat_interval
            attempts += 1
            local_candidates = [
                t for t in candidate_pool
                if abs(t - target_time) <= seek_window and not _has_near(t, completed, existing_tol)
            ]
            if not local_candidates:
                continue
            selected_time = min(local_candidates, key=lambda t: abs(t - target_time))
            if _has_near(selected_time, completed, existing_tol):
                continue
            completed.append(selected_time)
            added.append(selected_time)
            current_positions.add(pos)
            added_here += 1

    if verbose:
        print(
            f"[DrumGen][этап] loop4_reinforce=+{len(added)} attempts={attempts} "
            f"seek_window={seek_window:.3f} fill_skip={len(fill_indices)}"
        )
    return sorted(completed)


def _recover_fill_single_misses(
    events: List[float],
    candidate_events: List[float],
    beats: np.ndarray,
    bpm: float,
    mode: str,
    preset: Dict,
    verbose: bool,
) -> List[float]:
    if not events or not bool(preset.get("fill_recover", False)):
        return events
    if beats is None or len(beats) < 8:
        return events

    buckets, first_measure_start, beat_interval = _measure_position_buckets(events, beats, bpm)
    if not buckets:
        return events
    candidate_buckets, _cand_start, _cand_beat = _measure_position_buckets(candidate_events, beats, bpm)
    if not candidate_buckets:
        return events

    fill_indices = _detect_fill_candidate_measures(events, candidate_events, beats, bpm)
    if not fill_indices:
        return events

    measure_duration = beat_interval * 4.0
    existing_tol = max(min(0.04, beat_interval * 0.12), min(float(preset.get("hit_cluster_window", 0.0) or 0.0), beat_interval * 0.28))
    seek_window = 0.07
    max_notes_per_measure = int(preset.get("max_notes_per_measure", 0) or 0)
    fill_anchor_positions = [round(i * 0.25, 2) for i in range(16)]

    completed = sorted(set(events))
    candidate_pool = sorted(set(candidate_events or []))
    added: List[float] = []
    attempts = 0

    for measure_idx in sorted(fill_indices):
        current_positions = set(buckets.get(measure_idx, set()))
        if max_notes_per_measure > 0 and len(current_positions) >= max_notes_per_measure:
            continue
        candidate_positions = set(candidate_buckets.get(measure_idx, set()))
        if not candidate_positions:
            continue

        missing_positions = [pos for pos in fill_anchor_positions if pos in candidate_positions and pos not in current_positions]
        if not missing_positions:
            continue

        measure_start = first_measure_start + measure_idx * measure_duration
        missing_positions.sort(key=lambda pos: min(abs(pos - 1.75), abs(pos - 2.0), abs(pos - 2.25), abs(pos - 2.5)))
        for pos in missing_positions:
            attempts += 1
            target_time = measure_start + pos * beat_interval
            local_candidates = [
                t for t in candidate_pool
                if abs(t - target_time) <= seek_window and not _has_near(t, completed, existing_tol)
            ]
            if not local_candidates:
                continue
            selected_time = min(local_candidates, key=lambda t: abs(t - target_time))
            if _has_near(selected_time, completed, existing_tol):
                continue
            completed.append(selected_time)
            added.append(selected_time)
            break

    if verbose:
        print(
            f"[DrumGen][этап] fill_recover=+{len(added)} attempts={attempts} "
            f"seek_window={seek_window:.3f}"
        )
    return sorted(completed)


def _apply_basic_section_timing_correction(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    mode: str,
    preset: Dict,
    filtered_events: List[float],
    verbose: bool,
) -> List[float]:
    if not events:
        return events
    if beats is None or len(beats) < 16:
        return events
    if not bool(preset.get("section_timing_correction", False)):
        return events

    strength = float(preset.get("section_timing_correction_strength", 0.45) or 0.45)
    cap_ms = float(preset.get("section_timing_correction_cap_ms", 14.0) or 14.0)
    min_events = int(preset.get("section_timing_correction_min_events", 8) or 8)
    median_ignore_below_ms = float(preset.get("section_timing_correction_ignore_below_ms", 5.0) or 5.0)
    if strength <= 0 or cap_ms <= 0:
        return events

    beat_interval = float(np.median(np.diff(beats))) if len(beats) >= 2 else 60.0 / max(1.0, bpm)
    if beat_interval <= 0:
        return events

    first_measure_start = float(beats[0])
    measure_duration = beat_interval * 4.0
    window_duration = measure_duration * 4.0
    sorted_ev = sorted(events)
    fill_skip = (
        _detect_fill_candidate_measures(events, filtered_events, beats, bpm)
        if filtered_events
        else set()
    )

    def measure_idx_for(t: float) -> int:
        return int(np.floor((t - first_measure_start) / measure_duration))

    max_time = max(sorted_ev[-1], float(beats[-1]))
    sixteenth_grid = np.arange(first_measure_start, max_time + beat_interval, beat_interval / 4.0)
    if len(sixteenth_grid) == 0:
        return events

    t0 = first_measure_start
    num_blocks = max(1, int(np.ceil((sorted_ev[-1] - t0) / window_duration)) + 1)

    block_corr_ms: List[float] = []
    for b in range(num_blocks):
        start_w = t0 + b * window_duration
        end_w = start_w + window_duration
        offsets_ms: List[float] = []
        for t in sorted_ev:
            if not (start_w <= t < end_w):
                continue
            if fill_skip and measure_idx_for(t) in fill_skip:
                continue
            nearest = float(sixteenth_grid[int(np.argmin(np.abs(sixteenth_grid - t)))])
            offsets_ms.append((t - nearest) * 1000.0)
        if len(offsets_ms) >= min_events:
            med = float(np.median(np.array(offsets_ms, dtype=float)))
            if abs(med) < median_ignore_below_ms:
                block_corr_ms.append(0.0)
            else:
                corr = -strength * med
                block_corr_ms.append(float(max(-cap_ms, min(cap_ms, corr))))
        else:
            block_corr_ms.append(0.0)

    adjusted: List[float] = []
    max_abs_ms = 0.0
    last_b = len(block_corr_ms) - 1
    for t in sorted_ev:
        if fill_skip and measure_idx_for(t) in fill_skip:
            adjusted.append(t)
            continue
        b = int(np.floor((t - t0) / window_duration))
        if b < 0:
            b = 0
        elif b > last_b:
            b = last_b
        corr_ms = block_corr_ms[b]
        max_abs_ms = max(max_abs_ms, abs(corr_ms))
        adjusted.append(t + corr_ms / 1000.0)

    if verbose:
        print(f"[DrumGen][этап] section_timing_corr max_abs_ms={max_abs_ms:.1f} blocks={num_blocks} piecewise=1")
    return sorted(adjusted)


def _rhythm_diagnostics_enabled() -> bool:
    return os.getenv("RF_RHYTHM_DIAG", "1") == "1"


def _signature_label(signature: tuple) -> str:
    if not signature:
        return "-"
    return ",".join(f"{pos:g}" for pos in signature)


def _print_timing_offset_diagnostics(
    label: str,
    events: List[float],
    beats: np.ndarray,
    beat_interval: float,
) -> None:
    if not events or beats is None or len(beats) < 2 or beat_interval <= 0:
        return

    max_time = max(float(events[-1]), float(beats[-1]))
    eighth_step = beat_interval / 2.0
    sixteenth_step = beat_interval / 4.0
    eighth_grid = np.arange(float(beats[0]), max_time + beat_interval, eighth_step)
    sixteenth_grid = np.arange(float(beats[0]), max_time + beat_interval, sixteenth_step)
    if len(eighth_grid) == 0 or len(sixteenth_grid) == 0:
        return

    offsets_8_ms: List[float] = []
    offsets_16_ms: List[float] = []
    for t in sorted(events):
        e8 = float(eighth_grid[int(np.argmin(np.abs(eighth_grid - t)))])
        e16 = float(sixteenth_grid[int(np.argmin(np.abs(sixteenth_grid - t)))])
        offsets_8_ms.append((t - e8) * 1000.0)
        offsets_16_ms.append((t - e16) * 1000.0)

    arr8 = np.array(offsets_8_ms, dtype=float)
    arr16 = np.array(offsets_16_ms, dtype=float)
    late8 = float(np.mean(arr8 > 0.0) * 100.0)
    late16 = float(np.mean(arr16 > 0.0) * 100.0)

    print(
        f"[RhythmDiag]   timing_{label}_8th_ms median={np.median(arr8):.1f} "
        f"p10={np.percentile(arr8, 10):.1f} p90={np.percentile(arr8, 90):.1f} "
        f"late={late8:.0f}%"
    )
    print(
        f"[RhythmDiag]   timing_{label}_16th_ms median={np.median(arr16):.1f} "
        f"p10={np.percentile(arr16, 10):.1f} p90={np.percentile(arr16, 90):.1f} "
        f"late={late16:.0f}%"
    )


def _print_section_timing_offset_diagnostics(
    label: str,
    events: List[float],
    beats: np.ndarray,
    beat_interval: float,
) -> None:
    if not events or beats is None or len(beats) < 16 or beat_interval <= 0:
        return

    measure_duration = beat_interval * 4.0
    first_measure_start = float(beats[0])
    total_measures = int(np.ceil((float(events[-1]) - first_measure_start) / measure_duration))
    if total_measures < 8:
        return

    sixteenth_step = beat_interval / 4.0
    max_time = max(float(events[-1]), float(beats[-1]))
    sixteenth_grid = np.arange(first_measure_start, max_time + beat_interval, sixteenth_step)
    if len(sixteenth_grid) == 0:
        return

    window_measures = 4
    window_stats: List[tuple[int, float, float, int]] = []
    sorted_events = sorted(events)
    for start_measure in range(0, max(1, total_measures - window_measures + 1)):
        start_t = first_measure_start + start_measure * measure_duration
        end_t = start_t + window_measures * measure_duration
        section_events = [t for t in sorted_events if start_t <= t < end_t]
        if len(section_events) < 8:
            continue
        offsets = []
        for t in section_events:
            nearest = float(sixteenth_grid[int(np.argmin(np.abs(sixteenth_grid - t)))])
            offsets.append((t - nearest) * 1000.0)
        arr = np.array(offsets, dtype=float)
        window_stats.append((start_measure, float(np.median(arr)), float(np.mean(arr > 0.0) * 100.0), len(section_events)))

    if not window_stats:
        return

    most_late = sorted(window_stats, key=lambda item: item[1], reverse=True)[:2]
    most_early = sorted(window_stats, key=lambda item: item[1])[:2]
    print(f"[RhythmDiag]   timing_{label}_sections_4bar total={len(window_stats)}")
    for start, median_ms, late_pct, n in most_late:
        print(
            f"[RhythmDiag]     late m{start}-{start + window_measures - 1}: "
            f"median={median_ms:.1f}ms late={late_pct:.0f}% n={n}"
        )
    for start, median_ms, late_pct, n in most_early:
        print(
            f"[RhythmDiag]     early m{start}-{start + window_measures - 1}: "
            f"median={median_ms:.1f}ms late={late_pct:.0f}% n={n}"
        )


def _print_rhythm_diagnostics(
    label: str,
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    mode: str,
    preset_id: str,
) -> None:
    if not _rhythm_diagnostics_enabled() or not events or beats is None or len(beats) < 8:
        return

    beat_interval = float(np.median(np.diff(beats))) if len(beats) >= 2 else 60.0 / max(1.0, bpm)
    if beat_interval <= 0:
        return

    measure_duration = beat_interval * 4
    first_measure_start = float(beats[0])
    buckets: Dict[int, List[float]] = {}
    for event in sorted(events):
        measure_idx = int(np.floor((event - first_measure_start) / measure_duration))
        if measure_idx < 0:
            continue
        rel_beats = (event - (first_measure_start + measure_idx * measure_duration)) / beat_interval
        if rel_beats < -0.1 or rel_beats >= 4.1:
            continue
        quantized = round(max(0.0, min(3.75, rel_beats)) * 4.0) / 4.0
        bucket = buckets.setdefault(measure_idx, [])
        if quantized not in bucket:
            bucket.append(quantized)

    if not buckets:
        return

    signatures: Dict[tuple, int] = {}
    counts: List[int] = []
    for positions in buckets.values():
        signature = tuple(sorted(positions))
        signatures[signature] = signatures.get(signature, 0) + 1
        counts.append(len(signature))

    top = sorted(signatures.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[:5]
    median_count = float(np.median(counts)) if counts else 0.0
    top_count = top[0][1] if top else 0
    stable_measures = sum(count for _, count in top[:2])
    dense_measures = [
        idx for idx, positions in sorted(buckets.items())
        if len(positions) >= max(5, int(np.ceil(median_count + 3)))
    ][:8]
    sparse_measures = [
        idx for idx, positions in sorted(buckets.items())
        if len(positions) <= max(1, int(np.floor(median_count - 2)))
    ][:8]

    print(
        f"[RhythmDiag] {label} mode={mode} preset={preset_id} "
        f"events={len(events)} measures={len(buckets)} median_notes={median_count:.1f} "
        f"stable_top2={stable_measures}/{len(buckets)}"
    )
    for i, (signature, count) in enumerate(top, start=1):
        print(f"[RhythmDiag]   top{i}: x{count} notes={len(signature)} sig={_signature_label(signature)}")
    if dense_measures:
        print(f"[RhythmDiag]   dense_measure_candidates={dense_measures}")
    if sparse_measures and top_count < len(buckets):
        print(f"[RhythmDiag]   sparse_measure_candidates={sparse_measures}")
    _print_timing_offset_diagnostics(label, sorted(events), beats, beat_interval)
    _print_section_timing_offset_diagnostics(label, sorted(events), beats, beat_interval)


def _append_events_with_density_guardrails(
    base_events: List[float],
    added_events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
) -> List[float]:
    kept = sorted(set(base_events))
    for event in sorted(set(added_events)):
        if event in kept:
            continue
        candidate = sorted(kept + [event])
        if len(_apply_density_guardrails(candidate, beats, bpm, preset)) == len(candidate):
            kept = candidate
    return kept


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
    preset_id: Optional[str] = None,
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
    preset = resolve_generation_preset(preset_id, generation_mode)
    preset_id = str(preset.get("preset_id", preset_id or generation_mode or "basic"))
    mode = str(preset.get("mode", generation_mode or "basic")).lower()
    fill, groove, density, accent_strong_beats, genre_template_strength = _presets_for_mode(
        mode,
        fill,
        groove,
        density,
        accent_strong_beats,
        genre_template_strength,
        preset=preset,
    )
    genre_template_strength = int(max(0, min(100, int(genre_template_strength))))
    if grid_snap_strength is None or not bool(preset.get("allow_client_overrides", False)):
        grid_snap_strength = int(preset.get("grid_snap_strength", 35))
    grid_snap_strength = int(max(0, min(100, int(grid_snap_strength))))
    grid_snap_enabled = grid_snap_strength > 0

    if verbose:
        print(f"[DrumGen] preset={preset_id} режим={mode} fill={fill} groove={groove} density={density} grid_snap_strength={grid_snap_strength} accent_strong_beats={accent_strong_beats} genre_template_strength={genre_template_strength} bpm={bpm} lanes={lanes}")

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

    raw_events = _select_raw_events(
        kick_times,
        snare_times,
        dominant_onsets,
        str(preset.get("dominant_onsets_policy", "dominant_onsets")),
    )
    if not raw_events:
        return None
    _print_rhythm_diagnostics("source", raw_events, beats, bpm, mode, preset_id)

    if "sync_tolerance_multiplier" in genre_params:
        sync_tolerance = float(sync_tolerance) * float(genre_params.get("sync_tolerance_multiplier", 1.0))

    drum_start_window = float(genre_params.get("drum_start_window", 4.0))
    drum_density_threshold = float(genre_params.get("drum_density_threshold", 0.5))
    drum_section_start = detect_drum_section_start(raw_events, drum_start_window, drum_density_threshold)
    filtered_events = [t for t in raw_events if t >= drum_section_start]
    core_sources = sorted(set(t for t in (kick_times + snare_times) if t >= drum_section_start))

    min_note_distance = float(genre_params.get("min_note_distance", 0.05))
    if mode == "custom":
        min_note_distance = _density_to_min_distance(min_note_distance, density)
    elif mode == "minimal":
        min_note_distance = min(0.22, max(0.06, min_note_distance * 1.35))
    min_note_distance = max(min_note_distance, float(preset.get("min_note_distance_floor", 0.0) or 0.0))

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
    before_cluster = len(events_after_timing)
    events_after_timing = _cluster_hit_events(events_after_timing, beats, bpm, preset)
    if verbose and len(events_after_timing) != before_cluster:
        print(f"[DrumGen][этап] hit_cluster={before_cluster}->{len(events_after_timing)}")
    events_after_timing = _complete_groove_from_neighbors(events_after_timing, beats, bpm, preset, verbose)
    events_after_timing = _apply_expected_groove_grid(
        events_after_timing,
        filtered_events,
        beats,
        bpm,
        preset,
        verbose,
    )
    events_after_timing = _reinforce_repeating_measure_hits(
        events_after_timing,
        filtered_events,
        beats,
        bpm,
        mode,
        preset,
        verbose,
    )
    events_after_timing = _reinforce_four_bar_loop_hits(
        events_after_timing,
        filtered_events,
        beats,
        bpm,
        mode,
        preset,
        verbose,
    )
    events_after_timing = _recover_fill_single_misses(
        events_after_timing,
        filtered_events,
        beats,
        bpm,
        mode,
        preset,
        verbose,
    )
    events_after_timing = _apply_basic_section_timing_correction(
        events_after_timing,
        beats,
        bpm,
        mode,
        preset,
        filtered_events,
        verbose,
    )
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
    if mode == "basic":
        print(
            f"[DrumGen] Basic caps | hits/sec={_effective_max_hits_per_second(preset, bpm)} "
            f"notes/measure={_effective_max_notes_per_measure(preset, bpm)}"
        )

    if cancel_cb:
        cancel_cb()

    before_guardrails = len(events_after_timing)
    base_times = _apply_density_guardrails_preserving_core(
        list(events_after_timing),
        core_sources,
        beats,
        bpm,
        preset,
    )
    if verbose and len(base_times) != before_guardrails:
        print(f"[DrumGen][этап] density_guardrails={before_guardrails}->{len(base_times)}")
    if verbose and bool(preset.get("preserve_core_hits", False)):
        core_kept, extra_kept = _split_core_and_extra_events(base_times, core_sources, preset)
        print(f"[DrumGen][этап] core_hits={len(core_kept)} extra_hits={len(extra_kept)}")
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

    all_times = _append_events_with_density_guardrails(base_times, added_times, beats, bpm, preset)
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
            all_times = _append_events_with_density_guardrails(all_times, extra, beats, bpm, preset)
    _print_rhythm_diagnostics("final", all_times, beats, bpm, mode, preset_id)

    if status_cb:
        status_cb("Назначение линий...")
    if cancel_cb:
        cancel_cb()

    all_events = [{"type": NoteType.DRUM, "time": t} for t in all_times]
    notes = assign_lanes_to_notes(all_events, lanes=lanes, song_offset=0.0)

    if verbose:
        print(
            f"[DrumGen] Итого: {len(notes)} (обнаружено={len(base_times)}, "
            f"добавлено={max(0, len(all_times) - len(base_times))})"
        )

    return notes if notes else None
