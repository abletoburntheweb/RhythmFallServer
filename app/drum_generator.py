# app/drum_generator.py
import os
from copy import deepcopy
import numpy as np
from typing import List, Dict, Optional, Sequence, Tuple, Any

from .audio_analysis import analyze_audio, LIBROSA_AVAILABLE

if LIBROSA_AVAILABLE:
    import librosa
else:
    librosa = None
from . import stem_memory_cache
from .bass_transforms import diversify_lane_runs
from .drum_utils import (
    apply_temporal_filter,
    apply_groove_pattern,
    assign_lanes_to_notes,
    assign_lanes_by_drum_class,
    assign_lanes_ergonomic,
    chart_variant_suffix,
    detect_drum_section_start,
    load_drum_augment_profiles,
    save_drums_notes,
)
from .measure_energy import (
    adtof_drum_counts,
    annotate_salience_roles,
    build_measure_map,
    filter_hits_by_salience_roles,
    is_loud_mix_quiet_drum,
    is_phantom_orphan_measure,
    is_quiet_mix_breakdown,
    log_measure_map,
    measure_map_enabled,
    measure_map_recap_line,
    print_generation_recap,
    relative_contour_per_measure,
    rms_per_measure,
    salience_recap_line,
    drum_entry_recap_line,
)
from .note_types import NoteType
from .stage_ledger import StageLedger, log_stage_ledger
from .difficulty_transforms import apply_difficulty_transform
from .style_builders import (
    StyleBuildContext,
    apply_style_post_passes,
    build_style_map,
    get_style_policy,
)
from .drum_hit_detector import resolve_drum_at_time
from .genre_detector import get_genre_config
from .generation_presets import resolve_generation_preset, FILL_ZONE_DEFAULTS
from .rhythm_dna import build_rhythm_dna

_last_rhythm_dna: Optional[Dict[str, Any]] = None
_last_stage_ledger: Optional[Dict[str, Any]] = None


def get_last_rhythm_dna() -> Optional[Dict[str, Any]]:
    return _last_rhythm_dna


def get_last_stage_ledger() -> Optional[Dict[str, Any]]:
    return _last_stage_ledger


# Per-genre note-count budgets and pattern grids used by _augment_notes()/
# _get_pattern_positions() live in drum_augment_profiles.json — these musical
# facts ("how many hits/measure is idiomatic for this genre", "which beat
# positions are typically hit") belong to data, not to Python. Only a minimal
# "default" tier stays here as a safety net if the JSON is missing/broken.
_DRUM_AUGMENT_PROFILES: Dict[str, Any] = load_drum_augment_profiles()

_HARD_CAPS: Dict[str, Dict] = _DRUM_AUGMENT_PROFILES.get("note_count_ranges") or {
    "default": {"min": 4, "max": 7, "per_measure": 3, "per_measure_break": 4, "cap_ratio": 0.25},
}

_GENRE_PATTERN_POSITIONS: Dict[str, Any] = _DRUM_AUGMENT_PROFILES.get("rhythmic_grids") or {
    "default": [0.0, 2.0, 0.5, 1.5, 2.5, 3.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
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


def _classified_times_for_classes(
    classified_hits: Optional[List[Dict]],
    rhythm_classes: Optional[Sequence[str]],
) -> List[float]:
    if not classified_hits:
        return []
    if not rhythm_classes:
        return sorted({float(h["time"]) for h in classified_hits})
    allowed = {str(c).strip().lower() for c in rhythm_classes if str(c).strip()}
    if not allowed:
        return sorted({float(h["time"]) for h in classified_hits})
    return sorted(
        {
            float(h["time"])
            for h in classified_hits
            if str(h.get("drum", "")).strip().lower() in allowed
        }
    )


def _select_raw_events(
    kick_times: List[float],
    snare_times: List[float],
    dominant_onsets: List[float],
    policy: str,
    classified_hits: Optional[List[Dict]] = None,
    rhythm_classes: Optional[Sequence[str]] = None,
) -> List[float]:
    drum_hits = sorted(set(kick_times + snare_times))
    dominant = sorted(set(dominant_onsets))
    classified_times = _classified_times_for_classes(classified_hits, rhythm_classes)
    core_classified = _classified_times_for_classes(classified_hits, ("kick", "snare"))

    if policy == "classified_hits":
        return classified_times or drum_hits or dominant
    if policy == "kick_snare_core":
        merged = sorted(set(drum_hits) | set(core_classified))
        return merged or dominant
    if policy == "drum_hits_with_dominant_fallback":
        merged = sorted(set(drum_hits) | set(classified_times))
        return merged or dominant
    if policy == "drum_hits_only":
        return drum_hits or dominant
    return dominant or classified_times or drum_hits


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
    if not bool(preset.get("bpm_adaptive_caps", True)):
        return configured
    mode = str(preset.get("mode", ""))
    sixteenth_rate = (4.0 * max(1.0, bpm)) / 60.0
    if mode in ("basic", "enhanced"):
        rate_mult = 0.90 if mode == "enhanced" else 0.85
        adaptive_floor = int(round(sixteenth_rate * rate_mult))
        return max(configured, adaptive_floor)
    if mode == "minimal" and bpm >= 165:
        adaptive_floor = int(round(sixteenth_rate * 0.45))
        return max(configured, adaptive_floor)
    return configured


def _effective_max_notes_per_measure(preset: Dict, bpm: float) -> int:
    configured = int(preset.get("max_notes_per_measure", 0) or 0)
    if configured <= 0:
        return 0
    if str(preset.get("mode", "")) not in ("basic", "enhanced"):
        return configured
    if not bool(preset.get("bpm_adaptive_caps", True)):
        return configured
    if bpm >= 170:
        floor = 14 if str(preset.get("mode", "")) == "enhanced" else 12
        return max(configured, floor)
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


def _measure_timing(beats: np.ndarray, bpm: float) -> tuple[float, float]:
    beat_interval = float(np.median(np.diff(beats))) if beats is not None and len(beats) >= 2 else 60.0 / max(1.0, bpm)
    first_measure_start = float(beats[0]) if beats is not None and len(beats) > 0 else 0.0
    measure_duration = beat_interval * 4.0
    return first_measure_start, measure_duration


def _times_per_measure(
    times: List[float],
    beats: np.ndarray,
    bpm: float,
) -> Dict[int, List[float]]:
    if not times:
        return {}
    first_measure_start, measure_duration = _measure_timing(beats, bpm)
    buckets: Dict[int, List[float]] = {}
    for t in sorted(times):
        idx = int(np.floor((float(t) - first_measure_start) / measure_duration))
        if idx < 0:
            continue
        buckets.setdefault(idx, []).append(float(t))
    return buckets


def _counts_per_measure(
    times: List[float],
    beats: np.ndarray,
    bpm: float,
) -> Dict[int, int]:
    return {idx: len(vals) for idx, vals in _times_per_measure(times, beats, bpm).items()}


def _kick_snare_counts_per_measure(
    kick_times: List[float],
    snare_times: List[float],
    beats: np.ndarray,
    bpm: float,
) -> Dict[int, int]:
    ks_times = sorted(set(float(t) for t in (kick_times or []) + (snare_times or [])))
    return _counts_per_measure(ks_times, beats, bpm)


def _last_kick_snare_time(kick_times: List[float], snare_times: List[float]) -> float:
    vals = [float(t) for t in (kick_times or []) + (snare_times or [])]
    return max(vals) if vals else 0.0


def _stem_rms_per_measure(
    audio_path: Optional[str],
    beats: np.ndarray,
    bpm: float,
    max_measure: int,
) -> Dict[int, float]:
    if not audio_path or max_measure < 0:
        return {}
    first_measure_start, measure_duration = _measure_timing(beats, bpm)
    return rms_per_measure(audio_path, first_measure_start, measure_duration, max_measure)


def _section_energy_contours(
    stem_audio_path: Optional[str],
    mix_audio_path: Optional[str],
    beats: np.ndarray,
    bpm: float,
    max_measure: int,
    preset: Dict,
) -> tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
    stem_rms: Dict[int, float] = {}
    mix_rms: Dict[int, float] = {}
    drum_rel: Dict[int, float] = {}
    mix_rel: Dict[int, float] = {}
    if max_measure < 0:
        return stem_rms, mix_rms, drum_rel, mix_rel
    if bool(preset.get("section_stem_energy_enabled", False)) and stem_audio_path:
        stem_rms = _stem_rms_per_measure(stem_audio_path, beats, bpm, max_measure)
    if bool(preset.get("section_dual_energy_enabled", False)) and mix_audio_path:
        mix_rms = _stem_rms_per_measure(mix_audio_path, beats, bpm, max_measure)
    radius = int(preset.get("section_contour_rolling_radius", 16) or 16)
    if stem_rms:
        drum_rel = relative_contour_per_measure(stem_rms, max_measure, rolling_radius=radius)
    if mix_rms:
        mix_rel = relative_contour_per_measure(mix_rms, max_measure, rolling_radius=radius)
    return stem_rms, mix_rms, drum_rel, mix_rel


def _is_loud_mix_quiet_drum_measure(
    measure_idx: int,
    drum_rel: Dict[int, float],
    mix_rel: Dict[int, float],
    preset: Dict,
) -> bool:
    if not bool(preset.get("section_dual_energy_enabled", False)):
        return False
    return is_loud_mix_quiet_drum(
        float(drum_rel.get(measure_idx, 0.0) or 0.0),
        float(mix_rel.get(measure_idx, 0.0) or 0.0),
        mix_loud_min=float(preset.get("section_mix_loud_rel_min", 0.85) or 0.85),
        drum_quiet_max=float(preset.get("section_drum_quiet_rel_max", 0.55) or 0.55),
    )


def _stem_rms_median(stem_rms: Dict[int, float]) -> float:
    vals = [float(v) for v in stem_rms.values() if v > 0]
    if not vals:
        return 0.0
    return float(np.median(vals))


def _is_stem_quiet_measure(
    stem_rms: Dict[int, float],
    measure_idx: int,
    stem_median: float,
    preset: Dict,
) -> bool:
    rms = float(stem_rms.get(measure_idx, 0.0) or 0.0)
    quiet_ratio = float(preset.get("section_stem_quiet_ratio", 0.42) or 0.42)
    quiet_floor = float(preset.get("section_stem_quiet_floor", 0.006) or 0.006)
    if rms <= quiet_floor:
        return True
    if stem_median > 0 and rms < stem_median * quiet_ratio:
        return True
    return False


def _dropout_measures_after_last_dense(
    event_counts: Dict[int, int],
    ks_counts: Dict[int, int],
    stem_rms: Dict[int, float],
    stem_median: float,
    max_measure: int,
    preset: Dict,
    drum_rel: Optional[Dict[int, float]] = None,
    mix_rel: Optional[Dict[int, float]] = None,
) -> set[int]:
    if not bool(preset.get("section_dropout_enabled", False)) or not stem_rms:
        return set()
    drum_rel = drum_rel or {}
    mix_rel = mix_rel or {}
    dense_ks = int(preset.get("section_dropout_last_dense_ks", 2) or 2)
    trigger_notes = int(preset.get("section_sparse_block_trigger_notes", 3) or 3)
    min_run = int(preset.get("section_dropout_min_measures", 2) or 2)
    last_dense = -1
    for measure_idx in range(0, max_measure + 1):
        if int(ks_counts.get(measure_idx, 0)) >= dense_ks:
            last_dense = measure_idx
    if last_dense < 0:
        return set()
    blocked: set[int] = set()
    run_start: Optional[int] = None
    run_len = 0
    for measure_idx in range(last_dense + 1, max_measure + 1):
        if _is_loud_mix_quiet_drum_measure(measure_idx, drum_rel, mix_rel, preset):
            run_start = None
            run_len = 0
            continue
        count = int(event_counts.get(measure_idx, 0))
        quiet = _is_stem_quiet_measure(stem_rms, measure_idx, stem_median, preset)
        hit = quiet and count >= trigger_notes
        if hit:
            if run_start is None:
                run_start = measure_idx
            run_len += 1
        elif run_len >= min_run and run_start is not None:
            for m in range(run_start, run_start + run_len):
                blocked.add(m)
            run_start = None
            run_len = 0
        else:
            run_start = None
            run_len = 0
    if run_len >= min_run and run_start is not None:
        for m in range(run_start, run_start + run_len):
            blocked.add(m)
    return blocked


def _quiet_core_measure_runs(
    core_counts: Dict[int, int],
    max_measure: int,
    quiet_max: int,
    min_run: int,
) -> set[int]:
    if min_run <= 0:
        return set()
    blocked: set[int] = set()
    run_start: Optional[int] = None
    run_len = 0
    for measure_idx in range(0, max_measure + 1):
        quiet = int(core_counts.get(measure_idx, 0)) <= quiet_max
        if quiet:
            if run_start is None:
                run_start = measure_idx
            run_len += 1
        elif run_len >= min_run and run_start is not None:
            for m in range(run_start, run_start + run_len):
                blocked.add(m)
            run_start = None
            run_len = 0
        else:
            run_start = None
            run_len = 0
    if run_len >= min_run and run_start is not None:
        for m in range(run_start, run_start + run_len):
            blocked.add(m)
    return blocked


def _neighbor_median_count(
    counts: Dict[int, int],
    measure_idx: int,
    radius: int,
) -> float:
    vals: List[int] = []
    for other in range(measure_idx - radius, measure_idx + radius + 1):
        if other == measure_idx:
            continue
        if other in counts:
            vals.append(int(counts[other]))
    if not vals:
        return float(counts.get(measure_idx, 0))
    return float(np.median(vals))


_DRUM_CLASS_PRIORITY = {
    "kick": 0,
    "snare": 1,
    "tom": 2,
    "hat": 3,
    "cymbal": 4,
    "perc": 5,
}


def _event_class_priority(
    event_time: float,
    classified_hits: Optional[List[Dict]],
    tolerance: float,
) -> int:
    if not classified_hits:
        return 3
    drum = resolve_drum_at_time(float(event_time), classified_hits, tolerance=tolerance)
    return int(_DRUM_CLASS_PRIORITY.get(str(drum or "hat").lower(), 3))


def _cap_measure_events_class_aware(
    measure_events: List[float],
    cap: int,
    classified_hits: Optional[List[Dict]],
    tolerance: float,
    beats: Optional[np.ndarray] = None,
    bpm: float = 120.0,
    preset: Optional[Dict] = None,
) -> List[float]:
    if cap <= 0:
        return []
    if len(measure_events) <= cap:
        return sorted(measure_events)
    preset = preset or {}

    def _rank(t: float) -> Tuple[int, int, int, int, float]:
        # Lower tuple sorts first and is kept.
        # Kick/snare outrank hats even off the "strong" 1/3 grid — otherwise a
        # four-on-the-floor / every-beat K|S pulse loses beats 2/4 to on-1 hats.
        cls = _event_class_priority(t, classified_hits, tolerance)
        is_core = 0 if cls <= 1 else 1
        on_grid = 0 if _is_near_beat_grid(t, beats, bpm, preset) else 1
        on_strong = 0 if _is_on_strong_beat(t, beats, bpm, preset) else 1
        return (is_core, on_grid if cls <= 1 else 1, cls, on_strong, float(t))

    ranked = sorted(measure_events, key=_rank)
    return sorted(ranked[:cap])


def _strong_beat_times(beats: Optional[np.ndarray]) -> np.ndarray:
    """Beats 1 and 3 in 4/4 (every other beat marker)."""
    if beats is None or len(beats) == 0:
        return np.asarray([], dtype=float)
    return np.asarray(beats[::2], dtype=float)


def _is_near_beat_grid(
    t: float,
    beats: Optional[np.ndarray],
    bpm: float,
    preset: Optional[Dict] = None,
) -> bool:
    """True if t sits on any beat marker (1/2/3/4), not only strong 1/3."""
    preset = preset or {}
    if beats is None or len(beats) == 0:
        return False
    beat_interval = 60.0 / max(1.0, float(bpm))
    tol = beat_interval * float(preset.get("strong_beat_tolerance_beats", 0.18) or 0.18)
    return bool(np.min(np.abs(np.asarray(beats, dtype=float) - float(t))) <= tol)


def _is_on_strong_beat(
    t: float,
    beats: Optional[np.ndarray],
    bpm: float,
    preset: Optional[Dict] = None,
) -> bool:
    preset = preset or {}
    if not bool(preset.get("preserve_strong_beats", True)):
        return False
    strong = _strong_beat_times(beats)
    if strong.size == 0:
        return False
    beat_interval = 60.0 / max(1.0, float(bpm))
    tol = beat_interval * float(preset.get("strong_beat_tolerance_beats", 0.18) or 0.18)
    return bool(np.min(np.abs(strong - float(t))) <= tol)


def _keep_beat_grid_pulse_events(
    measure_events: List[float],
    beats: Optional[np.ndarray],
    bpm: float,
    preset: Dict,
    classified_hits: Optional[List[Dict]] = None,
) -> List[float]:
    """In sparse blocks keep kick/snare/tom on any beat-grid step; drop texture hats/cymbal."""
    if not measure_events:
        return []
    if not bool(preset.get("preserve_strong_beats", True)):
        return []
    drum_tol = float(preset.get("drum_class_tolerance", 0.06) or 0.06)
    pulse_classes = {"kick", "snare", "tom"}
    survivors: List[float] = []
    for t in measure_events:
        if not _is_near_beat_grid(t, beats, bpm, preset):
            continue
        if not classified_hits:
            survivors.append(t)
            continue
        drum = str(resolve_drum_at_time(float(t), classified_hits, tolerance=drum_tol) or "").lower()
        if drum in pulse_classes:
            survivors.append(t)
    return survivors


def _mix_rms_median(mix_rms: Dict[int, float]) -> float:
    vals = [float(v) for v in mix_rms.values() if v > 0]
    if not vals:
        return 0.0
    return float(np.median(vals))


def _classify_measures_for_section_pass(
    event_counts: Dict[int, int],
    ks_counts: Dict[int, int],
    preset: Dict,
    stem_rms: Optional[Dict[int, float]] = None,
    max_notes_per_measure: int = 0,
    dropout_measures: Optional[set[int]] = None,
    drum_rel: Optional[Dict[int, float]] = None,
    mix_rel: Optional[Dict[int, float]] = None,
    mix_rms: Optional[Dict[int, float]] = None,
) -> Dict[int, str]:
    if not event_counts:
        return {}

    quiet_max = int(preset.get("section_core_quiet_max", 1) or 1)
    block_run = int(preset.get("section_sparse_block_measures", 2) or 2)
    trigger_notes = int(preset.get("section_sparse_block_trigger_notes", 3) or 3)
    orphan_strip = bool(preset.get("section_ks_orphan_strip", True))
    runaway_ratio = float(preset.get("section_runaway_ratio", 2.0) or 2.0)
    runaway_min = int(preset.get("section_runaway_min_events", 10) or 10)
    radius = int(preset.get("section_runaway_neighbor_radius", 2) or 2)
    stem_enabled = bool(preset.get("section_stem_energy_enabled", False))
    chart_stem_strip = bool(preset.get("section_chart_stem_strip", False))
    weak_max = int(preset.get("section_ks_weak_max", 1) or 1)
    runaway_over_cap = bool(preset.get("section_runaway_over_cap", False))
    over_cap_mult = float(preset.get("section_runaway_over_cap_mult", 1.0) or 1.0)
    stem_rms = stem_rms or {}
    drum_rel = drum_rel or {}
    mix_rel = mix_rel or {}
    stem_median = _stem_rms_median(stem_rms) if stem_enabled and stem_rms else 0.0
    dropout_measures = dropout_measures or set()
    mix_loud_min = float(preset.get("section_mix_loud_rel_min", 0.85) or 0.85)
    drum_quiet_max = float(preset.get("section_drum_quiet_rel_max", 0.55) or 0.55)
    dual_energy = bool(preset.get("section_dual_energy_enabled", False))
    mix_quiet_gate = bool(preset.get("section_mix_quiet_gate_enabled", False))
    mix_quiet_rel_max = float(preset.get("section_mix_quiet_rel_max", 0.48) or 0.48)
    mix_quiet_trigger = int(preset.get("section_mix_quiet_trigger_notes", 1) or 1)
    phantom_orphan = bool(preset.get("section_phantom_orphan_enabled", False))
    phantom_mix_rel_max = float(preset.get("section_phantom_mix_rel_max", 0.62) or 0.62)
    phantom_abs_ratio = float(preset.get("section_phantom_mix_absolute_ratio", 0.40) or 0.40)
    phantom_min_notes = int(preset.get("section_phantom_min_notes", 1) or 1)
    mix_rms = mix_rms or {}
    mix_median = _mix_rms_median(mix_rms) if mix_rms else 0.0

    max_measure = max(max(event_counts.keys(), default=0), max(ks_counts.keys(), default=0))
    quiet_runs = _quiet_core_measure_runs(ks_counts, max_measure, quiet_max, block_run)

    classes: Dict[int, str] = {}
    for measure_idx in range(0, max_measure + 1):
        count = int(event_counts.get(measure_idx, 0))
        ks = int(ks_counts.get(measure_idx, 0))
        if count <= 0:
            continue
        split_section = dual_energy and is_loud_mix_quiet_drum(
            float(drum_rel.get(measure_idx, 0.0) or 0.0),
            float(mix_rel.get(measure_idx, 0.0) or 0.0),
            mix_loud_min=mix_loud_min,
            drum_quiet_max=drum_quiet_max,
        )
        if (
            mix_quiet_gate
            and dual_energy
            and not split_section
            and is_quiet_mix_breakdown(
                float(drum_rel.get(measure_idx, 0.0) or 0.0),
                float(mix_rel.get(measure_idx, 0.0) or 0.0),
                mix_quiet_max=mix_quiet_rel_max,
                mix_loud_min=mix_loud_min,
                drum_quiet_max=drum_quiet_max,
            )
            and count >= mix_quiet_trigger
        ):
            classes[measure_idx] = "sparse_block"
            continue
        stem_quiet = (
            stem_enabled
            and stem_rms
            and _is_stem_quiet_measure(stem_rms, measure_idx, stem_median, preset)
        )
        if (
            phantom_orphan
            and dual_energy
            and not split_section
            and is_phantom_orphan_measure(
                ks=ks,
                note_count=count,
                drum_rel=float(drum_rel.get(measure_idx, 0.0) or 0.0),
                mix_rel=float(mix_rel.get(measure_idx, 0.0) or 0.0),
                mix_rms_val=float(mix_rms.get(measure_idx, 0.0) or 0.0),
                mix_median=mix_median,
                stem_quiet=bool(stem_quiet),
                mix_quiet_rel_max=mix_quiet_rel_max,
                phantom_mix_rel_max=phantom_mix_rel_max,
                mix_absolute_ratio=phantom_abs_ratio,
                mix_loud_min=mix_loud_min,
                drum_quiet_max=drum_quiet_max,
                phantom_min_notes=phantom_min_notes,
            )
        ):
            classes[measure_idx] = "sparse_block"
            continue
        if measure_idx in dropout_measures and not split_section:
            classes[measure_idx] = "sparse_block"
            continue
        if stem_quiet and chart_stem_strip and count >= trigger_notes and not split_section:
            classes[measure_idx] = "sparse_block"
            continue
        if measure_idx in quiet_runs and count >= trigger_notes and ks <= quiet_max and not split_section:
            classes[measure_idx] = "sparse_block"
            continue
        if orphan_strip and ks <= 0 and count >= trigger_notes and not split_section:
            classes[measure_idx] = "sparse_block"
            continue
        if orphan_strip and stem_quiet and ks <= weak_max and count >= trigger_notes and not split_section:
            classes[measure_idx] = "sparse_block"
            continue
        cap_limit = 0
        if max_notes_per_measure > 0 and runaway_over_cap:
            cap_limit = max(2, int(round(max_notes_per_measure * over_cap_mult)))
        if cap_limit > 0 and count > cap_limit:
            classes[measure_idx] = "runaway"
            continue
        neighbor_med = _neighbor_median_count(event_counts, measure_idx, radius)
        effective_runaway_ratio = runaway_ratio
        effective_runaway_min = runaway_min
        if split_section:
            effective_runaway_ratio = max(1.15, runaway_ratio * 0.82)
            effective_runaway_min = max(runaway_min + 2, int(round(runaway_min * 1.15)))
        if count >= effective_runaway_min and neighbor_med > 0 and count >= neighbor_med * effective_runaway_ratio:
            classes[measure_idx] = "runaway"
        elif count >= max(5, neighbor_med + 4):
            classes[measure_idx] = "fill"
        elif count <= max(1, int(np.floor(neighbor_med * 0.45))):
            classes[measure_idx] = "sparse"
        else:
            classes[measure_idx] = "dense"
    return classes


def _strip_events_after_kick_snare_tail(
    events: List[float],
    kick_times: List[float],
    snare_times: List[float],
    preset: Dict,
) -> tuple[List[float], int]:
    grace = float(preset.get("section_eof_tail_grace_sec", 0.0) or 0.0)
    if grace <= 0 or not events:
        return events, 0
    last_ks = _last_kick_snare_time(kick_times, snare_times)
    if last_ks <= 0:
        return events, 0
    cutoff = last_ks + grace
    kept = [float(t) for t in events if float(t) <= cutoff]
    return kept, max(0, len(events) - len(kept))


def _apply_section_context_pass(
    events: List[float],
    kick_times: List[float],
    snare_times: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    classified_hits: Optional[List[Dict]] = None,
    verbose: bool = False,
    stem_audio_path: Optional[str] = None,
    mix_audio_path: Optional[str] = None,
    max_notes_per_measure: int = 0,
) -> List[float]:
    if not events or not bool(preset.get("section_pass_enabled", False)):
        return events
    if beats is None or len(beats) < 4:
        return events

    event_buckets = _times_per_measure(events, beats, bpm)
    event_counts = {idx: len(vals) for idx, vals in event_buckets.items()}
    ks_counts = _kick_snare_counts_per_measure(kick_times, snare_times, beats, bpm)
    max_measure = max(max(event_counts.keys(), default=0), max(ks_counts.keys(), default=0))
    stem_rms, mix_rms, drum_rel, mix_rel = _section_energy_contours(
        stem_audio_path,
        mix_audio_path,
        beats,
        bpm,
        max_measure,
        preset,
    )
    stem_median = _stem_rms_median(stem_rms) if stem_rms else 0.0
    dropout_measures = _dropout_measures_after_last_dense(
        event_counts,
        ks_counts,
        stem_rms,
        stem_median,
        max_measure,
        preset,
        drum_rel=drum_rel,
        mix_rel=mix_rel,
    )
    classes = _classify_measures_for_section_pass(
        event_counts,
        ks_counts,
        preset,
        stem_rms=stem_rms,
        max_notes_per_measure=max_notes_per_measure,
        dropout_measures=dropout_measures,
        drum_rel=drum_rel,
        mix_rel=mix_rel,
        mix_rms=mix_rms,
    )
    if not classes and not float(preset.get("section_eof_tail_grace_sec", 0.0) or 0.0):
        return events

    cap_mult = float(preset.get("section_runaway_cap_mult", 1.35) or 1.35)
    drum_tol = float(preset.get("drum_class_tolerance", 0.06) or 0.06)
    radius = int(preset.get("section_runaway_neighbor_radius", 2) or 2)
    kept: List[float] = []
    stripped_sparse = 0
    capped_runaway = 0
    removed_detail: List[Tuple[int, str, float, str]] = []  # (measure_idx, kind, time, drum)

    def _removed_tag(t: float) -> str:
        drum = resolve_drum_at_time(float(t), classified_hits, tolerance=drum_tol) if classified_hits else None
        return str(drum or "?")

    for measure_idx in sorted(event_buckets.keys()):
        measure_events = list(event_buckets[measure_idx])
        kind = classes.get(measure_idx, "dense")
        if kind == "sparse_block":
            # Already-sparse measures: pulse filter can erase the only detector hit (P2_drop).
            keep_all_max = int(preset.get("section_sparse_keep_all_max", 2) or 2)
            if keep_all_max > 0 and len(measure_events) <= keep_all_max:
                kept.extend(measure_events)
                continue
            # Keep pulse classes on every beat step — not only downbeats 1/3.
            survivors = _keep_beat_grid_pulse_events(
                measure_events, beats, bpm, preset, classified_hits=classified_hits
            )
            removed_now = [t for t in measure_events if t not in survivors]
            stripped_sparse += len(removed_now)
            for t in removed_now:
                removed_detail.append((measure_idx, "sparse_block", t, _removed_tag(t)))
            kept.extend(survivors)
            continue
        if kind == "runaway":
            neighbor_med = _neighbor_median_count(event_counts, measure_idx, radius)
            cap = max(2, int(round(neighbor_med * cap_mult)) + 1)
            before_events = list(measure_events)
            measure_events = _cap_measure_events_class_aware(
                measure_events,
                cap,
                classified_hits,
                drum_tol,
                beats=beats,
                bpm=bpm,
                preset=preset,
            )
            removed_now = [t for t in before_events if t not in measure_events]
            capped_runaway += len(removed_now)
            for t in removed_now:
                removed_detail.append((measure_idx, "runaway", t, _removed_tag(t)))
        kept.extend(measure_events)

    kept, stripped_tail = _strip_events_after_kick_snare_tail(kept, kick_times, snare_times, preset)
    stripped_sparse += stripped_tail

    log_always = bool(preset.get("section_log_always", True))
    if log_always or (verbose and (stripped_sparse or capped_runaway)):
        summary: Dict[str, int] = {}
        for kind in classes.values():
            summary[kind] = summary.get(kind, 0) + 1
        print(
            f"[DrumGen][section] classes={summary} stripped_sparse_block={stripped_sparse} "
            f"runaway_trimmed={capped_runaway} {len(events)}->{len(kept)}"
        )
        for measure_idx, kind in sorted(classes.items()):
            if kind in ("sparse_block", "runaway"):
                energy_tag = ""
                if stem_rms:
                    energy_tag = f" rms={stem_rms.get(measure_idx, 0.0):.4f}"
                if drum_rel or mix_rel:
                    energy_tag += (
                        f" d_rel={drum_rel.get(measure_idx, 0.0):.2f}"
                        f" x_rel={mix_rel.get(measure_idx, 0.0):.2f}"
                    )
                print(
                    f"[DrumGen][section]   m{measure_idx + 1} {kind} "
                    f"events={event_counts.get(measure_idx, 0)} ks={ks_counts.get(measure_idx, 0)}{energy_tag}"
                )
        if removed_detail:
            core_removed = [d for d in removed_detail if d[3] in ("kick", "snare")]
            tag = " (incl. kick/snare!)" if core_removed else ""
            print(
                f"[DrumGen][section]   removed_detail{tag}: "
                + ", ".join(
                    f"m{mi + 1}/{kind}/{drum}@{t:.3f}" for mi, kind, t, drum in removed_detail
                )
            )
        if stripped_tail:
            print(
                f"[DrumGen][section]   tail_strip after last kick/snare "
                f"+{float(preset.get('section_eof_tail_grace_sec', 0)):.0f}s removed={stripped_tail}"
            )
    return sorted(kept)


def _class_position_buckets(
    events: List[float],
    classified_hits: Optional[List[Dict]],
    beats: np.ndarray,
    bpm: float,
    drum_tol: float,
    rhythm_classes: Optional[Sequence[str]],
) -> Dict[int, Dict[str, set]]:
    if not events or not classified_hits:
        return {}
    allowed = {str(c).strip().lower() for c in (rhythm_classes or []) if str(c).strip()}
    if not allowed:
        allowed = {"kick", "snare", "hat"}
    first_measure_start, measure_duration = _measure_timing(beats, bpm)
    beat_interval = measure_duration / 4.0
    buckets: Dict[int, Dict[str, set]] = {}
    for event in sorted(events):
        drum = resolve_drum_at_time(float(event), classified_hits, tolerance=drum_tol)
        if not drum:
            continue
        drum_key = str(drum).strip().lower()
        if drum_key not in allowed:
            continue
        measure_idx = int(np.floor((float(event) - first_measure_start) / measure_duration))
        if measure_idx < 0:
            continue
        rel_beats = (float(event) - (first_measure_start + measure_idx * measure_duration)) / beat_interval
        if rel_beats < -0.1 or rel_beats >= 4.1:
            continue
        pos = round(max(0.0, min(3.75, rel_beats)) * 4.0) / 4.0
        buckets.setdefault(measure_idx, {}).setdefault(drum_key, set()).add(pos)
    return buckets


def _classify_measures_for_consistency_pass(
    event_counts: Dict[int, int],
    ks_counts: Dict[int, int],
    section_classes: Dict[int, str],
    preset: Dict,
) -> Dict[int, str]:
    if not event_counts:
        return {}
    radius = int(preset.get("rhythm_consistency_neighbor_radius", 2) or 2)
    outlier_ratio = float(preset.get("rhythm_consistency_outlier_ratio", 0.45) or 0.45)
    skip_fill = bool(preset.get("rhythm_consistency_skip_fill", True))
    max_measure = max(max(event_counts.keys(), default=0), max(section_classes.keys(), default=0))
    classes: Dict[int, str] = {}

    for measure_idx in range(0, max_measure + 1):
        section_kind = section_classes.get(measure_idx, "")
        if section_kind == "sparse_block":
            classes[measure_idx] = "sparse_block"
            continue
        if section_kind == "fill" and skip_fill:
            classes[measure_idx] = "fill"
            continue

        count = int(event_counts.get(measure_idx, 0))
        neighbor_med = _neighbor_median_count(event_counts, measure_idx, radius)
        if neighbor_med >= 4.0:
            if count == 0:
                classes[measure_idx] = "outlier_sparse"
                continue
            if count <= max(1, int(np.floor(neighbor_med * outlier_ratio))):
                classes[measure_idx] = "outlier_sparse"
                continue

        if section_kind:
            classes[measure_idx] = section_kind
        elif count >= 4:
            classes[measure_idx] = "dense"
        elif count <= 1:
            classes[measure_idx] = "sparse"
        else:
            classes[measure_idx] = "dense"
    return classes


def _apply_rhythm_consistency_pass(
    events: List[float],
    classified_hits: Optional[List[Dict]],
    kick_times: List[float],
    snare_times: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    verbose: bool = False,
) -> List[float]:
    if not bool(preset.get("rhythm_consistency", False)) or not events:
        return events
    if not classified_hits or beats is None or len(beats) < 8:
        return events

    radius = int(preset.get("rhythm_consistency_radius", 4) or 4)
    min_support = int(preset.get("rhythm_consistency_min_support", 3) or 3)
    max_add_per_measure = int(preset.get("rhythm_consistency_max_add_per_measure", 2) or 2)
    max_add_per_class = int(preset.get("rhythm_consistency_max_add_per_class", 1) or 1)
    seek_window = float(preset.get("rhythm_consistency_seek_window", 0.09) or 0.09)
    max_notes_per_measure = int(preset.get("max_notes_per_measure", 0) or 0)
    drum_tol = float(preset.get("drum_class_tolerance", 0.06) or 0.06)
    rhythm_classes = preset.get("rhythm_hit_classes") or ["kick", "snare", "hat"]
    allowed_classes = [str(c).strip().lower() for c in rhythm_classes if str(c).strip()]
    if not allowed_classes:
        allowed_classes = ["kick", "snare", "hat"]
    if radius <= 0 or min_support <= 0 or max_add_per_measure <= 0:
        return events

    event_buckets = _times_per_measure(events, beats, bpm)
    event_counts = {idx: len(vals) for idx, vals in event_buckets.items()}
    ks_counts = _kick_snare_counts_per_measure(kick_times, snare_times, beats, bpm)
    section_classes = _classify_measures_for_section_pass(event_counts, ks_counts, preset)
    measure_classes = _classify_measures_for_consistency_pass(
        event_counts,
        ks_counts,
        section_classes,
        preset,
    )
    if not measure_classes:
        return events

    class_buckets = _class_position_buckets(
        events,
        classified_hits,
        beats,
        bpm,
        drum_tol,
        allowed_classes,
    )
    first_measure_start, measure_duration = _measure_timing(beats, bpm)
    beat_interval = measure_duration / 4.0
    cluster_window = float(preset.get("hit_cluster_window", 0.0) or 0.0)
    existing_tol = max(min(0.04, beat_interval * 0.12), min(cluster_window, beat_interval * 0.28))

    candidate_pool = [
        h
        for h in classified_hits
        if str(h.get("drum", "")).strip().lower() in allowed_classes
    ]
    completed = sorted(set(float(t) for t in events))
    added_log: List[str] = []
    total_added = 0
    recoverable_kinds = {"dense", "outlier_sparse"}

    for measure_idx in sorted(set(measure_classes.keys()) | set(class_buckets.keys())):
        kind = measure_classes.get(measure_idx, "dense")
        if kind not in recoverable_kinds:
            continue

        measure_start = first_measure_start + measure_idx * measure_duration
        current_count = len(event_buckets.get(measure_idx, []))
        if max_notes_per_measure > 0 and current_count >= max_notes_per_measure:
            continue

        measure_added = 0
        room = max_add_per_measure
        if max_notes_per_measure > 0:
            room = min(room, max(0, max_notes_per_measure - current_count))
        if room <= 0:
            continue

        for drum_class in allowed_classes:
            if measure_added >= room:
                break
            current_positions = set(class_buckets.get(measure_idx, {}).get(drum_class, set()))
            neighbor_votes: Dict[float, int] = {}
            for other_idx in range(measure_idx - radius, measure_idx + radius + 1):
                if other_idx == measure_idx:
                    continue
                other_kind = measure_classes.get(other_idx, "dense")
                if other_kind in ("sparse_block", "fill"):
                    continue
                for pos in class_buckets.get(other_idx, {}).get(drum_class, set()):
                    neighbor_votes[pos] = neighbor_votes.get(pos, 0) + 1

            if not neighbor_votes:
                continue

            candidates = [
                (pos, count)
                for pos, count in neighbor_votes.items()
                if count >= min_support and pos not in current_positions
            ]
            candidates.sort(key=lambda item: (-item[1], item[0]))
            class_added = 0

            for pos, votes in candidates:
                if measure_added >= room or class_added >= max_add_per_class:
                    break
                target_time = measure_start + pos * beat_interval
                if target_time < 0:
                    continue
                local_hits = [
                    h
                    for h in candidate_pool
                    if str(h.get("drum", "")).strip().lower() == drum_class
                    and abs(float(h["time"]) - target_time) <= seek_window
                ]
                if not local_hits:
                    continue
                selected_time = float(
                    min(local_hits, key=lambda h: abs(float(h["time"]) - target_time))["time"]
                )
                if _has_near(selected_time, completed, existing_tol):
                    continue
                completed.append(selected_time)
                class_buckets.setdefault(measure_idx, {}).setdefault(drum_class, set()).add(pos)
                event_buckets.setdefault(measure_idx, []).append(selected_time)
                measure_added += 1
                class_added += 1
                total_added += 1
                added_log.append(
                    f"m{measure_idx + 1} {kind} +{drum_class}@{pos:.2f} votes={votes}"
                )

    log_always = bool(preset.get("rhythm_consistency_log_always", True))
    if total_added and (log_always or verbose):
        summary: Dict[str, int] = {}
        for kind in measure_classes.values():
            summary[kind] = summary.get(kind, 0) + 1
        print(
            f"[DrumGen][consistency] classes={summary} added={total_added} "
            f"{len(events)}->{len(completed)}"
        )
        for line in added_log[:24]:
            print(f"[DrumGen][consistency]   {line}")
        if len(added_log) > 24:
            print(f"[DrumGen][consistency]   ... +{len(added_log) - 24} more")
    elif log_always and verbose:
        summary: Dict[str, int] = {}
        for kind in measure_classes.values():
            summary[kind] = summary.get(kind, 0) + 1
        print(f"[DrumGen][consistency] classes={summary} added=0")

    return sorted(completed)


# Keys owned by the Goal axis (_GOAL_POLICY in generation_intents.py) — genre
# preferences must never set these directly, no matter what a genre profile's
# "preferences" block contains. Goal decides *whether a feature is allowed at
# all*; genre preferences only decide *how dense/strict it is within that*.
_GOAL_PROTECTED_KEYS = frozenset({
    "groove_completion", "loop_reinforce", "loop4_reinforce", "fill_recover", "fill",
    "dominant_onsets_policy", "rhythm_hit_classes",
    "arcade_policy", "arcade_tension_map", "arcade_phantom_gate", "arcade_backbeat",
    "arcade_texture_downsample", "ergonomic_router",
})


def _apply_genre_preferences(
    preset: Dict,
    preferences: Dict,
    mode: str,
    verbose: bool = False,
) -> None:
    """Mode-agnostic genre preference layer.

    Interprets a small set of high-level semantic knobs (relative multipliers,
    not absolute pipeline settings) on top of whatever the goal×difficulty axis
    already put into `preset` — so it behaves consistently across basic/
    standard/dense/custom instead of only affecting the `basic` mode like the
    old `_apply_genre_generation_settings()` did. See docs for the vocabulary.
    """
    if not isinstance(preferences, dict) or not preferences:
        return

    leaked = _GOAL_PROTECTED_KEYS.intersection(preferences.keys())
    if leaked:
        if verbose:
            print(f"[DrumGen][genre_prefs] игнорирую goal-защищённые ключи: {sorted(leaked)}")
        preferences = {k: v for k, v in preferences.items() if k not in leaked}

    def _as_float(value, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    if "prefer_fills" in preferences:
        preset["fill_zone_enabled"] = bool(preferences["prefer_fills"])

    if "fill_intensity" in preferences and bool(preset.get("fill_zone_enabled", True)):
        intensity = max(0.0, min(1.0, _as_float(preferences["fill_intensity"], 0.5)))
        # 0.5 == neutral (matches base preset defaults untouched).
        scale = 0.5 + intensity
        for key in ("fill_zone_min_events", "fill_zone_metal_min_events", "fill_zone_spike_min_events"):
            base = FILL_ZONE_DEFAULTS.get(key)
            if base is not None:
                preset[key] = max(4, min(40, round(base / scale)))
        for key in (
            "fill_zone_cluster_mult", "fill_zone_flam_mult",
            "fill_zone_metal_cluster_mult", "fill_zone_metal_flam_mult",
            "fill_zone_halo_cluster_mult", "fill_zone_halo_flam_mult",
            "fill_zone_metal_halo_cluster_mult", "fill_zone_metal_halo_flam_mult",
        ):
            base = FILL_ZONE_DEFAULTS.get(key)
            if base is not None:
                preset[key] = max(0.15, min(0.95, base * scale))
        for key in ("fill_zone_spike_ratio", "fill_zone_metal_spike_ratio"):
            base = FILL_ZONE_DEFAULTS.get(key)
            if base is not None:
                preset[key] = max(1.05, min(2.2, base / scale))
        # Groove-completion knobs are goal-gated (only tuned once the goal
        # policy already turned groove_completion on) but "how eagerly gaps
        # get filled in" is genre taste, so fill_intensity governs it too.
        if "groove_completion_min_support" in preset:
            preset["groove_completion_min_support"] = max(
                1, round(_as_float(preset["groove_completion_min_support"], 3) / scale)
            )
        if "groove_completion_max_add_per_measure" in preset:
            preset["groove_completion_max_add_per_measure"] = max(
                0, round(_as_float(preset["groove_completion_max_add_per_measure"], 1) * scale)
            )

    if "density_cap_mult" in preferences:
        mult = max(0.3, min(3.0, _as_float(preferences["density_cap_mult"], 1.0)))
        for key in ("max_hits_per_second", "max_notes_per_measure"):
            if key in preset and preset[key]:
                preset[key] = max(1, round(_as_float(preset[key], 0.0) * mult))
        if "hit_cluster_window" in preset:
            preset["hit_cluster_window"] = max(0.02, min(0.30, _as_float(preset["hit_cluster_window"], 0.11) / mult))
        if "hit_cluster_beat_fraction" in preset:
            preset["hit_cluster_beat_fraction"] = max(
                0.1, min(0.6, _as_float(preset["hit_cluster_beat_fraction"], 0.30) * mult)
            )

    if "timing_strictness" in preferences:
        strictness = max(0.1, min(2.0, _as_float(preferences["timing_strictness"], 1.0)))
        if "grid_snap_strength" in preset:
            preset["grid_snap_strength"] = max(
                0, min(100, round(_as_float(preset["grid_snap_strength"], 40) * strictness))
            )
        if abs(strictness - 1.0) > 0.05:
            preset["accent_strong_beats"] = strictness > 1.0
        if "section_timing_correction_min_events" in preset:
            preset["section_timing_correction_min_events"] = max(3, round(
                _as_float(preset["section_timing_correction_min_events"], 8) / strictness
            ))

    if "arrangement_sensitivity" in preferences:
        # < 1 == loose (keep more, trim less aggressively); > 1 == strict.
        sens = max(0.3, min(2.0, _as_float(preferences["arrangement_sensitivity"], 1.0)))
        loosen = 2.0 - sens
        for key in ("section_runaway_cap_mult", "section_runaway_over_cap_mult", "section_runaway_ratio"):
            if key in preset:
                preset[key] = max(1.0, _as_float(preset[key], 1.35) * loosen)
        if "section_runaway_min_events" in preset:
            preset["section_runaway_min_events"] = max(4, round(
                _as_float(preset["section_runaway_min_events"], 10) / sens
            ))
        for key in (
            "section_mix_quiet_rel_max", "section_phantom_mix_rel_max",
            "section_phantom_mix_absolute_ratio", "section_stem_quiet_ratio",
        ):
            if key in preset:
                preset[key] = max(0.05, min(0.95, _as_float(preset[key], 0.5) * sens))
        for key in (
            "section_sparse_block_trigger_notes", "section_core_quiet_max", "section_ks_weak_max",
            "section_mix_quiet_trigger_notes", "section_phantom_min_notes",
        ):
            if key in preset:
                preset[key] = max(0, round(_as_float(preset[key], 1) / sens))
        if sens <= 0.5:
            for flag in (
                "section_ks_orphan_strip", "section_stem_energy_enabled", "section_chart_stem_strip",
                "section_runaway_over_cap", "section_dropout_enabled",
            ):
                if flag in preset:
                    preset[flag] = False

    if "texture_strictness" in preferences:
        preset["salience_texture_strictness"] = _as_float(preferences["texture_strictness"], 1.0)
    if "include_roles" in preferences and isinstance(preferences["include_roles"], list):
        preset["salience_include_roles"] = list(preferences["include_roles"])

    if "micro_timing_looseness" in preferences:
        preset["flam_merge_sec"] = max(0.0, _as_float(preferences["micro_timing_looseness"], 0.11))

    if verbose:
        print(f"[DrumGen][genre_prefs] mode={mode} preferences={preferences}")


def _is_metal_genre(genre_label: str) -> bool:
    label = (genre_label or "").strip().lower()
    metal_tokens = ("metal", "hardcore", "thrash", "death", "grind", "dragon")
    return any(token in label for token in metal_tokens)


def _fill_zone_spike_allowed(preset: Dict, genre_label: str, bpm: float) -> bool:
    if not bool(preset.get("fill_zone_enabled", False)):
        return False
    if _is_metal_genre(genre_label) and not bool(preset.get("fill_zone_metal_enabled", True)):
        return False
    max_bpm = float(preset.get("fill_zone_spike_max_bpm", 240) or 240)
    return float(bpm) < max_bpm


def _prune_consecutive_fill_candidates(
    candidates: set[int],
    buckets: Dict[int, List[float]],
    max_consecutive: int,
) -> set[int]:
    if max_consecutive <= 0 or not candidates:
        return candidates
    sorted_idxs = sorted(candidates)
    result: set[int] = set()
    i = 0
    while i < len(sorted_idxs):
        run = [sorted_idxs[i]]
        j = i + 1
        while j < len(sorted_idxs) and sorted_idxs[j] == run[-1] + 1:
            run.append(sorted_idxs[j])
            j += 1
        if len(run) <= max_consecutive:
            result.update(run)
        else:
            ranked = sorted(run, key=lambda idx: len(buckets.get(idx, [])), reverse=True)
            result.update(ranked[:max_consecutive])
        i = j
    return result


def _fill_zone_halo_measures(
    core: set[int],
    buckets: Dict[int, List[float]],
    preset: Dict,
    min_abs: int,
) -> set[int]:
    halo_radius = int(preset.get("fill_zone_spike_halo_measures", 1) or 1)
    if halo_radius <= 0 or not core:
        return set()
    halo: set[int] = set()
    slack = int(preset.get("fill_zone_spike_halo_min_slack", 2) or 2)
    min_halo = max(6, min_abs - slack)
    for idx in core:
        for offset in range(-halo_radius, halo_radius + 1):
            if offset == 0:
                continue
            neighbor_idx = idx + offset
            if neighbor_idx in core or neighbor_idx not in buckets:
                continue
            if len(buckets[neighbor_idx]) >= min_halo:
                halo.add(neighbor_idx)
    return halo


def _fill_zone_measure_tiers(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    genre_label: str = "",
) -> Tuple[set[int], set[int]]:
    if not events:
        return set(), set()
    buckets = _times_per_measure(events, beats, bpm)
    if not buckets:
        return set(), set()

    if not _fill_zone_spike_allowed(preset, genre_label, bpm):
        return set(), set()

    metal = _is_metal_genre(genre_label)
    min_events = int(
        preset.get("fill_zone_metal_min_events" if metal else "fill_zone_min_events", 12) or 12
    )
    min_abs = int(preset.get("fill_zone_spike_min_events", 10) or 10)
    ratio = float(
        preset.get("fill_zone_metal_spike_ratio" if metal else "fill_zone_spike_ratio", 1.42) or 1.42
    )
    radius = int(preset.get("fill_zone_spike_neighbor_radius", 2) or 2)
    min_delta = int(preset.get("fill_zone_spike_min_delta", 3) or 3)
    require_peak = bool(preset.get("fill_zone_spike_require_peak", True))
    max_consecutive = int(preset.get("fill_zone_spike_max_consecutive", 2) or 2)

    candidates: set[int] = set()
    for idx, vals in buckets.items():
        count = len(vals)
        if count < min_events or count < min_abs:
            continue
        neighbor_counts: List[int] = []
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            neighbor_idx = idx + offset
            if neighbor_idx in buckets:
                neighbor_counts.append(len(buckets[neighbor_idx]))
        if not neighbor_counts:
            continue
        baseline = float(np.median(neighbor_counts))
        if baseline < 4.0:
            baseline = max(baseline, 6.0)
        if count < baseline * ratio or (count - baseline) < min_delta:
            continue
        if require_peak:
            left = len(buckets.get(idx - 1, []))
            right = len(buckets.get(idx + 1, []))
            if not (count >= left and count >= right and (count > left or count > right)):
                continue
        candidates.add(idx)

    core = _prune_consecutive_fill_candidates(candidates, buckets, max_consecutive)
    halo = _fill_zone_halo_measures(core, buckets, preset, min_abs)
    halo -= core
    return core, halo


def _fill_zone_cluster_mult_at(
    preset: Dict,
    genre_label: str,
    measure_idx: int,
    fill_core: set[int],
    fill_halo: Optional[set[int]] = None,
) -> float:
    fill_halo = fill_halo or set()
    metal = _is_metal_genre(genre_label)
    if measure_idx in fill_core:
        if metal:
            return float(preset.get("fill_zone_metal_cluster_mult", 0.58) or 0.58)
        return float(preset.get("fill_zone_cluster_mult", 0.42) or 0.42)
    if measure_idx in fill_halo:
        if metal:
            return float(preset.get("fill_zone_metal_halo_cluster_mult", 0.68) or 0.68)
        return float(preset.get("fill_zone_halo_cluster_mult", 0.72) or 0.72)
    return 1.0


def _fill_zone_flam_mult_at(
    preset: Dict,
    genre_label: str,
    measure_idx: int,
    fill_core: set[int],
    fill_halo: Optional[set[int]] = None,
) -> float:
    fill_halo = fill_halo or set()
    metal = _is_metal_genre(genre_label)
    if measure_idx in fill_core:
        if metal:
            return float(preset.get("fill_zone_metal_flam_mult", 0.50) or 0.50)
        return float(preset.get("fill_zone_flam_mult", 0.35) or 0.35)
    if measure_idx in fill_halo:
        if metal:
            return float(preset.get("fill_zone_metal_halo_flam_mult", 0.58) or 0.58)
        return float(preset.get("fill_zone_halo_flam_mult", 0.62) or 0.62)
    return 1.0


def _measure_index_for_time(event_time: float, beats: np.ndarray, bpm: float) -> int:
    first_measure_start, measure_duration = _measure_timing(beats, bpm)
    return int(np.floor((float(event_time) - first_measure_start) / measure_duration))


def _sixteenth_merge_cap_sec(bpm: float, preset: Dict) -> float:
    """Upper bound so true 16th-note streams do not chain-merge into one hit.

    At 200 BPM a 16th is ~75 ms. hit_cluster_beat_fraction=0.30 → ~90 ms window, which
    links every consecutive 16th into one giant cluster (blast metal P2_drop). Cap just
    under a 16th. Class-level (BPM), never per-track.
    """
    beat_interval = 60.0 / max(1.0, float(bpm) or 1.0)
    frac = float(preset.get("hit_cluster_sixteenth_cap", 0.85) or 0.85)
    frac = max(0.5, min(0.95, frac))
    return beat_interval * frac / 4.0


def _effective_cluster_window_sec(preset: Dict, bpm: float) -> float:
    cluster_window = float(preset.get("hit_cluster_window", 0.0) or 0.0)
    if cluster_window <= 0:
        return 0.0
    beat_interval = 60.0 / max(1.0, float(bpm) or 1.0)
    beat_fraction = float(preset.get("hit_cluster_beat_fraction", 0.16) or 0.16)
    max_musical_window = beat_interval * beat_fraction
    return min(cluster_window, max_musical_window, _sixteenth_merge_cap_sec(bpm, preset))


def _effective_flam_merge_sec(preset: Dict, bpm: float) -> float:
    flam_sec = float(preset.get("flam_merge_sec", 0.0) or 0.0)
    if flam_sec <= 0:
        return 0.0
    return min(flam_sec, _sixteenth_merge_cap_sec(bpm, preset))


def _cluster_hit_events(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    fill_measures: Optional[set[int]] = None,
    fill_halo_measures: Optional[set[int]] = None,
    core_sources: Optional[List[float]] = None,
    genre_label: str = "",
) -> List[float]:
    if not events:
        return events
    window = _effective_cluster_window_sec(preset, bpm)
    if window <= 0:
        return sorted(events)

    fill_core = fill_measures or set()
    fill_halo = fill_halo_measures or set()
    core_tol = float(preset.get("core_hit_tolerance", 0.10) or 0.10)
    core_sorted = sorted(core_sources or [])
    preserve_core = bool(preset.get("preserve_core_hits", False)) and bool(core_sorted)

    def _is_core_hit(t: float) -> bool:
        return preserve_core and _has_near(t, core_sorted, core_tol)

    def _in_fill_zone(m: int) -> bool:
        return m in fill_core or m in fill_halo

    def _window_at(t: float) -> float:
        if not fill_core and not fill_halo:
            return window
        m = _measure_index_for_time(t, beats, bpm)
        mult = _fill_zone_cluster_mult_at(preset, genre_label, m, fill_core, fill_halo)
        if mult >= 1.0:
            return window
        return max(0.012, window * mult)

    clusters: List[List[float]] = []
    current: List[float] = []
    for event in sorted(events):
        if not current:
            current = [event]
            continue
        if _is_core_hit(event) and not _in_fill_zone(_measure_index_for_time(event, beats, bpm)):
            clusters.append(current)
            current = [event]
            continue
        if _is_core_hit(current[-1]) and not _in_fill_zone(_measure_index_for_time(current[-1], beats, bpm)):
            clusters.append(current)
            current = [event]
            continue
        pair_window = min(_window_at(current[-1]), _window_at(event))
        if event - current[-1] <= pair_window:
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
            best = min(
                cluster,
                key=lambda t: (
                    0 if _is_on_strong_beat(t, beats, bpm, preset) else 1,
                    float(np.min(np.abs(beats - t))),
                ),
            )
            clustered.append(best)
        else:
            clustered.append(cluster[0])
    return sorted(clustered)


def _merge_flam_events(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    fill_measures: Optional[set[int]] = None,
    fill_halo_measures: Optional[set[int]] = None,
    core_sources: Optional[List[float]] = None,
    genre_label: str = "",
) -> List[float]:
    if not events:
        return events
    flam_sec = _effective_flam_merge_sec(preset, bpm)
    if flam_sec <= 0:
        return sorted(events)

    fill_core = fill_measures or set()
    fill_halo = fill_halo_measures or set()
    core_tol = float(preset.get("core_hit_tolerance", 0.10) or 0.10)
    core_sorted = sorted(core_sources or [])
    preserve_core = bool(preset.get("preserve_core_hits", False)) and bool(core_sorted)

    def _is_core_hit(t: float) -> bool:
        return preserve_core and _has_near(t, core_sorted, core_tol)

    def _in_fill_zone(m: int) -> bool:
        return m in fill_core or m in fill_halo

    def _flam_at(t: float) -> float:
        if not fill_core and not fill_halo:
            return flam_sec
        m = _measure_index_for_time(t, beats, bpm)
        mult = _fill_zone_flam_mult_at(preset, genre_label, m, fill_core, fill_halo)
        if mult >= 1.0:
            return flam_sec
        return max(0.01, flam_sec * mult)

    merged: List[float] = []
    cluster: List[float] = []
    for event in sorted(events):
        if not cluster:
            cluster = [event]
            continue
        if _is_core_hit(event) and not _in_fill_zone(_measure_index_for_time(event, beats, bpm)):
            if len(cluster) == 1:
                merged.append(cluster[0])
            elif beats is not None and len(beats) > 0:
                merged.append(min(cluster, key=lambda t: float(np.min(np.abs(beats - t)))))
            else:
                merged.append(cluster[0])
            cluster = [event]
            continue
        if _is_core_hit(cluster[-1]) and not _in_fill_zone(_measure_index_for_time(cluster[-1], beats, bpm)):
            merged.append(cluster[-1])
            cluster = [event]
            continue
        pair_flam = min(_flam_at(cluster[-1]), _flam_at(event))
        if event - cluster[-1] <= pair_flam:
            cluster.append(event)
        else:
            if len(cluster) == 1:
                merged.append(cluster[0])
            elif beats is not None and len(beats) > 0:
                merged.append(min(cluster, key=lambda t: float(np.min(np.abs(beats - t)))))
            else:
                merged.append(cluster[0])
            cluster = [event]
    if cluster:
        if len(cluster) == 1:
            merged.append(cluster[0])
        elif beats is not None and len(beats) > 0:
            merged.append(min(cluster, key=lambda t: float(np.min(np.abs(beats - t)))))
        else:
            merged.append(cluster[0])
    return sorted(merged)


def _apply_critic_intro_no_add(
    times: List[float],
    drum_section_start: float,
    kick_times: List[float],
    snare_times: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    verbose: bool = False,
    classified_hits: Optional[List[Dict]] = None,
    dominant_onsets: Optional[List[float]] = None,
) -> List[float]:
    if not times or not bool(preset.get("critic_intro_no_add", False)):
        return times
    beat_interval = 60.0 / max(1.0, bpm)
    grace_beats = float(preset.get("critic_intro_grace_beats", 1.0) or 1.0)
    intro_end = float(drum_section_start) + grace_beats * beat_interval
    tol = float(preset.get("drum_entry_merge_tol", 0.04) or 0.04)
    # Grid snap can move onsets by ~half an 8th; keep detector-aligned notes after snap.
    match_tol = max(tol, beat_interval * 0.15)
    lookback = max(
        beat_interval * grace_beats,
        float(preset.get("drum_entry_grace_sec", 0.65) or 0.65),
        float(preset.get("drum_entry_preamble_sec", 0.0) or 0.0),
    )
    core = sorted(float(t) for t in (kick_times + snare_times))
    adtof_times: List[float] = []
    if classified_hits:
        adtof_times = [float(h.get("time", 0.0)) for h in classified_hits if h.get("time") is not None]
    elif dominant_onsets:
        adtof_times = [float(t) for t in dominant_onsets]
    # Full detector pool: invented intro notes won't match; real hits will (incl. preamble).
    match_pool = sorted(set(core + adtof_times))
    preserve_first = bool(preset.get("critic_intro_preserve_first_onset", True))
    sparse_max = int(preset.get("critic_intro_sparse_max_notes", 4) or 4)
    intro_scope = [
        t for t in times
        if float(drum_section_start) - lookback <= t < intro_end
    ]
    intro_buckets = _times_per_measure(intro_scope, beats, bpm) if intro_scope else {}
    earliest_pool = None
    if preserve_first and match_pool:
        after_start = [c for c in match_pool if c >= float(drum_section_start) - lookback]
        if after_start:
            earliest_pool = min(after_start)
    kept: List[float] = []
    stripped = 0
    first_preserved = False
    for t in sorted(times):
        if t < intro_end:
            if t >= float(drum_section_start) - lookback:
                measure_idx = _measure_index_for_time(t, beats, bpm)
                if sparse_max > 0 and len(intro_buckets.get(measure_idx, [])) <= sparse_max:
                    kept.append(t)
                    continue
            if (
                preserve_first
                and not first_preserved
                and earliest_pool is not None
                and t >= float(drum_section_start) - lookback
                and abs(t - earliest_pool) <= match_tol
            ):
                kept.append(t)
                first_preserved = True
                continue
            matched = any(abs(t - c) <= match_tol for c in match_pool)
            if matched:
                kept.append(t)
            else:
                stripped += 1
        else:
            kept.append(t)
    if verbose and stripped:
        print(
            f"[DrumGen][critic] intro_no_add stripped={stripped} "
            f"window=[{drum_section_start:.3f},{intro_end:.3f})"
        )
    return kept


def _apply_playability_linter(
    notes: List[Dict],
    lanes: int,
    preset: Dict,
    verbose: bool = False,
) -> List[Dict]:
    if not notes or not bool(preset.get("critic_playability_lint", False)):
        return notes
    min_gap = float(preset.get("critic_lint_min_lane_gap_sec", 0.04) or 0.04)
    lane_buckets: Dict[int, List[Dict]] = {}
    for note in notes:
        lane_buckets.setdefault(int(note.get("lane", 0)), []).append(note)

    deduped: List[Dict] = []
    removed_gap = 0
    for lane_notes in lane_buckets.values():
        lane_notes.sort(key=lambda n: float(n.get("time", 0.0)))
        last_t = -999.0
        for note in lane_notes:
            t = float(note.get("time", 0.0))
            if t - last_t < min_gap:
                removed_gap += 1
                continue
            deduped.append(note)
            last_t = t

    deduped.sort(key=lambda n: float(n.get("time", 0.0)))
    bucket_tol = 0.05
    simultaneous_removed = 0
    i = 0
    capped: List[Dict] = []
    lane_cap = max(1, int(lanes))
    while i < len(deduped):
        j = i + 1
        while j < len(deduped) and float(deduped[j]["time"]) - float(deduped[i]["time"]) <= bucket_tol:
            j += 1
        group = deduped[i:j]
        if len(group) <= lane_cap:
            capped.extend(group)
        else:
            priority = {"kick": 0, "snare": 1, "hat": 2, "tom": 3, "cymbal": 4, "perc": 5}
            group.sort(
                key=lambda n: (
                    priority.get(str(n.get("drum", "")).lower(), 9),
                    float(n.get("time", 0.0)),
                )
            )
            capped.extend(group[:lane_cap])
            simultaneous_removed += len(group) - lane_cap
        i = j

    removed = removed_gap + simultaneous_removed
    if verbose and removed:
        print(
            f"[DrumGen][critic] playability lint removed={removed} "
            f"(lane_gap={removed_gap} stack_cap={simultaneous_removed})"
        )
    return sorted(capped, key=lambda n: float(n.get("time", 0.0)))


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


def _recover_drum_entry_hits(
    events: List[float],
    drum_section_start: float,
    kick_times: List[float],
    snare_times: List[float],
    bpm: float,
    preset: Dict,
    verbose: bool,
    classified_hits: Optional[List[Dict]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """Re-attach kick/snare onsets near drum entry that detect_drum_section_start dropped.

    Two windows:
    - grace: classic ~0.5–1s miss just before section_start
    - preamble: sparse kick/snare further back (intro hits before the dense groove)
    Hats never enter the recover pool.
    """
    recap: Dict[str, Any] = {
        "enabled": bool(preset.get("drum_entry_recovery", True)),
        "section_start": float(drum_section_start),
        "recovered": 0,
        "times": [],
        "window": None,
        "skip": "",
    }
    if not recap["enabled"]:
        recap["skip"] = "disabled"
        return events, recap
    if drum_section_start <= 0.0:
        recap["skip"] = "no_section_start"
        return events, recap

    beat_interval = 60.0 / max(1.0, bpm)
    grace_beats = float(preset.get("drum_entry_grace_beats", 1.25) or 1.25)
    grace_sec_floor = float(preset.get("drum_entry_grace_sec", 0.65) or 0.65)
    # High-BPM tracks: 1.25 beats ≈ 0.4s — too short for the classic ~0.5s miss.
    grace_before = max(beat_interval * grace_beats, grace_sec_floor)
    entry_span = beat_interval * float(preset.get("drum_entry_recovery_beats", 2.0) or 2.0)
    max_grace = int(preset.get("drum_entry_max_recover", 3) or 3)
    preamble_sec = float(preset.get("drum_entry_preamble_sec", 12.0) or 12.0)
    max_preamble = int(preset.get("drum_entry_preamble_max", 8) or 8)
    merge_tol = float(preset.get("drum_entry_merge_tol", 0.04) or 0.04)

    grace_start = max(0.0, drum_section_start - grace_before)
    preamble_start = max(0.0, drum_section_start - max(grace_before, preamble_sec))
    window_end = drum_section_start + entry_span
    recap["window"] = (preamble_start, window_end)

    # Kick/snare only — never pull hats into the entry recover pool (they ate max_add).
    core_pool = sorted({float(t) for t in (kick_times or []) + (snare_times or [])})
    if bool(preset.get("drum_entry_use_classified", True)) and classified_hits:
        core_pool = sorted(
            set(core_pool)
            | set(_classified_times_for_classes(classified_hits, ("kick", "snare")))
        )
    if not core_pool:
        recap["skip"] = "no_core_pool"
        return events, recap

    recovered = sorted(set(float(t) for t in events))
    added: List[float] = []

    # 1) Classic: earliest kick/snare just before section_start.
    pre_near = [t for t in core_pool if grace_start <= t < drum_section_start]
    # 2) Sparse intro further back (beyond grace).
    pre_far = [t for t in core_pool if preamble_start <= t < grace_start]
    # 3) Early hits inside the dense section window.
    in_window = [t for t in core_pool if drum_section_start <= t < window_end]

    def _try_add(pool: List[float], budget: int) -> None:
        nonlocal recovered, added
        for t in pool:
            if len(added) >= budget:
                break
            if _has_near(t, recovered, merge_tol):
                continue
            recovered.append(t)
            added.append(t)

    _try_add(sorted(pre_near), max_grace)
    # After grace budget, allow preamble fills up to preamble_max extra.
    _try_add(sorted(pre_far), max_grace + max_preamble)
    _try_add(sorted(in_window), max_grace + max_preamble)

    if not added:
        recap["skip"] = "none_needed"
        return events, recap

    recovered.sort()
    recap["recovered"] = len(added)
    recap["times"] = [round(t, 3) for t in added]
    if verbose:
        print(
            f"[DrumGen][critic] drum_entry: section_start={drum_section_start:.3f} "
            f"window=[{preamble_start:.3f},{window_end:.3f}) grace={grace_before:.3f}s "
            f"preamble={preamble_sec:.3f}s recovered=+{len(added)} "
            f"times={[round(t, 3) for t in added]}"
        )
    return recovered, recap


def _ensure_first_core_onset(
    events: List[float],
    drum_section_start: float,
    kick_times: List[float],
    snare_times: List[float],
    bpm: float,
    preset: Dict,
    verbose: bool,
    classified_hits: Optional[List[Dict]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """Re-insert the first kick/snare if later passes (cluster/section/caps) ate it.

    Listen/hako pattern: section_start≈1.79 matches stem peak, but final chart
    started at 2.13 — entry recovery saw nothing to add, then a mid-pipeline
    pass dropped the onset. This runs *after* those passes.
    """
    recap: Dict[str, Any] = {
        "enabled": bool(preset.get("drum_entry_ensure_first", True)),
        "section_start": float(drum_section_start),
        "added": 0,
        "time": None,
        "skip": "",
    }
    if not recap["enabled"]:
        recap["skip"] = "disabled"
        return events, recap
    if drum_section_start < 0.0:
        recap["skip"] = "no_section_start"
        return events, recap

    beat_interval = 60.0 / max(1.0, bpm)
    grace_beats = float(preset.get("drum_entry_grace_beats", 2.0) or 2.0)
    grace_sec = float(preset.get("drum_entry_grace_sec", 0.65) or 0.65)
    lookback = max(beat_interval * grace_beats, grace_sec)
    # Keep search tight after section_start — only the entry downbeat family.
    lookahead = beat_interval * float(preset.get("drum_entry_ensure_beats", 1.5) or 1.5)
    merge_tol = float(preset.get("drum_entry_merge_tol", 0.04) or 0.04)

    window_start = max(0.0, drum_section_start - lookback)
    window_end = drum_section_start + lookahead

    core_pool = sorted({float(t) for t in (kick_times or []) + (snare_times or [])})
    if bool(preset.get("drum_entry_use_classified", True)) and classified_hits:
        core_pool = sorted(
            set(core_pool)
            | set(_classified_times_for_classes(classified_hits, ("kick", "snare")))
        )
    candidates = [t for t in core_pool if window_start <= t < window_end]
    if not candidates:
        recap["skip"] = "no_core_in_window"
        return events, recap

    target = min(candidates)
    existing = sorted(set(float(t) for t in events))
    if _has_near(target, existing, merge_tol * 2.0):
        # Already have something near the first core hit (maybe snapped slightly).
        recap["skip"] = "already_present"
        return events, recap

    existing.append(target)
    existing.sort()
    recap["added"] = 1
    recap["time"] = round(target, 3)
    if verbose:
        print(
            f"[DrumGen][critic] first_onset_ensure: section_start={drum_section_start:.3f} "
            f"added={target:.3f} window=[{window_start:.3f},{window_end:.3f})"
        )
    return existing, recap


def _recover_sparse_measure_core_hits(
    events: List[float],
    kick_times: List[float],
    snare_times: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    verbose: bool,
) -> List[float]:
    """Pull quiet kick/snare hits back into measures that look under-detected."""
    if not events or beats is None or len(beats) < 4:
        return events

    buckets, first_measure_start, beat_interval = _measure_position_buckets(events, beats, bpm)
    if not buckets:
        return events

    measure_duration = beat_interval * 4.0
    counts = [len(buckets[i]) for i in buckets]
    median_count = float(np.median(counts)) if counts else 2.0
    sparse_threshold = max(2, int(np.floor(median_count * 0.55)))
    recovery_seconds = float(preset.get("sparse_recovery_seconds", 120.0) or 120.0)
    merge_tol = max(0.035, float(preset.get("sparse_recovery_tolerance", 0.045) or 0.045))
    max_add_per_measure = int(preset.get("sparse_recovery_max_add", 4) or 4)

    core_pool = sorted(set(float(t) for t in (kick_times or []) + (snare_times or [])))
    if not core_pool:
        return events

    recovered = sorted(set(events))
    added = 0
    max_measure = int(np.ceil((max(core_pool[-1], recovered[-1]) - first_measure_start) / measure_duration))

    for measure_idx in range(0, max_measure + 1):
        measure_start = first_measure_start + measure_idx * measure_duration
        if measure_start > recovery_seconds:
            break
        current = len(buckets.get(measure_idx, set()))
        if current >= sparse_threshold:
            continue

        neighbor_counts = []
        for other_idx in range(max(0, measure_idx - 2), measure_idx + 3):
            if other_idx in buckets:
                neighbor_counts.append(len(buckets[other_idx]))
        if neighbor_counts and max(neighbor_counts) >= sparse_threshold + 1 and current >= 1:
            continue

        measure_end = measure_start + measure_duration
        candidates = [t for t in core_pool if measure_start - 0.02 <= t < measure_end + 0.02]
        candidates.sort()
        slots_left = max(0, max_add_per_measure - current)
        for t in candidates:
            if slots_left <= 0:
                break
            if _has_near(t, recovered, merge_tol):
                continue
            recovered.append(t)
            buckets.setdefault(measure_idx, set()).add(round((t - measure_start) / beat_interval * 4) / 4)
            added += 1
            slots_left -= 1

    if verbose and added:
        print(f"[DrumGen][этап] sparse_recovery=+{added} threshold={sparse_threshold}")
    return sorted(recovered)


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

    from .arcade_passes import arcade_should_skip_sparse_add

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
        if arcade_should_skip_sparse_add(
            current_count=len(current_positions),
            neighbor_sizes=neighbor_sizes,
            preset=preset,
            measure_idx=measure_idx,
        ):
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

    from .arcade_passes import arcade_should_skip_sparse_add

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
        if arcade_should_skip_sparse_add(
            current_count=len(current_positions),
            neighbor_sizes=[len(prev_positions)],
            preset=preset,
            measure_idx=measure_idx,
        ):
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

    from .arcade_passes import arcade_should_skip_sparse_add

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
        if arcade_should_skip_sparse_add(
            current_count=len(current_positions),
            neighbor_sizes=[len(reference_positions)],
            preset=preset,
            measure_idx=measure_idx,
        ):
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
    base = _GENRE_PATTERN_POSITIONS.get(genre_label, _GENRE_PATTERN_POSITIONS.get("default", []))
    if isinstance(base, dict):
        return base.get("break" if is_break else "normal") or base.get("normal", [])
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
    preset: Optional[Dict] = None,
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

        dense_boost = bool((preset or {}).get("fill_dense_boost", False))
        if dense_boost:
            need_fill = energy > 0 and len(base_in_measure) < target_max
        else:
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
    generation_preset: Optional[Dict] = None,
    fill: Optional[int] = None,
    groove: Optional[int] = None,
    density: Optional[int] = None,
    grid_snap_strength: Optional[int] = None,
    accent_strong_beats: Optional[bool] = None,
    genre_template_strength: Optional[int] = None,
    include_hi_hats: Optional[bool] = None,
    track_info: Optional[Dict] = None,
    auto_identify_track: bool = False,
    use_filename_for_genres: bool = False,
    provided_genres: Optional[List[str]] = None,
    provided_primary_genre: Optional[str] = None,
    verbose: bool = True,
    status_cb=None,
    cancel_cb=None,
    chart_id: str = "",
) -> Optional[List[Dict]]:
    if generation_preset is not None:
        preset = deepcopy(generation_preset)
        preset_id = str(preset.get("preset_id", preset_id or generation_mode or "basic"))
        mode = str(preset.get("mode", generation_mode or "basic")).lower()
    else:
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

    if cancel_cb:
        cancel_cb()
    if status_cb and not stem_memory_cache.get_cached_stem(str(song_path), "drums"):
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
        chart_id=chart_id,
    )
    if cancel_cb:
        cancel_cb()

    bpm = analysis.get("bpm", bpm)
    beats = np.array(analysis.get("beats", []))
    kick_times: List[float] = analysis.get("kick_times", [])
    snare_times: List[float] = analysis.get("snare_times", [])
    hat_times: List[float] = analysis.get("hat_times", [])
    dominant_onsets: List[float] = analysis.get("dominant_onsets", [])
    classified_hits: List[Dict] = analysis.get("classified_hits", [])
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
    _apply_genre_preferences(
        preset,
        genre_params.get("preferences") or {},
        mode,
        verbose=verbose,
    )
    grid_snap_strength = int(preset.get("grid_snap_strength", grid_snap_strength))
    accent_strong_beats = bool(preset.get("accent_strong_beats", accent_strong_beats))
    if include_hi_hats is not None:
        preset["rhythm_hit_classes"] = (
            ["kick", "snare", "hat"] if include_hi_hats else ["kick", "snare"]
        )
    grid_snap_enabled = grid_snap_strength > 0

    if verbose:
        chart_intent = str(preset.get("chart_intent", "") or "")
        intent_suffix = f" intent={chart_intent}" if chart_intent else ""
        print(
            f"[DrumGen] preset={preset_id} режим={mode}{intent_suffix} policy={preset.get('dominant_onsets_policy')} "
            f"fill={fill} groove={groove} density={density} grid_snap_strength={grid_snap_strength} "
            f"accent_strong_beats={accent_strong_beats} rhythm_hit_classes={preset.get('rhythm_hit_classes')} "
            f"flam_merge={preset.get('flam_merge_sec', 0)} "
            f"caps={_effective_max_hits_per_second(preset, bpm)}/{_effective_max_notes_per_measure(preset, bpm)} "
            f"bpm={bpm} lanes={lanes}"
        )
        print(f"[DrumGen] Жанр: {genre_label} | уникальные: {unique_genres}")

    if status_cb:
        status_cb("Детекция ударных...")
    if cancel_cb:
        cancel_cb()

    if verbose:
        print(
            f"[DrumGen][этап] beats={len(beats)} kick={len(kick_times)} "
            f"snare={len(snare_times)} dominant={len(dominant_onsets)} classified={len(classified_hits)}"
        )

    salience_stats: Optional[Dict[str, float]] = None
    classified_for_source = classified_hits
    drum_stem_path = analysis.get("analysis_path")
    if classified_hits and drum_stem_path:
        texture_strictness = float(preset.get("salience_texture_strictness", 1.0) or 1.0)
        classified_hits, salience_stats = annotate_salience_roles(
            classified_hits,
            drum_stem_path,
            texture_strictness=texture_strictness,
        )
        include_roles = preset.get("salience_include_roles")
        if include_roles:
            classified_for_source = filter_hits_by_salience_roles(classified_hits, include_roles)
            if verbose:
                print(
                    f"[DrumGen][salience] roles={list(include_roles)} "
                    f"kept={len(classified_for_source)}/{len(classified_hits)} "
                    f"rhythm={int(salience_stats.get('rhythm', 0))} "
                    f"texture={int(salience_stats.get('texture', 0))}"
                )

    raw_events = _select_raw_events(
        kick_times,
        snare_times,
        dominant_onsets,
        str(preset.get("dominant_onsets_policy", "dominant_onsets")),
        classified_hits=classified_for_source,
        rhythm_classes=preset.get("rhythm_hit_classes"),
    )
    if not raw_events:
        return None
    ledger = StageLedger.create(beats, bpm)
    ledger.record("cand", raw_events)
    raw_events = _recover_sparse_measure_core_hits(
        raw_events,
        kick_times,
        snare_times,
        beats,
        bpm,
        preset,
        verbose,
    )
    ledger.record("recover_sparse", raw_events)
    _print_rhythm_diagnostics("source", raw_events, beats, bpm, mode, preset_id)
    source_event_count = len(raw_events)

    if "sync_tolerance_multiplier" in genre_params:
        sync_tolerance = float(sync_tolerance) * float(genre_params.get("sync_tolerance_multiplier", 1.0))

    drum_start_window = float(genre_params.get("drum_start_window", 4.0))
    drum_density_threshold = float(genre_params.get("drum_density_threshold", 0.5))
    drum_section_start = detect_drum_section_start(raw_events, drum_start_window, drum_density_threshold)
    if bool(preset.get("keep_pre_section_hits", False)):
        # Original documentary: keep detector hits before the dense groove.
        filtered_events = list(raw_events)
        if verbose and drum_section_start > 0.0:
            print(
                f"[DrumGen][этап] keep_pre_section_hits=1 "
                f"(section_start={drum_section_start:.3f} unused for trim)"
            )
        drum_entry_recap = {"enabled": False, "skip": "keep_pre_section_hits", "recovered": 0}
    else:
        filtered_events = [t for t in raw_events if t >= drum_section_start]
        before_entry = len(filtered_events)
        filtered_events, drum_entry_recap = _recover_drum_entry_hits(
            filtered_events,
            drum_section_start,
            kick_times,
            snare_times,
            bpm,
            preset,
            verbose,
            classified_hits,
        )
    ledger.record("section_cut", filtered_events)
    core_sources = sorted(set(t for t in (kick_times + snare_times) if t >= drum_section_start))

    min_note_distance = float(genre_params.get("min_note_distance", 0.05))
    if mode == "custom":
        min_note_distance = _density_to_min_distance(min_note_distance, density)
    elif mode == "minimal":
        min_note_distance = min(0.22, max(0.06, min_note_distance * 1.35))
    else:
        min_note_distance = _density_to_min_distance(min_note_distance, density)
    min_note_distance = max(min_note_distance, float(preset.get("min_note_distance_floor", 0.0) or 0.0))

    apply_groove = False
    use_grid_sync = False

    if verbose:
        entry_note = ""
        recovered_n = int(drum_entry_recap.get("recovered", 0) or 0)
        if recovered_n > 0:
            entry_note = f" entry_recover=+{recovered_n}"
        print(
            f"[DrumGen][этап] mode={mode} raw={len(raw_events)} after_start={len(filtered_events)} "
            f"start={drum_section_start:.3f}{entry_note} min_dist={min_note_distance:.3f} "
            f"sync={use_grid_sync} grid_strength={grid_snap_strength} groove={apply_groove} tol={sync_tolerance:.3f}"
        )

    events = apply_temporal_filter(sorted(filtered_events), min_note_distance)
    ledger.record("temporal", events)
    if mode == "custom":
        if groove <= 40:
            use_grid_sync = True
            apply_groove = False
        elif groove >= 60:
            apply_groove = True
        if groove >= 80:
            use_grid_sync = False
    use_grid_sync = bool(use_grid_sync and grid_snap_enabled)

    grooved_events = apply_groove_pattern(events, bpm=bpm) if apply_groove else events
    synced_events = _pull_to_grid(grooved_events, beats, sync_tolerance, grid_snap_strength) if use_grid_sync else grooved_events
    events_after_timing = synced_events
    ledger.record("groove", grooved_events)
    ledger.record("grid", synced_events)
    if accent_strong_beats:
        events_after_timing = _accent_to_strong_beats(events_after_timing, beats, sync_tolerance, 70)
    ledger.record("accent", events_after_timing)
    before_cluster = len(events_after_timing)
    style_policy = get_style_policy(preset.get("generation_goal"), preset)
    if style_policy.minimize_fill_zone:
        fill_core, fill_halo = set(), set()
    else:
        fill_core, fill_halo = _fill_zone_measure_tiers(
            events_after_timing, beats, bpm, preset, genre_label=genre_label
        )
    fill_measures = fill_core | fill_halo
    fill_core_list = sorted(fill_core)
    fill_halo_list = sorted(fill_halo)
    if verbose and fill_measures:
        sample = sorted(fill_core)[:8]
        print(
            f"[DrumGen][fill_zone] genre={genre_label} bpm={bpm:g} "
            f"core={sample}{'…' if len(fill_core) > 8 else ''} core_n={len(fill_core)} "
            f"halo_n={len(fill_halo)} total_n={len(fill_measures)}"
        )
    elif verbose and bool(preset.get("fill_zone_enabled", False)):
        print(f"[DrumGen][fill_zone] skipped genre={genre_label} bpm={bpm:g}")
    events_after_timing = _cluster_hit_events(
        events_after_timing,
        beats,
        bpm,
        preset,
        fill_measures=fill_core,
        fill_halo_measures=fill_halo,
        core_sources=core_sources,
        genre_label=genre_label,
    )
    ledger.record("cluster", events_after_timing)
    if verbose and len(events_after_timing) != before_cluster:
        print(f"[DrumGen][этап] hit_cluster={before_cluster}->{len(events_after_timing)}")
    before_flam = len(events_after_timing)
    events_after_timing = _merge_flam_events(
        events_after_timing,
        beats,
        bpm,
        preset,
        fill_measures=fill_core,
        fill_halo_measures=fill_halo,
        core_sources=core_sources,
        genre_label=genre_label,
    )
    ledger.record("flam", events_after_timing)
    if verbose and len(events_after_timing) != before_flam:
        print(f"[DrumGen][этап] flam_merge={before_flam}->{len(events_after_timing)}")

    style_ctx = StyleBuildContext(
        events_after_timing=list(events_after_timing),
        filtered_events=filtered_events,
        beats=beats,
        bpm=bpm,
        preset=preset,
        mode=mode,
        genre_label=genre_label,
        classified_hits=classified_hits,
        kick_times=kick_times,
        snare_times=snare_times,
        dominant_onsets=dominant_onsets,
        analysis=analysis,
        fill=fill,
        genre_template_strength=genre_template_strength,
        verbose=verbose,
        fill_core=fill_core,
        fill_halo=fill_halo,
        cancel_cb=cancel_cb,
    )
    style_result = build_style_map(style_ctx)
    events_after_timing = style_result.times
    ledger.record("style", events_after_timing)
    fill_core_list = style_result.fill_core_list
    fill_halo_list = style_result.fill_halo_list
    if verbose:
        print(
            f"[DrumGen][style] goal={style_policy.goal} "
            f"events={len(events_after_timing)} recap={style_result.recap}"
        )

    events_after_timing, _difficulty_recap = apply_difficulty_transform(
        events_after_timing,
        difficulty=preset.get("generation_difficulty"),
        goal=preset.get("generation_goal"),
        beats=beats,
        bpm=bpm,
        preset=preset,
        classified_hits=classified_hits,
        verbose=verbose,
    )
    ledger.record("difficulty", events_after_timing)

    if verbose:
        print(
            f"[DrumGen][этап] after_filter={len(events)} after_groove={len(grooved_events)} "
            f"after_sync={len(synced_events)} style_goal={style_policy.goal}"
        )
    if mode == "basic":
        print(
            f"[DrumGen] Basic caps | hits/sec={_effective_max_hits_per_second(preset, bpm)} "
            f"notes/measure={_effective_max_notes_per_measure(preset, bpm)}"
        )

    if measure_map_enabled():
        mm_rows = build_measure_map(
            events_after_timing,
            beats,
            bpm,
            analysis.get("analysis_path"),
            analysis.get("original_path"),
        )
        log_measure_map(
            "pre_section",
            mm_rows,
            drum_path=analysis.get("analysis_path"),
            mix_path=analysis.get("original_path"),
        )
    else:
        mm_rows = []

    if cancel_cb:
        cancel_cb()

    before_section = len(events_after_timing)
    events_after_timing = _apply_section_context_pass(
        list(events_after_timing),
        kick_times,
        snare_times,
        beats,
        bpm,
        preset,
        classified_hits=classified_hits,
        verbose=verbose,
        stem_audio_path=analysis.get("analysis_path"),
        mix_audio_path=analysis.get("original_path"),
        max_notes_per_measure=_effective_max_notes_per_measure(preset, bpm),
    )
    ledger.record("section_pass", events_after_timing)
    if verbose and len(events_after_timing) != before_section:
        print(f"[DrumGen][этап] section_pass={before_section}->{len(events_after_timing)}")
    post_section_count = len(events_after_timing)

    before_guardrails = len(events_after_timing)
    base_times = _apply_density_guardrails_preserving_core(
        list(events_after_timing),
        core_sources,
        beats,
        bpm,
        preset,
    )
    ledger.record("guardrails", base_times)
    if verbose and len(base_times) != before_guardrails:
        print(f"[DrumGen][этап] density_guardrails={before_guardrails}->{len(base_times)}")
    if verbose and bool(preset.get("preserve_core_hits", False)):
        core_kept, extra_kept = _split_core_and_extra_events(base_times, core_sources, preset)
        print(f"[DrumGen][этап] core_hits={len(core_kept)} extra_hits={len(extra_kept)}")
    # Goal×difficulty preset owns fill; do not zero dense fill on basic mode.
    if mode == "natural":
        fill = 0

    added_times: List[float] = []
    if fill > 0 and style_policy.run_fill_augmentation:
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
            preset=preset,
        )

    all_times = _append_events_with_density_guardrails(base_times, added_times, beats, bpm, preset)
    ledger.record("fill_add", all_times)

    all_times, _post_style_recap = apply_style_post_passes(all_times, style_ctx, base_times)
    ledger.record("style_post", all_times)
    all_times = _apply_critic_intro_no_add(
        all_times,
        drum_section_start,
        kick_times,
        snare_times,
        beats,
        bpm,
        preset,
        verbose=verbose,
        classified_hits=classified_hits,
        dominant_onsets=dominant_onsets,
    )
    ledger.record("critic_intro", all_times)
    all_times, first_onset_recap = _ensure_first_core_onset(
        all_times,
        drum_section_start,
        kick_times,
        snare_times,
        bpm,
        preset,
        verbose,
        classified_hits,
    )
    if first_onset_recap.get("added"):
        # Reflect ensure in drum_entry recap so GenRecap is honest.
        prev = int(drum_entry_recap.get("recovered", 0) or 0)
        drum_entry_recap["recovered"] = prev + int(first_onset_recap["added"])
        times = list(drum_entry_recap.get("times") or [])
        times.append(first_onset_recap.get("time"))
        drum_entry_recap["times"] = [t for t in times if t is not None]
        drum_entry_recap["ensure_first"] = first_onset_recap.get("time")
        if drum_entry_recap.get("skip") in ("none_needed", ""):
            drum_entry_recap["skip"] = ""
    if style_policy.run_enhanced_topup and mode == "enhanced" and base_times:
        min_target = int(len(base_times) * 1.25)
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
                preset=preset,
            )
            all_times = _append_events_with_density_guardrails(all_times, extra, beats, bpm, preset)
    ledger.record("topup", all_times)
    _print_rhythm_diagnostics("final", all_times, beats, bpm, mode, preset_id)

    all_events = [{"type": NoteType.DRUM, "time": t} for t in all_times]
    lane_by_drum = bool(preset.get("lane_by_drum", False))
    ergonomic_env = os.environ.get("RFALL_ERGONOMIC_ROUTER", "").strip().lower()
    if ergonomic_env in ("0", "off", "false", "no"):
        use_ergonomic = False
    elif ergonomic_env in ("1", "on", "true", "yes"):
        use_ergonomic = True
    else:
        use_ergonomic = bool(preset.get("ergonomic_router", False))
    if use_ergonomic and classified_hits:
        notes = assign_lanes_ergonomic(
            all_events,
            classified_hits,
            lanes=lanes,
            song_offset=0.0,
            bpm=bpm,
            beats=beats,
            phrase_bars=int(preset.get("arcade_lane_phrase_bars", 4) or 4),
            phrase_mode=str(preset.get("arcade_lane_phrase_mode", "mirror") or "mirror"),
            phrase_bias=float(preset.get("arcade_lane_phrase_bias", 1.25) or 1.25),
        )
        if verbose:
            print(f"[DrumGen][этап] lane_router=ergonomic(arcade) notes={len(notes)}")
    elif lane_by_drum and classified_hits:
        notes = assign_lanes_by_drum_class(all_events, classified_hits, lanes=lanes, song_offset=0.0)
    else:
        notes = assign_lanes_to_notes(all_events, lanes=lanes, song_offset=0.0)

    # Break unplayable same-lane jacks (Original + Arcade). Times unchanged — lane only.
    if lanes >= 2 and notes:
        before_div = len(notes)
        notes, div_recap = diversify_lane_runs(notes, lanes=lanes, max_same_run=3)
        if verbose and int(div_recap.get("nudged", 0)) > 0:
            print(
                f"[DrumGen][этап] lane_diversify notes={before_div} "
                f"nudged={div_recap.get('nudged')} max_run={div_recap.get('max_run')}"
            )

    drum_tol = float(preset.get("drum_class_tolerance", 0.06) or 0.06)
    for note in notes:
        drum = resolve_drum_at_time(float(note["time"]), classified_hits, tolerance=drum_tol)
        if drum:
            note["drum"] = drum

    before_lint = len(notes)
    notes = _apply_playability_linter(notes, lanes, preset, verbose=verbose)
    ledger.record_notes("playability", notes)
    ledger.record_notes("final", notes)
    if verbose and len(notes) != before_lint:
        print(f"[DrumGen][этап] playability_lint={before_lint}->{len(notes)}")

    if verbose:
        print(
            f"[DrumGen] Итого: {len(notes)} (обнаружено={len(base_times)}, "
            f"добавлено={max(0, len(all_times) - len(base_times))})"
        )

    drum_counts = adtof_drum_counts(classified_hits)
    track_label = os.path.basename(str(song_path))
    variant_suffix = chart_variant_suffix()
    chart_variant = variant_suffix[1:] if variant_suffix.startswith("_") else variant_suffix
    print_generation_recap(
        track=track_label,
        genre=genre_label,
        bpm=float(bpm),
        mode=mode,
        preset_id=preset_id,
        adtof_unique=len(dominant_onsets),
        adtof_kick=len(kick_times) or drum_counts.get("kick", 0),
        adtof_snare=len(snare_times) or drum_counts.get("snare", 0),
        adtof_hat=len(hat_times) or drum_counts.get("hat", 0),
        adtof_tom=drum_counts.get("tom", 0),
        adtof_cymbal=drum_counts.get("cymbal", 0),
        adtof_rows=len(classified_hits),
        source_events=source_event_count,
        pre_section_events=before_section,
        post_section_events=post_section_count,
        final_events=len(all_times),
        caps_hps=_effective_max_hits_per_second(preset, bpm),
        caps_npm=_effective_max_notes_per_measure(preset, bpm),
        measure_map_line=measure_map_recap_line(mm_rows) if mm_rows else "",
        salience_line=salience_recap_line(
            salience_stats,
            filtered_kept=len(classified_for_source),
            filtered_total=len(classified_hits),
        )
        if salience_stats
        else "",
        critic_line=drum_entry_recap_line(drum_entry_recap),
        chart_variant=chart_variant,
        generation_goal=str(preset.get("generation_goal", "")),
        generation_difficulty=str(preset.get("generation_difficulty", "")),
        chart_stem=str(preset.get("chart_stem", preset_id)),
    )

    track_artist = ""
    track_title = ""
    if isinstance(track_info, dict):
        track_artist = str(track_info.get("artist", "") or "")
        track_title = str(track_info.get("title", "") or "")

    global _last_rhythm_dna
    chart_intent = str(preset.get("chart_intent", "") or "").strip().lower()
    _last_rhythm_dna = build_rhythm_dna(
        track=track_label,
        artist=track_artist,
        title=track_title,
        genre=genre_label,
        bpm=float(bpm),
        mode=chart_intent or mode,
        preset_id=preset_id,
        lanes=int(lanes),
        source_events=source_event_count,
        pre_section_events=before_section,
        post_section_events=post_section_count,
        final_events=len(all_times),
        final_notes=len(notes),
        adtof_unique=len(dominant_onsets),
        adtof_kick=len(kick_times) or drum_counts.get("kick", 0),
        adtof_snare=len(snare_times) or drum_counts.get("snare", 0),
        adtof_hat=len(hat_times) or drum_counts.get("hat", 0),
        adtof_rows=len(classified_hits),
        caps_hps=_effective_max_hits_per_second(preset, bpm),
        caps_npm=_effective_max_notes_per_measure(preset, bpm),
        measure_rows=mm_rows,
        salience_stats={
            **(salience_stats or {}),
            "filter_kept": len(classified_for_source),
            "filter_total": len(classified_hits),
        },
        drum_entry_recap=drum_entry_recap,
        fill_core_measures=fill_core_list,
        fill_halo_measures=fill_halo_list,
        playability_lint_removed=max(0, before_lint - len(notes)),
        chart_variant=chart_variant,
        mix_audio_path=analysis.get("original_path"),
    )

    global _last_stage_ledger
    _last_stage_ledger = ledger.build(
        measure_rows=mm_rows,
        meta={
            "label": track_label,
            "artist": track_artist,
            "title": track_title,
            "genre": genre_label,
            "chart_id": chart_id,
            "chart_intent": chart_intent or mode,
            "preset_id": preset_id,
            "lanes": int(lanes),
            "variant": chart_variant,
            "drum_stem": analysis.get("analysis_path"),
        },
    )
    log_stage_ledger(_last_stage_ledger)

    return notes if notes else None
