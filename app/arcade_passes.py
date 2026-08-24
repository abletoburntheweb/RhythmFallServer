# app/arcade_passes.py — Arcade playability passes (see docs/arcade_mode.md).
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .drum_hit_detector import resolve_drum_at_time
from .measure_energy import relative_contour_per_measure, rms_per_measure

TEXTURE_CLASSES = frozenset({"hat", "cymbal"})
BACKBEAT_SNARE_GENRES = frozenset({
    "house", "techno", "trance", "edm", "electronic", "dance", "pop",
    "rap", "hip hop", "hip-hop", "hiphop", "chillwave", "lo-fi", "lofi",
    "funk", "disco", "garage", "drum and bass", "dnb",
})
METAL_KICK_GENRES = frozenset({"metal", "rock", "hard rock", "punk"})


def _beat_interval(beats: np.ndarray, bpm: float) -> float:
    if beats is not None and len(beats) >= 2:
        return float(np.median(np.diff(beats)))
    return 60.0 / max(1.0, float(bpm))


def _measure_timing(beats: np.ndarray, bpm: float) -> Tuple[float, float]:
    beat_interval = _beat_interval(beats, bpm)
    first = float(beats[0]) if beats is not None and len(beats) > 0 else 0.0
    return first, beat_interval * 4.0


def _max_measure_index(events: Sequence[float], first_measure_start: float, measure_duration: float) -> int:
    if not events or measure_duration <= 0:
        return 0
    return int(max(0, np.floor((max(events) - first_measure_start) / measure_duration)))


def _measure_index(time_val: float, first_measure_start: float, measure_duration: float) -> int:
    if measure_duration <= 0:
        return 0
    return int(np.floor((float(time_val) - first_measure_start) / measure_duration))


def _has_near(t: float, existing: Sequence[float], tol: float) -> bool:
    for other in existing:
        if abs(float(other) - float(t)) <= tol:
            return True
    return False


def arcade_sparse_restraint_enabled(preset: Optional[Dict]) -> bool:
    """Arcade-only: block densify adds into already-sparse / quiet measures."""
    if not preset:
        return False
    if str(preset.get("generation_goal", "") or "").strip().lower() != "arcade":
        return False
    return bool(preset.get("arcade_sparse_restraint", True))


def arcade_should_skip_sparse_add(
    *,
    current_count: int,
    neighbor_sizes: Optional[Sequence[int]] = None,
    tension: Optional[Dict[int, float]] = None,
    measure_idx: int = 0,
    preset: Optional[Dict] = None,
) -> bool:
    """Return True when Arcade must not add notes into this measure.

    Sparse bars (≤ arcade_sparse_max_notes) stay sparse unless neighbors are
    similarly light. Low mix tension also blocks adds. Stops groove/loop/backbeat
    from pulling quiet singles up to dense neighboring measures.
    """
    if not arcade_sparse_restraint_enabled(preset):
        return False
    sparse_max = int((preset or {}).get("arcade_sparse_max_notes", 2) or 2)
    if current_count > sparse_max:
        return False

    tension_min = float((preset or {}).get("arcade_add_tension_min", 0.28) or 0.28)
    if tension is not None and tension:
        score = float(tension.get(int(measure_idx), 0.5) or 0.5)
        if score < tension_min:
            return True

    if neighbor_sizes:
        sizes = [int(s) for s in neighbor_sizes if int(s) >= 0]
        if sizes:
            med = float(np.median(np.asarray(sizes, dtype=float)))
            # Completing a consistently light groove is OK; do not pull quiet
            # bars up toward much denser neighbors.
            if med <= float(sparse_max) + 1.0:
                return False
            return True

    # No neighbor evidence on an already-sparse bar → do not invent density
    # (covers metal backbeat +2 kicks on single-hit measures).
    return True


def build_tension_map(
    *,
    mix_audio_path: Optional[str],
    beats: np.ndarray,
    bpm: float,
    max_measure: int,
    rolling_radius: int = 16,
) -> Dict[int, float]:
    """Per-measure tension in [0, 1] from full-mix RMS contour (arcade_mode pass 1)."""
    if max_measure < 0 or not mix_audio_path:
        return {}
    first_start, measure_duration = _measure_timing(beats, bpm)
    mix_rms = rms_per_measure(mix_audio_path, first_start, measure_duration, max_measure)
    if not mix_rms:
        return {}
    mix_rel = relative_contour_per_measure(mix_rms, max_measure, rolling_radius=max(1, rolling_radius))
    out: Dict[int, float] = {}
    for idx in range(0, max_measure + 1):
        rel = float(mix_rel.get(idx, 0.0) or 0.0)
        # 1.0 ≈ typical local level; squash into [0, 1].
        out[idx] = float(min(1.0, max(0.0, rel / 1.25)))
    return out


def apply_phantom_gate(
    events: List[float],
    tension: Dict[int, float],
    *,
    beats: np.ndarray,
    bpm: float,
    classified_hits: Optional[List[Dict]] = None,
    tension_cutoff: float = 0.10,
    protect_kick_snare: bool = True,
    drum_tolerance: float = 0.06,
) -> Tuple[List[float], int]:
    """Drop notes in low-tension (quiet) sections (arcade_mode pass 2)."""
    if not events or not tension:
        return events, 0
    first_start, measure_duration = _measure_timing(beats, bpm)
    kept: List[float] = []
    removed = 0
    for t in sorted(events):
        m = _measure_index(t, first_start, measure_duration)
        score = float(tension.get(m, 0.5))
        if score >= tension_cutoff:
            kept.append(t)
            continue
        if protect_kick_snare and classified_hits:
            drum = resolve_drum_at_time(t, classified_hits, tolerance=drum_tolerance) or ""
            if drum in ("kick", "snare"):
                kept.append(t)
                continue
        removed += 1
    return kept, removed


def apply_backbeat_reconstruction(
    events: List[float],
    tension: Dict[int, float],
    *,
    candidate_events: List[float],
    beats: np.ndarray,
    bpm: float,
    genre_label: str,
    preset: Dict,
    snare_times: Optional[List[float]] = None,
) -> Tuple[List[float], int]:
    """Add missing snare/clap on 2/4 when section has energy (arcade_mode pass 3)."""
    if not events or beats is None or len(beats) < 8:
        return events, 0
    genre = str(genre_label or "").strip().lower()
    use_snare_backbeat = any(g in genre for g in BACKBEAT_SNARE_GENRES)
    use_metal_kick = any(g in genre for g in METAL_KICK_GENRES)
    if not use_snare_backbeat and not use_metal_kick:
        return events, 0

    tension_min = float(preset.get("arcade_backbeat_tension_min", 0.30) or 0.30)
    max_add = int(preset.get("arcade_backbeat_max_add_per_measure", 2) or 2)
    seek_window = float(preset.get("arcade_backbeat_seek_window", 0.08) or 0.08)
    beat_interval = _beat_interval(beats, bpm)
    first_start, measure_duration = _measure_timing(beats, bpm)
    cluster_tol = max(min(0.04, beat_interval * 0.12), beat_interval * 0.08)

    buckets: Dict[int, set] = {}
    for t in sorted(events):
        m = _measure_index(t, first_start, measure_duration)
        if m < 0:
            continue
        rel = (t - (first_start + m * measure_duration)) / beat_interval
        if rel < -0.1 or rel >= 4.1:
            continue
        q = round(max(0.0, min(3.75, rel)) * 4.0) / 4.0
        buckets.setdefault(m, set()).add(q)

    completed = sorted(set(events))
    pool = sorted(set(candidate_events or []))
    snare_set = {round(float(t), 4) for t in (snare_times or [])}
    added = 0

    backbeat_positions = [1.0, 3.0] if use_snare_backbeat else []
    if use_metal_kick and not use_snare_backbeat:
        backbeat_positions = [0.0, 2.0]

    for m in sorted(buckets.keys()):
        if float(tension.get(m, 0.0) or 0.0) < tension_min:
            continue
        positions = buckets.get(m, set())
        if arcade_should_skip_sparse_add(
            current_count=len(positions),
            neighbor_sizes=None,
            tension=tension,
            measure_idx=m,
            preset=preset,
        ):
            continue
        measure_start = first_start + m * measure_duration
        added_here = 0
        for pos in backbeat_positions:
            if added_here >= max_add:
                break
            if pos in positions:
                continue
            target = measure_start + pos * beat_interval
            if any(abs(target - s) <= seek_window for s in snare_set):
                continue
            selected = None
            for cand in pool:
                if abs(cand - target) <= seek_window and not _has_near(cand, completed, cluster_tol):
                    selected = cand
                    break
            if selected is None:
                selected = target
            if _has_near(selected, completed, cluster_tol):
                continue
            completed.append(selected)
            positions.add(pos)
            added += 1
            added_here += 1

    return sorted(completed), added


def apply_texture_downsampling(
    events: List[float],
    *,
    classified_hits: Optional[List[Dict]],
    beats: np.ndarray,
    bpm: float,
    genre_label: str,
    preset: Dict,
    drum_tolerance: float = 0.06,
) -> Tuple[List[float], int]:
    """Thin hat/cymbal spam into readable patterns (arcade_mode pass 4)."""
    if not events or not classified_hits or len(events) < 3:
        return events, 0
    min_run = int(preset.get("arcade_texture_min_run", 3) or 3)
    beat_interval = _beat_interval(beats, bpm)
    max_dt = float(preset.get("arcade_texture_max_dt", 0.22) or 0.22)
    genre = str(genre_label or "").strip().lower()
    pattern = str(preset.get("arcade_texture_pattern", "") or "").strip().lower()
    if not pattern:
        if any(g in genre for g in ("metal", "rock", "jazz")):
            return events, 0
        if any(g in genre for g in ("house", "techno", "trance", "edm")):
            pattern = "trill"
        elif any(g in genre for g in ("chillwave", "lo-fi", "lofi")):
            pattern = "gallop"
        else:
            pattern = "trill"

    tagged: List[Tuple[float, str]] = []
    for t in sorted(events):
        drum = str(resolve_drum_at_time(t, classified_hits, tolerance=drum_tolerance) or "perc").lower()
        tagged.append((t, drum))

    drop: set = set()
    i = 0
    while i < len(tagged):
        if tagged[i][1] not in TEXTURE_CLASSES:
            i += 1
            continue
        j = i + 1
        while j < len(tagged) and tagged[j][1] in TEXTURE_CLASSES:
            if tagged[j][0] - tagged[j - 1][0] > max_dt:
                break
            j += 1
        run_len = j - i
        if run_len >= min_run:
            keep_idx = _texture_keep_indices(run_len, pattern)
            for k in range(i, j):
                if (k - i) not in keep_idx:
                    drop.add(tagged[k][0])
        i = j if run_len >= min_run else i + 1

    if not drop:
        return events, 0
    kept = [t for t in events if t not in drop]
    return kept, len(drop)


def _texture_keep_indices(run_len: int, pattern: str) -> set:
    if pattern == "gallop":
        # 1-x-3-x within 8-step grid
        return {idx for idx in range(run_len) if idx % 4 in (0, 2)}
    if pattern == "offbeat":
        return {idx for idx in range(run_len) if idx % 4 in (1, 3)}
    # trill: keep every other 16th
    return {idx for idx in range(run_len) if idx % 2 == 0}


def apply_arcade_passes(
    events: List[float],
    *,
    candidate_events: List[float],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    classified_hits: Optional[List[Dict]] = None,
    snare_times: Optional[List[float]] = None,
    kick_times: Optional[List[float]] = None,
    genre_label: str = "",
    mix_audio_path: Optional[str] = None,
    drum_audio_path: Optional[str] = None,
    verbose: bool = False,
    phase: str = "pre_section",
) -> Tuple[List[float], Dict[str, int]]:
    """Run enabled arcade passes; returns events and recap counters.

    pre_section: phantom gate + texture trim (before section pass).
    post_section: backbeat reconstruction (after fill/guardrails).
    """
    recap = {"phantom_removed": 0, "backbeat_added": 0, "texture_removed": 0}
    if not bool(preset.get("arcade_policy", False)) or not events:
        return events, recap

    phase_key = str(phase or "pre_section").strip().lower()
    first_start, measure_duration = _measure_timing(beats, bpm)
    max_m = _max_measure_index(events, first_start, measure_duration)

    tension: Dict[int, float] = {}
    needs_tension = (
        (phase_key == "pre_section" and bool(preset.get("arcade_phantom_gate", True)))
        or (phase_key == "post_section" and bool(preset.get("arcade_backbeat", True)))
    )
    if needs_tension and bool(preset.get("arcade_tension_map", True)):
        tension = build_tension_map(
            mix_audio_path=mix_audio_path or drum_audio_path,
            beats=beats,
            bpm=bpm,
            max_measure=max_m,
            rolling_radius=int(preset.get("section_contour_rolling_radius", 16) or 16),
        )

    out = list(events)

    if phase_key == "pre_section":
        if bool(preset.get("arcade_phantom_gate", True)) and tension:
            out, recap["phantom_removed"] = apply_phantom_gate(
                out,
                tension,
                beats=beats,
                bpm=bpm,
                classified_hits=classified_hits,
                tension_cutoff=float(preset.get("arcade_phantom_tension_max", 0.10) or 0.10),
            )

        if bool(preset.get("arcade_texture_downsample", True)) and classified_hits:
            out, recap["texture_removed"] = apply_texture_downsampling(
                out,
                classified_hits=classified_hits,
                beats=beats,
                bpm=bpm,
                genre_label=genre_label,
                preset=preset,
            )

    elif phase_key == "post_section":
        if bool(preset.get("arcade_backbeat", True)) and tension:
            out, recap["backbeat_added"] = apply_backbeat_reconstruction(
                out,
                tension,
                candidate_events=candidate_events,
                beats=beats,
                bpm=bpm,
                genre_label=genre_label,
                preset=preset,
                snare_times=snare_times,
            )

    if verbose and any(recap.values()):
        print(
            f"[DrumGen][arcade][{phase_key}] phantom=-{recap['phantom_removed']} "
            f"backbeat=+{recap['backbeat_added']} texture=-{recap['texture_removed']}"
        )
    return out, recap
