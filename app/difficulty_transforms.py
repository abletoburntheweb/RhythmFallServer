# app/difficulty_transforms.py — Difficulty axis: pattern transforms after style build.
"""See docs/gen_styles.md — Difficulty adapts the raw style map, not just caps."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .drum_hit_detector import resolve_drum_at_time

TEXTURE_CLASSES = frozenset({"hat", "cymbal", "tom"})
CORE_CLASSES = frozenset({"kick", "snare"})

_CLASS_PRIORITY = {
    "kick": 100,
    "snare": 95,
    "tom": 60,
    "hat": 35,
    "cymbal": 30,
}


def _event_priority(
    time_val: float,
    classified_hits: Optional[List[Dict]],
    tolerance: float,
) -> float:
    drum = str(resolve_drum_at_time(time_val, classified_hits, tolerance=tolerance) or "").strip().lower()
    return float(_CLASS_PRIORITY.get(drum, 50))


def _has_near_time(time_val: float, others: Sequence[float], tol: float) -> bool:
    for other in others:
        if abs(float(other) - float(time_val)) <= tol:
            return True
    return False


def trim_to_event_ratio(
    events: List[float],
    *,
    ratio: float,
    classified_hits: Optional[List[Dict]],
    tolerance: float = 0.06,
) -> Tuple[List[float], int]:
    """Keep highest-priority events when standard must sit below dense without cap binding."""
    if not events:
        return [], 0
    target_ratio = max(0.5, min(1.0, float(ratio)))
    target_count = max(1, int(round(len(events) * target_ratio)))
    if target_count >= len(events):
        return sorted(events), 0

    scored = [
        (_event_priority(t, classified_hits, tolerance), float(t))
        for t in events
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    kept = sorted(t for _prio, t in scored[:target_count])
    return kept, len(events) - len(kept)


def topup_texture_events(
    events: List[float],
    *,
    classified_hits: Optional[List[Dict]],
    beats: np.ndarray,
    bpm: float,
    topup_fraction: float = 0.12,
    tolerance: float = 0.06,
) -> Tuple[List[float], int]:
    """Add off-detector texture hits so dense stays above standard when caps do not bind."""
    if not classified_hits or topup_fraction <= 0:
        return sorted(events), 0

    beat_interval = _beat_interval(beats, bpm)
    merge_tol = max(0.04, beat_interval * 0.08)
    kept = sorted(set(events))
    max_add = max(1, int(round(len(kept) * float(topup_fraction))))
    candidates: List[float] = []
    for hit in classified_hits:
        drum = str(hit.get("drum", "")).strip().lower()
        if drum not in TEXTURE_CLASSES:
            continue
        try:
            t = float(hit.get("time", 0.0))
        except (TypeError, ValueError):
            continue
        if _has_near_time(t, kept, merge_tol) or _has_near_time(t, candidates, merge_tol):
            continue
        candidates.append(t)

    candidates.sort()
    added = 0
    for t in candidates:
        if added >= max_add:
            break
        kept.append(t)
        added += 1
    return sorted(kept), added


def normalize_difficulty(value: Optional[str]) -> str:
    key = str(value or "standard").strip().lower()
    if key in ("relaxed", "easy"):
        return "relaxed"
    if key in ("dense", "hard", "expert"):
        return "dense"
    return "standard"


def _beat_interval(beats: np.ndarray, bpm: float) -> float:
    if beats is not None and len(beats) >= 2:
        return float(np.median(np.diff(beats)))
    return 60.0 / max(1.0, float(bpm))


def _quantize_time_to_grid(time_val: float, beats: np.ndarray, bpm: float, division: int) -> float:
    if beats is None or len(beats) < 2 or division <= 0:
        return time_val
    beat_interval = _beat_interval(beats, bpm)
    step = beat_interval / float(division)
    first = float(beats[0])
    idx = round((float(time_val) - first) / step)
    return first + idx * step


def quantize_fast_runs(
    events: List[float],
    *,
    beats: np.ndarray,
    bpm: float,
    min_run: int = 3,
    fast_division: int = 16,
    target_division: int = 8,
    tol: float = 0.012,
) -> Tuple[List[float], int]:
    """Merge runs of >= min_run events closer than 1/16 beat into 1/8 grid."""
    if not events or fast_division <= target_division:
        return sorted(events), 0

    beat_interval = _beat_interval(beats, bpm)
    fast_step = beat_interval / float(fast_division)
    sorted_ev = sorted(events)
    out: List[float] = []
    merged = 0
    i = 0
    while i < len(sorted_ev):
        run = [sorted_ev[i]]
        j = i + 1
        while j < len(sorted_ev) and (sorted_ev[j] - run[-1]) <= fast_step + tol:
            run.append(sorted_ev[j])
            j += 1
        if len(run) >= min_run:
            for t in run:
                qt = _quantize_time_to_grid(t, beats, bpm, target_division)
                if not out or abs(qt - out[-1]) > tol:
                    out.append(qt)
            merged += max(0, len(run) - len({round(x, 4) for x in out[-len(run):]}))
        else:
            out.extend(run)
        i = j
    return sorted(out), merged


def strip_ghost_notes(
    events: List[float],
    *,
    classified_hits: Optional[List[Dict]],
    beats: np.ndarray,
    bpm: float,
    tolerance: float = 0.06,
    aggressive: bool = False,
) -> Tuple[List[float], int]:
    """Drop weak texture hits that sit between strong beats without kick/snare support."""
    if not events or not classified_hits:
        return sorted(events), 0

    beat_interval = _beat_interval(beats, bpm)
    strong_tol = beat_interval * 0.22
    kept: List[float] = []
    removed = 0

    for t in sorted(events):
        drum = resolve_drum_at_time(t, classified_hits, tolerance=tolerance)
        drum_l = str(drum or "").strip().lower()
        if drum_l in CORE_CLASSES or not drum_l:
            kept.append(t)
            continue
        if drum_l not in TEXTURE_CLASSES:
            kept.append(t)
            continue

        on_strong = False
        if beats is not None and len(beats) > 0:
            for b in beats:
                if abs(float(t) - float(b)) <= strong_tol:
                    on_strong = True
                    break
                if abs(float(t) - float(b) - beat_interval * 2.0) <= strong_tol:
                    on_strong = True
                    break

        near_core = False
        for other in kept:
            if abs(other - t) > beat_interval:
                continue
            other_drum = resolve_drum_at_time(other, classified_hits, tolerance=tolerance)
            if str(other_drum or "").strip().lower() in CORE_CLASSES:
                near_core = True
                break

        if aggressive and not on_strong and not near_core:
            removed += 1
            continue
        if not aggressive and not on_strong and not near_core and drum_l in ("hat", "cymbal"):
            removed += 1
            continue
        kept.append(t)

    return sorted(kept), removed


def sparsify_by_beats(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    *,
    max_per_beat: int = 1,
) -> List[float]:
    if not events:
        return []
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
    from .drum_utils import apply_temporal_filter
    return apply_temporal_filter(sorted(events), beat_interval * 0.95)


def thin_standard_texture(
    events: List[float],
    *,
    classified_hits: Optional[List[Dict]],
    beats: np.ndarray,
    bpm: float,
    tolerance: float = 0.06,
) -> Tuple[List[float], int]:
    """Drop a share of off-beat texture hits so standard sits below dense."""
    if not events or not classified_hits:
        return sorted(events), 0

    beat_interval = _beat_interval(beats, bpm)
    strong_tol = beat_interval * 0.18
    kept: List[float] = []
    removed = 0
    off_beat_texture_idx = 0

    for t in sorted(events):
        drum = resolve_drum_at_time(t, classified_hits, tolerance=tolerance)
        drum_l = str(drum or "").strip().lower()
        if drum_l in CORE_CLASSES or drum_l not in TEXTURE_CLASSES:
            kept.append(t)
            continue

        on_strong = False
        if beats is not None and len(beats) > 0:
            for b in beats:
                if abs(float(t) - float(b)) <= strong_tol:
                    on_strong = True
                    break

        if on_strong:
            kept.append(t)
            continue

        off_beat_texture_idx += 1
        if off_beat_texture_idx % 2 == 0:
            removed += 1
            continue
        kept.append(t)

    return kept, removed


def apply_difficulty_transform(
    events: List[float],
    *,
    difficulty: Optional[str],
    goal: Optional[str],
    beats: np.ndarray,
    bpm: float,
    preset: Dict,
    classified_hits: Optional[List[Dict]] = None,
    verbose: bool = False,
) -> Tuple[List[float], Dict]:
    """Transform the style-built map for relaxed / standard / dense."""
    diff = normalize_difficulty(difficulty or preset.get("generation_difficulty"))
    goal_n = str(goal or preset.get("generation_goal", "original")).strip().lower()
    recap: Dict = {"difficulty": diff, "goal": goal_n}

    out = list(events)
    before = len(out)

    if diff == "relaxed":
        out, q_merged = quantize_fast_runs(out, beats=beats, bpm=bpm)
        recap["quantize_fast_runs"] = q_merged
        out, ghosts = strip_ghost_notes(
            out,
            classified_hits=classified_hits,
            beats=beats,
            bpm=bpm,
            aggressive=True,
        )
        recap["ghost_strip"] = ghosts
        if goal_n == "arcade":
            out, q2 = quantize_fast_runs(out, beats=beats, bpm=bpm, min_run=2)
            recap["arcade_relaxed_quantize"] = q2
        max_per_beat = 2 if bpm >= 165 else 1
        out = sparsify_by_beats(out, beats, bpm, max_per_beat=max_per_beat)
        recap["sparsify_max_per_beat"] = max_per_beat

    elif diff == "standard":
        out, q_merged = quantize_fast_runs(
            out,
            beats=beats,
            bpm=bpm,
            min_run=4 if bpm >= 165 else 5,
        )
        recap["quantize_fast_runs"] = q_merged
        out, ghosts = strip_ghost_notes(
            out,
            classified_hits=classified_hits,
            beats=beats,
            bpm=bpm,
            aggressive=False,
        )
        recap["ghost_strip"] = ghosts
        out, thinned = thin_standard_texture(
            out,
            classified_hits=classified_hits,
            beats=beats,
            bpm=bpm,
        )
        recap["standard_texture_thin"] = thinned
        ratio = float(preset.get("difficulty_event_ratio", 0.0) or 0.0)
        if ratio > 0.0 and ratio < 1.0:
            out, trimmed = trim_to_event_ratio(
                out,
                ratio=ratio,
                classified_hits=classified_hits,
            )
            recap["event_ratio_trim"] = trimmed
            recap["event_ratio"] = ratio

    else:
        recap["dense_passthrough"] = True
        if goal_n == "arcade":
            recap["arcade_dense"] = True
        topup = float(preset.get("difficulty_texture_topup", 0.0) or 0.0)
        if topup > 0.0:
            out, added = topup_texture_events(
                out,
                classified_hits=classified_hits,
                beats=beats,
                bpm=bpm,
                topup_fraction=topup,
            )
            recap["texture_topup"] = added
            recap["texture_topup_fraction"] = topup

    recap["before"] = before
    recap["after"] = len(out)
    if verbose:
        print(f"[DrumGen][difficulty] {diff} {before}->{len(out)} recap={recap}")
    return out, recap
