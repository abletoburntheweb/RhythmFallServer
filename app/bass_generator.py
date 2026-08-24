# app/bass_generator.py — Bass line chart generation (v0.1: onsets + pitch, pyin fallback).
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.bass_utils import pitch_to_lane, robust_midi_range, save_bass_notes
from app.drum_utils import CANONICAL_MAX_LANES, dedupe_notes_same_lane_same_time

# Per-generate caches — HPSS/onset_strength and RMS profiles are expensive on long stems.
_ONSET_ENV_CACHE: Dict[Tuple[int, int, int, int], Tuple[Any, int]] = {}
_STEM_ENERGY_CACHE: Dict[Tuple[int, int, int, int], Tuple[Any, Any, float]] = {}


def _clear_bass_analysis_caches() -> None:
    _ONSET_ENV_CACHE.clear()
    _STEM_ENERGY_CACHE.clear()


def _beat_interval(bpm: float) -> float:
    return 60.0 / max(float(bpm), 60.0)


def _quantize_time(
    t: float,
    bpm: float,
    strength: float = 0.85,
    *,
    phase: float = 0.0,
    prefer_eighths: bool = False,
    prefer_quarters: bool = False,
    prefer_downbeats: bool = False,
    bar_phase: Optional[float] = None,
    meters: int = 4,
) -> float:
    """Soft-snap ``t`` toward a beat-phased grid.

    Bass often lands on beats / downbeats. Preference order when enabled:
    downbeat (1) → quarter → 8th → 16th.
    """
    beat = _beat_interval(bpm)
    if beat <= 0 or strength <= 0.0:
        return float(t)
    strength = float(max(0.0, min(1.0, strength)))
    sixteenth = beat / 4.0
    eighth = beat / 2.0
    rel = float(t) - float(phase)
    grid_16 = float(phase) + round(rel / sixteenth) * sixteenth
    grid_8 = float(phase) + round(rel / eighth) * eighth
    grid_4 = float(phase) + round(rel / beat) * beat
    grid = grid_16

    if prefer_downbeats and bar_phase is not None and meters > 0:
        bar = beat * float(meters)
        if bar > 0:
            rel_bar = float(t) - float(bar_phase)
            grid_1 = float(bar_phase) + round(rel_bar / bar) * bar
            # ~1/3 of a beat: close enough that the ear still hears "the one".
            if abs(float(t) - grid_1) <= beat * 0.34:
                grid = grid_1
                return float(t * (1.0 - strength) + grid * strength)

    if prefer_quarters and abs(float(t) - grid_4) <= sixteenth * 1.15:
        grid = grid_4
    elif prefer_eighths and abs(float(t) - grid_8) <= sixteenth * 0.65:
        grid = grid_8
    elif prefer_eighths:
        grid = grid_16 if abs(float(t) - grid_16) <= abs(float(t) - grid_8) else grid_8
    else:
        grid = grid_16
    return float(t * (1.0 - strength) + grid * strength)


def _estimate_downbeat_phase(
    beats: Any,
    onsets: List[float],
    *,
    beat_interval: float,
    meters: int = 4,
    stem_times: Optional[Any] = None,
    stem_rms: Optional[Any] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Pick which beat-tracker index is musical '1' via bass onset agreement.

    When onset scores for ``k`` and ``k+2`` (half-bar apart) are close, break the
    tie with stem RMS energy on those candidate downbeats — avoids locking the
    chart to musical beat 3 as if it were 1.
    """
    recap: Dict[str, Any] = {"meters": meters, "k": 0, "score": 0.0}
    try:
        beats_arr = np.asarray(beats, dtype=float)
    except Exception:
        return 0.0, {**recap, "reason": "no_beats"}
    if len(beats_arr) < meters * 3 or beat_interval <= 0:
        phase0 = float(beats_arr[0]) if len(beats_arr) else 0.0
        return phase0, {**recap, "reason": "short_beats", "bar_phase": round(phase0, 4)}
    onset_arr = np.asarray(sorted(float(t) for t in onsets), dtype=float)
    if onset_arr.size < 8:
        phase0 = float(beats_arr[0])
        return phase0, {**recap, "reason": "few_onsets", "bar_phase": round(phase0, 4)}

    window = beat_interval * 0.28
    best_k = 0
    best_score = -1.0
    scores: List[float] = []
    for k in range(meters):
        downs = beats_arr[k::meters]
        score = 0.0
        for d in downs:
            dist = np.abs(onset_arr - float(d))
            near = dist[dist <= window]
            if near.size:
                score += float(np.sum(np.exp(-near * 35.0)))
        scores.append(round(score, 2))
        if score > best_score:
            best_score = score
            best_k = k

    tie_k = (best_k + 2) % meters
    tie_score = scores[tie_k] if tie_k < len(scores) else 0.0
    used_stem_tiebreak = False
    if (
        best_score > 0
        and tie_score > 0
        and abs(best_score - tie_score) / max(best_score, 1e-6) < 0.12
        and stem_times is not None
        and stem_rms is not None
    ):
        energy: List[float] = []
        for k in range(meters):
            downs = beats_arr[k::meters]
            vals = [
                _stem_rms_near(stem_times, stem_rms, float(d), window_sec=0.04)
                for d in downs[:48]
            ]
            energy.append(float(np.median(vals)) if vals else 0.0)
        e_best = energy[best_k]
        e_tie = energy[tie_k]
        recap["stem_energy"] = [round(e, 5) for e in energy]
        if e_tie > e_best * 1.08:
            best_k = tie_k
            best_score = scores[best_k]
            used_stem_tiebreak = True

    bar_phase = float(beats_arr[best_k])
    recap.update(
        {
            "k": best_k,
            "score": round(best_score, 2),
            "scores": scores,
            "bar_phase": round(bar_phase, 4),
            "stem_tiebreak": int(used_stem_tiebreak),
        }
    )
    return bar_phase, recap


def _grid_median_stem_rms(
    times: Any,
    rms: Any,
    *,
    phase: float,
    step: float,
    t_max: float,
    window_sec: float = 0.04,
    max_points: int = 64,
) -> float:
    """Median stem RMS at ``phase + n*step`` grid points covering ``[0, t_max]``."""
    if times is None or rms is None or step <= 0:
        return 0.0
    start = float(phase)
    while start > 0.0:
        start -= step
    vals: List[float] = []
    t = start
    while t < t_max + step and len(vals) < max_points:
        if t >= -0.02:
            vals.append(_stem_rms_near(times, rms, t, window_sec=window_sec))
        t += step
    if not vals:
        return 0.0
    return float(np.median(vals))


def _maybe_flip_half_bar_phase(
    bar_phase: float,
    notes: List[Dict[str, Any]],
    y,
    *,
    sr: int,
    bpm: float,
    meters: int = 4,
    onsets: Optional[List[float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """If stem energy is louder on beat-3 grid than beat-1, flip phase by half bar."""
    recap: Dict[str, Any] = {"flipped": 0}
    if y is None or not notes:
        return bar_phase, {**recap, "reason": "no_data"}
    times, rms, thresh = _stem_energy_profile(y, sr=sr)
    if times is None or rms is None or thresh <= 0.0:
        return bar_phase, {**recap, "reason": "no_energy"}
    beat = _beat_interval(bpm)
    bar = beat * float(max(1, meters))
    t_max = max(float(n.get("time", 0.0)) for n in notes) + bar
    e0 = _grid_median_stem_rms(
        times, rms, phase=float(bar_phase), step=bar, t_max=t_max, window_sec=0.04
    )
    e2 = _grid_median_stem_rms(
        times,
        rms,
        phase=float(bar_phase) + beat * 2.0,
        step=bar,
        t_max=t_max,
        window_sec=0.04,
    )
    recap.update({"rms_beat1": round(e0, 5), "rms_beat3": round(e2, 5)})
    # Require clear RMS win; optionally confirm with onset counts near each grid.
    rms_wins = e2 > e0 * 1.18 and e2 >= thresh * 1.08
    onset_ok = True
    if onsets and rms_wins:
        onset_arr = np.asarray(sorted(float(t) for t in onsets), dtype=float)
        win = beat * 0.22

        def _count_near(phase: float) -> int:
            if onset_arr.size == 0:
                return 0
            start = float(phase)
            while start > 0:
                start -= bar
            n = 0
            t = start
            while t < t_max + bar:
                if t >= -0.02 and np.any(np.abs(onset_arr - t) <= win):
                    n += 1
                t += bar
            return n

        c0 = _count_near(float(bar_phase))
        c2 = _count_near(float(bar_phase) + beat * 2.0)
        recap["onset_beat1"] = c0
        recap["onset_beat3"] = c2
        onset_ok = c2 >= c0  # don't flip if current '1' already has more attacks
    if rms_wins and onset_ok:
        new_phase = float(bar_phase) + beat * 2.0
        recap["flipped"] = 1
        recap["bar_phase"] = round(new_phase, 4)
        return new_phase, recap
    return bar_phase, recap


def _anchor_notes_to_downbeats(
    notes: List[Dict[str, Any]],
    onsets: List[float],
    *,
    bar_phase: float,
    bpm: float,
    meters: int = 4,
    pull_strength: float = 0.92,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """If a bass attack sits on '1', pull the nearest chart note onto that downbeat."""
    if not notes or bar_phase is None:
        return notes, {"anchored": 0}
    beat = _beat_interval(bpm)
    bar = beat * float(max(1, meters))
    if bar <= 0:
        return notes, {"anchored": 0}
    onset_arr = np.asarray(sorted(float(t) for t in onsets), dtype=float)
    if onset_arr.size == 0:
        return notes, {"anchored": 0, "reason": "no_onsets"}

    times = [float(n.get("time", 0.0)) for n in notes]
    t_max = max(times) + bar
    # Build downbeat grid covering the chart.
    start = float(bar_phase)
    while start > -bar:
        start -= bar
    downs = np.arange(start, t_max + bar, bar)
    out = [dict(n) for n in notes]
    anchored = 0
    used_note: set = set()
    attack_window = beat * 0.30
    note_window = beat * 0.45

    for d in downs:
        d = float(d)
        if d < -0.05:
            continue
        near_on = onset_arr[np.abs(onset_arr - d) <= attack_window]
        if near_on.size == 0:
            continue
        # Audible attack really is on/near this downbeat.
        best_i = -1
        best_dist = note_window
        for i, n in enumerate(out):
            if i in used_note:
                continue
            dist = abs(float(n.get("time", 0.0)) - d)
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i < 0:
            continue
        note = out[best_i]
        old_t = float(note.get("time", 0.0))
        if abs(old_t - d) < 0.006:
            used_note.add(best_i)
            continue
        new_t = old_t * (1.0 - pull_strength) + d * pull_strength
        delta = new_t - old_t
        note["time"] = round(max(0.0, new_t), 4)
        if note.get("end") is not None:
            try:
                note["end"] = round(
                    max(float(note["end"]) + delta, float(note["time"]) + 0.02), 4
                )
            except (TypeError, ValueError):
                pass
        out[best_i] = note
        used_note.add(best_i)
        anchored += 1

    return out, {"anchored": anchored, "downbeats_checked": int(len(downs))}


def _note_event_times(notes: List[Dict[str, Any]]) -> List[float]:
    times: List[float] = []
    for note in notes:
        try:
            times.append(float(note.get("time", 0.0)))
        except (TypeError, ValueError):
            continue
    return times


def _print_bass_timing_offset_diagnostics(
    label: str,
    events: List[float],
    beats: Any,
    beat_interval: float,
) -> None:
    """Median signed lag of note times vs beat-phased 8th/16th grids (ms)."""
    if not events or beats is None:
        return
    try:
        beats_arr = np.asarray(beats, dtype=float)
    except Exception:
        return
    if len(beats_arr) < 2 or beat_interval <= 0:
        return

    max_time = max(float(events[-1]), float(beats_arr[-1]))
    phase = float(beats_arr[0])
    eighth_step = beat_interval / 2.0
    sixteenth_step = beat_interval / 4.0
    # Extend BEFORE beats[0] — intro notes must compare to the same phased grid
    # that quantize uses (round of negative offsets), not only arange(phase, …).
    def _phased_grid(step: float) -> np.ndarray:
        if step <= 0:
            return np.array([])
        start = phase
        while start > -step:
            start -= step
        return np.arange(start, max_time + beat_interval, step)

    eighth_grid = _phased_grid(eighth_step)
    sixteenth_grid = _phased_grid(sixteenth_step)
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
        f"[BassGen][RhythmDiag] timing_{label}_8th_ms median={np.median(arr8):.1f} "
        f"p10={np.percentile(arr8, 10):.1f} p90={np.percentile(arr8, 90):.1f} "
        f"late={late8:.0f}%"
    )
    print(
        f"[BassGen][RhythmDiag] timing_{label}_16th_ms median={np.median(arr16):.1f} "
        f"p10={np.percentile(arr16, 10):.1f} p90={np.percentile(arr16, 90):.1f} "
        f"late={late16:.0f}%"
    )


def _shift_note_times(notes: List[Dict[str, Any]], delta_sec: float) -> List[Dict[str, Any]]:
    """Apply a constant time shift to note heads (and ends)."""
    if abs(delta_sec) < 1e-6:
        return notes
    out: List[Dict[str, Any]] = []
    for note in notes:
        n = dict(note)
        t = float(n.get("time", 0.0)) + delta_sec
        n["time"] = round(max(0.0, t), 4)
        if n.get("end") is not None:
            try:
                n["end"] = round(max(float(n["end"]) + delta_sec, float(n["time"]) + 0.02), 4)
            except (TypeError, ValueError):
                pass
        out.append(n)
    return out


def _phased_grid_times(
    phase: float,
    step: float,
    t_max: float,
) -> np.ndarray:
    """Beat-phased grid extending before ``phase`` (same as quantize / diag)."""
    if step <= 0:
        return np.array([], dtype=float)
    start = float(phase)
    while start > -step:
        start -= step
    return np.arange(start, float(t_max) + step, step)


def _median_grid_offset_sec(
    events: List[float],
    *,
    phase: float,
    step: float,
    t_max: Optional[float] = None,
) -> float:
    if not events or step <= 0:
        return 0.0
    tmax = float(t_max) if t_max is not None else max(float(events[-1]), 1.0)
    grid = _phased_grid_times(phase, step, tmax)
    if grid.size == 0:
        return 0.0
    offsets = []
    for t in events:
        g = float(grid[int(np.argmin(np.abs(grid - float(t))))])
        offsets.append(float(t) - g)
    if not offsets:
        return 0.0
    return float(np.median(np.asarray(offsets, dtype=float)))


def _median_16th_offset_sec(events: List[float], beats: Any, beat_interval: float) -> float:
    if not events or beats is None or beat_interval <= 0:
        return 0.0
    try:
        beats_arr = np.asarray(beats, dtype=float)
    except Exception:
        return 0.0
    if len(beats_arr) < 2:
        return 0.0
    phase = float(beats_arr[0])
    max_time = max(float(events[-1]), float(beats_arr[-1]))
    return _median_grid_offset_sec(
        events, phase=phase, step=beat_interval / 4.0, t_max=max_time
    )


def _median_onset_lag_sec(
    events: List[float],
    onsets: List[float],
    *,
    window_sec: float = 0.090,
) -> float:
    """Median (note - nearest onset); positive ⇒ chart late vs stem attacks."""
    if not events or not onsets or window_sec <= 0:
        return 0.0
    onset_arr = np.asarray(sorted(float(t) for t in onsets), dtype=float)
    lags: List[float] = []
    for t in events:
        idx = int(np.argmin(np.abs(onset_arr - float(t))))
        lag = float(t) - float(onset_arr[idx])
        if abs(lag) <= window_sec:
            lags.append(lag)
    if len(lags) < max(6, len(events) // 8):
        return 0.0
    return float(np.median(np.asarray(lags, dtype=float)))


def _shape_counts(notes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"tap": 0, "hold": 0, "slide": 0, "ghost": 0}
    for note in notes:
        shape = str(note.get("shape", "tap")).lower()
        if shape in counts:
            counts[shape] += 1
        if note.get("ghost"):
            counts["ghost"] += 1
    return counts


def _local_event_density(times: List[float], t: float, window: float) -> int:
    if not times or window <= 0:
        return 0
    return sum(1 for x in times if abs(float(x) - float(t)) <= window)


def _demote_holds_for_density(
    notes: List[Dict[str, Any]],
    *,
    bpm: float,
    groove_class: str = "mixed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Only demote tiny stub holds — never shred the shared backbone.

    Dense difficulty must ADD notes (gap_fill / accents), not rewrite holds→taps.
    """
    if not notes:
        return [], {"demoted_short": 0, "demoted_dense": 0, "after": 0}
    beat = _beat_interval(bpm)
    # Stub only — real holds stay across relaxed/standard/dense.
    short_hold_max = beat * 0.22 if groove_class == "sustain" else beat * 0.28
    sorted_notes = sorted(notes, key=lambda n: float(n.get("time", 0.0)))
    out: List[Dict[str, Any]] = []
    demoted_short = 0
    for note in sorted_notes:
        shape = str(note.get("shape", "tap")).strip().lower()
        if shape != "hold":
            out.append(note)
            continue
        t0 = float(note.get("time", 0.0))
        t1 = float(note.get("end", t0))
        dur = max(0.0, t1 - t0)
        if dur <= short_hold_max + 1e-6:
            tap = {k: v for k, v in note.items() if k not in ("end", "lane_end", "curve")}
            tap["shape"] = "tap"
            if tap.get("type") == "BassHoldNote":
                tap["type"] = "BassTapNote"
            out.append(tap)
            demoted_short += 1
            continue
        out.append(note)
    return out, {
        "demoted_short": demoted_short,
        "demoted_dense": 0,
        "after": len(out),
        "hold_left": sum(1 for n in out if str(n.get("shape", "")).lower() == "hold"),
        "groove": groove_class,
        "policy": "backbone_preserve",
    }


def _harden_quant_in_dense_windows(
    notes: List[Dict[str, Any]],
    *,
    bpm: float,
    phase: float = 0.0,
    min_density: int = 3,
    strength: float = 0.92,
) -> Tuple[List[Dict[str, Any]], int]:
    """Snap busy notes harder onto the 8th/16th grid so metal grooves don't float."""
    if not notes:
        return [], 0
    beat = _beat_interval(bpm)
    window = beat * 0.55
    times = [float(n.get("time", 0.0)) for n in notes]
    moved = 0
    out: List[Dict[str, Any]] = []
    strength = float(max(0.0, min(1.0, strength)))
    for note in notes:
        # Stem peak-lock wins — don't pull locked heads back onto the grid.
        if note.get("peak_locked"):
            out.append(note)
            continue
        t0 = float(note.get("time", 0.0))
        dens = _local_event_density(times, t0, window)
        if dens < min_density:
            out.append(note)
            continue
        # Spam runs (4+ in a half-beat): lock to 16ths; lighter runs keep 8th preference.
        prefer_8 = dens < 4
        snapped = _quantize_time(
            t0,
            bpm,
            strength=strength if dens < 4 else min(0.98, strength + 0.04),
            phase=phase,
            prefer_eighths=prefer_8,
            prefer_quarters=False,
            prefer_downbeats=False,
        )
        if dens >= 4:
            # Force pure 16th magnet for gallops / 16th spam.
            sixteenth = beat / 4.0
            rel = t0 - float(phase)
            grid_16 = float(phase) + round(rel / sixteenth) * sixteenth
            snapped = float(t0 * (1.0 - min(0.98, strength + 0.04)) + grid_16 * min(0.98, strength + 0.04))
        if abs(snapped - t0) < 0.0005:
            out.append(note)
            continue
        n = dict(note)
        n["time"] = round(snapped, 4)
        if n.get("end") is not None:
            dur = float(note.get("end", t0)) - t0
            n["end"] = round(snapped + max(0.02, dur), 4)
        out.append(n)
        moved += 1
    return out, moved


def _lane_distribution(notes: List[Dict[str, Any]], lanes: int = 5) -> Dict[int, int]:
    dist: Dict[int, int] = {i: 0 for i in range(max(1, lanes))}
    for note in notes:
        lane_raw = note.get("lane")
        if lane_raw is None and isinstance(note.get("lanes"), list) and note["lanes"]:
            lane_raw = note["lanes"][0]
        lane = int(lane_raw if lane_raw is not None else 0)
        lane = max(0, min(lanes - 1, lane))
        dist[lane] = dist.get(lane, 0) + 1
    return dist


def _timing_stats(
    notes: List[Dict[str, Any]],
    bpm: float,
    *,
    phase: float = 0.0,
) -> Dict[str, Any]:
    if not notes:
        return {}
    times = [float(n.get("time", 0.0)) for n in notes]
    beat = _beat_interval(bpm)
    step = beat / 4.0
    grid_ms: List[float] = []
    for t in times:
        if step > 0:
            rel = float(t) - float(phase)
            grid = float(phase) + round(rel / step) * step
            grid_ms.append(abs(t - grid) * 1000.0)
    hold_ms: List[float] = []
    for note in notes:
        shape = str(note.get("shape", "tap")).lower()
        if shape in ("hold", "slide") and note.get("end") is not None:
            hold_ms.append(max(0.0, float(note["end"]) - float(note.get("time", 0.0))) * 1000.0)
    out: Dict[str, Any] = {
        "first_s": round(min(times), 3),
        "last_s": round(max(times), 3),
        "span_s": round(max(times) - min(times), 3),
        "phase_s": round(float(phase), 4),
    }
    if grid_ms:
        out["grid_median_ms"] = round(float(np.median(grid_ms)), 1)
        out["grid_p90_ms"] = round(float(np.percentile(grid_ms, 90)), 1)
    if hold_ms:
        out["hold_median_ms"] = round(float(np.median(hold_ms)), 1)
        out["hold_p90_ms"] = round(float(np.percentile(hold_ms, 90)), 1)
    return out


def _print_bass_recap(
    *,
    notes: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    segment_source: str,
    bpm: float,
    lanes: int,
    goal: str,
    difficulty: str,
    style_recap: Dict[str, Any],
    diff_recap: Dict[str, Any],
    onsets: int = 0,
    phase: float = 0.0,
) -> None:
    shapes = _shape_counts(notes)
    lanes_dist = _lane_distribution(notes, lanes)
    timing = _timing_stats(notes, bpm, phase=phase)
    print("[BassGen] ---------- bass summary (copy from here) ----------")
    print(
        f"[BassGen] goal={goal} difficulty={difficulty} bpm={bpm:g} lanes={lanes} "
        f"source={segment_source} segments={len(segments)} onsets={onsets}"
    )
    print(f"[BassGen] shapes={shapes} lane_dist={lanes_dist}")
    if timing:
        print(
            f"[BassGen] timing first={timing.get('first_s')}s last={timing.get('last_s')}s "
            f"span={timing.get('span_s')}s phase={timing.get('phase_s')}s "
            f"grid_med={timing.get('grid_median_ms', '-')}ms "
            f"hold_med={timing.get('hold_median_ms', '-')}ms"
        )
    print(f"[BassGen][style] {style_recap}")
    print(f"[BassGen][difficulty] {diff_recap}")
    if notes:
        sample = notes[:3]
        print(f"[BassGen] sample_notes={sample}")
    print("[BassGen] ---------- end bass summary ----------")


def _load_audio(audio_path: str, *, sr: int = 22050) -> Tuple[Any, int]:
    try:
        import librosa
    except ImportError:
        return None, sr
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=900)
    except Exception:
        return None, sr
    return y, sr


def _extract_pitch_track(
    y,
    *,
    sr: int,
    bpm: float,
) -> List[Tuple[float, float, float]]:
    """Return (time_s, midi, rms) samples on a coarse grid."""
    try:
        import librosa
    except ImportError:
        return []
    if y is None or len(y) < sr:
        return []
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("E1"),
        fmax=librosa.note_to_hz("G4"),
        sr=sr,
        hop_length=hop,
    )
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    out: List[Tuple[float, float, float]] = []
    for i, t in enumerate(times):
        if i >= len(f0):
            break
        voiced = True
        if voiced_flag is not None:
            voiced = bool(voiced_flag[i])
        if not voiced or f0[i] is None or np.isnan(f0[i]):
            continue
        hz = float(f0[i])
        if hz <= 0:
            continue
        midi = float(librosa.hz_to_midi(hz))
        amp = float(rms[i]) if i < len(rms) else 0.0
        out.append((float(t), midi, amp))
    return out


def _pitch_at_time(
    samples: List[Tuple[float, float, float]],
    center_t: float,
    *,
    window_s: float = 0.14,
) -> Tuple[Optional[float], float]:
    lo = center_t - 0.02
    hi = center_t + window_s
    pitches: List[float] = []
    amps: List[float] = []
    for t, midi, amp in samples:
        if t < lo or t > hi:
            continue
        pitches.append(midi)
        amps.append(amp)
    if not pitches:
        return None, 0.0
    return float(np.median(pitches)), float(max(amps) if amps else 0.0)


def _bass_onset_envelope(y, *, sr: int, hop: int = 256) -> Tuple[Any, int]:
    """Percussive+full onset-strength curve for pluck localization.

    Cached per audio buffer — HPSS is expensive and was run twice (onsets + envelope fuse).
    """
    try:
        import librosa
    except ImportError:
        return None, hop
    if y is None or len(y) < max(sr // 4, hop * 4):
        return None, hop
    cache_key = (id(y), int(sr), int(hop), int(len(y)))
    cached = _ONSET_ENV_CACHE.get(cache_key)
    if cached is not None:
        return cached
    t0 = time.perf_counter()
    # margin=1.5 is enough for pluck emphasis; 2.5 was much slower for little gain.
    _, y_perc = librosa.effects.hpss(y, margin=1.5)
    env = librosa.onset.onset_strength(
        y=y_perc, sr=sr, hop_length=hop, aggregate=np.median
    )
    env_full = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop, aggregate=np.median
    )
    if env_full.size == env.size:
        env = np.maximum(env, env_full * 0.9)
    if env.size == 0 or float(np.max(env)) <= 0:
        return None, hop
    print(f"[BassGen][perf] onset_envelope={time.perf_counter() - t0:.1f}s frames={env.size}")
    _ONSET_ENV_CACHE[cache_key] = (env, hop)
    return env, hop


def _extract_bass_onsets(y, *, sr: int, bpm: float) -> List[float]:
    """Pluck / attack times on the bass stem."""
    try:
        import librosa
    except ImportError:
        return []
    hop = 256
    onset_env, hop = _bass_onset_envelope(y, sr=sr, hop=hop)
    if onset_env is None:
        return []
    wait = max(2, int((sr / hop) * 0.035))
    delta = float(onset_env.max()) * 0.045
    onset_frames = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=delta,
        wait=wait,
    )
    if len(onset_frames) == 0:
        return []
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    beat = _beat_interval(bpm)
    min_gap = beat * 0.08
    filtered: List[float] = []
    for t in sorted(float(x) for x in onset_times):
        if not filtered or (t - filtered[-1]) >= min_gap:
            filtered.append(t)
    return filtered


def _bass_transcription_is_sparse(segments: List[Dict[str, Any]]) -> bool:
    """True when BP almost missed the line (typical for 808 / sub-bass sine)."""
    if not segments:
        return True
    starts = [float(s.get("start", 0.0)) for s in segments]
    ends = [
        float(s.get("end", s.get("start", 0.0)))
        for s in segments
    ]
    span = max(ends) - min(starts) if starts else 0.0
    n = len(segments)
    if n < 24:
        return True
    if span >= 30.0 and (span / float(n)) > 4.0:
        # Less than one note every 4 seconds across a long span.
        return True
    if span >= 60.0 and n < max(40, int(span / 3.0)):
        return True
    return False


def _is_808_pulse_groove(
    segments: List[Dict[str, Any]],
    *,
    bpm: float,
    groove_class: str = "mixed",
) -> bool:
    """Sparse low sustains — trap 808s that should sit on beats 1 & 3."""
    if not segments:
        return False
    beat = _beat_interval(bpm)
    midis = [float(s.get("midi", 60.0)) for s in segments]
    durs = [
        max(0.0, float(s.get("end", s.get("start", 0.0))) - float(s.get("start", 0.0)))
        for s in segments
    ]
    starts = sorted(float(s.get("start", 0.0)) for s in segments)
    span = max(1e-3, starts[-1] - starts[0]) if starts else 1.0
    npb = len(segments) / (span / beat)
    med_midi = float(np.median(np.asarray(midis, dtype=float)))
    med_dur = float(np.median(np.asarray(durs, dtype=float))) if durs else 0.0
    # Sub-bass register (≤ ~G1) + sparse + held = classic 808 pulse.
    if med_midi <= 43.0 and npb <= 0.60 and med_dur >= beat * 0.35:
        return True
    if groove_class == "sustain" and npb <= 0.38 and med_dur >= beat * 0.45:
        return True
    if _bass_transcription_is_sparse(segments) and med_midi <= 48.0:
        return True
    return False


def _snap_notes_to_odd_quarters(
    notes: List[Dict[str, Any]],
    *,
    bpm: float,
    bar_phase: float,
    strength: float = 0.85,
    onsets: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Magnet heads to beats 1 and 3 (half-note grid from downbeat)."""
    if not notes or strength <= 0.0:
        return notes, 0
    beat = _beat_interval(bpm)
    half = beat * 2.0
    if half <= 0:
        return notes, 0
    strength = float(max(0.0, min(1.0, strength)))
    onset_arr = (
        np.asarray(sorted(float(t) for t in onsets), dtype=float)
        if onsets
        else None
    )
    peak_guard = beat * 0.08
    moved = 0
    out: List[Dict[str, Any]] = []
    for note in notes:
        n = dict(note)
        t0 = float(n.get("time", 0.0))
        # Already on a clear stem pluck — don't yank toward 1/3 grid.
        if onset_arr is not None and onset_arr.size:
            if float(np.min(np.abs(onset_arr - t0))) <= peak_guard:
                out.append(n)
                continue
        rel = t0 - float(bar_phase)
        grid = float(bar_phase) + round(rel / half) * half
        # Narrow window: avoid pulling beat-2 material onto 1/3.
        if abs(t0 - grid) <= beat * 0.28:
            snapped = float(t0 * (1.0 - strength) + grid * strength)
            if abs(snapped - t0) >= 0.001:
                dur = 0.0
                if n.get("end") is not None:
                    dur = max(0.02, float(n["end"]) - t0)
                n["time"] = round(max(0.0, snapped), 4)
                if n.get("end") is not None:
                    n["end"] = round(float(n["time"]) + dur, 4)
                moved += 1
        out.append(n)
    return out, moved


def _estimate_bass_groove_class(
    segments: List[Dict[str, Any]],
    onsets: List[float],
    *,
    bpm: float,
) -> Dict[str, Any]:
    """Classify groove for *additive* densify strength — never to shred the backbone.

    Philosophy: one shared skeleton (BP holds/taps). Difficulty / plucky fixes only
    ADD notes on top. ``plucky`` means “many stem attacks inside BP sustains”
    (metal) → allow accent taps. ``sustain`` means real held notes (R&B) → don't.
    """
    beat = _beat_interval(bpm)
    if not segments:
        return {"class": "mixed", "reason": "empty"}
    durs = [
        max(0.0, float(s.get("end", s.get("start", 0.0))) - float(s.get("start", 0.0)))
        for s in segments
    ]
    starts = sorted(float(s.get("start", 0.0)) for s in segments)
    med_dur = float(np.median(np.asarray(durs, dtype=float))) if durs else 0.0
    gaps = np.diff(np.asarray(starts, dtype=float)) if len(starts) > 1 else np.asarray([beat])
    med_gap = float(np.median(gaps)) if gaps.size else beat
    long_ratio = sum(1 for d in durs if d >= beat * 0.55) / max(len(durs), 1)
    short_ratio = sum(1 for d in durs if d < beat * 0.28) / max(len(durs), 1)
    span = max(1e-3, starts[-1] - starts[0]) if starts else 1.0
    onset_per_beat = (len(onsets) / (span / beat)) if onsets else 0.0
    notes_per_beat = len(segments) / (span / beat)

    onset_arr = np.asarray(sorted(float(t) for t in onsets), dtype=float) if onsets else np.asarray([])
    long_segs = 0
    internal_hits = 0
    edge = max(0.055, beat * 0.12)
    for seg, dur in zip(segments, durs):
        if dur < beat * 0.50:
            continue
        long_segs += 1
        if onset_arr.size == 0:
            continue
        start = float(seg.get("start", 0.0))
        end = start + dur
        lo = start + edge
        hi = end - max(0.04, beat * 0.08)
        if hi <= lo:
            continue
        if np.any((onset_arr >= lo) & (onset_arr <= hi)):
            internal_hits += 1
    internal_ratio = internal_hits / max(long_segs, 1)

    # Metal: BP often one long MIDI while stem has several attacks inside.
    # Plucky first (DOMINATION ~0.22). Sustain next — Sundays was 0.177 and
    # fell into mixed, where fuse/gate/peak-lock shredded R&B holds.
    if internal_ratio >= 0.20 or internal_hits >= 40:
        klass = "plucky"
        reason = "internal_onsets"
    elif med_gap <= beat * 0.36 and onset_per_beat >= 0.9:
        klass = "plucky"
        reason = "tight_gaps"
    elif long_ratio >= 0.40 and internal_ratio < 0.20:
        klass = "sustain"
        reason = "held_line_few_internals"
    elif long_ratio >= 0.35 and internal_ratio < 0.16 and med_dur >= beat * 0.40:
        klass = "sustain"
        reason = "held_line_med_dur"
    else:
        klass = "mixed"
        reason = "balanced"

    return {
        "class": klass,
        "reason": reason,
        "med_dur_ms": round(med_dur * 1000.0, 1),
        "med_gap_ms": round(med_gap * 1000.0, 1),
        "long_ratio": round(long_ratio, 3),
        "short_ratio": round(short_ratio, 3),
        "internal_ratio": round(internal_ratio, 3),
        "internal_hits": internal_hits,
        "long_segs": long_segs,
        "onset_per_beat": round(onset_per_beat, 3),
        "notes_per_beat": round(notes_per_beat, 3),
        "beat_ms": round(beat * 1000.0, 1),
    }


def _add_internal_pluck_accents(
    segments: List[Dict[str, Any]],
    onsets: List[float],
    *,
    bpm: float,
    groove_class: str = "mixed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ADD short taps on stem attacks inside a BP sustain — keep the hold intact.

    Backbone philosophy: never rewrite a hold into N short notes. Dense / plucky
    layers put accents on top (midi nudged so lane mapping can differ from the hold).
    """
    if not segments or not onsets:
        return segments, {"added": 0, "reason": "skip"}
    if groove_class == "sustain":
        return segments, {"added": 0, "reason": "sustain_backbone"}
    onset_arr = np.asarray(sorted(float(t) for t in onsets), dtype=float)
    beat = _beat_interval(bpm)
    if groove_class == "plucky":
        min_dur = max(0.16, beat * 0.42)
        edge_pad = max(0.045, beat * 0.10)
        min_piece = max(0.055, beat * 0.12)
        max_add = 120
    else:
        min_dur = max(0.28, beat * 0.70)
        edge_pad = max(0.07, beat * 0.16)
        min_piece = max(0.08, beat * 0.18)
        max_add = 48

    starts = [float(s.get("start", 0.0)) for s in segments]
    extras: List[Dict[str, Any]] = []
    touched = 0
    for seg in segments:
        if len(extras) >= max_add:
            break
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.05))
        dur = end - start
        if dur < min_dur:
            continue
        lo = start + edge_pad
        hi = end - max(0.035, beat * 0.08)
        if hi <= lo:
            continue
        internals = onset_arr[(onset_arr >= lo) & (onset_arr <= hi)].tolist()
        cleaned: List[float] = []
        for t in internals:
            if not cleaned or (t - cleaned[-1]) >= min_piece:
                cleaned.append(float(t))
        if groove_class == "mixed" and len(cleaned) < 2:
            continue
        if not cleaned:
            continue
        touched += 1
        midi = float(seg.get("midi", 40.0))
        amp = float(seg.get("amp", seg.get("amp_mean", 0.5)))
        for t in cleaned:
            if len(extras) >= max_add:
                break
            # Skip if a BP head already sits here.
            if any(abs(t - s) <= min_piece * 0.85 for s in starts):
                continue
            if any(abs(t - float(e.get("start", 0.0))) <= min_piece * 0.85 for e in extras):
                continue
            extras.append(
                {
                    "start": round(t, 4),
                    "end": round(t + max(0.06, beat * 0.14), 4),
                    # Nudge pitch so accent tends to land on a neighbor lane (on top).
                    "midi": midi + 1.0,
                    "amp": amp,
                    "amp_mean": amp,
                    "accent": True,
                }
            )
    if not extras:
        return segments, {"added": 0, "touched_holds": 0, "after": len(segments), "groove": groove_class}
    merged = sorted(list(segments) + extras, key=lambda s: float(s.get("start", 0.0)))
    return merged, {
        "added": len(extras),
        "touched_holds": touched,
        "after": len(merged),
        "groove": groove_class,
        "mode": "additive",
    }


def _fuse_segments_to_pluck_onsets(
    segments: List[Dict[str, Any]],
    onsets: List[float],
    *,
    lookback_sec: float = 0.130,
    lookahead_sec: float = 0.018,
    groove_class: str = "mixed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pull Basic Pitch starts back to discrete pluck peaks."""
    if not segments:
        return segments, {"pulled": 0, "candidates": 0}
    if not onsets:
        return segments, {"pulled": 0, "candidates": 0, "reason": "no_onsets"}

    # Sustain grooves: soft envelope bumps ≠ attacks — only tiny corrections.
    if groove_class == "sustain":
        return segments, {
            "pulled": 0,
            "candidates": len(segments),
            "reason": "bp_faithful_sustain",
            "groove": groove_class,
        }
    if groove_class == "mixed":
        lookback_sec = min(lookback_sec, 0.090)

    onset_arr = np.asarray(sorted(float(t) for t in onsets), dtype=float)
    out: List[Dict[str, Any]] = []
    pulls_ms: List[float] = []
    pulled = 0
    for seg in segments:
        note = dict(seg)
        start = float(note.get("start", 0.0))
        end = float(note.get("end", start + 0.05))
        lo = start - lookback_sec
        hi = start + lookahead_sec
        idxs = np.where((onset_arr >= lo) & (onset_arr <= hi))[0]
        if idxs.size > 0:
            cands = onset_arr[idxs]
            scores = []
            for t in cands:
                early = max(0.0, start - float(t))
                late = max(0.0, float(t) - start)
                if groove_class == "sustain":
                    # Prefer nearest peak; don't yank holds 70–100ms early.
                    scores.append(-abs(float(t) - start) * 3.0 - late * 1.5 - early * 0.4)
                else:
                    # Prefer earlier pluck peaks — BP bodies sit late on metal.
                    scores.append(early * 1.55 - late * 2.4 - abs(float(t) - start) * 0.20)
            best = float(cands[int(np.argmax(scores))])
            delta = best - start
            min_pull = 0.010 if groove_class == "sustain" else 0.004
            max_pull = 0.045 if groove_class == "sustain" else 0.130
            if min_pull <= abs(delta) <= max_pull:
                note["start"] = round(max(0.0, best), 4)
                note["end"] = round(max(float(note["start"]) + 0.03, end + delta * 0.15), 4)
                pulls_ms.append(delta * 1000.0)
                pulled += 1
        out.append(note)

    recap: Dict[str, Any] = {
        "pulled": pulled,
        "candidates": len(segments),
        "onset_pool": int(onset_arr.size),
        "method": "peaks",
        "groove": groove_class,
    }
    if pulls_ms:
        arr = np.asarray(pulls_ms, dtype=float)
        recap["pull_median_ms"] = round(float(np.median(arr)), 1)
        recap["pull_p10_ms"] = round(float(np.percentile(arr, 10)), 1)
        recap["pull_p90_ms"] = round(float(np.percentile(arr, 90)), 1)
    return out, recap


def _fuse_segments_to_onset_envelope(
    segments: List[Dict[str, Any]],
    y,
    *,
    sr: int,
    lookback_sec: float = 0.100,
    lookahead_sec: float = 0.015,
    only_unpulled: bool = False,
    already_pulled: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-note: move start to max onset-strength in a left-biased window.

    Works even when global peak_pick finds few/no discrete onsets (soft R&B bass).
    """
    try:
        import librosa
    except ImportError:
        return segments, {"pulled": 0, "reason": "no_librosa"}
    if not segments or y is None:
        return segments, {"pulled": 0, "candidates": 0}

    hop = 256
    env, hop = _bass_onset_envelope(y, sr=sr, hop=hop)
    if env is None:
        return segments, {"pulled": 0, "reason": "no_env"}

    frame_times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    pulled_set = already_pulled or set()
    out: List[Dict[str, Any]] = []
    pulls_ms: List[float] = []
    pulled = 0
    for i, seg in enumerate(segments):
        note = dict(seg)
        start = float(note.get("start", 0.0))
        end = float(note.get("end", start + 0.05))
        if only_unpulled and i in pulled_set:
            out.append(note)
            continue
        lo = start - lookback_sec
        hi = start + lookahead_sec
        mask = (frame_times >= lo) & (frame_times <= hi)
        if not np.any(mask):
            out.append(note)
            continue
        idxs = np.flatnonzero(mask)
        local = env[idxs].astype(float)
        peak_val = float(np.max(local))
        if peak_val <= 1e-8:
            out.append(note)
            continue
        # Prefer the earliest frame that reaches 85% of local peak (attack, not body).
        thresh = peak_val * 0.85
        strong = idxs[local >= thresh]
        best_i = int(strong[0]) if strong.size else int(idxs[int(np.argmax(local))])
        best_t = float(frame_times[best_i])
        delta = best_t - start
        # Only pull earlier (or tiny forward jitter) — BP late sustain is the bug.
        if delta <= -0.006 or (0.0 < delta <= 0.008):
            if abs(delta) >= 0.004:
                note["start"] = round(max(0.0, best_t), 4)
                note["end"] = round(max(float(note["start"]) + 0.03, end + min(0.0, delta) * 0.15), 4)
                pulls_ms.append(delta * 1000.0)
                pulled += 1
        out.append(note)

    recap: Dict[str, Any] = {
        "pulled": pulled,
        "candidates": len(segments),
        "method": "envelope",
    }
    if pulls_ms:
        arr = np.asarray(pulls_ms, dtype=float)
        recap["pull_median_ms"] = round(float(np.median(arr)), 1)
        recap["pull_p10_ms"] = round(float(np.percentile(arr, 10)), 1)
        recap["pull_p90_ms"] = round(float(np.percentile(arr, 90)), 1)
    return out, recap


def _stem_energy_profile(
    y,
    *,
    sr: int,
    hop: int = 512,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Return (frame_times, rms, silence_thresh). thresh=0 means unusable."""
    try:
        import librosa
    except ImportError:
        return None, None, 0.0
    if y is None or len(y) < max(sr // 4, 1024):
        return None, None, 0.0
    cache_key = (id(y), int(sr), int(hop), int(len(y)))
    cached = _STEM_ENERGY_CACHE.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    if rms.size < 4:
        return None, None, 0.0
    times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop)
    p90 = float(np.percentile(rms, 90))
    p15 = float(np.percentile(rms, 15))
    if p90 <= 1e-9:
        return times, rms, 0.0
    # Below this ≈ “player hears silence on the bass stem”.
    thresh = max(p15 * 1.15, p90 * 0.07)
    result = (times, rms, float(thresh))
    _STEM_ENERGY_CACHE[cache_key] = result
    return result


def _stem_rms_near(
    times: np.ndarray,
    rms: np.ndarray,
    t: float,
    *,
    window_sec: float = 0.03,
) -> float:
    if times is None or rms is None or len(times) == 0:
        return 0.0
    lo = float(t) - window_sec
    hi = float(t) + window_sec
    i0 = int(np.searchsorted(times, lo, side="left"))
    i1 = int(np.searchsorted(times, hi, side="right"))
    if i1 <= i0:
        idx = int(np.clip(np.searchsorted(times, float(t)), 0, len(times) - 1))
        return float(rms[idx])
    return float(np.median(rms[i0:i1]))


def _stem_is_audible(
    times: np.ndarray,
    rms: np.ndarray,
    thresh: float,
    t: float,
    *,
    window_sec: float = 0.03,
) -> bool:
    if thresh <= 0.0:
        return True
    return _stem_rms_near(times, rms, t, window_sec=window_sec) >= thresh


def _trim_end_at_silence(
    times: np.ndarray,
    rms: np.ndarray,
    thresh: float,
    start: float,
    end: float,
    *,
    min_silent_sec: float = 0.07,
    hop_guess: float = 0.0116,
) -> float:
    """Cut sustain when stem goes quiet for ~min_silent_sec (philosophy: silence wins)."""
    if thresh <= 0.0 or end <= start + 0.06 or times is None or len(times) == 0:
        return end
    # Only walk frames inside [start, end] — full-track scans per note hung dense charts.
    i0 = int(np.searchsorted(times, start + 0.04, side="left"))
    i1 = int(np.searchsorted(times, end, side="right"))
    if i1 <= i0:
        return end
    silent_run = 0.0
    cut_at = end
    prev_t = float(times[max(0, i0 - 1)]) if i0 > 0 else float(times[i0])
    for i in range(i0, i1):
        ft = float(times[i])
        dt = max(hop_guess, ft - prev_t)
        prev_t = ft
        if float(rms[i]) < thresh:
            silent_run += dt
            if silent_run >= min_silent_sec:
                cut_at = ft - silent_run
                break
        else:
            silent_run = 0.0
    return max(start + 0.05, min(end, cut_at))


def _gate_segments_by_stem_energy(
    segments: List[Dict[str, Any]],
    y,
    *,
    sr: int,
    bpm: float,
    groove_class: str = "mixed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Trim holds through silence. Drop silent heads only on plucky/mixed.

    Sustain / R&B: keep BP heads (soft notes near the floor) — dropping them
    was the main regression vs raw Basic Pitch.
    """
    times, rms, thresh = _stem_energy_profile(y, sr=sr)
    if times is None or rms is None or thresh <= 0.0:
        return segments, {"skipped": True}
    beat = _beat_interval(bpm)
    hop_guess = float(times[1] - times[0]) if len(times) > 1 else 0.012
    drop_heads = groove_class != "sustain"
    head_thresh = thresh * (0.55 if groove_class == "sustain" else 1.0)
    out: List[Dict[str, Any]] = []
    dropped = 0
    trimmed = 0
    for seg in segments:
        note = dict(seg)
        start = float(note.get("start", 0.0))
        end = float(note.get("end", start + 0.05))
        if drop_heads and not _stem_is_audible(times, rms, head_thresh, start, window_sec=0.035):
            dropped += 1
            continue
        new_end = _trim_end_at_silence(
            times,
            rms,
            thresh,
            start,
            end,
            min_silent_sec=max(0.06, beat * 0.18),
            hop_guess=hop_guess,
        )
        if new_end < end - 0.025:
            note["end"] = round(new_end, 4)
            trimmed += 1
        out.append(note)
    return out, {
        "dropped_silent_heads": dropped,
        "trimmed_holds": trimmed,
        "after": len(out),
        "silence_thresh": round(thresh, 6),
        "groove": groove_class,
        "drop_heads": drop_heads,
    }


def _inject_orphan_stem_onsets(
    segments: List[Dict[str, Any]],
    onsets: List[float],
    y,
    *,
    sr: int,
    bpm: float,
    groove_class: str = "mixed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Add short notes where the stem has clear plucks but BP left a hole.

    Dense / pulse intros: BP often skips audible attacks; sparse grooves rarely need this.
    """
    if groove_class == "sustain":
        return segments, {"added": 0, "reason": "sustain_groove"}
    if not onsets:
        return segments, {"added": 0, "reason": "no_onsets"}
    times, rms, thresh = _stem_energy_profile(y, sr=sr)
    if times is None or rms is None or thresh <= 0.0:
        return segments, {"added": 0, "reason": "no_energy"}
    beat = _beat_interval(bpm)
    match_win = max(0.038, beat * 0.10)
    starts = sorted(float(s.get("start", 0.0)) for s in segments)
    # Pitch hint from nearest BP segment (or a safe bass default).
    def _midi_near(t: float) -> float:
        if not segments:
            return 40.0
        best = min(segments, key=lambda s: abs(float(s.get("start", 0.0)) - t))
        return float(best.get("midi", 40.0))

    def _has_near(sorted_times: List[float], t: float, win: float) -> bool:
        if not sorted_times:
            return False
        import bisect

        i = bisect.bisect_left(sorted_times, t)
        for j in (i - 1, i):
            if 0 <= j < len(sorted_times) and abs(sorted_times[j] - t) <= win:
                return True
        return False

    extras: List[Dict[str, Any]] = []
    max_add = min(96, max(8, len(onsets) // 3))
    if groove_class == "mixed":
        max_add = min(max_add, max(4, len(onsets) // 8))
    # Require stronger attacks for orphans — weak HPSS bumps on R&B aren't hits.
    aud_mult = 1.35 if groove_class == "mixed" else 1.0
    for t in sorted(float(x) for x in onsets):
        if len(extras) >= max_add:
            break
        if not _stem_is_audible(times, rms, thresh * aud_mult, t, window_sec=0.028):
            continue
        if _has_near(starts, t, match_win):
            continue
        if any(abs(t - float(e.get("start", 0.0))) <= match_win for e in extras):
            continue
        midi = _midi_near(t)
        amp = _stem_rms_near(times, rms, t, window_sec=0.028)
        extras.append(
            {
                "start": round(t, 4),
                "end": round(t + max(0.07, beat * 0.16), 4),
                "midi": midi,
                "amp": float(amp),
                "amp_mean": float(amp),
            }
        )
    if not extras:
        return segments, {"added": 0, "after": len(segments)}
    merged = sorted(list(segments) + extras, key=lambda s: float(s.get("start", 0.0)))
    return merged, {"added": len(extras), "after": len(merged), "groove": groove_class}


def _peak_lock_chart_notes(
    notes: List[Dict[str, Any]],
    onsets: List[float],
    *,
    bpm: float,
    window_sec: Optional[float] = None,
    lookback_sec: Optional[float] = None,
    lookahead_sec: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Snap note heads to stem pluck peaks (prefer earlier; stem wins over grid)."""
    if not notes or not onsets:
        return notes, {"locked": 0}
    onset_arr = np.asarray(sorted(float(t) for t in onsets), dtype=float)
    beat = _beat_interval(bpm)
    if lookback_sec is None and lookahead_sec is None and window_sec is not None:
        lookback_sec = float(window_sec)
        lookahead_sec = float(window_sec) * 0.55
    if lookback_sec is None:
        lookback_sec = max(0.036, beat * 0.12)
    if lookahead_sec is None:
        lookahead_sec = max(0.018, beat * 0.05)
    locked = 0
    shifts_ms: List[float] = []
    out: List[Dict[str, Any]] = []
    for note in notes:
        n = dict(note)
        t0 = float(n.get("time", 0.0))
        lo = t0 - float(lookback_sec)
        hi = t0 + float(lookahead_sec)
        idxs = np.where((onset_arr >= lo) & (onset_arr <= hi))[0]
        if idxs.size == 0:
            out.append(n)
            continue
        cands = onset_arr[idxs]
        scores = []
        for t in cands:
            early = max(0.0, t0 - float(t))
            late = max(0.0, float(t) - t0)
            # Prefer earlier plucks — BP / grid often sit after the hit.
            scores.append(early * 1.55 - late * 2.4 - abs(float(t) - t0) * 0.20)
        best = float(cands[int(np.argmax(scores))])
        delta = best - t0
        if abs(delta) < 0.004:
            n["peak_locked"] = True
            out.append(n)
            continue
        n["time"] = round(max(0.0, best), 4)
        if str(n.get("shape", "tap")).lower() in ("hold", "slide") and n.get("end") is not None:
            n["end"] = round(max(float(n["time"]) + 0.05, float(n["end"]) + delta), 4)
        n["peak_locked"] = True
        locked += 1
        shifts_ms.append(delta * 1000.0)
        out.append(n)
    recap: Dict[str, Any] = {
        "locked": locked,
        "lookback_ms": round(float(lookback_sec) * 1000.0, 1),
        "lookahead_ms": round(float(lookahead_sec) * 1000.0, 1),
    }
    if shifts_ms:
        arr = np.asarray(shifts_ms, dtype=float)
        recap["shift_median_ms"] = round(float(np.median(arr)), 1)
    return out, recap


def _bar_energy_mask16(
    times: Any,
    rms: Any,
    *,
    bar_start: float,
    bar: float,
) -> np.ndarray:
    """16-bin relative energy mask for one bar (stem RMS per sixteenth)."""
    mask = np.zeros(16, dtype=float)
    if times is None or rms is None or bar <= 0:
        return mask
    step = bar / 16.0
    for i in range(16):
        t = bar_start + (i + 0.5) * step
        mask[i] = _stem_rms_near(times, rms, t, window_sec=max(0.012, step * 0.45))
    peak = float(np.max(mask))
    if peak > 1e-12:
        mask = mask / peak
    return mask


def _notes_in_bar(
    notes: List[Dict[str, Any]],
    *,
    bar_start: float,
    bar: float,
) -> List[Dict[str, Any]]:
    lo = float(bar_start)
    hi = lo + float(bar)
    out: List[Dict[str, Any]] = []
    for n in notes:
        t = float(n.get("time", 0.0))
        if lo <= t < hi:
            out.append(n)
    return out


def _lane_blocked_by_hold(
    notes: List[Dict[str, Any]],
    *,
    t: float,
    lane: int,
) -> bool:
    for n in notes:
        shape = str(n.get("shape", "tap")).lower()
        if shape not in ("hold", "slide"):
            continue
        t0 = float(n.get("time", 0.0))
        end = n.get("end")
        if end is None:
            continue
        lanes_raw = n.get("lanes")
        if isinstance(lanes_raw, list) and lanes_raw:
            note_lanes = [int(x) for x in lanes_raw]
        else:
            note_lanes = [int(n.get("lane", 0))]
        if int(lane) not in note_lanes:
            continue
        if t0 - 0.001 <= float(t) <= float(end) + 0.001:
            return True
    return False


def _fill_bars_from_template(
    notes: List[Dict[str, Any]],
    y,
    *,
    sr: int,
    bpm: float,
    bar_phase: float,
    lanes: int = 5,
    goal: str = "arcade",
    groove_class: str = "mixed",
    meters: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Copy taps from similar louder bars into sparse bars (monotone bass riffs)."""
    recap: Dict[str, Any] = {"added": 0, "bars_matched": 0}
    if groove_class == "sustain":
        return notes, {**recap, "reason": "sustain_groove"}
    if y is None or not notes or bar_phase is None:
        return notes, {**recap, "reason": "no_data"}
    times, rms, thresh = _stem_energy_profile(y, sr=sr)
    if times is None or rms is None or thresh <= 0.0:
        return notes, {**recap, "reason": "no_energy"}

    beat = _beat_interval(bpm)
    bar = beat * float(max(1, meters))
    if bar <= 0:
        return notes, {**recap, "reason": "bad_bar"}
    t_max = max(float(n.get("time", 0.0)) for n in notes) + bar
    goal_n = str(goal or "arcade").strip().lower()
    corr_min = 0.92 if goal_n == "original" else 0.85
    max_add = 16 if goal_n == "original" else 48
    if groove_class == "mixed":
        max_add = min(max_add, 24)
    match_win = max(0.034, beat * 0.09)

    # Align first bar start to cover chart start.
    start = float(bar_phase)
    while start > 0.0:
        start -= bar
    bar_starts: List[float] = []
    t = start
    while t < t_max + bar * 0.25:
        if t + bar > 0.05:
            bar_starts.append(t)
        t += bar
    if len(bar_starts) < 2:
        return notes, {**recap, "reason": "few_bars"}

    masks = [
        _bar_energy_mask16(times, rms, bar_start=bs, bar=bar) for bs in bar_starts
    ]
    bar_notes = [
        _notes_in_bar(notes, bar_start=bs, bar=bar) for bs in bar_starts
    ]
    counts = [len(bn) for bn in bar_notes]

    out = [dict(n) for n in notes]
    existing_times = sorted(float(n.get("time", 0.0)) for n in out)
    added = 0
    matched = 0

    def _has_near(sorted_times: List[float], t: float, win: float) -> bool:
        if not sorted_times:
            return False
        import bisect

        i = bisect.bisect_left(sorted_times, t)
        for j in (i - 1, i):
            if 0 <= j < len(sorted_times) and abs(sorted_times[j] - t) <= win:
                return True
        return False

    def _nearest_lane_midi(t: float) -> Tuple[int, float]:
        if not out:
            return 0, 40.0
        best = min(out, key=lambda n: abs(float(n.get("time", 0.0)) - t))
        lane = int(best.get("lane", 0))
        if best.get("lanes") and isinstance(best["lanes"], list):
            lane = int(best["lanes"][0])
        midi = float(best.get("midi", 40.0))
        return max(0, min(max(1, lanes) - 1, lane)), midi

    for i, bs in enumerate(bar_starts):
        if added >= max_add:
            break
        if counts[i] == 0 and float(np.mean(masks[i])) < 0.12:
            continue  # silent bar
        # Find best denser template with similar stem mask.
        best_j = -1
        best_corr = corr_min
        for j, _bs_j in enumerate(bar_starts):
            if j == i or counts[j] <= counts[i]:
                continue
            a = masks[i]
            b = masks[j]
            if float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(corr):
                continue
            if corr >= best_corr:
                best_corr = corr
                best_j = j
        if best_j < 0:
            continue
        matched += 1
        tmpl = bar_notes[best_j]
        tmpl_start = bar_starts[best_j]
        for src in tmpl:
            if added >= max_add:
                break
            # Clone attack *times* as taps only (never sustain bodies / slides).
            rel = float(src.get("time", 0.0)) - tmpl_start
            if rel < 0.0 or rel >= bar:
                continue
            t_new = bs + rel
            if t_new < 0.02:
                continue
            if not _stem_is_audible(times, rms, thresh, t_new, window_sec=0.028):
                continue
            if _has_near(existing_times, t_new, match_win):
                continue
            lane, midi = _nearest_lane_midi(t_new)
            lanes_raw = src.get("lanes")
            if isinstance(lanes_raw, list) and lanes_raw:
                try:
                    lane = int(lanes_raw[0])
                except (TypeError, ValueError):
                    pass
            elif src.get("lane") is not None:
                try:
                    lane = int(src["lane"])
                except (TypeError, ValueError):
                    pass
            lane = max(0, min(max(1, lanes) - 1, lane))
            if _lane_blocked_by_hold(out, t=t_new, lane=lane):
                continue
            amp = _stem_rms_near(times, rms, t_new, window_sec=0.028)
            out.append(
                {
                    "time": round(t_new, 4),
                    "lane": lane,
                    "shape": "tap",
                    "midi": midi,
                    "amp": float(amp),
                }
            )
            existing_times.append(t_new)
            existing_times.sort()
            added += 1
        counts[i] = len(_notes_in_bar(out, bar_start=bs, bar=bar))

    if added:
        out.sort(key=lambda n: float(n.get("time", 0.0)))
    recap.update({"added": added, "bars_matched": matched, "corr_min": corr_min})
    return out, recap


def _strip_peak_locked_flags(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for n in notes:
        d = dict(n)
        d.pop("peak_locked", None)
        out.append(d)
    return out


def _gate_chart_notes_by_stem_energy(
    notes: List[Dict[str, Any]],
    y,
    *,
    sr: int,
    bpm: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Final pass: no taps in silence; no holds spanning stem silence."""
    times, rms, thresh = _stem_energy_profile(y, sr=sr)
    if times is None or rms is None or thresh <= 0.0:
        return notes, {"skipped": True}
    beat = _beat_interval(bpm)
    hop_guess = float(times[1] - times[0]) if len(times) > 1 else 0.012
    out: List[Dict[str, Any]] = []
    dropped = 0
    trimmed = 0
    for note in notes:
        n = dict(note)
        t0 = float(n.get("time", 0.0))
        if not _stem_is_audible(times, rms, thresh, t0, window_sec=0.032):
            dropped += 1
            continue
        shape = str(n.get("shape", "tap")).strip().lower()
        if shape in ("hold", "slide") and n.get("end") is not None:
            end = float(n["end"])
            new_end = _trim_end_at_silence(
                times,
                rms,
                thresh,
                t0,
                end,
                min_silent_sec=max(0.06, beat * 0.18),
                hop_guess=hop_guess,
            )
            if new_end < end - 0.025:
                # Short remainder → tap (don't leave a stub hold).
                if new_end - t0 < max(0.11, beat * 0.28):
                    n.pop("end", None)
                    n.pop("lane_end", None)
                    n.pop("curve", None)
                    n["shape"] = "tap"
                    if n.get("type") == "BassHoldNote":
                        n["type"] = "BassTapNote"
                else:
                    n["end"] = round(new_end, 4)
                trimmed += 1
        out.append(n)
    return out, {
        "dropped_silent": dropped,
        "trimmed_or_demoted": trimmed,
        "after": len(out),
    }


def _apply_constant_start_bias(
    segments: List[Dict[str, Any]], bias_sec: float
) -> List[Dict[str, Any]]:
    if abs(bias_sec) < 1e-6 or not segments:
        return segments
    out: List[Dict[str, Any]] = []
    for seg in segments:
        note = dict(seg)
        start = float(note.get("start", 0.0)) + bias_sec
        end = float(note.get("end", start + 0.05)) + bias_sec
        note["start"] = round(max(0.0, start), 4)
        note["end"] = round(max(float(note["start"]) + 0.03, end), 4)
        out.append(note)
    return out


def _detect_bass_entry_sec(y, *, sr: int, bpm: float) -> float:
    """First real bass activity on the stem (skip bleed / empty intro).

    Dense / plucky intros: use short energy spikes, not only long sustains —
    requiring ~350ms continuous energy wrongly pushes entry past pulse intros
    or returns 0 while mid-track silence still needs phantom-trim elsewhere
    (silence-gate handles mid-song; entry only trims lead-in bleed).
    """
    try:
        import librosa
    except ImportError:
        return 0.0
    if y is None or len(y) < max(sr // 2, 2048):
        return 0.0
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    if rms.size < 8:
        return 0.0
    times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop)
    p90 = float(np.percentile(rms, 90))
    p50 = float(np.percentile(rms, 50))
    if p90 <= 1e-8:
        return 0.0
    # Soft threshold: intro bleed is usually << active groove.
    thresh = max(p50 * 1.05, p90 * 0.14)
    beat = _beat_interval(bpm)
    # ~90ms pulse OR ~280ms sustain — covers gallops and held notes.
    need_pulse = max(2, int((0.09 * float(sr)) / hop))
    need_sustain = max(3, int((0.28 * float(sr)) / hop))
    run = 0
    for i, val in enumerate(rms):
        if float(val) >= thresh:
            run += 1
            # Strong single-frame spike after quiet: treat as entry (pluck).
            strong = float(val) >= p90 * 0.45
            if run >= need_sustain or (strong and run >= need_pulse):
                t0 = float(times[max(0, i - run + 1)])
                # Don't trim a true early bass entry in the first ~1.5 beats.
                return 0.0 if t0 < beat * 1.5 else t0
        else:
            run = 0
    return 0.0


def _refine_bass_entry_with_bp(
    entry_sec: float,
    segments: List[Dict[str, Any]],
    y,
    *,
    sr: int,
    bpm: float,
) -> float:
    """If Basic Pitch heard several notes before RMS-entry, trust earlier pocket.

    Soft R&B intros often sit below the global RMS wake threshold until the
    loud groove; entry then jumps to ~10s and the chart looks empty at the start.
    """
    if entry_sec <= 0.05 or not segments or y is None:
        return entry_sec
    try:
        import librosa
    except ImportError:
        return entry_sec
    beat = _beat_interval(bpm)
    early = [
        float(s.get("start", 0.0))
        for s in segments
        if float(s.get("start", 0.0)) < entry_sec - beat * 0.5
    ]
    if len(early) < 4:
        return entry_sec
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    if rms.size < 8:
        return entry_sec
    p50 = float(np.percentile(rms, 50))
    soft = max(p50 * 0.55, float(np.percentile(rms, 90)) * 0.06)
    audible = 0
    for t in early[:40]:
        frame = int(round((t * float(sr)) / hop))
        frame = max(0, min(int(rms.size) - 1, frame))
        if float(rms[frame]) >= soft:
            audible += 1
    if audible < max(3, len(early) // 5):
        return entry_sec
    first = min(early)
    refined = max(0.0, first - beat * 0.1)
    if refined < entry_sec - beat:
        print(
            f"[BassGen][этап] bass_entry_refine {entry_sec:.3f}s->{refined:.3f}s "
            f"(bp_early={len(early)} audible={audible})"
        )
        return refined
    return entry_sec


def _trim_before_entry(
    items: List[Dict[str, Any]],
    entry_sec: float,
    *,
    time_key: str = "time",
    grace_sec: float = 0.04,
) -> Tuple[List[Dict[str, Any]], int]:
    if entry_sec <= 0.05 or not items:
        return items, 0
    cut = float(entry_sec) - float(grace_sec)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for item in items:
        t = float(item.get(time_key, item.get("start", 0.0)))
        if t >= cut:
            kept.append(item)
        else:
            dropped += 1
    return kept, dropped


def _onsets_to_segments(
    onsets: List[float],
    samples: List[Tuple[float, float, float]],
    *,
    bpm: float,
) -> List[Dict[str, Any]]:
    if not onsets:
        return []
    beat = _beat_interval(bpm)
    segments: List[Dict[str, Any]] = []
    for i, start in enumerate(onsets):
        midi, amp = _pitch_at_time(samples, start)
        if midi is None:
            continue
        if i + 1 < len(onsets):
            end = max(start + 0.04, float(onsets[i + 1]) - 0.025)
        else:
            end = start + beat * 0.65
            for t, seg_midi, _ in samples:
                if t < start:
                    continue
                if t > start + beat * 1.25:
                    break
                if abs(seg_midi - midi) > 1.25:
                    end = max(start + 0.06, t - 0.01)
                    break
        if end <= start:
            end = start + 0.06
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "midi": float(midi),
                "amp": float(amp),
                "amp_mean": float(amp),
            }
        )
    return segments


def _merge_pitch_runs(
    samples: List[Tuple[float, float, float]],
    *,
    bpm: float,
    min_run_ms: float = 90.0,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    hop_t = max(0.02, _beat_interval(bpm) / 16.0)
    segments: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for t, midi, amp in samples:
        if cur is None:
            cur = {"start": t, "end": t, "midi_sum": midi, "n": 1, "amp_max": amp, "amp_sum": amp}
            continue
        same_pitch = abs(midi - (cur["midi_sum"] / cur["n"])) <= 0.85
        gap = t - float(cur["end"])
        if same_pitch and gap <= hop_t * 2.5:
            cur["end"] = t
            cur["midi_sum"] += midi
            cur["n"] += 1
            cur["amp_max"] = max(float(cur["amp_max"]), amp)
            cur["amp_sum"] += amp
        else:
            segments.append(cur)
            cur = {"start": t, "end": t, "midi_sum": midi, "n": 1, "amp_max": amp, "amp_sum": amp}
    if cur:
        segments.append(cur)
    min_run = max(min_run_ms / 1000.0, hop_t)
    out: List[Dict[str, Any]] = []
    for seg in segments:
        dur = float(seg["end"]) - float(seg["start"])
        if dur < min_run * 0.25:
            continue
        out.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "midi": float(seg["midi_sum"]) / max(int(seg["n"]), 1),
                "amp": float(seg["amp_max"]),
                "amp_mean": float(seg["amp_sum"]) / max(int(seg["n"]), 1),
            }
        )
    return out


def _collapse_same_lane_taps(
    notes: List[Dict[str, Any]],
    *,
    bpm: float,
) -> List[Dict[str, Any]]:
    if not notes:
        return []
    beat = _beat_interval(bpm)
    merge_gap = beat * 0.38
    min_hold = beat * 0.32
    out: List[Dict[str, Any]] = []
    for note in sorted(notes, key=lambda n: float(n.get("time", 0.0))):
        shape = str(note.get("shape", "tap")).lower()
        if (
            out
            and shape == "tap"
            and str(out[-1].get("shape", "tap")).lower() == "tap"
            and int(note.get("lane", -1)) == int(out[-1].get("lane", -2))
            and not note.get("ghost")
            and not out[-1].get("ghost")
        ):
            gap = float(note["time"]) - float(out[-1]["time"])
            if gap <= merge_gap:
                out[-1] = {
                    "time": round(float(out[-1]["time"]), 4),
                    "end": round(float(note["time"]) + min_hold * 0.35, 4),
                    "lane": int(out[-1]["lane"]),
                    "shape": "hold",
                    "ghost": False,
                }
                continue
        out.append(note)
    return out


def _transcription_segments_to_chart_notes(
    segments: List[Dict[str, Any]],
    *,
    bpm: float,
    lanes: int = 5,
    allow_multi_lane: bool = True,
    allow_slides: bool = True,
    quant_strength: float = 0.0,
    phase: float = 0.0,
    prefer_eighths: bool = True,
    prefer_quarters: bool = True,
    bar_phase: Optional[float] = None,
    groove_class: str = "mixed",
) -> List[Dict[str, Any]]:
    """Convert Basic Pitch segments; soft-snap onsets to a beat-phased grid."""
    if not segments:
        return []
    midis = [float(s["midi"]) for s in segments]
    midi_min, midi_max = robust_midi_range(midis)
    if midi_max - midi_min < 4.0:
        pad = max(2.0, (4.0 - (midi_max - midi_min)) * 0.5)
        midi_min -= pad
        midi_max += pad
    amps = sorted(float(s.get("amp_mean", s.get("amp", 0.0))) for s in segments)
    # Bottom ~12%: quiet short taps may become arcade ghost spice.
    ghost_cut = amps[max(0, min(len(amps) - 1, int(len(amps) * 0.12)))] if amps else 0.0
    amp_strong = amps[int(len(amps) * 0.65)] if amps else 0.0
    beat = _beat_interval(bpm)
    min_hold = max(0.11, beat * 0.2)
    if groove_class == "sustain":
        # R&B: keep BP holds — don't require big gaps / punish local density.
        min_hold = max(0.10, beat * 0.16)
    min_tap = max(0.045, beat * 0.07)
    slide_gap = beat * 1.6
    notes: List[Dict[str, Any]] = []
    multi_lane_quota = max(1, int(len(segments) * 0.04)) if allow_multi_lane else 0
    seg_starts = [float(s.get("start", 0.0)) for s in segments]
    density_window = beat * 0.55

    def _quant_opts(seg: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """Return (strength, prefer_quarters, prefer_downbeats) for this head."""
        dur = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        amp = float(seg.get("amp_mean", seg.get("amp", 0.0)))
        strong = dur >= beat * 0.50 or amp >= amp_strong
        medium = dur >= beat * 0.28
        local = _local_event_density(seg_starts, float(seg.get("start", 0.0)), density_window)
        if local >= 3:
            # Dense metal/pluck lines: snap harder to 8/16, avoid quarter pull.
            base = max(quant_strength, 0.78)
            return min(0.95, base + 0.18), False, False
        if strong:
            return min(0.95, quant_strength + 0.40), True, True
        if medium:
            return min(0.90, quant_strength + 0.22), True, bool(bar_phase is not None)
        return quant_strength, prefer_quarters, False

    i = 0
    while i < len(segments):
        seg = segments[i]
        q_str, q_qr, q_db = _quant_opts(seg)
        start = round(
            _quantize_time(
                float(seg["start"]),
                bpm,
                strength=q_str,
                phase=phase,
                prefer_eighths=prefer_eighths,
                prefer_quarters=q_qr,
                prefer_downbeats=q_db,
                bar_phase=bar_phase,
            ),
            4,
        )
        end = round(
            max(
                _quantize_time(
                    float(seg["end"]),
                    bpm,
                    strength=q_str * 0.65,
                    phase=phase,
                    prefer_eighths=prefer_eighths,
                    prefer_quarters=q_qr,
                    prefer_downbeats=False,
                    bar_phase=bar_phase,
                ),
                start + 0.02,
            ),
            4,
        )
        midi = float(seg["midi"])
        lane = pitch_to_lane(midi, midi_min, midi_max, lanes)
        dur = max(end - start, 0.0)
        amp = float(seg.get("amp_mean", seg.get("amp", 0.0)))
        quiet = amp <= ghost_cut
        local = _local_event_density(seg_starts, float(seg.get("start", 0.0)), density_window)
        next_gap = float("inf")
        if i + 1 < len(seg_starts):
            next_gap = float(seg_starts[i + 1]) - float(seg.get("start", 0.0))
        # In dense pluck runs, BP sustain bodies look long — force taps.
        # Sustain/R&B: trust BP durations (that was the good baseline).
        if groove_class == "sustain":
            hold_ok = dur >= min_hold and next_gap >= beat * 0.32
        else:
            hold_ok = (
                dur >= (min_hold * (1.75 if local >= 3 else 1.0))
                and next_gap >= beat * 0.55
            )
            if local >= 3:
                # Metal 16ths: only keep holds that are clearly sustained pad notes.
                hold_ok = hold_ok and dur >= beat * 0.90 and next_gap >= beat * 0.75
        if allow_slides and i + 1 < len(segments):
            nxt = segments[i + 1]
            n_str, n_qr, n_db = _quant_opts(nxt)
            nxt_start = round(
                _quantize_time(
                    float(nxt["start"]),
                    bpm,
                    strength=n_str,
                    phase=phase,
                    prefer_eighths=prefer_eighths,
                    prefer_quarters=n_qr,
                    prefer_downbeats=n_db,
                    bar_phase=bar_phase,
                ),
                4,
            )
            nxt_lane = pitch_to_lane(float(nxt["midi"]), midi_min, midi_max, lanes)
            gap = nxt_start - start
            if (
                lane != nxt_lane
                and gap > min_tap
                and gap < slide_gap
                and dur >= min_tap
                and local < 3
            ):
                slide_end = round(max(end, nxt_start), 4)
                notes.append(
                    {
                        "time": start,
                        "end": slide_end,
                        "lane": lane,
                        "lane_end": nxt_lane,
                        "shape": "slide",
                        "curve": "linear",
                        "ghost": False,
                    }
                )
                if nxt_start <= end + beat * 0.12:
                    i += 2
                else:
                    i += 1
                continue
        if (
            multi_lane_quota > 0
            and dur < min_hold
            and lane <= lanes - 3
            and not quiet
            and amp >= ghost_cut * 1.15
        ):
            upper_lane = min(lanes - 1, lane + 3)
            notes.append(
                {
                    "time": start,
                    "lanes": [lane, upper_lane],
                    "shape": "tap",
                    "ghost": False,
                }
            )
            multi_lane_quota -= 1
            i += 1
            continue
        if hold_ok:
            notes.append(
                {
                    "time": start,
                    "end": end,
                    "lane": lane,
                    "shape": "hold",
                    "ghost": False,
                }
            )
        else:
            notes.append(
                {
                    "time": start,
                    "lane": lane,
                    "shape": "tap",
                    # Ghost only on short quiet taps (arcade spice; Original strips later).
                    "ghost": quiet,
                }
            )
        i += 1
    return notes


def _segments_to_chart_notes(
    segments: List[Dict[str, Any]],
    *,
    bpm: float,
    lanes: int = 5,
    quant_strength: float = 0.85,
    allow_multi_lane: bool = True,
    allow_slides: bool = True,
    phase: float = 0.0,
    prefer_eighths: bool = True,
) -> List[Dict[str, Any]]:
    if not segments:
        return []
    midis = [float(s["midi"]) for s in segments]
    midi_min, midi_max = robust_midi_range(midis)
    if midi_max - midi_min < 4.0:
        pad = max(2.0, (4.0 - (midi_max - midi_min)) * 0.5)
        midi_min -= pad
        midi_max += pad
    amps = sorted(float(s.get("amp_mean", s.get("amp", 0.0))) for s in segments)
    ghost_cut = amps[max(0, min(len(amps) - 1, int(len(amps) * 0.12)))] if amps else 0.0
    notes: List[Dict[str, Any]] = []
    multi_lane_quota = max(1, int(len(segments) * 0.05)) if allow_multi_lane else 0
    beat = _beat_interval(bpm)
    min_hold = beat * 0.32
    slide_gap = beat * 2.0
    seg_starts = [float(s.get("start", 0.0)) for s in segments]
    density_window = beat * 0.55
    i = 0
    while i < len(segments):
        seg = segments[i]
        local = _local_event_density(seg_starts, float(seg.get("start", 0.0)), density_window)
        q_str = min(0.95, quant_strength + (0.12 if local >= 4 else 0.0))
        start = _quantize_time(
            float(seg["start"]),
            bpm,
            strength=q_str,
            phase=phase,
            prefer_eighths=prefer_eighths,
        )
        end = _quantize_time(
            float(seg["end"]),
            bpm,
            strength=q_str,
            phase=phase,
            prefer_eighths=prefer_eighths,
        )
        midi = float(seg["midi"])
        lane = pitch_to_lane(midi, midi_min, midi_max, lanes)
        dur = max(end - start, 0.0)
        amp = float(seg.get("amp_mean", seg.get("amp", 0.0)))
        quiet = amp <= ghost_cut
        next_gap = float("inf")
        if i + 1 < len(seg_starts):
            next_gap = float(seg_starts[i + 1]) - float(seg.get("start", 0.0))
        hold_ok = dur >= (min_hold * (1.4 if local >= 4 else 1.0)) and next_gap >= beat * 0.42
        if allow_slides and i + 1 < len(segments) and local < 4:
            nxt = segments[i + 1]
            nxt_lane = pitch_to_lane(float(nxt["midi"]), midi_min, midi_max, lanes)
            nxt_start = _quantize_time(
                float(nxt["start"]),
                bpm,
                strength=quant_strength,
                phase=phase,
                prefer_eighths=prefer_eighths,
            )
            if (
                lane != nxt_lane
                and dur >= min_hold * 0.65
                and 0.0 < (nxt_start - start) < slide_gap
            ):
                slide_end = max(end, nxt_start)
                notes.append(
                    {
                        "time": round(start, 4),
                        "end": round(slide_end, 4),
                        "lane": lane,
                        "lane_end": nxt_lane,
                        "shape": "slide",
                        "curve": "linear",
                        "ghost": False,
                    }
                )
                if nxt_start <= end + beat * 0.12:
                    i += 2
                else:
                    i += 1
                continue
        if (
            multi_lane_quota > 0
            and dur < min_hold * 0.75
            and lane <= lanes - 3
            and not quiet
            and amp >= ghost_cut * 1.15
        ):
            upper_lane = min(lanes - 1, lane + 3)
            notes.append(
                {
                    "time": round(start, 4),
                    "lanes": [lane, upper_lane],
                    "shape": "tap",
                    "ghost": False,
                }
            )
            multi_lane_quota -= 1
            i += 1
            continue
        if hold_ok:
            notes.append(
                {
                    "time": round(start, 4),
                    "end": round(max(end, start + min_hold * 0.45), 4),
                    "lane": lane,
                    "shape": "hold",
                    "ghost": False,
                }
            )
        else:
            notes.append(
                {
                    "time": round(start, 4),
                    "lane": lane,
                    "shape": "tap",
                    "ghost": quiet,
                }
            )
        i += 1
    return notes


def generate_bass_notes(
    song_path: str,
    bpm: float,
    *,
    lanes: int = CANONICAL_MAX_LANES,
    use_stems: bool = True,
    chart_id: str = "",
    goal: str = "original",
    difficulty: str = "standard",
    status_cb: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], None]] = None,
) -> List[Dict[str, Any]]:
    from pathlib import Path
    from app import audio_analysis, song_storage
    from app.bass_transforms import (
        apply_bass_difficulty,
        apply_bass_style,
        demote_slides_to_holds,
        diversify_lane_runs,
        spread_simultaneous_same_lane,
        strip_same_lane_hold_overlaps,
        _cap_ghost_notes,
        _strip_ghosts,
    )
    from app.generation_intents import normalize_difficulty, normalize_goal

    def report(msg: str) -> None:
        if status_cb:
            status_cb(msg)

    def check_cancel() -> None:
        if cancel_cb:
            cancel_cb()

    lanes = max(3, min(int(lanes), CANONICAL_MAX_LANES))
    goal_n = normalize_goal(goal)
    diff_n = normalize_difficulty(difficulty)
    style_quant = 0.38 if goal_n == "original" else 0.62
    allow_multi_lane = goal_n == "arcade"
    import os

    allow_slides = os.environ.get("RFALL_BASS_SLIDES", "").strip().lower() in ("1", "true", "yes")
    _clear_bass_analysis_caches()
    min_run_ms = 110.0 if diff_n == "relaxed" else (70.0 if diff_n == "dense" else 85.0)
    if goal_n == "original" and diff_n == "dense":
        min_run_ms = 60.0
        style_quant = 0.32
    path = Path(song_path)
    cid = str(chart_id or "").strip() or song_storage.chart_id_from_song_path(str(path))
    song_folder = song_storage.song_dir(cid) if cid else Path("temp_uploads") / path.stem
    song_folder.mkdir(parents=True, exist_ok=True)

    def _bass_stem_on_disk() -> bool:
        from app import song_storage

        splitter_folder = song_folder / "splitter"
        if cid:
            stem_name = song_storage.stem_wav_name(cid, "bass")
        else:
            stem_name = f"{path.stem}_bass.wav"
        return (splitter_folder / stem_name).is_file()

    audio_path = str(path)
    if use_stems:
        if not _bass_stem_on_disk():
            report("Разделение на стемы...")
        check_cancel()
        try:
            audio_path = audio_analysis.separate_stems(str(path), song_folder, stem_type="bass", cancel_cb=cancel_cb)
        except Exception as exc:
            print(f"[BassGen] stem failed, using mix: {exc}")

    report("Анализ басовой линии...")
    check_cancel()

    from app.bass_transcriber import try_transcribe_basic_pitch

    print("[BassGen][этап] basic_pitch start…")
    _bp_t0 = time.perf_counter()
    segments = try_transcribe_basic_pitch(audio_path, cancel_cb=check_cancel)
    print(
        f"[BassGen][этап] basic_pitch done in {time.perf_counter() - _bp_t0:.1f}s "
        f"notes={(len(segments) if segments else 0)}"
    )
    report("Сборка бас-чарта...")
    check_cancel()
    segment_source = "basic_pitch"
    onsets: List[float] = []
    _build_t0 = time.perf_counter()
    y = None
    sr = 22050
    if segments is None:
        y, sr = _load_audio(audio_path)
        samples = _extract_pitch_track(y, sr=sr, bpm=bpm)
        if not samples:
            print("[BassGen] no pitch samples — empty chart")
            return []
        onsets = _extract_bass_onsets(y, sr=sr, bpm=bpm)
        segment_source = "onsets"
        segments = _onsets_to_segments(onsets, samples, bpm=bpm)
        if len(segments) < max(8, len(onsets) // 3):
            segment_source = "pyin"
            segments = _merge_pitch_runs(samples, bpm=bpm, min_run_ms=min_run_ms)
    elif not segments:
        print("[BassGen] basic_pitch returned 0 notes — heuristic rescue")
        y, sr = _load_audio(audio_path)
        samples = _extract_pitch_track(y, sr=sr, bpm=bpm)
        if not samples:
            print("[BassGen] no pitch samples — empty chart")
            return []
        onsets = _extract_bass_onsets(y, sr=sr, bpm=bpm)
        segment_source = "onsets"
        segments = _onsets_to_segments(onsets, samples, bpm=bpm)
        if len(segments) < max(8, len(onsets) // 3):
            segment_source = "pyin"
            segments = _merge_pitch_runs(samples, bpm=bpm, min_run_ms=min_run_ms)
    elif _bass_transcription_is_sparse(segments):
        # 808 / sub-bass: Basic Pitch often returns a handful of phantom notes.
        print(
            f"[BassGen][этап] sparse_bp={len(segments)} — sensitive retry + onset rescue"
        )
        sensitive = try_transcribe_basic_pitch(
            audio_path,
            cancel_cb=check_cancel,
            onset_threshold=0.28,
            frame_threshold=0.20,
        )
        if sensitive and len(sensitive) > len(segments) * 1.5:
            print(
                f"[BassGen][этап] sparse_bp_retry {len(segments)}->{len(sensitive)} "
                "(sensitive thresholds)"
            )
            segments = sensitive
        if _bass_transcription_is_sparse(segments):
            if y is None:
                y, sr = _load_audio(audio_path)
            samples = _extract_pitch_track(y, sr=sr, bpm=bpm)
            onsets = _extract_bass_onsets(y, sr=sr, bpm=bpm)
            rescued: List[Dict[str, Any]] = []
            rescue_src = "onsets"
            if samples:
                rescued = _onsets_to_segments(onsets, samples, bpm=bpm)
                if len(rescued) < max(8, len(onsets) // 3):
                    rescued = _merge_pitch_runs(samples, bpm=bpm, min_run_ms=min_run_ms)
                    rescue_src = "pyin"
            if len(rescued) > len(segments) * 1.35:
                print(
                    f"[BassGen][этап] sparse_rescue {len(segments)}->{len(rescued)} "
                    f"source={rescue_src}"
                )
                segments = rescued
                segment_source = rescue_src
            else:
                print(
                    f"[BassGen][этап] sparse_rescue_keep_bp "
                    f"bp={len(segments)} rescue={len(rescued)}"
                )

    print(
        f"[BassGen][этап] transcribe segments={len(segments)} source={segment_source} "
        f"onsets={len(onsets)}"
    )

    # Beat phase from the full mix (better groove than isolated bass stem).
    beat_interval = _beat_interval(bpm)
    phase = 0.0
    beats = np.array([])
    try:
        beats = audio_analysis.extract_beats(str(path), bpm=float(bpm))
        if beats is not None and len(beats) > 0:
            phase = float(beats[0])
            print(
                f"[BassGen][этап] beats={len(beats)} phase={phase:.4f}s "
                f"beat={beat_interval:.4f}s"
            )
        else:
            print("[BassGen][этап] beats=empty — grid phase falls back to t=0")
    except Exception as exc:
        print(f"[BassGen] beat track failed: {exc}")

    # Pull BP note heads to audible plucks BEFORE grid snap — otherwise we
    # quantize a late sustain body and the groove still feels "after" the hit.
    bar_phase: Optional[float] = None
    pulled_total = 0
    groove_class = "mixed"
    if segment_source == "basic_pitch":
        # HPSS onset (~5–11s) is why "Сборка бас-чарта" feels slow —
        # drums fold similar work into "Детекция ударных". Clear R&B/sustain
        # does not need the pool; skip it and only load stem for gate/gap.
        if y is None:
            y, sr = _load_audio(audio_path)
        groove_bp = _estimate_bass_groove_class(segments, [], bpm=bpm)
        long_r = float(groove_bp.get("long_ratio", 0.0))
        short_r = float(groove_bp.get("short_ratio", 0.0))
        med_dur_s = float(groove_bp.get("med_dur_ms", 0.0)) / 1000.0
        # med_dur gate: metal BP medians are short (~200ms); R&B holds ~300ms+.
        # Without this, DOMINATION (high long_ratio, no onsets yet) looks "sustain".
        clear_sustain = (
            str(groove_bp.get("class")) == "sustain"
            and med_dur_s >= 0.28
            and long_r >= 0.40
            and short_r < 0.35
        )
        if clear_sustain:
            groove_info = dict(groove_bp)
            groove_info["class"] = "sustain"
            print(
                "[BassGen][этап] onset_pool=skip "
                f"(clear sustain long={long_r:.2f} short={short_r:.2f} "
                f"med_dur={med_dur_s * 1000.0:.0f}ms — avoids HPSS)"
            )
        elif not onsets and y is not None:
            print("[BassGen][этап] onset_pool=extract (HPSS — slow path)")
            onsets = _extract_bass_onsets(y, sr=sr, bpm=bpm)
            print(f"[BassGen][этап] onset_pool={len(onsets)} (post-BP extract)")
            groove_info = _estimate_bass_groove_class(segments, onsets, bpm=bpm)
        else:
            groove_info = _estimate_bass_groove_class(segments, onsets, bpm=bpm)
            print(f"[BassGen][этап] onset_pool={len(onsets)} (pre-existing)")
        groove_class = str(groove_info.get("class", "mixed"))
        print(f"[BassGen][этап] groove={groove_info}")
        raw_seg_times = [float(s.get("start", 0.0)) for s in segments]
        _print_bass_timing_offset_diagnostics(
            "bp_raw", raw_seg_times, beats, beat_interval
        )
        segments, fusion_peaks = _fuse_segments_to_pluck_onsets(
            segments, onsets, groove_class=groove_class
        )
        print(f"[BassGen][этап] pluck_fuse={fusion_peaks}")
        # Envelope fuse helps soft R&B only with a short window; on sustain it
        # still tends to chase noise — skip there. Plucky/mixed keep it.
        if y is not None and groove_class != "sustain":
            lookback = 0.100 if groove_class == "plucky" else 0.070
            segments, fusion_env = _fuse_segments_to_onset_envelope(
                segments, y, sr=sr, lookback_sec=lookback
            )
            print(f"[BassGen][этап] pluck_env={fusion_env}")
            pulled_total = int(fusion_peaks.get("pulled", 0)) + int(
                fusion_env.get("pulled", 0)
            )
        else:
            pulled_total = int(fusion_peaks.get("pulled", 0))
            if groove_class == "sustain":
                print("[BassGen][этап] pluck_env=skip (sustain groove — BP-faithful)")
        # ADD accent taps inside BP sustains (metal) — never rewrite the hold away.
        segments, accent_recap = _add_internal_pluck_accents(
            segments, onsets, bpm=bpm, groove_class=groove_class
        )
        if int(accent_recap.get("added", 0)) > 0 or accent_recap.get("reason"):
            print(f"[BassGen][этап] pluck_accents={accent_recap}")
        # Stem-first: fill audible plucks BP skipped (dense intros), then cut silence.
        if y is not None and onsets:
            before_inj = len(segments)
            segments, inj_recap = _inject_orphan_stem_onsets(
                segments, onsets, y, sr=sr, bpm=bpm, groove_class=groove_class
            )
            if int(inj_recap.get("added", 0)) > 0 or inj_recap.get("reason"):
                print(
                    f"[BassGen][этап] orphan_onsets={before_inj}->{len(segments)} "
                    f"recap={inj_recap}"
                )
        if y is not None:
            before_gate = len(segments)
            segments, gate_recap = _gate_segments_by_stem_energy(
                segments, y, sr=sr, bpm=bpm, groove_class=groove_class
            )
            print(
                f"[BassGen][этап] stem_gate_seg={before_gate}->{len(segments)} "
                f"recap={gate_recap}"
            )
        # Stem wins: when pluck fuse already parked heads, don't fight with a fixed early bias.
        # Sustain: never apply constant early bias — soft peaks already mislead.
        if groove_class == "sustain":
            print("[BassGen][этап] attack_bias=skip (sustain groove)")
        elif pulled_total < max(8, max(1, len(segments) // 5)):
            bias = -0.016
            segments = _apply_constant_start_bias(segments, bias)
            print(
                f"[BassGen][этап] attack_bias={bias * 1000.0:.0f}ms "
                f"(weak pluck match pulled={pulled_total}/{len(segments)})"
            )
        else:
            print(
                f"[BassGen][этап] attack_bias=skip "
                f"(pluck match ok pulled={pulled_total}/{len(segments)})"
            )
        _print_bass_timing_offset_diagnostics(
            "post_pluck",
            [float(s.get("start", 0.0)) for s in segments],
            beats,
            beat_interval,
        )

    # Drop phantom BP notes before the bass actually speaks on the stem.
    # Sustain/R&B: soft intro bass is real — entry RMS often wakes only when
    # the loud groove hits (~10s on Sundays) and wrongly wipes the pocket.
    bass_entry = 0.0
    if y is not None and groove_class == "sustain":
        print("[BassGen][этап] bass_entry=skip (BP-faithful sustain)")
    elif y is not None:
        bass_entry = _detect_bass_entry_sec(y, sr=sr, bpm=bpm)
        bass_entry = _refine_bass_entry_with_bp(
            bass_entry, segments, y, sr=sr, bpm=bpm
        )
        if bass_entry > 0.05:
            before_n = len(segments)
            segments, dropped_seg = _trim_before_entry(
                segments, bass_entry, time_key="start", grace_sec=beat_interval * 0.15
            )
            print(
                f"[BassGen][этап] bass_entry={bass_entry:.3f}s "
                f"trim_segments=-{dropped_seg} ({before_n}->{len(segments)})"
            )
        else:
            print("[BassGen][этап] bass_entry=0 (early/active from start)")

    _stem_times_db = None
    _stem_rms_db = None
    if y is not None:
        _stem_times_db, _stem_rms_db, _ = _stem_energy_profile(y, sr=sr)
    if len(beats) > 0:
        bar_phase, bar_recap = _estimate_downbeat_phase(
            beats,
            onsets,
            beat_interval=beat_interval,
            meters=4,
            stem_times=_stem_times_db,
            stem_rms=_stem_rms_db,
        )
        print(f"[BassGen][этап] downbeat={bar_recap}")
    else:
        bar_phase = phase

    # Strong/long heads lock to quarters+downbeats; weak ones keep some looseness.
    bp_quant = 0.55 if goal_n == "original" else 0.90
    if goal_n == "original" and diff_n == "dense":
        bp_quant = 0.48
    if goal_n == "original" and diff_n == "relaxed":
        bp_quant = 0.62
    if goal_n == "arcade" and diff_n == "dense":
        bp_quant = 0.94
    if goal_n == "arcade" and diff_n == "relaxed":
        bp_quant = 0.84
    if goal_n == "original":
        style_quant = max(style_quant, 0.48)
    else:
        style_quant = max(style_quant, 0.80)

    transcription_faithful = segment_source == "basic_pitch"
    pulse_808 = _is_808_pulse_groove(
        segments, bpm=bpm, groove_class=groove_class
    )
    # Sustain / R&B: stay near Basic Pitch times (that was the good baseline).
    # Exception: trap 808s — soft BP times float between beats; force grid magnet.
    if pulse_808:
        bp_quant = max(bp_quant, 0.88 if goal_n == "arcade" else 0.78)
        print(
            f"[BassGen][этап] bp_quant_808_pulse={bp_quant:.2f} "
            "(snap to beats 1/3)"
        )
    elif groove_class == "sustain":
        bp_quant = min(bp_quant, 0.28 if goal_n == "original" else 0.40)
        print(f"[BassGen][этап] bp_quant_bp_faithful={bp_quant:.2f} (sustain)")
    elif (
        transcription_faithful
        and groove_class == "plucky"
        and pulled_total >= max(8, max(1, len(segments) // 5))
    ):
        # Stem fuse already parked heads on plucks — keep a strong grid magnet so
        # 8th/16th spam reads as rhythm (softening to ~0.58 left metal floating).
        softer = 0.55 if goal_n == "original" else 0.86
        if bp_quant > softer:
            print(
                f"[BassGen][этап] bp_quant_soften {bp_quant:.2f}->{softer:.2f} "
                f"(stem pluck match + grid magnet)"
            )
            bp_quant = softer
        elif bp_quant < softer:
            print(
                f"[BassGen][этап] bp_quant_grid {bp_quant:.2f}->{softer:.2f} "
                f"(plucky spam magnet)"
            )
            bp_quant = softer
    if transcription_faithful:
        notes = _transcription_segments_to_chart_notes(
            segments,
            bpm=bpm,
            lanes=lanes,
            allow_multi_lane=allow_multi_lane,
            allow_slides=allow_slides,
            quant_strength=bp_quant,
            phase=phase,
            prefer_eighths=not pulse_808,
            prefer_quarters=True,
            bar_phase=bar_phase,
            groove_class="mixed" if pulse_808 else groove_class,
        )
        print(
            f"[BassGen][этап] bp_quant={bp_quant:.2f} phase={phase:.4f}s "
            f"bar_phase={(bar_phase if bar_phase is not None else -1):.4f}s "
            f"prefer_4/8={'1/3' if pulse_808 else '1'} pulse_808={int(pulse_808)}"
        )
    else:
        notes = _segments_to_chart_notes(
            segments,
            bpm=bpm,
            lanes=lanes,
            quant_strength=style_quant,
            allow_multi_lane=allow_multi_lane,
            allow_slides=allow_slides,
            phase=phase,
            prefer_eighths=True,
        )
        notes = _collapse_same_lane_taps(notes, bpm=bpm)
        print(f"[BassGen][этап] heuristic_quant={style_quant:.2f} phase={phase:.4f}s prefer_8th=1")

    if onsets and bar_phase is not None and groove_class != "sustain":
        notes, anchor_recap = _anchor_notes_to_downbeats(
            notes,
            onsets,
            bar_phase=float(bar_phase),
            bpm=bpm,
            meters=4,
            pull_strength=0.90 if goal_n == "arcade" else 0.82,
        )
        print(f"[BassGen][этап] downbeat_anchor={anchor_recap}")
    elif pulse_808 and bar_phase is not None:
        # 808s still need '1' lock even when classified sustain.
        notes, anchor_recap = _anchor_notes_to_downbeats(
            notes,
            onsets if onsets else _note_event_times(notes),
            bar_phase=float(bar_phase),
            bpm=bpm,
            meters=4,
            pull_strength=0.95,
        )
        print(f"[BassGen][этап] downbeat_anchor={anchor_recap} (808 pulse)")
    elif groove_class == "sustain":
        print("[BassGen][этап] downbeat_anchor=skip (BP-faithful sustain)")

    # Half-bar sanity: if stem is louder on "beat 3" grid, flip phase and re-anchor.
    if (
        y is not None
        and bar_phase is not None
        and groove_class != "sustain"
        and notes
    ):
        new_phase, flip_recap = _maybe_flip_half_bar_phase(
            float(bar_phase),
            notes,
            y,
            sr=sr,
            bpm=bpm,
            meters=4,
            onsets=onsets,
        )
        if int(flip_recap.get("flipped", 0)) == 1:
            bar_phase = new_phase
            print(f"[BassGen][этап] half_bar_flip=1 recap={flip_recap}")
            anchor_onsets = onsets if onsets else _note_event_times(notes)
            notes, anchor_recap = _anchor_notes_to_downbeats(
                notes,
                anchor_onsets,
                bar_phase=float(bar_phase),
                bpm=bpm,
                meters=4,
                pull_strength=0.92 if goal_n == "arcade" else 0.85,
            )
            print(f"[BassGen][этап] downbeat_anchor={anchor_recap} (after half_bar_flip)")
        else:
            print(f"[BassGen][этап] half_bar_flip=0 recap={flip_recap}")

    if pulse_808 and bar_phase is not None:
        notes, odd_moved = _snap_notes_to_odd_quarters(
            notes,
            bpm=bpm,
            bar_phase=float(bar_phase),
            strength=0.85 if goal_n == "arcade" else 0.78,
            onsets=onsets,
        )
        print(f"[BassGen][этап] odd_quarter_snap={odd_moved} (808 beats 1/3)")

    # Systematic lag: pull whole chart onto grid / stem attacks (keeps pattern shape).
    event_times = _note_event_times(notes)
    _print_bass_timing_offset_diagnostics("pre_shift", event_times, beats, beat_interval)
    grid_phase = float(bar_phase) if bar_phase is not None else (
        float(np.asarray(beats, dtype=float)[0]) if len(beats) else 0.0
    )
    t_max_ev = max(event_times) if event_times else 1.0
    median_16 = _median_grid_offset_sec(
        event_times, phase=grid_phase, step=beat_interval / 4.0, t_max=t_max_ev
    )
    median_8 = _median_grid_offset_sec(
        event_times, phase=grid_phase, step=beat_interval / 2.0, t_max=t_max_ev
    )
    # Prefer the coarser grid when the groove is clearly off an 8th (whole pattern drift).
    median_off = median_16
    off_src = "16th"
    if abs(median_8) >= 0.012 and abs(median_8) > abs(median_16) * 0.85:
        median_off = median_8
        off_src = "8th"
    onset_lag = (
        _median_onset_lag_sec(event_times, onsets, window_sec=max(0.070, beat_interval * 0.18))
        if onsets
        else 0.0
    )
    # Blend: stem attacks win when they agree with a clear lag direction.
    shift = 0.0
    if abs(onset_lag) >= 0.010 and (
        abs(median_off) < 0.008 or (onset_lag * median_off) >= 0.0
    ):
        shift = -onset_lag * 0.90
        off_src = f"onset+{off_src}"
    elif abs(median_off) >= 0.008:
        shift = -median_off * 0.92
    # Allow up to ~1/5 beat so a whole riff can land back on the strong pulse.
    max_shift = min(0.100, beat_interval * 0.22)
    if abs(shift) >= 0.006:
        shift = max(-max_shift, min(max_shift, shift))
        notes = _shift_note_times(notes, shift)
        print(
            f"[BassGen][этап] global_shift={shift * 1000.0:.1f}ms "
            f"(src={off_src} grid={median_off * 1000.0:.1f}ms "
            f"onset_lag={onset_lag * 1000.0:.1f}ms)"
        )
        _print_bass_timing_offset_diagnostics(
            "post_shift", _note_event_times(notes), beats, beat_interval
        )

    before_dedupe = len(notes)
    notes = dedupe_notes_same_lane_same_time(notes)
    if len(notes) != before_dedupe:
        print(f"[BassGen][этап] dedupe={before_dedupe}->{len(notes)}")
    before_dense = len(notes)
    notes, dense_hold_recap = _demote_holds_for_density(
        notes, bpm=bpm, groove_class=groove_class
    )
    print(f"[BassGen][этап] density_holds {before_dense}->{len(notes)} recap={dense_hold_recap}")
    if groove_class == "plucky":
        notes, dense_snap = _harden_quant_in_dense_windows(
            notes,
            bpm=bpm,
            phase=phase,
            min_density=2,
            strength=0.96 if diff_n == "dense" else 0.93,
        )
        if dense_snap:
            print(f"[BassGen][этап] dense_grid_snap={dense_snap} (plucky)")
    elif groove_class != "sustain":
        notes, dense_snap = _harden_quant_in_dense_windows(notes, bpm=bpm, phase=phase)
        if dense_snap:
            print(f"[BassGen][этап] dense_grid_snap={dense_snap}")
    else:
        print("[BassGen][этап] dense_grid_snap=skip (sustain groove)")
    print(
        f"[BassGen][этап] shape_convert notes={len(notes)} shapes={_shape_counts(notes)} "
        f"lanes={_lane_distribution(notes, lanes)}"
    )
    before_style = len(notes)
    allow_gap_at = None
    if y is not None:
        _gap_times, _gap_rms, _gap_thresh = _stem_energy_profile(y, sr=sr)
        if _gap_times is not None and _gap_rms is not None and _gap_thresh > 0.0:

            def allow_gap_at(t: float, _tt=_gap_times, _rr=_gap_rms, _th=_gap_thresh) -> bool:
                return _stem_is_audible(_tt, _rr, _th, t, window_sec=0.035)

    notes, style_recap = apply_bass_style(
        notes,
        goal=goal_n,
        bpm=bpm,
        lanes=lanes,
        allow_gap_fill_at=allow_gap_at,
        groove_class=groove_class,
    )
    if len(notes) != before_style:
        print(f"[BassGen][этап] style={before_style}->{len(notes)} recap={style_recap}")
    else:
        print(f"[BassGen][этап] style={before_style} recap={style_recap}")
    before_diff = len(notes)
    notes, diff_recap = apply_bass_difficulty(
        notes,
        goal=goal_n,
        difficulty=diff_n,
        bpm=bpm,
        lanes=lanes,
        transcription_faithful=transcription_faithful,
        allow_gap_fill_at=allow_gap_at,
    )
    before_overlap = len(notes)
    notes, overlap_recap = strip_same_lane_hold_overlaps(notes)
    if goal_n == "arcade":
        before_div = len(notes)
        notes, div_recap = diversify_lane_runs(notes, lanes=lanes, max_same_run=3)
        if int(div_recap.get("nudged", 0)) > 0:
            print(
                f"[BassGen][этап] lane_diversify nudged={div_recap.get('nudged')} "
                f"({before_div} notes, max_run={div_recap.get('max_run')})"
            )
    before_spread = len(notes)
    notes, spread_recap = spread_simultaneous_same_lane(
        notes, lanes=lanes, bpm=bpm
    )
    if int(spread_recap.get("nudged", 0)) > 0:
        print(
            f"[BassGen][этап] same_time_lane_spread={before_spread} recap={spread_recap}"
        )
    # Peak-lock toward stem, then re-magnet to grid on plucky spam (order matters).
    if onsets and groove_class != "sustain":
        before_lock = len(notes)
        lookback = max(0.036, beat_interval * 0.12)
        lookahead = max(0.018, beat_interval * 0.05)
        if groove_class == "plucky" and diff_n == "dense":
            # Slightly tighter: avoid yanking into noise peaks on dense spam.
            lookback = max(0.028, beat_interval * 0.10)
            lookahead = max(0.014, beat_interval * 0.04)
        elif groove_class == "mixed":
            lookback = max(0.032, beat_interval * 0.10)
            lookahead = max(0.016, beat_interval * 0.045)
        notes, lock_recap = _peak_lock_chart_notes(
            notes,
            onsets,
            bpm=bpm,
            lookback_sec=lookback,
            lookahead_sec=lookahead,
        )
        if int(lock_recap.get("locked", 0)) > 0:
            print(f"[BassGen][этап] peak_lock={before_lock} recap={lock_recap}")
        if groove_class == "plucky":
            notes, re_snap = _harden_quant_in_dense_windows(
                notes,
                bpm=bpm,
                phase=phase,
                min_density=2,
                strength=0.94 if diff_n == "dense" else 0.90,
            )
            if re_snap:
                print(f"[BassGen][этап] dense_grid_resnap={re_snap} (after peak_lock)")
    elif groove_class == "sustain":
        print("[BassGen][этап] peak_lock=skip (sustain groove)")

    if y is not None and bar_phase is not None:
        before_pat = len(notes)
        notes, pat_recap = _fill_bars_from_template(
            notes,
            y,
            sr=sr,
            bpm=bpm,
            bar_phase=float(bar_phase),
            lanes=lanes,
            goal=goal_n,
            groove_class=groove_class,
        )
        print(
            f"[BassGen][этап] pattern_copy={before_pat}->{len(notes)} recap={pat_recap}"
        )

    # Pattern copy / peak-lock can reintroduce same-lane piles — spread again.
    before_spread2 = len(notes)
    notes, spread2 = spread_simultaneous_same_lane(notes, lanes=lanes, bpm=bpm)
    if int(spread2.get("nudged", 0)) > 0:
        print(
            f"[BassGen][этап] same_time_lane_spread_final={before_spread2} recap={spread2}"
        )

    if y is not None:
        before_cg = len(notes)
        notes, cg_recap = _gate_chart_notes_by_stem_energy(notes, y, sr=sr, bpm=bpm)
        print(
            f"[BassGen][этап] stem_gate_notes={before_cg}->{len(notes)} recap={cg_recap}"
        )
    notes = _strip_peak_locked_flags(notes)
    ghost_marked = sum(1 for n in notes if n.get("ghost"))
    if goal_n == "original":
        notes, ghost_stripped = _strip_ghosts(notes, aggressive=True)
        print(
            f"[BassGen][этап] ghost_marked={ghost_marked} kept=0 strip=-{ghost_stripped} (original)"
        )
    else:
        notes, ghost_cap = _cap_ghost_notes(notes, max_ratio=0.08)
        ghost_kept = sum(1 for n in notes if n.get("ghost"))
        print(
            f"[BassGen][этап] ghost_marked={ghost_marked} kept={ghost_kept} "
            f"cap_demote={ghost_cap}"
        )
    notes, slide_demote = demote_slides_to_holds(notes)
    if slide_demote:
        print(f"[BassGen][этап] slide_demote={slide_demote} (RFALL_BASS_SLIDES=1 to keep slides)")
    if not allow_slides and slide_demote == 0:
        print("[BassGen] slides=off (beta — hold-only; set RFALL_BASS_SLIDES=1 to experiment)")
    if len(notes) != before_diff:
        print(f"[BassGen][этап] difficulty={before_diff}->{before_overlap} recap={diff_recap}")
    else:
        print(f"[BassGen][этап] difficulty={before_diff} recap={diff_recap}")
    if len(notes) != before_overlap:
        print(f"[BassGen][этап] lane_overlap={before_overlap}->{len(notes)} recap={overlap_recap}")

    if bass_entry > 0.05:
        before_entry = len(notes)
        notes, dropped_notes = _trim_before_entry(
            notes, bass_entry, time_key="time", grace_sec=beat_interval * 0.15
        )
        if dropped_notes:
            print(
                f"[BassGen][этап] bass_entry_trim notes=-{dropped_notes} "
                f"({before_entry}->{len(notes)})"
            )

    shapes = _shape_counts(notes)
    print(
        f"[BassGen][perf] chart_build={time.perf_counter() - _build_t0:.1f}s "
        f"(post-BP: onsets/HPSS/fuse/gate/style)"
    )
    print(
        f"[BassGen] Итого: {len(notes)} notes (segments={len(segments)}, source={segment_source}) "
        f"shapes={shapes} goal={goal_n} difficulty={diff_n}"
    )
    _print_bass_recap(
        notes=notes,
        segments=segments,
        segment_source=segment_source,
        bpm=bpm,
        lanes=lanes,
        goal=goal_n,
        difficulty=diff_n,
        style_recap=style_recap,
        diff_recap=diff_recap,
        onsets=len(onsets),
        phase=phase,
    )
    _print_bass_timing_offset_diagnostics(
        "final", _note_event_times(notes), beats, beat_interval
    )
    return notes


def save_generated_bass(
    notes: List[Dict[str, Any]],
    song_path: str,
    *,
    chart_stem: str = "original",
    lanes: int = CANONICAL_MAX_LANES,
    artist: str = "",
    title: str = "",
    chart_id: str = "",
) -> bool:
    return save_bass_notes(
        notes,
        song_path,
        chart_intent=chart_stem,
        chart_stem=chart_stem,
        lanes=lanes,
        artist=artist,
        title=title,
        chart_id=chart_id,
    )
