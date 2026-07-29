"""Automated chart quality metrics (Tier 1 + 2 regression helpers).

Analyzes .rf charts without claiming "good chart" — flags suspicious regions
and compares to baseline / expected behaviour profiles.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

from app.rfc_chart_codec import read_file


@dataclass
class RegressionWarning:
    code: str
    severity: str  # FAIL | WARN
    message: str
    time_sec: Optional[float] = None
    measure: Optional[int] = None  # 1-based for humans

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChartAnalysis:
    chart_path: str
    bpm: float
    note_count: int
    duration_sec: float
    peak_notes_per_second: float
    median_notes_per_measure: float
    lane_histogram: Dict[str, int] = field(default_factory=dict)
    drum_histogram: Optional[Dict[str, int]] = None
    rms_density_correlation: Optional[float] = None
    warnings: List[RegressionWarning] = field(default_factory=list)
    measure_counts: Dict[int, int] = field(default_factory=dict)

    def fail_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "FAIL")

    def warn_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "WARN")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_path": self.chart_path,
            "bpm": self.bpm,
            "note_count": self.note_count,
            "duration_sec": self.duration_sec,
            "peak_notes_per_second": self.peak_notes_per_second,
            "median_notes_per_measure": self.median_notes_per_measure,
            "lane_histogram": self.lane_histogram,
            "drum_histogram": self.drum_histogram,
            "rms_density_correlation": self.rms_density_correlation,
            "fail_count": self.fail_count(),
            "warn_count": self.warn_count(),
            "warnings": [w.to_dict() for w in self.warnings],
            "measure_counts": {str(k): v for k, v in sorted(self.measure_counts.items())},
        }


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy is required for chart regression")


def beat_interval_from_bpm(bpm: float) -> float:
    return 60.0 / max(1.0, float(bpm))


def measure_duration_from_bpm(bpm: float) -> float:
    return beat_interval_from_bpm(bpm) * 4.0


def measure_index(
    time_sec: float,
    bpm: float,
    measure_offset: float = 0.0,
) -> int:
    md = measure_duration_from_bpm(bpm)
    return int(math.floor((float(time_sec) - float(measure_offset)) / md))


def counts_per_measure(
    notes: Sequence[dict[str, Any]],
    bpm: float,
    measure_offset: float = 0.0,
) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    md = measure_duration_from_bpm(bpm)
    for note in notes:
        t = float(note.get("time", 0.0))
        idx = int(math.floor((t - float(measure_offset)) / md))
        if idx < 0:
            continue
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def times_per_measure(
    notes: Sequence[dict[str, Any]],
    bpm: float,
    measure_offset: float = 0.0,
) -> Dict[int, List[float]]:
    buckets: Dict[int, List[float]] = {}
    md = measure_duration_from_bpm(bpm)
    for note in notes:
        t = float(note.get("time", 0.0))
        idx = int(math.floor((t - float(measure_offset)) / md))
        if idx < 0:
            continue
        buckets.setdefault(idx, []).append(t)
    return buckets


def lane_histogram(notes: Sequence[dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for note in notes:
        lane = int(note.get("lane", 0))
        key = f"lane_{lane}"
        out[key] = out.get(key, 0) + 1
    return out


def load_hits_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"hits json must be object: {path}")
    return data


def drum_histogram_from_hits(hits: dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if isinstance(hits.get("classified_hits"), list):
        for h in hits["classified_hits"]:
            if isinstance(h, dict):
                drum = str(h.get("drum", "unknown")).lower()
                out[drum] = out.get(drum, 0) + 1
        return out
    for key in ("kick", "snare", "hat", "tom", "cymbal", "perc"):
        times = hits.get(f"{key}_times") or hits.get(f"{key}s") or hits.get(key)
        if isinstance(times, list):
            out[key] = len(times)
    return out


def peak_notes_per_second(
    notes: Sequence[dict[str, Any]],
    window_sec: float = 1.0,
) -> Tuple[float, Optional[float]]:
    if not notes:
        return 0.0, None
    times = sorted(float(n.get("time", 0.0)) for n in notes)
    best = 0
    best_center = times[0]
    left = 0
    for right, t_end in enumerate(times):
        while times[right] - times[left] > window_sec:
            left += 1
        count = right - left + 1
        if count > best:
            best = count
            best_center = (times[left] + times[right]) / 2.0
    return float(best), best_center


def check_impossible_bursts(
    notes: Sequence[dict[str, Any]],
    max_nps: float,
    window_sec: float = 1.0,
) -> List[RegressionWarning]:
    if not notes:
        return []
    times = sorted(float(n.get("time", 0.0)) for n in notes)
    warnings: List[RegressionWarning] = []
    left = 0
    for right, _ in enumerate(times):
        while times[right] - times[left] > window_sec:
            left += 1
        count = right - left + 1
        if count > max_nps:
            center = (times[left] + times[right]) / 2.0
            warnings.append(
                RegressionWarning(
                    code="IMPOSSIBLE_BURST",
                    severity="FAIL",
                    message=f"{count} notes in {window_sec:.1f}s (max {max_nps:.0f})",
                    time_sec=center,
                )
            )
            break
    peak, peak_t = peak_notes_per_second(notes, window_sec)
    if peak > max_nps * 0.9 and not warnings:
        warnings.append(
            RegressionWarning(
                code="HIGH_BURST",
                severity="WARN",
                message=f"peak {peak:.0f} notes/{window_sec:.1f}s (limit {max_nps:.0f})",
                time_sec=peak_t,
            )
        )
    return warnings


def check_density_variance(
    measure_counts: Dict[int, int],
    neighbor_window: int = 2,
    neighbor_min_avg: float = 6.0,
    drop_ratio: float = 0.45,
) -> List[RegressionWarning]:
    if not measure_counts:
        return []
    warnings: List[RegressionWarning] = []
    for m, count in sorted(measure_counts.items()):
        neighbors: List[int] = []
        for d in range(-neighbor_window, neighbor_window + 1):
            if d == 0:
                continue
            v = measure_counts.get(m + d)
            if v is not None:
                neighbors.append(v)
        if len(neighbors) < 2:
            continue
        avg = sum(neighbors) / len(neighbors)
        if avg < neighbor_min_avg:
            continue
        if count < avg * drop_ratio:
            warnings.append(
                RegressionWarning(
                    code="DENSITY_DROP",
                    severity="WARN",
                    message=f"m{m + 1}: {count} notes (neighbors avg {avg:.1f})",
                    measure=m + 1,
                )
            )
        elif count > avg * (1.0 / max(drop_ratio, 0.01)):
            warnings.append(
                RegressionWarning(
                    code="DENSITY_SPIKE",
                    severity="WARN",
                    message=f"m{m + 1}: {count} notes (neighbors avg {avg:.1f})",
                    measure=m + 1,
                )
            )
    return warnings


def check_broken_patterns(
    notes: Sequence[dict[str, Any]],
    bpm: float,
    measure_offset: float = 0.0,
    dense_min_notes: int = 8,
    slots: int = 16,
) -> List[RegressionWarning]:
    """Single 16th-slot hole surrounded by hits in a dense measure."""
    warnings: List[RegressionWarning] = []
    md = measure_duration_from_bpm(bpm)
    slot_dur = md / float(slots)
    by_measure = times_per_measure(notes, bpm, measure_offset)
    for m, times in sorted(by_measure.items()):
        if len(times) < dense_min_notes:
            continue
        occupied = set()
        for t in times:
            local = t - measure_offset - m * md
            slot = int(round(local / slot_dur))
            slot = max(0, min(slots - 1, slot))
            occupied.add(slot)
        for s in range(1, slots - 1):
            if s in occupied:
                continue
            if (s - 1) in occupied and (s + 1) in occupied:
                if (s - 2) in occupied or (s + 2) in occupied:
                    t_sec = measure_offset + m * md + (s + 0.5) * slot_dur
                    warnings.append(
                        RegressionWarning(
                            code="BROKEN_PATTERN",
                            severity="WARN",
                            message=f"m{m + 1} slot {s + 1}/{slots} isolated gap in dense measure ({len(times)} notes)",
                            measure=m + 1,
                            time_sec=t_sec,
                        )
                    )
                    break
    return warnings


def stem_rms_per_measure(
    audio_path: Path,
    bpm: float,
    max_measure: int,
    measure_offset: float = 0.0,
) -> Dict[int, float]:
    if librosa is None or not audio_path.is_file():
        return {}
    _require_numpy()
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True, dtype="float32")
    except Exception:
        return {}
    if len(y) < 1 or sr <= 0:
        return {}
    md = measure_duration_from_bpm(bpm)
    out: Dict[int, float] = {}
    for m in range(0, max_measure + 1):
        start = measure_offset + m * md
        end = start + md
        s0 = max(0, int(round(start * sr)))
        s1 = min(len(y), int(round(end * sr)))
        if s1 <= s0:
            continue
        chunk = y[s0:s1]
        out[m] = float(np.sqrt(np.mean(chunk * chunk)))
    return out


def check_silence_overchart(
    measure_counts: Dict[int, int],
    stem_rms: Dict[int, float],
    quiet_ratio: float = 0.42,
    quiet_floor: float = 0.006,
    min_notes: int = 4,
) -> List[RegressionWarning]:
    if not stem_rms:
        return []
    vals = [v for v in stem_rms.values() if v > 0]
    median = float(sorted(vals)[len(vals) // 2]) if vals else 0.0
    warnings: List[RegressionWarning] = []
    for m, count in sorted(measure_counts.items()):
        if count < min_notes:
            continue
        rms = float(stem_rms.get(m, 0.0) or 0.0)
        quiet = rms <= quiet_floor or (median > 0 and rms < median * quiet_ratio)
        if quiet:
            warnings.append(
                RegressionWarning(
                    code="SILENCE_OVERCHART",
                    severity="WARN",
                    message=f"m{m + 1}: {count} notes but low stem RMS ({rms:.4f})",
                    measure=m + 1,
                )
            )
    return warnings


def rms_density_correlation(
    measure_counts: Dict[int, int],
    stem_rms: Dict[int, float],
) -> Optional[float]:
    if not stem_rms or not measure_counts:
        return None
    _require_numpy()
    keys = sorted(set(measure_counts.keys()) & set(stem_rms.keys()))
    if len(keys) < 6:
        return None
    x = np.array([stem_rms[k] for k in keys], dtype=float)
    y = np.array([float(measure_counts.get(k, 0)) for k in keys], dtype=float)
    if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def check_rms_correlation(
    correlation: Optional[float],
    min_corr: float = 0.35,
    min_measures: int = 16,
    measure_counts: Optional[Dict[int, int]] = None,
) -> List[RegressionWarning]:
    if correlation is None:
        return []
    if measure_counts and len(measure_counts) < min_measures:
        return []
    if correlation >= min_corr:
        return []
    return [
        RegressionWarning(
            code="LOW_RMS_DENSITY_CORR",
            severity="WARN",
            message=f"stem RMS vs note density r={correlation:.2f} (expected >= {min_corr:.2f})",
        )
    ]


def check_under_overchart_windows(
    measure_counts: Dict[int, int],
    stem_rms: Dict[int, float],
    window: int = 2,
    energy_delta_ratio: float = 0.35,
    density_delta_ratio: float = 0.35,
) -> List[RegressionWarning]:
    """Energy ↑ notes ↓ or energy ↓ notes ↑ across neighboring measure groups."""
    if not stem_rms or not measure_counts:
        return []
    warnings: List[RegressionWarning] = []
    measures = sorted(set(measure_counts.keys()) & set(stem_rms.keys()))
    if len(measures) < window * 2 + 1:
        return []

    def block_avg(ms: List[int], src: Dict[int, float]) -> float:
        vals = [float(src.get(m, 0.0)) for m in ms]
        return sum(vals) / max(1, len(vals))

    for i in range(window, len(measures) - window):
        prev_m = measures[i - window : i]
        next_m = measures[i + 1 : i + window + 1]
        cur = measures[i]
        rms_prev = block_avg(prev_m, stem_rms)
        rms_next = block_avg(next_m, stem_rms)
        dens_prev = block_avg(prev_m, {k: float(v) for k, v in measure_counts.items()})
        dens_next = block_avg(next_m, {k: float(v) for k, v in measure_counts.items()})
        rms_cur = float(stem_rms.get(cur, 0.0))
        dens_cur = float(measure_counts.get(cur, 0))

        if rms_cur > max(rms_prev, rms_next) * (1.0 + energy_delta_ratio):
            if dens_cur < min(dens_prev, dens_next) * (1.0 - density_delta_ratio):
                md = measure_duration_from_bpm(120)  # time hint only
                warnings.append(
                    RegressionWarning(
                        code="POTENTIAL_UNDERCHART",
                        severity="WARN",
                        message=f"m{cur + 1}: energy up, notes down ({dens_cur:.0f} vs ~{min(dens_prev, dens_next):.0f})",
                        measure=cur + 1,
                    )
                )
        if rms_cur < min(rms_prev, rms_next) * (1.0 - energy_delta_ratio):
            if dens_cur > max(dens_prev, dens_next) * (1.0 + density_delta_ratio):
                warnings.append(
                    RegressionWarning(
                        code="POTENTIAL_OVERCHART",
                        severity="WARN",
                        message=f"m{cur + 1}: energy down, notes up ({dens_cur:.0f} vs ~{max(dens_prev, dens_next):.0f})",
                        measure=cur + 1,
                    )
                )
    return warnings


def four_bar_signatures(measure_counts: Dict[int, int]) -> Dict[int, Tuple[int, ...]]:
    sigs: Dict[int, Tuple[int, ...]] = {}
    if not measure_counts:
        return sigs
    max_m = max(measure_counts.keys())
    for start in range(0, max_m + 1, 4):
        block = tuple(int(measure_counts.get(start + i, 0)) for i in range(4))
        if sum(block) == 0:
            continue
        sigs[start] = block
    return sigs


def check_repetition_consistency(
    measure_counts: Dict[int, int],
    similarity_tolerance: int = 1,
    note_delta_ratio: float = 0.25,
) -> List[RegressionWarning]:
    """Similar 4-bar density shapes far apart should have similar total notes."""
    sigs = four_bar_signatures(measure_counts)
    if len(sigs) < 2:
        return []
    warnings: List[RegressionWarning] = []
    entries = list(sigs.items())

    def similar(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
        if len(a) != len(b):
            return False
        return sum(abs(x - y) for x, y in zip(a, b)) <= similarity_tolerance * len(a)

    for i, (start_a, sig_a) in enumerate(entries):
        total_a = sum(sig_a)
        if total_a < 4:
            continue
        for start_b, sig_b in entries[i + 1 :]:
            if abs(start_b - start_a) < 8:
                continue
            if not similar(sig_a, sig_b):
                continue
            total_b = sum(sig_b)
            if total_a == 0:
                continue
            delta = abs(total_b - total_a) / float(total_a)
            if delta > note_delta_ratio:
                warnings.append(
                    RegressionWarning(
                        code="REPETITION_MISMATCH",
                        severity="WARN",
                        message=(
                            f"4-bar blocks m{start_a + 1} ({total_a} notes) vs "
                            f"m{start_b + 1} ({total_b} notes) look alike but differ {delta * 100:.0f}%"
                        ),
                        measure=start_b + 1,
                    )
                )
                break
    return warnings[:12]


def check_histogram_drift(
    current: Dict[str, int],
    baseline: Dict[str, int],
    max_ratio_change: float = 0.5,
    min_baseline_count: int = 20,
) -> List[RegressionWarning]:
    warnings: List[RegressionWarning] = []
    keys = set(current.keys()) | set(baseline.keys())
    for key in sorted(keys):
        b = int(baseline.get(key, 0))
        c = int(current.get(key, 0))
        if b < min_baseline_count:
            continue
        ratio = abs(c - b) / float(b)
        if ratio > max_ratio_change:
            warnings.append(
                RegressionWarning(
                    code="HISTOGRAM_DRIFT",
                    severity="WARN",
                    message=f"{key}: {b} -> {c} ({(c - b) / b * 100:+.0f}%)",
                )
            )
    return warnings


def check_expected_behaviour(
    measure_counts: Dict[int, int],
    note_count: int,
    peak_nps: float,
    expect: dict[str, Any],
) -> List[RegressionWarning]:
    warnings: List[RegressionWarning] = []
    if not expect:
        return warnings

    intro_measures = int(expect.get("intro_measures", 0) or 0)
    intro_max_per = int(expect.get("intro_max_notes_per_measure", 999) or 999)
    intro_max_total = int(expect.get("intro_max_total_notes", 999999) or 999999)
    if intro_measures > 0:
        intro_total = sum(measure_counts.get(m, 0) for m in range(intro_measures))
        if intro_total > intro_max_total:
            warnings.append(
                RegressionWarning(
                    code="EXPECTED_INTRO_SPARSE",
                    severity="WARN",
                    message=f"intro (m1–{intro_measures}): {intro_total} notes (max {intro_max_total})",
                )
            )
        for m in range(intro_measures):
            c = measure_counts.get(m, 0)
            if c > intro_max_per:
                warnings.append(
                    RegressionWarning(
                        code="EXPECTED_INTRO_SPARSE",
                        severity="WARN",
                        message=f"m{m + 1}: {c} notes (intro max {intro_max_per}/measure)",
                        measure=m + 1,
                    )
                )

    max_burst = expect.get("max_burst_notes_per_second")
    if max_burst is not None and peak_nps > float(max_burst):
        warnings.append(
            RegressionWarning(
                code="EXPECTED_MAX_BURST",
                severity="WARN",
                message=f"peak {peak_nps:.1f} nps exceeds expected max {float(max_burst):.1f}",
            )
        )

    blast_range = expect.get("blast_measure_range")
    blast_min = expect.get("blast_min_notes_per_measure")
    if isinstance(blast_range, list) and len(blast_range) >= 2 and blast_min is not None:
        lo, hi = int(blast_range[0]), int(blast_range[1])
        for m in range(lo - 1, hi):
            c = measure_counts.get(m, 0)
            if c > 0 and c < int(blast_min):
                warnings.append(
                    RegressionWarning(
                        code="EXPECTED_BLAST_DENSITY",
                        severity="WARN",
                        message=f"m{m + 1}: {c} notes (blast min {blast_min})",
                        measure=m + 1,
                    )
                )
                if len([w for w in warnings if w.code == "EXPECTED_BLAST_DENSITY"]) >= 5:
                    break

    max_median = expect.get("max_median_notes_per_measure")
    if max_median is not None and measure_counts:
        med = float(sorted(measure_counts.values())[len(measure_counts.values()) // 2])
        if med > float(max_median):
            warnings.append(
                RegressionWarning(
                    code="EXPECTED_MAX_MEDIAN_DENSITY",
                    severity="WARN",
                    message=f"median {med:.1f} notes/measure (max {float(max_median):.1f})",
                )
            )

    return warnings


def compare_to_baseline(
    current: ChartAnalysis,
    baseline: dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    b_count = int(baseline.get("note_count", 0))
    if b_count > 0:
        delta = (current.note_count - b_count) / float(b_count) * 100.0
        lines.append(f"Notes: {current.note_count} ({delta:+.1f}% vs baseline {b_count})")
    b_peak = float(baseline.get("peak_notes_per_second", 0) or 0)
    if b_peak > 0:
        delta = (current.peak_notes_per_second - b_peak) / b_peak * 100.0
        lines.append(
            f"Peak NPS: {current.peak_notes_per_second:.1f} ({delta:+.1f}% vs {b_peak:.1f})"
        )
    b_fail = int(baseline.get("fail_count", 0))
    b_warn = int(baseline.get("warn_count", 0))
    lines.append(
        f"Warnings: FAIL {current.fail_count()} ({current.fail_count() - b_fail:+d}), "
        f"WARN {current.warn_count()} ({current.warn_count() - b_warn:+d})"
    )
    b_lane = baseline.get("lane_histogram") or {}
    if b_lane:
        drift = check_histogram_drift(current.lane_histogram, b_lane)
        for w in drift[:5]:
            lines.append(f"  lane drift: {w.message}")
    b_drum = baseline.get("drum_histogram")
    if b_drum and current.drum_histogram:
        drift = check_histogram_drift(current.drum_histogram, b_drum)
        for w in drift[:5]:
            lines.append(f"  drum drift: {w.message}")
    return lines


def analyze_chart(
    chart_path: Path,
    bpm: float,
    *,
    measure_offset: float = 0.0,
    audio_path: Optional[Path] = None,
    hits_path: Optional[Path] = None,
    expect: Optional[dict[str, Any]] = None,
    baseline: Optional[dict[str, Any]] = None,
    defaults: Optional[dict[str, Any]] = None,
) -> ChartAnalysis:
    defaults = defaults or {}
    notes = read_file(chart_path)
    times = [float(n.get("time", 0.0)) for n in notes]
    duration = max(times) if times else 0.0
    measure_counts = counts_per_measure(notes, bpm, measure_offset)
    peak_nps, _ = peak_notes_per_second(notes)
    med = 0.0
    if measure_counts:
        med = float(sorted(measure_counts.values())[len(measure_counts.values()) // 2])

    drum_hist: Optional[Dict[str, int]] = None
    if hits_path and hits_path.is_file():
        drum_hist = drum_histogram_from_hits(load_hits_json(hits_path))

    analysis = ChartAnalysis(
        chart_path=str(chart_path),
        bpm=float(bpm),
        note_count=len(notes),
        duration_sec=duration,
        peak_notes_per_second=peak_nps,
        median_notes_per_measure=med,
        lane_histogram=lane_histogram(notes),
        drum_histogram=drum_hist,
        measure_counts=measure_counts,
    )

    max_nps = float(defaults.get("burst_max_notes_per_second", 20))
    if expect and expect.get("max_burst_notes_per_second") is not None:
        max_nps = max(max_nps, float(expect["max_burst_notes_per_second"]) * 1.5)

    analysis.warnings.extend(check_impossible_bursts(notes, max_nps=max_nps))
    analysis.warnings.extend(
        check_density_variance(
            measure_counts,
            neighbor_window=int(defaults.get("density_neighbor_window", 2)),
            neighbor_min_avg=float(defaults.get("density_neighbor_min_avg", 6)),
            drop_ratio=float(defaults.get("density_drop_ratio", 0.45)),
        )
    )
    analysis.warnings.extend(
        check_broken_patterns(notes, bpm, measure_offset=measure_offset)
    )
    analysis.warnings.extend(check_repetition_consistency(measure_counts))

    max_m = max(measure_counts.keys()) if measure_counts else 0
    stem_rms: Dict[int, float] = {}
    if audio_path:
        stem_rms = stem_rms_per_measure(audio_path, bpm, max_m, measure_offset)
        analysis.rms_density_correlation = rms_density_correlation(measure_counts, stem_rms)
        analysis.warnings.extend(
            check_rms_correlation(
                analysis.rms_density_correlation,
                min_corr=float(defaults.get("min_rms_correlation", 0.35)),
                measure_counts=measure_counts,
            )
        )
        analysis.warnings.extend(check_silence_overchart(measure_counts, stem_rms))
        analysis.warnings.extend(check_under_overchart_windows(measure_counts, stem_rms))

    if expect:
        analysis.warnings.extend(
            check_expected_behaviour(measure_counts, analysis.note_count, peak_nps, expect)
        )

    if baseline:
        lane_drift = check_histogram_drift(analysis.lane_histogram, baseline.get("lane_histogram") or {})
        drum_base = baseline.get("drum_histogram")
        if drum_base and analysis.drum_histogram:
            lane_drift.extend(check_histogram_drift(analysis.drum_histogram, drum_base))
        analysis.warnings.extend(lane_drift)

    return analysis


def format_analysis_report(
    analysis: ChartAnalysis,
    *,
    title: str = "",
    baseline: Optional[dict[str, Any]] = None,
) -> str:
    lines: List[str] = []
    if title:
        lines.append(f"=== {title} ===")
    lines.append(f"Chart: {analysis.chart_path}")
    lines.append(
        f"BPM: {analysis.bpm:g}  Notes: {analysis.note_count}  "
        f"Duration: {analysis.duration_sec:.0f}s  "
        f"Peak NPS: {analysis.peak_notes_per_second:.1f}  "
        f"Median/measure: {analysis.median_notes_per_measure:.1f}"
    )
    if analysis.rms_density_correlation is not None:
        lines.append(f"RMS vs density r: {analysis.rms_density_correlation:.2f}")
    if analysis.lane_histogram:
        parts = [f"{k}={v}" for k, v in sorted(analysis.lane_histogram.items())]
        lines.append("Lanes: " + " ".join(parts))
    if analysis.drum_histogram:
        parts = [f"{k}={v}" for k, v in sorted(analysis.drum_histogram.items())]
        lines.append("Drums (hits json): " + " ".join(parts))
    lines.append("")

    if not analysis.warnings:
        lines.append("OK - no automated warnings.")
    else:
        for w in analysis.warnings:
            loc = ""
            if w.measure is not None:
                loc = f" m{w.measure}"
            elif w.time_sec is not None:
                m = int(w.time_sec // 60)
                s = w.time_sec - m * 60
                loc = f" @ {m}:{s:04.1f}"
            lines.append(f"[{w.severity}] {w.code}{loc} - {w.message}")

    if baseline:
        lines.append("")
        lines.append("--- vs baseline ---")
        lines.extend(compare_to_baseline(analysis, baseline))

    status = "FAIL" if analysis.fail_count() else ("WARN" if analysis.warn_count() else "OK")
    lines.append("")
    lines.append(f"Summary: {status} ({analysis.fail_count()} fail, {analysis.warn_count()} warn)")
    return "\n".join(lines) + "\n"


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    return data


def match_track_manifest(chart_path: Path, manifest: dict[str, Any]) -> Optional[dict[str, Any]]:
    name = chart_path.as_posix().lower()
    for track in manifest.get("tracks", []):
        if not isinstance(track, dict):
            continue
        for fragment in track.get("match", []) or []:
            if str(fragment).lower() in name:
                return track
        if track.get("id") and str(track["id"]).lower() in name:
            return track
    return None
