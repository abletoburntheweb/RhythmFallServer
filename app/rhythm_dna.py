# app/rhythm_dna.py
"""Rhythm DNA v0 — structured generation report for client UI (no LLM)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DNA_VERSION = "0.2"
RFD_HEADER = "# RFD {version}\n# RhythmFall generation passport\n"


def serialize_rfd(payload: Dict[str, Any]) -> str:
    return RFD_HEADER.format(version=DNA_VERSION) + json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def parse_rfd(text: str) -> Dict[str, Any]:
    body = str(text or "").strip()
    if body.startswith("#"):
        lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped == "":
                continue
            lines.append(line)
        body = "\n".join(lines).strip()
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _level_high_med_low(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value <= low:
        return "low"
    return "medium"


_SEGMENT_LABEL_KEYS = {
    "quiet": "DNA_SEG_QUIET",
    "sparse": "DNA_SEG_SPARSE",
    "steady": "DNA_SEG_STEADY",
    "dense": "DNA_SEG_DENSE",
    "loud_quiet": "DNA_SEG_LOUD_QUIET",
}


def _measure_segment_kind(row: Dict[str, Any]) -> str:
    notes = int(row.get("notes") or 0)
    d_rel = float(row.get("drum_rel") or 0)
    x_rel = float(row.get("mix_rel") or 0)
    if x_rel >= 0.85 and d_rel < 0.55:
        return "loud_quiet"
    if notes <= 0 and d_rel < 0.45:
        return "quiet"
    if notes <= 2:
        return "sparse"
    if notes >= 6:
        return "dense"
    return "steady"


def _repeat_id_letters(measure_rows: List[Dict[str, Any]]) -> Dict[int, str]:
    id_to_letter: Dict[int, str] = {}
    for row in measure_rows:
        if not isinstance(row, dict):
            continue
        rid = int(row.get("repeat_id", -1) or -1)
        if rid >= 0 and rid not in id_to_letter:
            id_to_letter[rid] = chr(ord("A") + len(id_to_letter))
    return id_to_letter


def _dominant_block_letter(
    start_m: int,
    end_m: int,
    rows_by_m: Dict[int, Dict[str, Any]],
    id_to_letter: Dict[int, str],
) -> str:
    counts: Dict[str, int] = {}
    for m in range(start_m, end_m + 1):
        row = rows_by_m.get(m)
        if not row:
            continue
        rid = int(row.get("repeat_id", -1) or -1)
        if rid < 0:
            continue
        letter = id_to_letter.get(rid, "")
        if letter:
            counts[letter] = counts.get(letter, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda key: counts[key])


def build_structure_blocks_pattern(measure_rows: Optional[List[Dict[str, Any]]]) -> str:
    if not measure_rows:
        return ""
    id_to_letter = _repeat_id_letters(measure_rows)
    if not id_to_letter:
        return ""
    parts: List[str] = []
    last = ""
    for row in measure_rows:
        if not isinstance(row, dict):
            continue
        rid = int(row.get("repeat_id", -1) or -1)
        if rid < 0:
            continue
        letter = id_to_letter.get(rid, "")
        if letter and letter != last:
            parts.append(letter)
            last = letter
    return " · ".join(parts[:32])


def _finalize_timeline_segment(
    seg: Dict[str, Any],
    measure_duration: float,
    rows_by_m: Dict[int, Dict[str, Any]],
    id_to_letter: Dict[int, str],
) -> Dict[str, Any]:
    start_m = int(seg["start_m"])
    end_m = int(seg["end_m"])
    kind = str(seg["kind"])
    block = _dominant_block_letter(start_m, end_m, rows_by_m, id_to_letter)
    payload = {
        "kind": kind,
        "label_key": _SEGMENT_LABEL_KEYS.get(kind, "DNA_SEG_STEADY"),
        "start_s": round(start_m * measure_duration, 1),
        "end_s": round((end_m + 1) * measure_duration, 1),
        "measures": int(seg["measures"]),
        "notes": int(seg["notes"]),
    }
    if block:
        payload["block"] = block
    return payload


MIN_STRUCTURE_MEASURES = 4


def _coalesce_structure_timeline(
    segments: List[Dict[str, Any]],
    measure_duration: float,
) -> List[Dict[str, Any]]:
    if not segments or measure_duration <= 0:
        return segments
    min_sec = measure_duration * float(MIN_STRUCTURE_MEASURES)
    coalesced: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start_s = float(seg.get("start_s", 0.0) or 0.0)
        end_s = max(float(seg.get("end_s", start_s) or start_s), start_s)
        kind = str(seg.get("kind", "steady"))
        if not coalesced:
            coalesced.append({**seg, "start_s": start_s, "end_s": end_s, "kind": kind})
            continue
        last = coalesced[-1]
        group_start = float(last.get("start_s", 0.0) or 0.0)
        if start_s - group_start < min_sec:
            last["end_s"] = max(float(last.get("end_s", 0.0) or 0.0), end_s)
            last["measures"] = int(last.get("measures", 0) or 0) + int(seg.get("measures", 0) or 0)
            last["notes"] = int(last.get("notes", 0) or 0) + int(seg.get("notes", 0) or 0)
        else:
            coalesced.append({**seg, "start_s": start_s, "end_s": end_s, "kind": kind})
    return coalesced


def build_structure_timeline(
    measure_rows: Optional[List[Dict[str, Any]]],
    bpm: float,
) -> List[Dict[str, Any]]:
    if not measure_rows or bpm <= 0:
        return []
    measure_duration = (60.0 / float(bpm)) * 4.0
    rows_by_m = {int(row.get("m", 0) or 0): row for row in measure_rows if isinstance(row, dict)}
    id_to_letter = _repeat_id_letters(measure_rows)
    segments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for row in measure_rows:
        if not isinstance(row, dict):
            continue
        m = int(row.get("m", 0) or 0)
        kind = _measure_segment_kind(row)
        notes = int(row.get("notes") or 0)
        if current and current["kind"] == kind:
            current["end_m"] = m
            current["notes"] += notes
            current["measures"] += 1
        else:
            if current:
                segments.append(_finalize_timeline_segment(current, measure_duration, rows_by_m, id_to_letter))
            current = {"kind": kind, "start_m": m, "end_m": m, "measures": 1, "notes": notes}
    if current:
        segments.append(_finalize_timeline_segment(current, measure_duration, rows_by_m, id_to_letter))
    segments = _coalesce_structure_timeline(segments, measure_duration)
    return segments[:32]


def _median_notes_per_measure(measure_rows: Optional[List[Dict[str, Any]]]) -> float:
    if not measure_rows:
        return 1.0
    values = [float(row.get("notes", 0) or 0) for row in measure_rows if isinstance(row, dict)]
    if not values:
        return 1.0
    values.sort()
    return max(0.5, float(values[len(values) // 2]))


def _measure_index_for_time(time_s: float, measure_duration: float) -> int:
    if measure_duration <= 0:
        return 0
    return max(0, int(float(time_s) / measure_duration))


def _enrich_timeline_segments(
    segments: List[Dict[str, Any]],
    rows_by_m: Dict[int, Dict[str, Any]],
    measure_duration: float,
    median_nps: float,
) -> None:
    if not segments or measure_duration <= 0:
        return
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start_m = _measure_index_for_time(float(seg.get("start_s", 0.0) or 0.0), measure_duration)
        end_s = max(float(seg.get("end_s", 0.0) or 0.0), float(seg.get("start_s", 0.0) or 0.0))
        end_m = _measure_index_for_time(max(end_s - 0.01, 0.0), measure_duration)
        notes_sum = 0
        drum_rels: List[float] = []
        mix_rels: List[float] = []
        for m in range(start_m, end_m + 1):
            row = rows_by_m.get(m)
            if not row:
                continue
            notes_sum += int(row.get("notes", 0) or 0)
            drum_rels.append(float(row.get("drum_rel", 0) or 0))
            mix_rels.append(float(row.get("mix_rel", 0) or 0))
        measures = max(1, end_m - start_m + 1)
        nps = float(notes_sum) / float(measures)
        density = min(1.0, max(0.0, nps / max(median_nps, 0.5) / 1.5))
        seg["density"] = round(density, 3)
        seg["drum_energy"] = round(sum(drum_rels) / len(drum_rels), 3) if drum_rels else 0.0
        seg["mix_energy"] = round(sum(mix_rels) / len(mix_rels), 3) if mix_rels else 0.0
        if "boundary_source" not in seg:
            seg["boundary_source"] = "chart"


def _segment_activity(seg: Dict[str, Any]) -> float:
    if not isinstance(seg, dict):
        return 0.0
    density = float(seg.get("density", 0.0) or 0.0)
    mix_energy = float(seg.get("mix_energy", 0.0) or 0.0)
    drum_energy = float(seg.get("drum_energy", 0.0) or 0.0)
    return density * 0.45 + mix_energy * 0.35 + drum_energy * 0.20


def _arc_intensity_curve(rel: float) -> float:
    """Expected climactic weight by normalized track position (0=start, 1=end)."""
    t = max(0.0, min(1.0, float(rel)))
    if t <= 0.12:
        return 0.08 + t * 0.35
    if t <= 0.55:
        return 0.12 + (t - 0.12) * 0.95
    if t <= 0.82:
        return 0.53 + (t - 0.55) * 1.35
    return max(0.25, 1.0 - (t - 0.82) * 2.8)


def _block_occurrence_index(segments: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        block = str(seg.get("block", "")).strip()
        if block:
            counts[block] = counts.get(block, 0) + 1
    return counts


def _assign_segment_intensity(segments: List[Dict[str, Any]]) -> None:
    if not segments:
        return
    track_end = max(float(seg.get("end_s", 0.0) or 0.0) for seg in segments if isinstance(seg, dict))
    if track_end <= 0.0:
        return
    activities = [_segment_activity(seg) for seg in segments]
    peak_activity = max(activities) if activities else 0.01
    peak_activity = max(peak_activity, 0.01)
    block_counts = _block_occurrence_index(segments)
    seen_blocks: Dict[str, int] = {}
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        start_s = float(seg.get("start_s", 0.0) or 0.0)
        end_s = max(float(seg.get("end_s", start_s) or start_s), start_s)
        mid = (start_s + end_s) * 0.5
        rel = mid / track_end
        activity = activities[i]
        activity_norm = activity / peak_activity
        arc = _arc_intensity_curve(rel)
        contrast = 0.5
        if i > 0:
            delta = (activity - activities[i - 1]) / peak_activity
            contrast = max(0.0, min(1.0, 0.5 + delta * 0.55))
        block = str(seg.get("block", "")).strip()
        repeat_boost = 0.0
        if block and block_counts.get(block, 0) >= 2:
            seen_blocks[block] = seen_blocks.get(block, 0) + 1
            if seen_blocks[block] >= 2:
                repeat_boost = 0.55
            elif seen_blocks[block] == 1 and rel >= 0.2:
                repeat_boost = 0.25
        raw = arc * 0.40 + activity_norm * 0.18 + contrast * 0.17 + repeat_boost * 0.25
        if rel <= 0.15:
            raw *= 0.28
        elif rel <= 0.25:
            raw *= 0.52
        elif rel >= 0.90:
            raw *= max(0.35, 1.0 - (rel - 0.90) * 4.0)
        seg["intensity"] = round(max(0.0, min(1.0, raw)), 3)


def _assign_segment_roles(segments: List[Dict[str, Any]]) -> None:
    if not segments:
        return
    track_end = max(float(seg.get("end_s", 0.0) or 0.0) for seg in segments if isinstance(seg, dict))
    if track_end <= 0.0:
        return
    activities = [_segment_activity(seg) for seg in segments]
    intensities = [float(seg.get("intensity", 0.0) or 0.0) if isinstance(seg, dict) else 0.0 for seg in segments]
    peak_idx = 0
    peak_score = -1.0
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        score = intensities[i] * 0.65 + activities[i] * 0.35
        if score > peak_score:
            peak_score = score
            peak_idx = i
    block_counts = _block_occurrence_index(segments)
    seen_blocks: Dict[str, int] = {}
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        start_s = float(seg.get("start_s", 0.0) or 0.0)
        end_s = max(float(seg.get("end_s", start_s) or start_s), start_s)
        mid = (start_s + end_s) * 0.5
        rel = mid / track_end
        kind = str(seg.get("kind", "steady"))
        block = str(seg.get("block", "")).strip()
        activity = activities[i]
        intensity = intensities[i]
        if block:
            seen_blocks[block] = seen_blocks.get(block, 0) + 1
        if rel <= 0.15:
            role = "intro"
        elif rel >= 0.88:
            role = "outro"
        elif i == peak_idx and peak_score > 0.08:
            role = "chorus"
        elif block and block_counts.get(block, 0) >= 2 and intensity >= 0.45 and seen_blocks.get(block, 0) >= 2:
            role = "chorus"
        elif kind in ("quiet", "sparse") and activity < 0.38 and 0.18 < rel < 0.82:
            prev_act = activities[i - 1] if i > 0 else activity
            if prev_act >= 0.55 or (i > 0 and activities[i - 1] >= activity + 0.12):
                role = "breakdown"
            elif kind == "sparse" and 0.25 < rel < 0.75:
                role = "solo"
            else:
                role = "verse"
        elif (
            block
            and block_counts.get(block, 0) == 1
            and 0.22 < rel < 0.78
            and intensity >= 0.35
            and i > 0
            and i < len(segments) - 1
        ):
            role = "bridge"
        elif intensity >= 0.72 and rel >= 0.35:
            role = "chorus"
        else:
            role = "verse"
        seg["role"] = role


def _segment_confidence(seg: Dict[str, Any]) -> float:
    conf = 0.5
    measures = int(seg.get("measures", 0) or 0)
    if measures >= 4:
        conf += 0.12
    if measures >= 8:
        conf += 0.08
    if str(seg.get("block", "")).strip():
        conf += 0.08
    source = str(seg.get("boundary_source", "chart"))
    if source == "both":
        conf += 0.15
    elif source == "audio":
        conf += 0.07
    return round(min(1.0, conf), 2)


def build_audio_structure_boundaries(
    mix_audio_path: Optional[str],
    *,
    bpm: float = 0.0,
    max_boundaries: int = 14,
) -> List[float]:
    path = Path(str(mix_audio_path or "").strip())
    if not path.is_file():
        return []
    try:
        import librosa
        import numpy as np
    except ImportError:
        return []
    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=900)
    except Exception:
        return []
    if len(y) < sr * 8:
        return []
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    if rms.size < 8:
        return []
    rms_delta = np.abs(np.diff(rms, prepend=float(rms[0])))
    window = max(3, int(round(float(sr) / float(hop) / 10.0)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    smooth = np.convolve(rms_delta, kernel, mode="same")
    wait_frames = max(12, int(round((60.0 / max(bpm, 60.0)) * 4.0 * float(sr) / float(hop) * 0.5)))
    peaks = librosa.util.peak_pick(
        smooth,
        pre_max=wait_frames,
        post_max=wait_frames,
        pre_avg=wait_frames,
        post_avg=wait_frames,
        delta=float(np.percentile(smooth, 75)) * 0.35,
        wait=wait_frames,
    )
    if peaks.size == 0:
        return []
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    duration = float(len(y)) / float(sr)
    out: List[float] = []
    min_gap = max(4.0, (60.0 / max(bpm, 60.0)) * 4.0 * 2.0)
    for t in sorted(float(x) for x in times):
        if t <= min_gap or t >= duration - min_gap:
            continue
        if out and t - out[-1] < min_gap:
            continue
        out.append(round(t, 1))
        if len(out) >= max_boundaries:
            break
    return out


def merge_timeline_with_audio_boundaries(
    segments: List[Dict[str, Any]],
    boundaries_s: List[float],
    measure_duration: float,
) -> List[Dict[str, Any]]:
    if not segments or not boundaries_s or measure_duration <= 0:
        return segments
    tol = measure_duration * 1.5
    bound_set = sorted({round(float(b), 1) for b in boundaries_s if float(b) > 0.0})
    merged: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = float(seg.get("start_s", 0.0) or 0.0)
        end = max(float(seg.get("end_s", start) or start), start)
        inner = [b for b in bound_set if start + tol < b < end - tol]
        if not inner:
            merged.append(dict(seg))
            continue
        points = [start] + inner + [end]
        for i in range(len(points) - 1):
            sub = dict(seg)
            sub["start_s"] = round(points[i], 1)
            sub["end_s"] = round(points[i + 1], 1)
            split_at = points[i]
            near_chart = abs(split_at - start) < tol or abs(split_at - end) < tol
            if i == 0:
                sub["boundary_source"] = str(seg.get("boundary_source", "chart"))
            elif near_chart:
                sub["boundary_source"] = "both"
            else:
                sub["boundary_source"] = "audio"
            sub["measures"] = max(1, int(round((sub["end_s"] - sub["start_s"]) / measure_duration)))
            merged.append(sub)
    return merged


def apply_structure_enrichment(
    structure_timeline: List[Dict[str, Any]],
    measure_rows: Optional[List[Dict[str, Any]]],
    bpm: float,
    mix_audio_path: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not structure_timeline or bpm <= 0:
        return structure_timeline, None
    measure_duration = (60.0 / float(bpm)) * 4.0
    rows_by_m = {
        int(row.get("m", 0) or 0): row
        for row in (measure_rows or [])
        if isinstance(row, dict)
    }
    median_nps = _median_notes_per_measure(measure_rows)
    boundaries = build_audio_structure_boundaries(mix_audio_path, bpm=bpm)
    timeline = list(structure_timeline)
    enrichment: Optional[Dict[str, Any]] = None
    if boundaries:
        timeline = merge_timeline_with_audio_boundaries(timeline, boundaries, measure_duration)
        enrichment = {
            "version": "audio_v1",
            "boundaries_s": boundaries,
        }
    _enrich_timeline_segments(timeline, rows_by_m, measure_duration, median_nps)
    _assign_segment_intensity(timeline)
    _assign_segment_roles(timeline)
    for seg in timeline:
        if isinstance(seg, dict):
            seg["confidence"] = _segment_confidence(seg)
    return timeline[:32], enrichment


def _percussion_viability(
    *,
    adtof_kick: int,
    adtof_snare: int,
    final_events: int,
    mix_drum_ratio: float,
) -> str:
    drum_signal = int(adtof_kick) + int(adtof_snare)
    if final_events <= 8:
        if drum_signal < 24 or mix_drum_ratio > 6.0 or final_events <= 3:
            return "low"
        return "medium"
    if drum_signal < 40:
        return "medium"
    return "high"


def build_decision_gene(
    decisions: List[Dict[str, Any]],
    pipeline: Dict[str, Any],
) -> Dict[str, int]:
    removed_total = max(0, int(pipeline.get("removed_total", 0) or 0))
    added_net = max(0, int(pipeline.get("added_net", 0) or 0))
    added_gross = 0
    for item in decisions:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        if key == "DNA_DEC_DRUM_ENTRY":
            added_gross += max(0, int(args.get("count", 0) or 0))
        elif key == "DNA_DEC_FILL_ZONE":
            added_gross += 1
    if removed_total <= 0:
        for item in decisions:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            if key in (
                "DNA_DEC_SECTION_PASS",
                "DNA_DEC_SALIENCE",
                "DNA_DEC_PLAYABILITY",
                "DNA_DEC_CLUSTER",
            ):
                removed_total += max(
                    0,
                    int(args.get("count", args.get("removed", 0)) or 0),
                )
    saved = 0
    saved_hints = pipeline.get("saved_hints")
    if isinstance(saved_hints, list):
        saved += len(saved_hints)
    if any(
        isinstance(item, dict) and str(item.get("key", "")) == "DNA_DEC_MINIMAL"
        for item in decisions
    ):
        saved += 1
    if saved <= 0 and int(pipeline.get("final_notes", 0) or 0) > 0:
        saved = 1
    total = removed_total + added_net + saved
    if total <= 0:
        saved = 1
        total = 1
    return {
        "version": 2,
        "removed": int(removed_total),
        "added": int(added_net),
        "added_gross": int(added_gross),
        "saved": int(saved),
        "corrected": 0,
        "total": int(total),
    }


def build_rhythm_dna(
    *,
    track: str = "",
    artist: str = "",
    title: str = "",
    genre: str = "",
    bpm: float = 0.0,
    mode: str = "basic",
    preset_id: str = "basic",
    lanes: int = 4,
    source_events: int = 0,
    pre_section_events: int = 0,
    post_section_events: int = 0,
    final_events: int = 0,
    final_notes: int = 0,
    adtof_unique: int = 0,
    adtof_kick: int = 0,
    adtof_snare: int = 0,
    adtof_hat: int = 0,
    adtof_rows: int = 0,
    caps_hps: int = 0,
    caps_npm: int = 0,
    measure_rows: Optional[List[Dict[str, Any]]] = None,
    salience_stats: Optional[Dict[str, Any]] = None,
    drum_entry_recap: Optional[Dict[str, Any]] = None,
    fill_core_measures: Optional[List[int]] = None,
    fill_halo_measures: Optional[List[int]] = None,
    playability_lint_removed: int = 0,
    chart_variant: str = "",
    mix_audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    removed_section = max(0, pre_section_events - post_section_events)
    removed_total = max(0, source_events - final_events)
    added_net = max(0, final_events - source_events)

    mix_drum_ratio = 0.0
    loud_mix_quiet_drum = 0
    measure_count = 0
    if measure_rows:
        measure_count = len(measure_rows)
        drum_vals = [float(r.get("drum_rms", 0) or 0) for r in measure_rows if float(r.get("drum_rms", 0) or 0) > 0]
        mix_vals = [float(r.get("mix_rms", 0) or 0) for r in measure_rows if float(r.get("mix_rms", 0) or 0) > 0]
        if drum_vals and mix_vals:
            import numpy as np

            d_med = float(np.median(drum_vals))
            x_med = float(np.median(mix_vals))
            if d_med > 1e-9:
                mix_drum_ratio = x_med / d_med
        loud_mix_quiet_drum = sum(
            1
            for r in measure_rows
            if float(r.get("mix_rel", 0) or 0) >= 0.9 and float(r.get("drum_rel", 0) or 0) < 0.55
        )

    salience = salience_stats or {}
    rhythm_kept = int(salience.get("filter_kept", 0) or 0)
    rhythm_total = int(salience.get("filter_total", 0) or 0)
    if rhythm_total <= 0:
        rhythm_kept = int(salience.get("rhythm_kept", 0) or 0)
        rhythm_total = int(salience.get("rhythm_total", 0) or 0)

    drum_entry = drum_entry_recap or {}
    drum_entry_recovered = int(drum_entry.get("recovered", 0) or 0)

    percussion_viable = _percussion_viability(
        adtof_kick=adtof_kick,
        adtof_snare=adtof_snare,
        final_events=final_events,
        mix_drum_ratio=mix_drum_ratio,
    )
    structure_timeline = build_structure_timeline(measure_rows, bpm)
    structure_timeline, structure_enrichment = apply_structure_enrichment(
        structure_timeline,
        measure_rows,
        bpm,
        mix_audio_path,
    )
    structure_blocks_pattern = build_structure_blocks_pattern(measure_rows)

    found: List[Dict[str, Any]] = []
    if percussion_viable == "low":
        if bpm > 0:
            found.append({"key": "DNA_FOUND_BPM", "args": {"bpm": int(round(bpm))}})
        found.append({"key": "DNA_FOUND_ACTIVITY_ONLY", "args": {"final": int(final_events)}})
    else:
        if adtof_kick + adtof_snare > 0 and percussion_viable != "low":
            found.append({"key": "DNA_FOUND_KIT"})
        if bpm > 0:
            found.append({"key": "DNA_FOUND_BPM", "args": {"bpm": int(round(bpm))}})
        if adtof_kick > 0 and adtof_snare > 0 and percussion_viable == "high":
            found.append({"key": "DNA_FOUND_GROOVE"})

    warnings: List[Dict[str, Any]] = []
    if percussion_viable == "low":
        warnings.append({"key": "DNA_WARN_NO_DRUMS_CHART", "args": {"final": int(final_events)}})
    elif final_events <= 5 and mix_drum_ratio > 8.0:
        warnings.append({"key": "DNA_WARN_LOW_PERCUSSION", "args": {"final": final_events}})
    elif final_events <= 8:
        warnings.append({"key": "DNA_WARN_SPARSE_CHART", "args": {"final": final_events}})
    if loud_mix_quiet_drum >= 6:
        warnings.append({"key": "DNA_WARN_QUIET_DRUM_SECTIONS", "args": {"count": loud_mix_quiet_drum}})
    if adtof_snare <= max(3, int(adtof_kick * 0.15)) and genre in ("house", "chillwave", "electronic"):
        warnings.append({"key": "DNA_WARN_SNARE_WEAK"})

    decisions: List[Dict[str, Any]] = []
    if removed_section > 0:
        decisions.append({"key": "DNA_DEC_SECTION_PASS", "args": {"count": removed_section}})
    if rhythm_total > rhythm_kept > 0:
        decisions.append(
            {"key": "DNA_DEC_SALIENCE", "args": {"removed": rhythm_total - rhythm_kept, "total": rhythm_total}}
        )
    if fill_core_measures:
        decisions.append({"key": "DNA_DEC_FILL_ZONE", "args": {"core": len(fill_core_measures), "halo": len(fill_halo_measures or [])}})
    if drum_entry_recovered > 0:
        decisions.append({"key": "DNA_DEC_DRUM_ENTRY", "args": {"count": drum_entry_recovered}})
    if playability_lint_removed > 0:
        decisions.append({"key": "DNA_DEC_PLAYABILITY", "args": {"count": playability_lint_removed}})
    if removed_total > removed_section:
        decisions.append({"key": "DNA_DEC_CLUSTER", "args": {"count": removed_total - removed_section}})
    if not decisions:
        decisions.append({"key": "DNA_DEC_MINIMAL"})

    removed_hints: List[Dict[str, Any]] = []
    if removed_section > 0:
        removed_hints.append({"key": "DNA_HINT_REMOVED_SECTION"})
    if removed_total > removed_section:
        removed_hints.append({"key": "DNA_HINT_REMOVED_CLUSTER"})
    if rhythm_total > rhythm_kept > 0:
        removed_hints.append({"key": "DNA_HINT_REMOVED_SALIENCE"})
    if playability_lint_removed > 0:
        removed_hints.append({"key": "DNA_HINT_REMOVED_PLAYABILITY"})

    added_hints: List[Dict[str, Any]] = []
    if drum_entry_recovered > 0:
        added_hints.append({"key": "DNA_HINT_ADDED_ENTRY"})
    if fill_core_measures and added_net > 0:
        added_hints.append({"key": "DNA_HINT_ADDED_FILL"})

    saved_hints: List[Dict[str, Any]] = [{"key": "DNA_HINT_SAVED_PATTERN"}]
    if final_notes > 0:
        saved_hints.append({"key": "DNA_HINT_SAVED_DYNAMICS"})

    adtof_signal = min(1.0, (adtof_kick + adtof_snare) / max(1.0, float(adtof_unique or adtof_rows or 1)))
    genre_conf = 0.64 if genre else 0.45
    overall = int(round(100 * (0.35 * adtof_signal + 0.25 * min(1.0, final_events / 120.0) + 0.2 * genre_conf + 0.2)))
    overall = max(35, min(97, overall))
    if warnings:
        overall = max(35, overall - 8 * len(warnings))

    decision_gene = build_decision_gene(
        decisions,
        {
            "final_notes": int(final_notes),
            "removed_total": int(removed_total),
            "added_net": int(added_net),
            "saved_hints": saved_hints[:2],
        },
    )

    return {
        "version": DNA_VERSION,
        "track": {
            "label": track,
            "artist": artist,
            "title": title,
            "genre": genre,
            "bpm": float(bpm),
            "mode": mode,
            "chart_intent": mode,
            "preset_id": preset_id or mode,
            "instrument": "drums",
            "lanes": int(lanes),
            "variant": chart_variant,
        },
        "pipeline": {
            "source": int(source_events),
            "pre_section": int(pre_section_events),
            "post_section": int(post_section_events),
            "final_events": int(final_events),
            "final_notes": int(final_notes),
            "removed_section": int(removed_section),
            "removed_total": int(removed_total),
            "added_net": int(added_net),
            "removed_hints": removed_hints[:2],
            "added_hints": added_hints[:2],
            "saved_hints": saved_hints[:2],
        },
        "adtof": {
            "unique": int(adtof_unique),
            "kick": int(adtof_kick),
            "snare": int(adtof_snare),
            "hat": int(adtof_hat),
            "rows": int(adtof_rows),
        },
        "measure_map": {
            "measures": int(measure_count),
            "mix_drum_ratio": round(float(mix_drum_ratio), 2),
            "loud_mix_quiet_drum": int(loud_mix_quiet_drum),
        },
        "structure_timeline": structure_timeline,
        "structure_blocks_pattern": structure_blocks_pattern,
        "structure_enrichment": structure_enrichment,
        "caps": {"hps": int(caps_hps), "npm": int(caps_npm)},
        "found": found,
        "warnings": warnings,
        "decisions": decisions,
        "genes": {
            "rhythm": {
                "kit_detected": adtof_kick + adtof_snare > 0 and percussion_viable != "low",
                "percussion_viable": percussion_viable,
                "groove_stability": _level_high_med_low(float(adtof_kick + adtof_snare), 40.0, 180.0),
                "fill_density": _level_high_med_low(float(len(fill_core_measures or [])), 0.0, 8.0),
            },
            "structure": {
                "measures": int(measure_count),
                "mix_drum_ratio": round(float(mix_drum_ratio), 2),
                "quiet_drum_windows": int(loud_mix_quiet_drum),
            },
            "confidence": {
                "overall": overall,
                "drum_detection": int(round(min(97, 55 + adtof_signal * 42))),
                "beat_tracking": 90 if bpm > 0 else 50,
                "genre": int(round(genre_conf * 100)),
                "pattern": int(round(min(95, 60 + min(1.0, final_events / 200.0) * 35))),
            },
            "decision": decision_gene,
        },
    }


def chart_mode_stem_from_filename(filename: str) -> str:
    stem = filename
    if stem.endswith(".rf"):
        stem = stem[:-3]
    return re.sub(r"_lanes\d+$", "", stem)


def rhythm_dna_sidecar_path(notes_rf_path: Path) -> Path:
    mode_stem = chart_mode_stem_from_filename(notes_rf_path.name)
    return notes_rf_path.with_name(f"{mode_stem}.rfd")


def find_sidecar_on_disk(
    track_stem: str,
    mode: str = "basic",
    instrument: str = "drums",
    chart_id: str = "",
) -> Optional[Path]:
    from app import song_storage

    stem = str(track_stem or "").strip()
    if not stem:
        return None
    notes_dirs: List[Path] = []
    cid = str(chart_id or "").strip()
    if cid:
        notes_dirs.append(song_storage.song_dir(cid) / "notes")
    legacy = Path("temp_uploads") / stem / "notes"
    if legacy not in notes_dirs:
        notes_dirs.append(legacy)
    inst = str(instrument or "drums").strip().lower() or "drums"
    mode_key = str(mode or "basic").strip().lower() or "basic"
    preferred_names = [
        f"{stem}_{inst}_{mode_key}.rfd",
        f"{inst}_{mode_key}.rfd",
        f"{inst}_{mode_key}_lanes4.rfd",
    ]
    for notes_dir in notes_dirs:
        if not notes_dir.is_dir():
            continue
        for name in preferred_names:
            candidate = notes_dir / name
            if candidate.is_file():
                return candidate
        matches: List[Path] = []
        for path in notes_dir.glob("*.rfd"):
            text = path.name.lower()
            if mode_key in text and inst in text:
                matches.append(path)
        if not matches:
            matches = list(notes_dir.glob("*.rfd"))
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]
    return None


def load_sidecar_from_disk(
    track_stem: str,
    mode: str = "basic",
    instrument: str = "drums",
) -> Optional[Dict[str, Any]]:
    path = find_sidecar_on_disk(track_stem, mode=mode, instrument=instrument)
    if path is None:
        return None
    try:
        payload = parse_rfd(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return payload if isinstance(payload, dict) and payload else None


def legacy_rhythm_dna_sidecar_path(notes_rf_path: Path) -> Path:
    stem = notes_rf_path.name
    if stem.endswith(".rf"):
        stem = stem[:-3]
    return notes_rf_path.with_name(f"{stem}.rhythm_dna.json")


def is_full_rhythm_dna(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload or not isinstance(payload, dict):
        return False
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, dict):
        return False
    return int(pipeline.get("source", 0) or 0) > 0


def format_rhythm_dna_log(payload: Optional[Dict[str, Any]], *, context: str = "") -> str:
    prefix = f"[RhythmDNA] {context}: " if context else "[RhythmDNA] "
    if not payload or not isinstance(payload, dict):
        return prefix + "MISSING — client will save its own fallback .rfd"
    pipeline = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else {}
    source = int(pipeline.get("source", 0) or 0)
    final = int(pipeline.get("final_notes", 0) or pipeline.get("final_events", 0) or 0)
    decisions = payload.get("decisions") if isinstance(payload.get("decisions"), list) else []
    decision_keys = [
        str(item.get("key", ""))
        for item in decisions
        if isinstance(item, dict) and str(item.get("key", "")).strip()
    ]
    if is_full_rhythm_dna(payload):
        return (
            f"{prefix}FULL report ready "
            f"(pipeline.source={source}, final_notes={final}, decisions={len(decision_keys)})"
        )
    incomplete = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    reason = str(incomplete.get("reason", "")).strip()
    reason_suffix = f", meta.reason={reason}" if reason else ""
    return (
        f"{prefix}MINIMAL payload only "
        f"(final_notes={final}, pipeline.source={source}{reason_suffix})"
    )


def save_rhythm_dna_sidecar(notes_rf_path: Path, payload: Optional[Dict[str, Any]]) -> bool:
    path = rhythm_dna_sidecar_path(notes_rf_path)
    if not payload:
        print(format_rhythm_dna_log(None, context=f"sidecar skip ({path.name})"))
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = serialize_rfd(payload)
        path.write_text(body, encoding="utf-8")
        legacy = legacy_rhythm_dna_sidecar_path(notes_rf_path)
        if legacy.exists() and legacy != path:
            legacy.unlink(missing_ok=True)
        print(format_rhythm_dna_log(payload, context=f"sidecar saved -> {path} ({len(body)} bytes)"))
        return True
    except OSError as exc:
        print(format_rhythm_dna_log(payload, context=f"sidecar FAILED -> {path}"))
        print(f"[RhythmDNA] Save error: {exc}")
        return False
