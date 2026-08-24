# app/bass_transforms.py — Goal × difficulty passes for bass charts.
"""See docs/bass_mode.md — mirrors drums philosophy on continuous note shapes."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from app.generation_intents import normalize_difficulty, normalize_goal


def _beat_interval(bpm: float) -> float:
    return 60.0 / max(float(bpm), 60.0)


def _note_time(note: Dict[str, Any]) -> float:
    return float(note.get("time", 0.0))


def _note_amp_proxy(note: Dict[str, Any]) -> float:
    if note.get("ghost"):
        return 0.25
    shape = str(note.get("shape", "tap")).strip().lower()
    if shape == "hold":
        end = float(note.get("end", note.get("time", 0.0)))
        return max(0.5, end - _note_time(note))
    if shape == "slide":
        return 0.75
    return 0.4


def _sorted_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(notes, key=_note_time)


def _strip_ghosts(notes: List[Dict[str, Any]], *, aggressive: bool) -> Tuple[List[Dict[str, Any]], int]:
    if not notes:
        return [], 0
    removed = 0
    kept: List[Dict[str, Any]] = []
    for note in notes:
        if note.get("ghost"):
            if aggressive:
                removed += 1
                continue
            if note.get("shape") == "tap":
                removed += 1
                continue
            kept.append({**note, "ghost": False})
            removed += 1
            continue
        kept.append(note)
    return kept, removed


def _drop_weakest(notes: List[Dict[str, Any]], *, keep_ratio: float) -> Tuple[List[Dict[str, Any]], int]:
    if not notes or keep_ratio >= 1.0:
        return list(notes), 0
    ratio = max(0.35, min(1.0, float(keep_ratio)))
    target = max(1, int(round(len(notes) * ratio)))
    if target >= len(notes):
        return list(notes), 0
    scored = sorted(
        ((_note_amp_proxy(n), _note_time(n), n) for n in notes),
        key=lambda item: (-item[0], item[1]),
    )
    kept = sorted((item[2] for item in scored[:target]), key=_note_time)
    return kept, len(notes) - len(kept)


def _merge_same_lane_taps(
    notes: List[Dict[str, Any]],
    *,
    bpm: float,
    max_gap_beats: float = 0.25,
) -> Tuple[List[Dict[str, Any]], int]:
    if not notes:
        return [], 0
    gap = _beat_interval(bpm) * max_gap_beats
    merged: List[Dict[str, Any]] = []
    merged_count = 0
    for note in _sorted_notes(notes):
        if (
            merged
            and str(note.get("shape", "")).lower() == "tap"
            and str(merged[-1].get("shape", "")).lower() == "tap"
            and note.get("lane") == merged[-1].get("lane")
            and not note.get("ghost")
            and not merged[-1].get("ghost")
            and (_note_time(note) - _note_time(merged[-1])) <= gap
        ):
            merged_count += 1
            continue
        merged.append(note)
    return merged, merged_count


def _remove_multi_lane(notes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    removed = 0
    for note in notes:
        lanes = note.get("lanes")
        if isinstance(lanes, list) and len(lanes) > 1:
            lane = int(lanes[0])
            cleaned = {k: v for k, v in note.items() if k != "lanes"}
            cleaned["lane"] = lane
            cleaned["shape"] = "tap"
            out.append(cleaned)
            removed += 1
        else:
            out.append(note)
    return out, removed


def _gap_fill_passing_taps(
    notes: List[Dict[str, Any]],
    *,
    bpm: float,
    lanes: int,
    max_add_ratio: float = 0.08,
    gap_beats: float = 1.35,
    allow_time: Optional[Callable[[float], bool]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Insert passing taps in large gaps. Optional ``allow_time`` blocks silence (stem wins)."""
    if not notes:
        return [], 0
    beat = _beat_interval(bpm)
    gap_threshold = beat * float(gap_beats)
    sorted_notes = _sorted_notes(notes)
    max_add = max(1, int(round(len(sorted_notes) * max_add_ratio)))
    added = 0
    extras: List[Dict[str, Any]] = []
    for i in range(len(sorted_notes) - 1):
        if added >= max_add:
            break
        cur = sorted_notes[i]
        nxt = sorted_notes[i + 1]
        t0 = _note_time(cur)
        t1 = _note_time(nxt)
        if (t1 - t0) < gap_threshold:
            continue
        lane_a = int(cur.get("lane", cur.get("lanes", [0])[0] if isinstance(cur.get("lanes"), list) else 0))
        lane_b = int(nxt.get("lane", nxt.get("lanes", [0])[0] if isinstance(nxt.get("lanes"), list) else 0))
        # Prefer a step toward the next note — never spam the same lane as both neighbors.
        if lane_a != lane_b:
            fill_lane = int(round((lane_a + lane_b) * 0.5))
        else:
            fill_lane = lane_a + (1 if lane_a < lanes - 1 else -1)
        fill_lane = min(max(fill_lane, 0), lanes - 1)
        if fill_lane == lane_a and lanes > 1:
            fill_lane = (lane_a + 1) % lanes
        # Metal densify: land on next 8th; sparse fills one beat later.
        step = beat * 0.5 if gap_beats <= 0.85 else beat
        fill_time = round(t0 + step, 4)
        if fill_time >= t1 - beat * 0.28:
            continue
        if allow_time is not None and not allow_time(fill_time):
            continue
        extras.append(
            {
                "time": fill_time,
                "lane": fill_lane,
                "shape": "tap",
                "ghost": False,
            }
        )
        added += 1
    if not extras:
        return sorted_notes, 0
    return _sorted_notes(sorted_notes + extras), added


def demote_slides_to_holds(notes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Beta: slides need lane-glide UX — play as hold until client catches up."""
    out: List[Dict[str, Any]] = []
    demoted = 0
    for note in notes:
        n = dict(note)
        if str(n.get("shape", "tap")).strip().lower() == "slide":
            n["shape"] = "hold"
            n.pop("lane_end", None)
            n.pop("curve", None)
            if n.get("type") == "BassSlideNote":
                n["type"] = "BassHoldNote"
            demoted += 1
        out.append(n)
    return out, demoted


def _cap_ghost_notes(
    notes: List[Dict[str, Any]],
    *,
    max_ratio: float = 0.14,
) -> Tuple[List[Dict[str, Any]], int]:
    if not notes:
        return [], 0
    ghosts = [n for n in notes if n.get("ghost")]
    if not ghosts:
        return list(notes), 0
    limit = max(1, int(round(len(notes) * max_ratio)))
    if len(ghosts) <= limit:
        return list(notes), 0
    drop = len(ghosts) - limit
    scored = sorted(
        ((_note_amp_proxy(n), _note_time(n), n) for n in ghosts),
        key=lambda item: (item[0], item[1]),
    )
    drop_keys = {
        (_note_time(item[2]), int(item[2].get("lane", 0)), str(item[2].get("shape", "tap")))
        for item in scored[:drop]
    }
    out: List[Dict[str, Any]] = []
    for note in notes:
        key = (_note_time(note), int(note.get("lane", 0)), str(note.get("shape", "tap")))
        if note.get("ghost") and key in drop_keys:
            cleaned = dict(note)
            cleaned["ghost"] = False
            out.append(cleaned)
        else:
            out.append(note)
    return out, drop


def _note_end(note: Dict[str, Any]) -> float:
    shape = str(note.get("shape", "tap")).strip().lower()
    if shape in ("hold", "slide"):
        return float(note.get("end", _note_time(note)))
    return _note_time(note)


def diversify_lane_runs(
    notes: List[Dict[str, Any]],
    *,
    lanes: int = 5,
    max_same_run: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Break long same-lane tap streaks so arcade charts don't sit on one fret.

    Holds keep their mapped lane (sustain identity). Ghosts are left alone.
    """
    if not notes or lanes <= 1:
        return list(notes), {"nudged": 0, "max_run": max_same_run}
    sorted_notes = _sorted_notes(notes)
    out: List[Dict[str, Any]] = []
    run_lane = -1
    run_len = 0
    nudged = 0
    toggle = 1
    for note in sorted_notes:
        n = dict(note)
        shape = str(n.get("shape", "tap")).strip().lower()
        lane = int(n.get("lane", 0))
        lane = min(max(lane, 0), lanes - 1)
        if shape != "tap" or n.get("ghost"):
            out.append(n)
            run_lane = -1
            run_len = 0
            continue
        if lane == run_lane:
            run_len += 1
        else:
            run_lane = lane
            run_len = 1
            toggle = 1
        if run_len > max_same_run:
            alt = lane + toggle
            if alt < 0 or alt >= lanes:
                toggle = -toggle
                alt = lane + toggle
            alt = min(max(alt, 0), lanes - 1)
            if alt != lane:
                n["lane"] = alt
                nudged += 1
                run_lane = alt
                run_len = 1
                toggle = -toggle
        out.append(n)
    return out, {"nudged": nudged, "max_run": max_same_run}


def spread_simultaneous_same_lane(
    notes: List[Dict[str, Any]],
    *,
    lanes: int = 5,
    eps: float = 0.012,
    bpm: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """If two+ notes share a near-time bucket and lane, nudge extras to free lanes.

    Also breaks same-lane micro-stacks (stacked taps/hold-heads that look like one
    column pile) — not only exact same-time chords.
    """
    if not notes or lanes <= 1:
        return list(notes), {"nudged": 0}
    # Widen bucket for visual stacks: ~45ms or ~1/12 beat.
    cluster_eps = float(eps)
    if bpm is not None and float(bpm) > 0:
        beat = 60.0 / max(float(bpm), 60.0)
        cluster_eps = max(cluster_eps, min(0.055, beat * 0.12))
    else:
        cluster_eps = max(cluster_eps, 0.045)
    ordered = _sorted_notes(notes)
    out: List[Dict[str, Any]] = [dict(n) for n in ordered]
    nudged = 0
    i = 0
    while i < len(out):
        t0 = _note_time(out[i])
        j = i + 1
        while j < len(out) and abs(_note_time(out[j]) - t0) <= cluster_eps:
            j += 1
        group = out[i:j]
        used: Dict[int, int] = {}
        for idx, note in enumerate(group):
            lane = int(note.get("lane", 0))
            if isinstance(note.get("lanes"), list) and note["lanes"]:
                try:
                    lane = int(note["lanes"][0])
                except (TypeError, ValueError):
                    pass
            lane = min(max(lane, 0), lanes - 1)
            if lane not in used:
                used[lane] = idx
                note["lane"] = lane
                continue
            # Find a free lane near the conflict.
            alt = None
            for delta in (1, -1, 2, -2, 3, -3, 4, -4):
                cand = lane + delta
                if 0 <= cand < lanes and cand not in used:
                    alt = cand
                    break
            if alt is None:
                continue
            note["lane"] = alt
            if isinstance(note.get("lanes"), list) and len(note["lanes"]) == 1:
                note["lanes"] = [alt]
            used[alt] = idx
            nudged += 1
        i = j

    # Second pass: same-lane notes closer than cluster_eps even across group edges.
    out = _sorted_notes(out)
    last_t_by_lane: Dict[int, float] = {}
    for note in out:
        lane = int(note.get("lane", 0))
        lane = min(max(lane, 0), lanes - 1)
        t0 = _note_time(note)
        prev = last_t_by_lane.get(lane)
        if prev is not None and abs(t0 - prev) <= cluster_eps:
            alt = None
            occupied = {
                int(n.get("lane", 0))
                for n in out
                if abs(_note_time(n) - t0) <= cluster_eps
            }
            for delta in (1, -1, 2, -2, 3, -3, 4, -4):
                cand = lane + delta
                if 0 <= cand < lanes and cand not in occupied:
                    alt = cand
                    break
            if alt is not None:
                note["lane"] = alt
                if isinstance(note.get("lanes"), list) and len(note["lanes"]) == 1:
                    note["lanes"] = [alt]
                lane = alt
                nudged += 1
        last_t_by_lane[lane] = t0
    return out, {"nudged": nudged, "eps_ms": round(cluster_eps * 1000.0, 1)}


def strip_same_lane_hold_overlaps(notes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Drop notes that start inside an active hold/slide window on the same lane."""
    sorted_notes = _sorted_notes(notes)
    blocked: List[Tuple[float, float, int]] = []
    out: List[Dict[str, Any]] = []
    removed = 0
    for note in sorted_notes:
        lane_raw = note.get("lane")
        if isinstance(note.get("lanes"), list) and note.get("lanes"):
            lane = int(note["lanes"][0])
        else:
            lane = int(lane_raw if lane_raw is not None else 0)
        t0 = _note_time(note)
        blocked_by = False
        for b0, b1, bl in blocked:
            if bl == lane and t0 > b0 + 0.001 and t0 < b1 - 0.001:
                blocked_by = True
                break
        if blocked_by:
            removed += 1
            continue
        out.append(note)
        shape = str(note.get("shape", "tap")).strip().lower()
        if shape in ("hold", "slide"):
            end = _note_end(note)
            if end > t0 + 0.001:
                blocked.append((t0, end, lane))
    return out, {"removed": removed, "after": len(out)}


def apply_bass_style(
    notes: List[Dict[str, Any]],
    *,
    goal: Optional[str],
    bpm: float,
    lanes: int = 5,
    allow_gap_fill_at: Optional[Callable[[float], bool]] = None,
    groove_class: str = "mixed",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Goal axis after pitch→shape conversion (Original documentary vs Arcade playable)."""
    goal_n = normalize_goal(goal)
    out = _sorted_notes(notes)
    recap: Dict[str, Any] = {"goal": goal_n, "before": len(out), "groove": groove_class}

    if goal_n == "arcade":
        # Metal/plucky: tighter 8th-ish fills (full genre templates are drums-only).
        if groove_class == "plucky":
            out, added = _gap_fill_passing_taps(
                out,
                bpm=bpm,
                lanes=lanes,
                max_add_ratio=0.12,
                gap_beats=0.75,
                allow_time=allow_gap_fill_at,
            )
            recap["gap_fill"] = added
            recap["pattern"] = "plucky_eighths"
        else:
            out, added = _gap_fill_passing_taps(
                out,
                bpm=bpm,
                lanes=lanes,
                max_add_ratio=0.06,
                allow_time=allow_gap_fill_at,
            )
            recap["gap_fill"] = added
    else:
        out, removed = _remove_multi_lane(out)
        recap["strip_multi_lane"] = removed

    recap["after"] = len(out)
    return out, recap


def apply_bass_difficulty(
    notes: List[Dict[str, Any]],
    *,
    goal: Optional[str],
    difficulty: Optional[str],
    bpm: float,
    lanes: int = 5,
    transcription_faithful: bool = False,
    allow_gap_fill_at: Optional[Callable[[float], bool]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Difficulty axis after style build.

    ``transcription_faithful`` is recorded for diagnostics only — density ratios
    are the same whether the pitch track came from Basic Pitch or heuristic.
    """
    goal_n = normalize_goal(goal)
    diff = normalize_difficulty(difficulty)
    out = _sorted_notes(notes)
    recap: Dict[str, Any] = {"goal": goal_n, "difficulty": diff, "before": len(out)}

    if diff == "relaxed":
        out, ghosts = _strip_ghosts(out, aggressive=True)
        recap["ghost_strip"] = ghosts
        keep = 0.58 if goal_n == "original" else 0.68
        out, dropped = _drop_weakest(out, keep_ratio=keep)
        recap["drop_weakest"] = dropped
        out, merged = _merge_same_lane_taps(out, bpm=bpm, max_gap_beats=0.35)
        recap["merge_taps"] = merged
        if goal_n == "original":
            out, ml = _remove_multi_lane(out)
            recap["strip_multi_lane"] = ml
    elif diff == "standard":
        if goal_n == "original":
            keep = 0.82
            out, dropped = _drop_weakest(out, keep_ratio=keep)
            recap["drop_weakest"] = dropped
            out, merged = _merge_same_lane_taps(out, bpm=bpm, max_gap_beats=0.2)
            recap["merge_taps"] = merged
        else:
            keep = 0.92
            out, dropped = _drop_weakest(out, keep_ratio=keep)
            recap["drop_weakest"] = dropped
            out, added = _gap_fill_passing_taps(
                out,
                bpm=bpm,
                lanes=lanes,
                max_add_ratio=0.05,
                allow_time=allow_gap_fill_at,
            )
            recap["gap_fill"] = added
        recap["ghost_keep"] = sum(1 for n in out if n.get("ghost"))
    else:
        # dense: keep full line; arcade densifies gaps on the same line.
        recap["dense_passthrough"] = True
        recap["ghost_keep"] = sum(1 for n in out if n.get("ghost"))
        if goal_n == "arcade":
            # Mild densify only — heavy same-lane fill feels like spam on busy bass.
            out, added = _gap_fill_passing_taps(
                out,
                bpm=bpm,
                lanes=lanes,
                max_add_ratio=0.08,
                allow_time=allow_gap_fill_at,
            )
            recap["gap_fill"] = added

    if transcription_faithful:
        recap["transcription_faithful"] = True

    recap["after"] = len(out)
    return out, recap
