# app/stage_ledger.py — per-measure stage ledger (R0 наблюдаемости).
#
# Считает, сколько событий было после каждой стадии пайплайна, и на каждый такт
# ставит вердикт слоя: P1 (кандидата не было — вопрос к детектору) или P2
# (кандидат был, сломали дальше — и на какой стадии). Алгоритм генерации не
# трогает: только учёт. Протокол: docs/generation_debug_protocol.md.
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

LEDGER_VERSION = 1

# Порог: стем громкий, а кандидатов нет → проблема №1.
P1_LOUD_REL = 0.5
# Порог: на стеме тишина → отсутствие нот не баг.
SILENT_REL = 0.15
# Доля от кандидатов, ниже которой такт считаем выпотрошенным.
THIN_RATIO = 0.4
# Доля выше кандидатов, с которой считаем, что ноты выдуманы.
INVENT_RATIO = 1.3

VERDICT_P1 = "P1"
VERDICT_P1_MAYBE = "P1_maybe"
VERDICT_P2_DROP = "P2_drop"
VERDICT_P2_THIN = "P2_thin"
VERDICT_P2_INVENT = "P2_invent"
VERDICT_PATTERN_BREAK = "pattern_break"
VERDICT_OK = "ok"

CAND_STAGE = "cand"
FINAL_STAGE = "final"


def ledger_enabled() -> bool:
    return os.getenv("RF_STAGE_LEDGER", "1") == "1"


def _quantize_slots(
    events: Sequence[float],
    first_measure_start: float,
    measure_duration: float,
    beat_interval: float,
) -> Dict[int, List[float]]:
    """Slot signature на такт: позиции в долях, квантованные до 1/16."""
    out: Dict[int, List[float]] = {}
    if measure_duration <= 0 or beat_interval <= 0:
        return out
    for t in events:
        rel = float(t) - first_measure_start
        m = int(np.floor(rel / measure_duration))
        rel_beats = (rel - m * measure_duration) / beat_interval
        if rel_beats < -0.1 or rel_beats >= 4.1:
            continue
        slot = round(max(0.0, min(3.75, rel_beats)) * 4.0) / 4.0
        bucket = out.setdefault(m, [])
        if slot not in bucket:
            bucket.append(slot)
    return {m: sorted(v) for m, v in out.items()}


class StageLedger:
    """Собирает события после каждой стадии. Выключенный экземпляр — no-op."""

    def __init__(self, beats: Optional[Sequence[float]], bpm: float, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.stages: List[str] = []
        self._times: Dict[str, List[float]] = {}
        self.bpm = float(bpm or 0.0)
        self.beat_interval = 0.0
        self.first_measure_start = 0.0
        self.measure_duration = 0.0
        if not self.enabled:
            return
        if beats is None:
            raw: List[float] = []
        elif isinstance(beats, np.ndarray):
            raw = beats.astype(float, copy=False).ravel().tolist() if beats.size else []
        else:
            raw = [float(t) for t in beats]
        arr = np.asarray(raw, dtype=float)
        if arr.size >= 2:
            self.beat_interval = float(np.median(np.diff(arr)))
            self.first_measure_start = float(arr[0])
        elif self.bpm > 0:
            self.beat_interval = 60.0 / self.bpm
        if self.beat_interval <= 0:
            self.enabled = False
            return
        self.measure_duration = self.beat_interval * 4.0

    @classmethod
    def create(cls, beats: Optional[Sequence[float]], bpm: float, *, enabled: Optional[bool] = None) -> "StageLedger":
        return cls(beats, bpm, ledger_enabled() if enabled is None else enabled)

    def record(self, stage: str, events: Optional[Sequence[float]]) -> None:
        if not self.enabled:
            return
        if events is None:
            times: List[float] = []
        elif isinstance(events, np.ndarray):
            times = events.astype(float, copy=False).ravel().tolist() if events.size else []
        else:
            times = [float(t) for t in events]
        if stage in self._times:
            self._times[stage] = times
            return
        self.stages.append(stage)
        self._times[stage] = times

    def record_notes(self, stage: str, notes: Optional[Sequence[Dict[str, Any]]]) -> None:
        if not self.enabled:
            return
        self.record(stage, [float(n.get("time", 0.0)) for n in (notes or [])])

    def measure_of(self, seconds: float) -> int:
        if self.measure_duration <= 0:
            return 0
        return int(np.floor((float(seconds) - self.first_measure_start) / self.measure_duration))

    def _counts_per_measure(self, stage: str) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for t in self._times.get(stage, []):
            m = self.measure_of(t)
            counts[m] = counts.get(m, 0) + 1
        return counts

    def build(
        self,
        *,
        measure_rows: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled or CAND_STAGE not in self._times:
            return {}

        per_stage = {stage: self._counts_per_measure(stage) for stage in self.stages}
        rms_by_measure: Dict[int, Dict[str, Any]] = {}
        for row in measure_rows or []:
            rms_by_measure[int(row.get("m", 0))] = row

        touched = {m for counts in per_stage.values() for m in counts}
        touched |= set(rms_by_measure)
        if not touched:
            return {}
        m_min, m_max = min(touched), max(touched)

        final_stage = FINAL_STAGE if FINAL_STAGE in per_stage else self.stages[-1]
        final_sigs = _quantize_slots(
            self._times.get(final_stage, []),
            self.first_measure_start,
            self.measure_duration,
            self.beat_interval,
        )

        rows: List[Dict[str, Any]] = []
        for m in range(m_min, m_max + 1):
            counts = {stage: int(per_stage[stage].get(m, 0)) for stage in self.stages}
            info = rms_by_measure.get(m) or {}
            drum_rel = float(info.get("drum_rel", -1.0) if info else -1.0)
            verdict, blame, delta = self._verdict(counts, drum_rel)
            flags: List[str] = []
            if self._is_pattern_break(final_sigs, m):
                flags.append(VERDICT_PATTERN_BREAK)
                if verdict == VERDICT_OK:
                    verdict = VERDICT_PATTERN_BREAK
            rows.append(
                {
                    "m": m,
                    "sec": round(self.first_measure_start + m * self.measure_duration, 3),
                    "drum_rel": round(drum_rel, 3) if drum_rel >= 0 else None,
                    "mix_rel": round(float(info.get("mix_rel", 0.0)), 3) if info else None,
                    "sig": final_sigs.get(m, []),
                    "counts": counts,
                    "verdict": verdict,
                    "blame": blame,
                    "delta": delta,
                }
            )

        intervals = self._intervals(rows)
        verdict_hist = Counter(str(r["verdict"]) for r in rows)
        blame_hist = Counter(str(r["blame"]) for r in rows if r["blame"])
        worst = sorted(
            intervals,
            key=lambda it: (_verdict_weight(str(it["verdict"])), int(it["measures"])),
            reverse=True,
        )[:8]

        return {
            "version": LEDGER_VERSION,
            "track": dict(meta or {}),
            "grid": {
                "bpm": round(self.bpm, 3),
                "beat_interval": round(self.beat_interval, 5),
                "first_measure_start": round(self.first_measure_start, 3),
                "measure_duration": round(self.measure_duration, 5),
                "measures": len(rows),
                "rms_available": bool(rms_by_measure),
            },
            "stages": list(self.stages),
            "totals": {stage: len(self._times.get(stage, [])) for stage in self.stages},
            "summary": {
                "verdicts": dict(verdict_hist),
                "blame": dict(blame_hist.most_common()),
                "worst": worst,
            },
            "intervals": intervals,
            "rows": rows,
        }

    def _verdict(self, counts: Dict[str, int], drum_rel: float):
        cand = int(counts.get(CAND_STAGE, 0))
        final = int(counts.get(FINAL_STAGE, counts.get(self.stages[-1], 0)))

        if cand <= 0:
            if drum_rel < 0:
                return (VERDICT_P1_MAYBE if final <= 0 else VERDICT_P2_INVENT), self._blame(counts, +1), 0
            if drum_rel >= P1_LOUD_REL and final <= 0:
                return VERDICT_P1, "", 0
            if drum_rel < SILENT_REL:
                return (VERDICT_OK if final <= 0 else VERDICT_P2_INVENT), (self._blame(counts, +1) if final > 0 else ""), 0
            if final <= 0:
                return VERDICT_P1_MAYBE, "", 0
            return VERDICT_P2_INVENT, self._blame(counts, +1), final

        if final <= 0:
            blame, delta = self._blame_delta(counts, -1)
            return VERDICT_P2_DROP, blame, delta
        if final < THIN_RATIO * cand:
            blame, delta = self._blame_delta(counts, -1)
            return VERDICT_P2_THIN, blame, delta
        if final > INVENT_RATIO * cand:
            blame, delta = self._blame_delta(counts, +1)
            return VERDICT_P2_INVENT, blame, delta
        return VERDICT_OK, "", 0

    def _blame_delta(self, counts: Dict[str, int], sign: int):
        best_stage = ""
        best_delta = 0
        prev_stage = self.stages[0]
        for stage in self.stages[1:]:
            delta = counts.get(stage, 0) - counts.get(prev_stage, 0)
            if sign * delta > 0 and sign * delta > sign * best_delta:
                best_stage, best_delta = stage, delta
            prev_stage = stage
        return best_stage, int(best_delta)

    def _blame(self, counts: Dict[str, int], sign: int) -> str:
        return self._blame_delta(counts, sign)[0]

    @staticmethod
    def _is_pattern_break(sigs: Dict[int, List[float]], m: int) -> bool:
        """Две одинаковые фразы подряд, в третьей строго меньше слотов."""
        a, b, c = sigs.get(m - 2), sigs.get(m - 1), sigs.get(m)
        if not a or not b or c is None:
            return False
        if a != b or len(b) < 2:
            return False
        cur = set(c)
        if cur == set(b):
            return False
        return cur < set(b)

    @staticmethod
    def _intervals(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        run: List[Dict[str, Any]] = []

        def flush() -> None:
            if not run:
                return
            verdict = str(run[0]["verdict"])
            if verdict == VERDICT_OK:
                run.clear()
                return
            blames = Counter(str(r["blame"]) for r in run if r["blame"])
            cand = [int(r["counts"].get(CAND_STAGE, 0)) for r in run]
            final = [int(r["counts"].get(FINAL_STAGE, 0)) for r in run]
            out.append(
                {
                    "verdict": verdict,
                    "m_from": int(run[0]["m"]),
                    "m_to": int(run[-1]["m"]),
                    "sec_from": float(run[0]["sec"]),
                    "sec_to": round(float(run[-1]["sec"]), 3),
                    "measures": len(run),
                    "blame": blames.most_common(1)[0][0] if blames else "",
                    "cand_avg": round(sum(cand) / len(cand), 2),
                    "final_avg": round(sum(final) / len(final), 2),
                }
            )
            run.clear()

        for row in rows:
            if run and str(row["verdict"]) != str(run[-1]["verdict"]):
                flush()
            run.append(row)
        flush()
        return out


def _verdict_weight(verdict: str) -> int:
    return {
        VERDICT_P2_DROP: 5,
        VERDICT_P1: 4,
        VERDICT_P2_THIN: 3,
        VERDICT_P2_INVENT: 3,
        VERDICT_P1_MAYBE: 2,
        VERDICT_PATTERN_BREAK: 1,
    }.get(verdict, 0)


def format_interval(interval: Dict[str, Any]) -> str:
    return (
        f"m{int(interval['m_from'])}-{int(interval['m_to'])} "
        f"({_mmss(float(interval['sec_from']))}-{_mmss(float(interval['sec_to']))}) "
        f"{interval['verdict']}"
        + (f" blame={interval['blame']}" if interval.get("blame") else "")
        + f" cand={interval['cand_avg']} final={interval['final_avg']}"
    )


def _mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def log_stage_ledger(payload: Optional[Dict[str, Any]], *, limit: int = 6) -> None:
    if not payload:
        return
    summary = payload.get("summary") or {}
    verdicts = summary.get("verdicts") or {}
    blame = summary.get("blame") or {}
    grid = payload.get("grid") or {}
    order = [
        VERDICT_P1,
        VERDICT_P1_MAYBE,
        VERDICT_P2_DROP,
        VERDICT_P2_THIN,
        VERDICT_P2_INVENT,
        VERDICT_PATTERN_BREAK,
        VERDICT_OK,
    ]
    parts = [f"{name}={int(verdicts[name])}" for name in order if verdicts.get(name)]
    blame_top = ",".join(f"{k}:{v}" for k, v in list(blame.items())[:3])
    print(
        f"[StageLedger] measures={int(grid.get('measures', 0))} "
        + " ".join(parts)
        + (f" blame_top={blame_top}" if blame_top else "")
        + ("" if grid.get("rms_available") else " rms=unavailable(RF_MEASURE_MAP=0)")
    )
    for interval in (summary.get("worst") or [])[:limit]:
        print(f"[StageLedger]   {format_interval(interval)}")


def stage_ledger_path(notes_rf_path: Path) -> Path:
    stem = notes_rf_path.name
    if stem.endswith(".rf"):
        stem = stem[:-3]
    stem = re.sub(r"_lanes\d+$", "", stem)
    return notes_rf_path.with_name(f"{stem}.stages.json")


def save_stage_ledger(notes_rf_path: Path, payload: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not payload:
        return None
    try:
        path = stage_ledger_path(Path(notes_rf_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return path
    except Exception as exc:  # артефакт отладки не должен ломать генерацию
        print(f"[StageLedger] не сохранён: {exc}")
        return None


def load_stage_ledger(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
