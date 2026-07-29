"""Aggregate per-song generation problem reports into a compact summary.json.

Supports:
  - chart_regression batch reports (auto warnings from analyze_chart)
  - in-game F10 QA session files (issues[])
  - unified batch wrapper written by run_generation_test_batch.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

SUMMARY_VERSION = 1
SKIP_JSON_NAMES = frozenset({"summary.json", "generation_quality_reports.json"})

_HASH_RE = re.compile(r"^[0-9a-f]{16}$", re.I)


@dataclass
class Problem:
    type: str
    severity: str
    measure: int | None = None
    time_sec: float | None = None
    source: str = "auto"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type,
            "severity": self.severity,
            "source": self.source,
        }
        if self.measure is not None:
            out["measure"] = self.measure
        if self.time_sec is not None:
            out["time_sec"] = round(self.time_sec, 3)
        if self.message:
            out["message"] = self.message[:160]
        return out


@dataclass
class SongReport:
    song_id: str
    artist: str = ""
    title: str = ""
    genre: str = ""
    bpm: float = 0.0
    preset: str = ""
    lanes: int = 0
    instrument: str = ""
    report_kind: str = ""
    problems: list[Problem] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""

    @property
    def display_name(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or self.artist or self.song_id

    def problem_count(self) -> int:
        return len(self.problems)

    def severity_counts(self) -> dict[str, int]:
        c: Counter[str] = Counter()
        for p in self.problems:
            c[p.severity] += 1
        return dict(c)


def bpm_band(bpm: float) -> str:
    if bpm <= 0:
        return "unknown"
    lo = int(bpm // 20) * 20
    return f"{lo}-{lo + 19}"


def _parse_measure(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return int(value)


def _parse_time(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def detect_report_kind(data: dict[str, Any]) -> str:
    if data.get("report_kind") in ("chart_regression", "qa_session", "unified"):
        return str(data["report_kind"])
    if isinstance(data.get("issues"), list):
        return "qa_session"
    if isinstance(data.get("warnings"), list) or "fail_count" in data or "warn_count" in data:
        return "chart_regression"
    if isinstance(data.get("problems"), list):
        return "unified"
    return "unknown"


def _song_id_from_path(path: Path, data: dict[str, Any]) -> str:
    for key in ("song_hash", "song_id", "chart_id"):
        value = str(data.get(key, "")).strip()
        if value:
            return value
    parent = path.parent.name
    if _HASH_RE.match(parent):
        return parent
    stem = path.stem
    if _HASH_RE.match(stem):
        return stem
    return stem


def problems_from_chart_regression(data: dict[str, Any]) -> list[Problem]:
    out: list[Problem] = []
    for raw in data.get("warnings", []):
        if not isinstance(raw, dict):
            continue
        out.append(
            Problem(
                type=str(raw.get("code", "UNKNOWN")),
                severity=str(raw.get("severity", "WARN")).upper(),
                measure=_parse_measure(raw.get("measure")),
                time_sec=_parse_time(raw.get("time_sec")),
                source="auto",
                message=str(raw.get("message", "")),
            )
        )
    return out


def problems_from_qa_session(data: dict[str, Any]) -> list[Problem]:
    out: list[Problem] = []
    for raw in data.get("issues", []):
        if not isinstance(raw, dict):
            continue
        measure = _parse_measure(
            raw.get("marked_measure", raw.get("issue_measure", raw.get("measure")))
        )
        out.append(
            Problem(
                type=str(raw.get("issue", "UNKNOWN")),
                severity="MANUAL",
                measure=measure,
                time_sec=_parse_time(raw.get("marked_time", raw.get("time"))),
                source="manual",
                message=str(raw.get("comment", "")),
            )
        )
    return out


def problems_from_unified(data: dict[str, Any]) -> list[Problem]:
    out: list[Problem] = []
    for raw in data.get("problems", []):
        if not isinstance(raw, dict):
            continue
        out.append(
            Problem(
                type=str(raw.get("type", raw.get("code", raw.get("issue", "UNKNOWN")))),
                severity=str(raw.get("severity", "WARN")).upper(),
                measure=_parse_measure(raw.get("measure")),
                time_sec=_parse_time(raw.get("time_sec", raw.get("time"))),
                source=str(raw.get("source", "auto")),
                message=str(raw.get("message", raw.get("comment", ""))),
            )
        )
    return out


def metrics_from_chart_regression(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "note_count",
        "duration_sec",
        "peak_notes_per_second",
        "median_notes_per_measure",
        "rms_density_correlation",
        "fail_count",
        "warn_count",
        "lane_histogram",
        "drum_histogram",
    )
    metrics = {k: data[k] for k in keys if k in data}
    if "fail_count" not in metrics and "warnings" in data:
        metrics["fail_count"] = sum(
            1 for w in data["warnings"] if isinstance(w, dict) and w.get("severity") == "FAIL"
        )
    if "warn_count" not in metrics and "warnings" in data:
        metrics["warn_count"] = sum(
            1 for w in data["warnings"] if isinstance(w, dict) and w.get("severity") == "WARN"
        )
    return metrics


def normalize_song_report(data: dict[str, Any], path: Path | None = None) -> SongReport | None:
    if not isinstance(data, dict):
        return None
    kind = detect_report_kind(data)
    if kind == "unknown":
        return None

    src = str(path) if path else str(data.get("_source_file", ""))
    song_id = _song_id_from_path(path or Path(src or "song"), data)

    if kind == "chart_regression":
        problems = problems_from_chart_regression(data)
        metrics = metrics_from_chart_regression(data)
    elif kind == "qa_session":
        problems = problems_from_qa_session(data)
        metrics = {
            "manual_issue_count": len(problems),
            "with_comment": sum(1 for p in problems if p.message.strip()),
            "noticed_late": sum(
                1
                for issue in data.get("issues", [])
                if isinstance(issue, dict) and issue.get("reaction_lag") == "late"
            ),
        }
    else:
        problems = problems_from_unified(data)
        metrics = dict(data.get("metrics", {}))

    artist = str(data.get("artist", "")).strip()
    title = str(data.get("title", data.get("song", ""))).strip()
    genre = str(data.get("genre", data.get("primary_genre", "")) or "unknown").strip()
    bpm = float(data.get("bpm", 0) or 0)
    preset = str(data.get("preset", data.get("mode", ""))).strip()
    lanes = int(data.get("lanes", 0) or 0)
    instrument = str(data.get("instrument", "")).strip()

    return SongReport(
        song_id=song_id,
        artist=artist,
        title=title,
        genre=genre or "unknown",
        bpm=bpm,
        preset=preset,
        lanes=lanes,
        instrument=instrument,
        report_kind=kind,
        problems=problems,
        metrics=metrics,
        source_file=src,
    )


def load_reports_from_dir(
    root: Path,
    *,
    recursive: bool = True,
) -> list[SongReport]:
    if not root.is_dir():
        return []
    paths = sorted(root.rglob("*.json") if recursive else root.glob("*.json"))
    reports: list[SongReport] = []
    for path in paths:
        if path.name in SKIP_JSON_NAMES:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        data["_source_file"] = str(path)
        report = normalize_song_report(data, path)
        if report:
            reports.append(report)
    return reports


def _group_stats(reports: list[SongReport], key_fn) -> dict[str, Any]:
    grouped: dict[str, list[SongReport]] = defaultdict(list)
    for report in reports:
        grouped[key_fn(report)].append(report)
    out: dict[str, Any] = {}
    for label, items in sorted(grouped.items()):
        problems = sum(len(r.problems) for r in items)
        out[label] = {
            "songs": len(items),
            "problems": problems,
            "avg_problems_per_song": round(problems / len(items), 2) if items else 0.0,
        }
    return out


def _intro_vs_rest_measures(reports: list[SongReport], intro_max: int = 16) -> dict[str, int]:
    intro = 0
    rest = 0
    for report in reports:
        for problem in report.problems:
            if problem.measure is None:
                continue
            if problem.measure <= intro_max:
                intro += 1
            else:
                rest += 1
    return {"intro_measures_lte16": intro, "after_intro": rest}


def build_summary(reports: list[SongReport], *, source_dir: str = "") -> dict[str, Any]:
    songs_tested = len(reports)
    total_problems = sum(len(r.problems) for r in reports)
    by_type: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    measure_counter: Counter[int] = Counter()
    measure_songs: dict[int, set[str]] = defaultdict(set)
    type_songs: dict[str, set[str]] = defaultdict(set)

    for report in reports:
        for problem in report.problems:
            by_type[problem.type] += 1
            by_severity[problem.severity] += 1
            type_songs[problem.type].add(report.song_id)
            if problem.measure is not None:
                measure_counter[problem.measure] += 1
                measure_songs[problem.measure].add(report.song_id)

    ranked = sorted(reports, key=lambda r: (-len(r.problems), r.display_name.lower()))
    top_songs = []
    for report in ranked[:15]:
        sev = report.severity_counts()
        top_songs.append(
            {
                "song_id": report.song_id,
                "display": report.display_name,
                "artist": report.artist,
                "title": report.title,
                "genre": report.genre,
                "bpm": report.bpm,
                "problems": len(report.problems),
                "fail": sev.get("FAIL", 0),
                "warn": sev.get("WARN", 0),
                "manual": sev.get("MANUAL", 0),
                "preset": report.preset,
                "lanes": report.lanes,
            }
        )

    top_measures = [
        {
            "measure": measure,
            "count": count,
            "songs_affected": len(measure_songs[measure]),
        }
        for measure, count in measure_counter.most_common(20)
    ]

    bpms = [r.bpm for r in reports if r.bpm > 0]
    note_counts = [
        int(r.metrics["note_count"])
        for r in reports
        if isinstance(r.metrics.get("note_count"), (int, float))
    ]
    rms_values = [
        float(r.metrics["rms_density_correlation"])
        for r in reports
        if r.metrics.get("rms_density_correlation") is not None
    ]

    codes_insight: dict[str, Any] = {}
    for code, count in by_type.most_common():
        affected = len(type_songs[code])
        codes_insight[code] = {
            "count": count,
            "songs_affected": affected,
            "pct_songs_affected": round(affected / songs_tested * 100, 1) if songs_tested else 0.0,
        }

    songs_clean = sum(1 for r in reports if not r.problems)
    songs_with_fail = sum(1 for r in reports if any(p.severity == "FAIL" for p in r.problems))
    low_rms = sum(
        1
        for r in reports
        if r.metrics.get("rms_density_correlation") is not None
        and float(r.metrics["rms_density_correlation"]) < 0.35
    )

    manual_late = sum(
        int(r.metrics.get("noticed_late", 0))
        for r in reports
        if r.report_kind == "qa_session"
    )
    manual_with_comment = sum(
        int(r.metrics.get("with_comment", 0))
        for r in reports
        if r.report_kind == "qa_session"
    )

    return {
        "version": SUMMARY_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dir": source_dir,
        "songs_tested": songs_tested,
        "total_problems": total_problems,
        "avg_problems_per_song": round(total_problems / songs_tested, 2) if songs_tested else 0.0,
        "songs_clean": songs_clean,
        "songs_with_fail": songs_with_fail,
        "by_issue_type": dict(by_type.most_common()),
        "by_severity": dict(by_severity.most_common()),
        "issue_type_insights": codes_insight,
        "top_songs_by_problems": top_songs,
        "top_problem_measures": top_measures,
        "measure_hotspots_intro_vs_rest": _intro_vs_rest_measures(reports),
        "bpm_stats": {
            "count_with_bpm": len(bpms),
            "min": min(bpms) if bpms else None,
            "max": max(bpms) if bpms else None,
            "median": median(bpms) if bpms else None,
            "bands": _group_stats([r for r in reports if r.bpm > 0], lambda r: bpm_band(r.bpm)),
        },
        "genre_stats": _group_stats(reports, lambda r: r.genre or "unknown"),
        "preset_stats": _group_stats([r for r in reports if r.preset], lambda r: r.preset),
        "lanes_stats": _group_stats([r for r in reports if r.lanes], lambda r: str(r.lanes)),
        "report_kind_stats": _group_stats(reports, lambda r: r.report_kind),
        "chart_metrics": {
            "note_count_min": min(note_counts) if note_counts else None,
            "note_count_max": max(note_counts) if note_counts else None,
            "note_count_median": median(note_counts) if note_counts else None,
            "low_rms_correlation_songs": low_rms,
            "rms_correlation_median": median(rms_values) if rms_values else None,
        },
        "manual_qa_stats": {
            "sessions": sum(1 for r in reports if r.report_kind == "qa_session"),
            "noticed_late_marks": manual_late,
            "marks_with_comment": manual_with_comment,
        },
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        f"Generation problems summary ({summary.get('generated_at', '')})",
        f"Source: {summary.get('source_dir', '')}",
        f"Songs tested: {summary.get('songs_tested', 0)}",
        f"Total problems: {summary.get('total_problems', 0)} "
        f"(avg {summary.get('avg_problems_per_song', 0)}/song)",
        f"Clean songs: {summary.get('songs_clean', 0)} | With FAIL: {summary.get('songs_with_fail', 0)}",
        "",
        "By issue type:",
    ]
    for issue, count in summary.get("by_issue_type", {}).items():
        insight = summary.get("issue_type_insights", {}).get(issue, {})
        pct = insight.get("pct_songs_affected", 0)
        lines.append(f"  {issue}: {count} ({pct}% songs)")
    lines.append("")
    lines.append("Top songs:")
    for row in summary.get("top_songs_by_problems", [])[:8]:
        lines.append(f"  {row.get('display', row.get('song_id'))}: {row.get('problems', 0)}")
    lines.append("")
    lines.append("Top measures:")
    for row in summary.get("top_problem_measures", [])[:8]:
        lines.append(
            f"  m{row.get('measure')}: {row.get('count')} hits, "
            f"{row.get('songs_affected')} songs"
        )
    return "\n".join(lines) + "\n"
