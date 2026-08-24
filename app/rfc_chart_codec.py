"""RhythmFall RFC chart text format (v1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


RFC_VERSION = 1
BASS_SHAPES = frozenset({"tap", "hold", "slide"})
BASS_SHAPE_ALIASES = {"sustain": "hold", "octave": "tap"}
BASS_CURVES = frozenset({"linear", "bend", "gliss"})


def notes_to_spawn_array(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "TrackInfo":
            continue
        entry: Dict[str, Any] = {
            "lane": float(item.get("lane", 0)),
            "time": float(item.get("time", 0.0)),
            "type": "DrumNote",
        }
        drum = str(item.get("drum", "")).strip().lower()
        if drum:
            entry["drum"] = drum
        out.append(entry)
    out.sort(key=lambda n: float(n.get("time", 0.0)))
    return out


def _sanitize_header_value(raw: str) -> str:
    return str(raw or "").strip().replace("\n", " ").replace("\r", "")


def _track_comment_line(artist: str, title: str) -> str:
    a = _sanitize_header_value(artist)
    t = _sanitize_header_value(title)
    if a and t:
        return f"# {a} — {t}"
    if t:
        return f"# {t}"
    if a:
        return f"# {a}"
    return ""


def serialize(
    notes: List[Dict[str, Any]],
    instrument: str = "drums",
    intent: str = "groove",
    lanes: int = 4,
    artist: str = "",
    title: str = "",
    mode: Optional[str] = None,
) -> str:
    chart_intent = str(intent or mode or "groove").strip().lower() or "groove"
    spawn = notes_to_spawn_array(notes)
    a = _sanitize_header_value(artist)
    t = _sanitize_header_value(title)
    lines = [
        f"# RFC {RFC_VERSION}",
        "# RhythmFall chart",
    ]
    track_line = _track_comment_line(a, t)
    if track_line:
        lines.append(track_line)
    lines.append("")
    if a:
        lines.append(f"artist={a}")
    if t:
        lines.append(f"title={t}")
    lines.extend(
        [
            f"instrument={instrument.lower()}",
            f"intent={chart_intent}",
            f"lanes={lanes}",
            f"notes={len(spawn)}",
            "",
            "---",
        ]
    )
    has_drum = any(str(n.get("drum", "")).strip() for n in spawn)
    if has_drum:
        lines.append("# time(s)   lane   drum")
    else:
        lines.append("# time(s)   lane")
    for note in spawn:
        t = float(note["time"])
        lane = int(note["lane"])
        if has_drum:
            drum = str(note.get("drum", "")).strip().lower()
            if drum:
                lines.append(f"{t:9.4f}  {lane}  {drum}")
            else:
                lines.append(f"{t:9.4f}  {lane}")
        else:
            lines.append(f"{t:9.4f}  {lane}")
    return "\n".join(lines) + "\n"


def parse_header(text: str) -> Dict[str, str]:
    """Read key=value lines before the ``---`` body marker."""
    header: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "---":
            break
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip()
    return header


def read_header(path: Path | str) -> Dict[str, str]:
    chart_path = Path(path)
    if not chart_path.is_file():
        return {}
    return parse_header(chart_path.read_text(encoding="utf-8"))


def _parse_bool_flag(raw: str) -> bool:
    return str(raw or "").strip().lower() in ("1", "true", "yes", "ghost")


def _normalize_bass_shape(raw: str) -> str:
    s = str(raw or "tap").strip().lower()
    s = BASS_SHAPE_ALIASES.get(s, s)
    return s if s in BASS_SHAPES else "tap"


def _parse_lane_field(raw: str) -> tuple[List[int], float]:
    text = str(raw or "").strip()
    if "," in text:
        lanes = [int(x.strip()) for x in text.split(",") if x.strip()]
        if not lanes:
            return [0], 0.0
        return lanes, float(lanes[0])
    return [int(text)], float(int(text))


def _bass_shape_to_type(shape: str) -> str:
    s = _normalize_bass_shape(shape)
    if s == "hold":
        return "BassHoldNote"
    if s == "slide":
        return "BassSlideNote"
    return "BassTapNote"


def bass_notes_to_spawn_array(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        shape = _normalize_bass_shape(str(item.get("shape", "tap")))
        if isinstance(item.get("lanes"), list) and item.get("lanes"):
            lanes = [int(x) for x in item["lanes"]]
        else:
            lanes, _ = _parse_lane_field(str(item.get("lane", 0)))
        entry: Dict[str, Any] = {
            "time": float(item.get("time", 0.0)),
            "type": _bass_shape_to_type(shape),
            "shape": shape,
            "ghost": bool(item.get("ghost", False)),
            "lane": float(lanes[0]),
            "lanes": lanes,
        }
        if shape in ("hold", "slide"):
            end = float(item.get("end", item.get("time", 0.0)))
            entry["end"] = end
            entry["duration"] = max(end - float(entry["time"]), 0.05)
        if shape == "slide":
            entry["lane_end"] = float(item.get("lane_end", lanes[0]))
            curve = str(item.get("curve", "linear")).strip().lower()
            entry["curve"] = curve if curve in BASS_CURVES else "linear"
        out.append(entry)
    out.sort(key=lambda n: float(n.get("time", 0.0)))
    return out


def serialize_bass(
    notes: List[Dict[str, Any]],
    intent: str = "original_standard",
    lanes: int = 5,
    artist: str = "",
    title: str = "",
) -> str:
    spawn = bass_notes_to_spawn_array(notes)
    a = _sanitize_header_value(artist)
    t = _sanitize_header_value(title)
    lines = [
        f"# RFC {RFC_VERSION}",
        "# RhythmFall chart",
    ]
    track_line = _track_comment_line(a, t)
    if track_line:
        lines.append(track_line)
    lines.append("")
    if a:
        lines.append(f"artist={a}")
    if t:
        lines.append(f"title={t}")
    chart_intent = str(intent or "original_standard").strip().lower() or "original_standard"
    lines.extend(
        [
            "instrument=bass",
            f"intent={chart_intent}",
            f"lanes={lanes}",
            f"notes={len(spawn)}",
            "",
            "---",
            "# time(s)   lane   end(s)   lane_end   curve   shape   ghost",
        ]
    )
    for note in spawn:
        t0 = float(note["time"])
        shape = str(note.get("shape", "tap"))
        ghost = "1" if bool(note.get("ghost", False)) else ""
        lanes = note.get("lanes") or [int(note.get("lane", 0))]
        lane_cols = ",".join(str(int(x)) for x in lanes)
        end_s = float(note.get("end", 0.0)) if shape in ("hold", "slide") else 0.0
        end_txt = f"{end_s:7.4f}" if end_s > 0 else ""
        lane_end_txt = ""
        curve_txt = ""
        if shape == "slide":
            lane_end_txt = str(int(note.get("lane_end", lanes[0])))
            curve_txt = str(note.get("curve", "linear"))
        lines.append(
            f"{t0:9.4f}  {lane_cols}  {end_txt}  {lane_end_txt}  {curve_txt}  {shape}  {ghost}".rstrip()
        )
    return "\n".join(lines) + "\n"


def write_bass_file(
    path,
    notes: List[Dict[str, Any]],
    intent: str = "original_standard",
    lanes: int = 5,
    artist: str = "",
    title: str = "",
) -> None:
    path = path.with_suffix(".rf") if path.suffix.lower() != ".rf" else path
    body = serialize_bass(notes, intent=intent, lanes=lanes, artist=artist, title=title)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
        f.flush()
    temp_path.replace(path)


def _parse_bass_line(line: str) -> Optional[Dict[str, Any]]:
    parts = line.replace("\t", " ").split()
    if len(parts) < 2:
        return None
    try:
        t0 = float(parts[0])
    except ValueError:
        return None
    lanes, primary_lane = _parse_lane_field(parts[1])
    tail = [p.strip() for p in parts[2:]]
    shape = "tap"
    ghost = False
    curve = "linear"
    end_s = 0.0
    lane_end: Optional[int] = None
    floats: List[float] = []
    for token in tail:
        low = token.lower()
        if _parse_bool_flag(token):
            ghost = True
        elif low in BASS_SHAPES or low in BASS_SHAPE_ALIASES:
            shape = _normalize_bass_shape(low)
        elif low in BASS_CURVES:
            curve = low
        elif token.replace(".", "", 1).isdigit() or (
            token.count(".") == 1 and token.replace(".", "", 1).isdigit()
        ):
            try:
                floats.append(float(token))
            except ValueError:
                pass
        elif token.isdigit():
            lane_end = int(token)
    if floats:
        end_s = floats[0]
        if len(floats) > 1 and lane_end is None:
            lane_end = int(floats[1])
    entry: Dict[str, Any] = {
        "time": t0,
        "lane": primary_lane,
        "lanes": lanes,
        "shape": shape,
        "ghost": ghost,
        "type": _bass_shape_to_type(shape),
    }
    if shape in ("hold", "slide") and end_s > 0:
        entry["end"] = end_s
        entry["duration"] = max(end_s - t0, 0.05)
    if shape == "slide":
        entry["lane_end"] = float(lane_end if lane_end is not None else lanes[0])
        entry["curve"] = curve
    return entry


def parse(text: str) -> List[Dict[str, Any]]:
    header = parse_header(text)
    instrument = str(header.get("instrument", "drums")).strip().lower()
    if instrument == "bass":
        return _parse_bass_body(text)
    return _parse_drums_body(text)


def _parse_drums_body(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    in_body = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "---":
            in_body = True
            continue
        if not in_body or line.startswith("#"):
            continue
        parts = line.replace("\t", " ").split()
        if len(parts) < 2:
            continue
        entry: Dict[str, Any] = {
            "lane": float(int(parts[1])),
            "time": float(parts[0]),
            "type": "DrumNote",
        }
        if len(parts) >= 3:
            drum = str(parts[2]).strip().lower()
            if drum:
                entry["drum"] = drum
        out.append(entry)
    out.sort(key=lambda n: float(n.get("time", 0.0)))
    return out


def _parse_bass_body(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    in_body = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "---":
            in_body = True
            continue
        if not in_body or line.startswith("#"):
            continue
        entry = _parse_bass_line(line)
        if entry:
            out.append(entry)
    out.sort(key=lambda n: float(n.get("time", 0.0)))
    return out


def write_file(
    path,
    notes: List[Dict[str, Any]],
    instrument: str = "drums",
    intent: str = "groove",
    lanes: int = 4,
    artist: str = "",
    title: str = "",
    mode: Optional[str] = None,
) -> None:
    path = path.with_suffix(".rf") if path.suffix.lower() != ".rf" else path
    body = serialize(
        notes,
        instrument=instrument,
        intent=intent,
        lanes=lanes,
        artist=artist,
        title=title,
        mode=mode,
    )
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
        f.flush()
    temp_path.replace(path)


def read_file(path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return notes_to_spawn_array(data)
    with open(path, "r", encoding="utf-8") as f:
        return parse(f.read())
