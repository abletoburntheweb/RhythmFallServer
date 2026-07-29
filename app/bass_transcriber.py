# app/bass_transcriber.py — Basic Pitch (primary) bass note transcription.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Bass stem range (E1–G4) — matches bass_generator pyin bounds.
BASS_MIN_HZ = 41.2
BASS_MAX_HZ = 392.0

_MODEL = None
_MODEL_PATH: Optional[Path] = None

try:
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model, predict

    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    Model = None  # type: ignore
    predict = None  # type: ignore
    FilenameSuffix = None  # type: ignore
    build_icassp_2022_model_path = None  # type: ignore


def bass_backend_name() -> str:
    raw = os.environ.get("RFALL_BASS_BACKEND", "basic_pitch").strip().lower()
    if raw in ("basic_pitch", "basic-pitch", "bp", "basicpitch"):
        return "basic_pitch"
    if raw in ("heuristic", "pyin", "librosa", "legacy", "off", "0"):
        return "heuristic"
    return raw


def is_basic_pitch_available() -> bool:
    return BASIC_PITCH_AVAILABLE


def _resolve_model_path() -> Path:
    """Prefer ONNX on Windows/server stacks that already use onnxruntime."""
    env = os.environ.get("RFALL_BASIC_PITCH_MODEL", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    if not BASIC_PITCH_AVAILABLE:
        raise RuntimeError("basic-pitch not installed")
    assert build_icassp_2022_model_path is not None
    assert FilenameSuffix is not None
    onnx_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    if onnx_path.is_file():
        return onnx_path
    from basic_pitch import ICASSP_2022_MODEL_PATH

    return Path(ICASSP_2022_MODEL_PATH)


def _get_model():
    global _MODEL, _MODEL_PATH
    if not BASIC_PITCH_AVAILABLE:
        raise RuntimeError("basic-pitch not installed")
    path = _resolve_model_path()
    if _MODEL is not None and _MODEL_PATH == path:
        return _MODEL
    _MODEL = Model(path)
    _MODEL_PATH = path
    return _MODEL


def _note_events_to_segments(
    note_events: List[Tuple[float, float, int, float, Any]],
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for item in note_events:
        if len(item) < 4:
            continue
        start = float(item[0])
        end = float(item[1])
        midi = float(item[2])
        amp = float(item[3])
        if end <= start:
            end = start + 0.05
        segments.append(
            {
                "start": start,
                "end": end,
                "midi": midi,
                "amp": amp,
                "amp_mean": amp,
            }
        )
    return sorted(segments, key=lambda s: float(s["start"]))


def _monophonic_lowest(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep a single bass line when Basic Pitch returns overlapping pitches.

    Rapid successive plucks often overlap slightly in BP output — if note heads
    are clearly separated, keep both instead of dropping the second hit.
    """
    if len(segments) < 2:
        return segments
    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not out:
            out.append(seg)
            continue
        prev = out[-1]
        start_delta = abs(float(seg["start"]) - float(prev["start"]))
        overlap = min(float(prev["end"]), float(seg["end"])) - max(float(prev["start"]), float(seg["start"]))
        if overlap > 0.02 and start_delta < 0.045:
            if float(seg["midi"]) < float(prev["midi"]):
                out[-1] = seg
            elif float(seg["midi"]) == float(prev["midi"]) and float(seg["amp"]) > float(prev["amp"]):
                out[-1] = {
                    **prev,
                    "end": max(float(prev["end"]), float(seg["end"])),
                    "amp": max(float(prev["amp"]), float(seg["amp"])),
                    "amp_mean": max(float(prev.get("amp_mean", 0.0)), float(seg.get("amp_mean", 0.0))),
                }
            continue
        out.append(seg)
    return out


def try_transcribe_basic_pitch(
    audio_path: str,
    *,
    cancel_cb: Optional[Callable[[], None]] = None,
    onset_threshold: Optional[float] = None,
    frame_threshold: Optional[float] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Return segments when backend is basic_pitch and library is installed; else None."""
    if bass_backend_name() != "basic_pitch":
        return None
    if not is_basic_pitch_available():
        print("[BassTranscriber] basic-pitch not installed — heuristic fallback")
        return None
    try:
        segments = transcribe_bass_basic_pitch(
            audio_path,
            cancel_cb=cancel_cb,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )
        print(
            f"[BassTranscriber] basic_pitch notes={len(segments)} "
            f"model={_MODEL_PATH or _resolve_model_path()}"
        )
        return segments
    except Exception as exc:
        print(f"[BassTranscriber] basic_pitch failed: {exc}")
        return None


def transcribe_bass_basic_pitch(
    audio_path: str,
    *,
    cancel_cb: Optional[Callable[[], None]] = None,
    onset_threshold: Optional[float] = None,
    frame_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if not BASIC_PITCH_AVAILABLE or predict is None:
        return []
    if cancel_cb:
        cancel_cb()

    model = _get_model()
    if cancel_cb:
        cancel_cb()

    # Slightly more sensitive defaults help dense metal/pluck bass; override via env.
    min_note_ms = float(os.environ.get("RFALL_BASS_BP_MIN_NOTE_MS", "70"))
    onset = float(
        onset_threshold
        if onset_threshold is not None
        else os.environ.get("RFALL_BASS_BP_ONSET", "0.42")
    )
    frame = float(
        frame_threshold
        if frame_threshold is not None
        else os.environ.get("RFALL_BASS_BP_FRAME", "0.30")
    )

    _, _, note_events = predict(
        str(audio_path),
        model,
        onset_threshold=onset,
        frame_threshold=frame,
        minimum_note_length=min_note_ms,
        minimum_frequency=BASS_MIN_HZ,
        maximum_frequency=BASS_MAX_HZ,
        melodia_trick=True,
    )
    segments = _note_events_to_segments(note_events)
    segments = _monophonic_lowest(segments)
    segments = _merge_touching_same_pitch(segments)
    return segments


def _merge_touching_same_pitch(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Glue adjacent Basic Pitch notes on the same pitch into one sustain."""
    if len(segments) < 2:
        return segments
    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not out:
            out.append(dict(seg))
            continue
        prev = out[-1]
        same_pitch = abs(float(seg["midi"]) - float(prev["midi"])) <= 0.6
        gap = float(seg["start"]) - float(prev["end"])
        if same_pitch and gap <= 0.06:
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
            prev["amp"] = max(float(prev.get("amp", 0.0)), float(seg.get("amp", 0.0)))
            prev["amp_mean"] = max(
                float(prev.get("amp_mean", 0.0)), float(seg.get("amp_mean", 0.0))
            )
            continue
        out.append(dict(seg))
    return out
