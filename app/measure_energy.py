# app/measure_energy.py — per-measure energy + chart signatures (measure map v0).
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None  # type: ignore
    SOUNDFILE_AVAILABLE = False


def measure_map_enabled() -> bool:
    return os.getenv("RF_MEASURE_MAP", "1") == "1"


def measure_map_verbose_rows() -> bool:
    return os.getenv("RF_MEASURE_MAP_FULL", "0") == "1"


def load_mono_audio(audio_path: Optional[str]) -> Tuple[np.ndarray, int]:
    """Load mono float32 PCM. soundfile (WAV/FLAC) first, then ffmpeg for MP3 etc."""
    if not audio_path or not os.path.isfile(audio_path):
        return np.array([], dtype=np.float32), 0

    if SOUNDFILE_AVAILABLE:
        try:
            data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            if data.size == 0 or sr <= 0:
                raise ValueError("empty audio")
            mono = np.mean(data, axis=1).astype(np.float32, copy=False)
            return mono, int(sr)
        except Exception:
            pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return np.array([], dtype=np.float32), 0

    ffprobe = shutil.which("ffprobe")
    if not ffprobe and os.path.dirname(ffmpeg):
        sibling = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(sibling):
            ffprobe = sibling

    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                audio_path,
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-vn",
                "-",
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return np.array([], dtype=np.float32), 0
        y = np.frombuffer(proc.stdout, dtype=np.float32)
        if y.size == 0:
            return np.array([], dtype=np.float32), 0
        sr = 44100
        if ffprobe:
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip().isdigit():
                sr = int(probe.stdout.strip())
        return y, sr
    except Exception:
        return np.array([], dtype=np.float32), 0


def rms_per_measure(
    audio_path: Optional[str],
    first_measure_start: float,
    measure_duration: float,
    max_measure: int,
    *,
    y: Optional[np.ndarray] = None,
    sr: int = 0,
) -> Dict[int, float]:
    if max_measure < 0:
        return {}
    if y is None or sr <= 0:
        y, sr = load_mono_audio(audio_path)
    if len(y) < 1 or sr <= 0:
        return {}

    out: Dict[int, float] = {}
    for measure_idx in range(0, max_measure + 1):
        start = first_measure_start + measure_idx * measure_duration
        end = start + measure_duration
        s0 = max(0, int(round(start * sr)))
        s1 = min(len(y), int(round(end * sr)))
        if s1 <= s0:
            continue
        chunk = y[s0:s1]
        out[measure_idx] = float(np.sqrt(np.mean(chunk * chunk)))
    return out


def _rolling_median(values: Dict[int, float], center: int, radius: int) -> float:
    lo = center - radius
    hi = center + radius
    window = [float(values[i]) for i in range(lo, hi + 1) if i in values and values[i] > 0]
    if not window:
        return 0.0
    return float(np.median(window))


def relative_contour_per_measure(
    rms: Dict[int, float],
    max_measure: int,
    *,
    rolling_radius: int = 16,
) -> Dict[int, float]:
    """Per-measure energy vs rolling median (1.0 ≈ typical local level)."""
    if not rms or max_measure < 0:
        return {}
    radius = max(1, int(rolling_radius))
    out: Dict[int, float] = {}
    for measure_idx in range(0, max_measure + 1):
        rms_val = float(rms.get(measure_idx, 0.0) or 0.0)
        med = _rolling_median(rms, measure_idx, radius)
        out[measure_idx] = (rms_val / med) if med > 1e-9 else 0.0
    return out


def is_loud_mix_quiet_drum(
    drum_rel: float,
    mix_rel: float,
    *,
    mix_loud_min: float = 0.85,
    drum_quiet_max: float = 0.55,
) -> bool:
    """Mix swells while drum stem stays below its local baseline (Tycho-style split)."""
    if mix_rel < mix_loud_min or drum_rel <= 0.0:
        return False
    return drum_rel < drum_quiet_max


def is_quiet_mix_breakdown(
    drum_rel: float,
    mix_rel: float,
    *,
    mix_quiet_max: float = 0.48,
    mix_loud_min: float = 0.85,
    drum_quiet_max: float = 0.55,
) -> bool:
    """Full mix below local baseline (breakdown/bridge) — strip stem-artifact notes.

    Excludes Tycho-style loud-mix/quiet-drum splits protected by ``is_loud_mix_quiet_drum``.
    """
    if mix_rel <= 0.0:
        return False
    if is_loud_mix_quiet_drum(
        drum_rel,
        mix_rel,
        mix_loud_min=mix_loud_min,
        drum_quiet_max=drum_quiet_max,
    ):
        return False
    return mix_rel < mix_quiet_max


def is_phantom_orphan_measure(
    *,
    ks: int,
    note_count: int,
    drum_rel: float,
    mix_rel: float,
    mix_rms_val: float,
    mix_median: float,
    stem_quiet: bool,
    mix_quiet_rel_max: float = 0.52,
    phantom_mix_rel_max: float = 0.62,
    mix_absolute_ratio: float = 0.40,
    mix_loud_min: float = 0.85,
    drum_quiet_max: float = 0.55,
    phantom_min_notes: int = 1,
) -> bool:
    """Chart notes with no kick/snare in ADTOF — likely stem/mix artifact in a breakdown."""
    if ks > 0 or note_count < phantom_min_notes:
        return False
    if is_loud_mix_quiet_drum(
        drum_rel,
        mix_rel,
        mix_loud_min=mix_loud_min,
        drum_quiet_max=drum_quiet_max,
    ):
        return False
    if stem_quiet:
        return True
    if is_quiet_mix_breakdown(
        drum_rel,
        mix_rel,
        mix_quiet_max=mix_quiet_rel_max,
        mix_loud_min=mix_loud_min,
        drum_quiet_max=drum_quiet_max,
    ):
        return True
    if mix_median > 0.0 and mix_rms_val > 0.0 and mix_rms_val < mix_median * mix_absolute_ratio:
        return True
    if mix_rel > 0.0 and mix_rel < phantom_mix_rel_max and note_count <= 3:
        return True
    return False


def slot_signatures_per_measure(
    events: List[float],
    first_measure_start: float,
    measure_duration: float,
    beat_interval: float,
    max_measure: int,
) -> Dict[int, Tuple[tuple, int]]:
    buckets: Dict[int, List[float]] = {}
    for event in sorted(events):
        measure_idx = int(np.floor((event - first_measure_start) / measure_duration))
        if measure_idx < 0 or measure_idx > max_measure:
            continue
        rel_beats = (event - (first_measure_start + measure_idx * measure_duration)) / beat_interval
        if rel_beats < -0.1 or rel_beats >= 4.1:
            continue
        quantized = round(max(0.0, min(3.75, rel_beats)) * 4.0) / 4.0
        bucket = buckets.setdefault(measure_idx, [])
        if quantized not in bucket:
            bucket.append(quantized)

    out: Dict[int, Tuple[tuple, int]] = {}
    for idx, positions in buckets.items():
        sig = tuple(sorted(positions))
        out[idx] = (sig, len(positions))
    return out


def build_measure_map(
    events: List[float],
    beats: np.ndarray,
    bpm: float,
    drum_audio_path: Optional[str],
    mix_audio_path: Optional[str],
    *,
    rolling_radius: int = 16,
) -> List[Dict[str, object]]:
    if beats is None or len(beats) < 1:
        return []

    beat_interval = float(np.median(np.diff(beats))) if len(beats) >= 2 else 60.0 / max(1.0, bpm)
    first_measure_start = float(beats[0])
    measure_duration = beat_interval * 4.0
    max_measure = 0
    if events:
        max_measure = int(max(0, np.floor((max(events) - first_measure_start) / measure_duration)))

    y_drum, sr_drum = load_mono_audio(drum_audio_path)
    y_mix, sr_mix = load_mono_audio(mix_audio_path)
    drum_rms = rms_per_measure(
        drum_audio_path,
        first_measure_start,
        measure_duration,
        max_measure,
        y=y_drum,
        sr=sr_drum,
    )
    mix_rms = rms_per_measure(
        mix_audio_path,
        first_measure_start,
        measure_duration,
        max_measure,
        y=y_mix,
        sr=sr_mix,
    )

    signatures = slot_signatures_per_measure(
        events, first_measure_start, measure_duration, beat_interval, max_measure
    )

    sig_to_cluster: Dict[tuple, int] = {}
    rows: List[Dict[str, object]] = []
    for m in range(0, max_measure + 1):
        sig, note_count = signatures.get(m, (tuple(), 0))
        cluster = -1
        if sig:
            if sig not in sig_to_cluster:
                sig_to_cluster[sig] = len(sig_to_cluster)
            cluster = sig_to_cluster[sig]

        d_rms = float(drum_rms.get(m, 0.0) or 0.0)
        x_rms = float(mix_rms.get(m, 0.0) or 0.0)
        d_med = _rolling_median(drum_rms, m, rolling_radius)
        x_med = _rolling_median(mix_rms, m, rolling_radius)
        d_rel = (d_rms / d_med) if d_med > 1e-9 else 0.0
        x_rel = (x_rms / x_med) if x_med > 1e-9 else 0.0

        rows.append(
            {
                "m": m,
                "drum_rms": round(d_rms, 5),
                "mix_rms": round(x_rms, 5),
                "drum_rel": round(d_rel, 3),
                "mix_rel": round(x_rel, 3),
                "notes": note_count,
                "sig": sig,
                "repeat_id": cluster,
            }
        )
    return rows


def log_measure_map(
    label: str,
    rows: List[Dict[str, object]],
    *,
    drum_path: Optional[str] = None,
    mix_path: Optional[str] = None,
) -> None:
    if not rows:
        print(f"[MeasureMap] {label} (empty)")
        return

    drum_vals = [float(r["drum_rms"]) for r in rows if float(r["drum_rms"]) > 0]
    mix_vals = [float(r["mix_rms"]) for r in rows if float(r["mix_rms"]) > 0]
    note_vals = [int(r["notes"]) for r in rows]
    d_med = float(np.median(drum_vals)) if drum_vals else 0.0
    x_med = float(np.median(mix_vals)) if mix_vals else 0.0
    n_med = float(np.median(note_vals)) if note_vals else 0.0

    drum_name = os.path.basename(drum_path or "")
    mix_name = os.path.basename(mix_path or "")
    print(
        f"[MeasureMap] {label} measures={len(rows)} "
        f"drum_med={d_med:.4f} mix_med={x_med:.4f} notes_med={n_med:.1f} "
        f"drum={drum_name or '-'} mix={mix_name or '-'}"
    )

    sparse = [r for r in rows if int(r["notes"]) <= max(1, int(n_med - 2))]
    dense = [r for r in rows if int(r["notes"]) >= max(5, int(n_med + 3))]
    if sparse:
        sample = sparse[:6]
        print(
            "[MeasureMap]   sparse_sample: "
            + " | ".join(
                f"m{int(r['m'])} d={r['drum_rel']} x={r['mix_rel']} n={r['notes']}"
                for r in sample
            )
        )
    if dense:
        sample = dense[:6]
        print(
            "[MeasureMap]   dense_sample: "
            + " | ".join(
                f"m{int(r['m'])} d={r['drum_rel']} x={r['mix_rel']} n={r['notes']}"
                for r in sample
            )
        )

    if not measure_map_verbose_rows():
        return

    for r in rows:
        sig = r.get("sig") or tuple()
        print(
            f"[MeasureMap]   m={int(r['m']):3d} "
            f"drum={r['drum_rms']:.4f} mix={r['mix_rms']:.4f} "
            f"d_rel={r['drum_rel']:.2f} x_rel={r['mix_rel']:.2f} "
            f"notes={int(r['notes'])} rep={int(r['repeat_id'])} sig={_format_sig(sig)}"
        )


def _format_sig(sig: tuple) -> str:
    if not sig:
        return "-"
    return ",".join(f"{p:g}" for p in sig)


def measure_map_recap_line(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return "measure_map=unavailable"
    drum_vals = [float(r["drum_rms"]) for r in rows if float(r["drum_rms"]) > 0]
    mix_vals = [float(r["mix_rms"]) for r in rows if float(r["mix_rms"]) > 0]
    d_med = float(np.median(drum_vals)) if drum_vals else 0.0
    x_med = float(np.median(mix_vals)) if mix_vals else 0.0
    ratio = (x_med / d_med) if d_med > 1e-9 else 0.0
    split = sum(
        1
        for r in rows
        if float(r.get("mix_rel", 0) or 0) >= 0.9 and float(r.get("drum_rel", 0) or 0) < 0.55
    )
    return (
        f"measure_map measures={len(rows)} drum_med={d_med:.4f} mix_med={x_med:.4f} "
        f"mix/drum={ratio:.2f} loud_mix_quiet_drum={split}"
    )


_SALIENCE_RHYTHM_DRUMS = frozenset({"kick", "snare"})
_SALIENCE_TEXTURE_DRUMS = frozenset({"hat", "cymbal", "hihat", "ride", "crash"})


def hit_rms_energy(
    y: np.ndarray,
    sr: int,
    time_sec: float,
    *,
    window_ms: float = 30.0,
) -> float:
    if y is None or len(y) < 1 or sr <= 0:
        return 0.0
    half = max(1, int(round(window_ms * 0.001 * sr * 0.5)))
    center = int(round(float(time_sec) * sr))
    s0 = max(0, center - half)
    s1 = min(len(y), center + half)
    if s1 <= s0:
        return 0.0
    chunk = y[s0:s1]
    return float(np.sqrt(np.mean(chunk * chunk)))


def classify_salience_role(
    drum: str,
    energy: float,
    median_energy: float,
    *,
    texture_strictness: float = 1.0,
) -> str:
    name = str(drum or "unknown").strip().lower()
    if name in _SALIENCE_RHYTHM_DRUMS:
        return "RHYTHM"
    if name == "tom":
        threshold = max(median_energy * 0.55, 1e-9)
        return "RHYTHM" if energy >= threshold else "TEXTURE"
    if name in _SALIENCE_TEXTURE_DRUMS or not name or name == "unknown":
        threshold = max(median_energy * max(0.35, float(texture_strictness)), 1e-9)
        return "RHYTHM" if energy >= threshold else "TEXTURE"
    threshold = max(median_energy * max(0.45, float(texture_strictness) * 0.85), 1e-9)
    return "RHYTHM" if energy >= threshold else "TEXTURE"


def annotate_salience_roles(
    classified_hits: List[Dict],
    drum_audio_path: Optional[str],
    *,
    texture_strictness: float = 1.0,
) -> Tuple[List[Dict], Dict[str, float]]:
    if not classified_hits:
        return classified_hits, {"rhythm": 0.0, "texture": 0.0, "median_energy": 0.0}

    y, sr = load_mono_audio(drum_audio_path)
    energies = [hit_rms_energy(y, sr, float(h.get("time", 0.0) or 0.0)) for h in classified_hits]
    positive = [e for e in energies if e > 0]
    median_energy = float(np.median(positive)) if positive else 0.0

    rhythm = 0
    texture = 0
    for hit, energy in zip(classified_hits, energies):
        role = classify_salience_role(
            str(hit.get("drum") or hit.get("class") or ""),
            energy,
            median_energy,
            texture_strictness=texture_strictness,
        )
        hit["salience"] = role
        if role == "RHYTHM":
            rhythm += 1
        else:
            texture += 1

    return classified_hits, {
        "rhythm": float(rhythm),
        "texture": float(texture),
        "median_energy": median_energy,
    }


def filter_hits_by_salience_roles(
    classified_hits: List[Dict],
    include_roles: Optional[Sequence[str]],
) -> List[Dict]:
    if not classified_hits or not include_roles:
        return classified_hits
    allowed = {str(r).strip().upper() for r in include_roles if str(r).strip()}
    if not allowed:
        return classified_hits
    return [h for h in classified_hits if str(h.get("salience", "RHYTHM")).upper() in allowed]


def salience_recap_line(stats: Optional[Dict[str, float]], *, filtered_kept: int = 0, filtered_total: int = 0) -> str:
    if not stats:
        return ""
    rhythm = int(stats.get("rhythm", 0) or 0)
    texture = int(stats.get("texture", 0) or 0)
    if filtered_total > 0 and filtered_kept != filtered_total:
        return (
            f"salience rhythm={rhythm} texture={texture} "
            f"filter_kept={filtered_kept}/{filtered_total}"
        )
    return f"salience rhythm={rhythm} texture={texture} filter=off"


def adtof_drum_counts(classified_hits: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for hit in classified_hits or []:
        drum = str(hit.get("drum") or hit.get("class") or "unknown").lower()
        counts[drum] = counts.get(drum, 0) + 1
    return counts


def generation_recap_enabled() -> bool:
    return os.getenv("RF_GEN_RECAP", "1") == "1"


def drum_entry_recap_line(recap: Optional[Dict]) -> str:
    if not recap:
        return ""
    if not bool(recap.get("enabled", True)):
        return "critic drum_entry=off"
    skip = str(recap.get("skip", "")).strip()
    section_start = float(recap.get("section_start", 0.0) or 0.0)
    if skip == "no_section_start":
        return "critic drum_entry=skip(no_intro section_start=0)"
    if skip == "no_core_pool":
        return f"critic drum_entry=skip(no_kick_snare section_start={section_start:.3f})"
    recovered = int(recap.get("recovered", 0) or 0)
    if recovered <= 0:
        window = recap.get("window")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            return (
                f"critic drum_entry=0 section_start={section_start:.3f} "
                f"window=[{float(window[0]):.3f},{float(window[1]):.3f})"
            )
        return f"critic drum_entry=0 section_start={section_start:.3f}"
    times = recap.get("times") or []
    return (
        f"critic drum_entry=+{recovered} section_start={section_start:.3f} "
        f"times={times}"
    )


def print_generation_recap(
    *,
    track: str,
    genre: str,
    bpm: float,
    mode: str,
    preset_id: str,
    adtof_unique: int,
    adtof_kick: int,
    adtof_snare: int,
    adtof_hat: int,
    adtof_tom: int = 0,
    adtof_cymbal: int = 0,
    adtof_rows: int = 0,
    source_events: int,
    pre_section_events: int,
    post_section_events: int,
    final_events: int,
    caps_hps: int,
    caps_npm: int,
    measure_map_line: str = "",
    salience_line: str = "",
    critic_line: str = "",
    chart_variant: str = "",
    generation_goal: str = "",
    generation_difficulty: str = "",
    chart_stem: str = "",
) -> None:
    if not generation_recap_enabled():
        return
    print("[GenRecap] ---------- generation summary (copy from here) ----------")
    print(
        f"[GenRecap] track={track} genre={genre} bpm={bpm} mode={mode} preset={preset_id} "
        f"caps={caps_hps}/{caps_npm}"
        + (f" goal={generation_goal}" if generation_goal else "")
        + (f" difficulty={generation_difficulty}" if generation_difficulty else "")
        + (f" stem={chart_stem}" if chart_stem else "")
        + (f" variant={chart_variant}" if chart_variant else "")
    )
    print(
        f"[GenRecap] adtof unique={adtof_unique} kick={adtof_kick} snare={adtof_snare} "
        f"hat={adtof_hat} tom={adtof_tom} cymbal={adtof_cymbal} rows={adtof_rows}"
    )
    print(
        f"[GenRecap] pipeline source={source_events} pre_section={pre_section_events} "
        f"post_section={post_section_events} final={final_events}"
    )
    if measure_map_line:
        print(f"[GenRecap] {measure_map_line}")
    if salience_line:
        print(f"[GenRecap] {salience_line}")
    if critic_line:
        print(f"[GenRecap] {critic_line}")
    print("[GenRecap] ---------- end summary ----------")
