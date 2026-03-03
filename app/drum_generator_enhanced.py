# app/drum_generator_enhanced.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from .audio_analysis import analyze_audio
from .drum_utils import (
    apply_temporal_filter,
    assign_lanes_to_notes,
    detect_drum_section_start,
    save_drums_notes,
    load_drums_notes
)
from .note_types import NoteType
from . import drum_generator_basic

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _coefficient_of_variation(values: List[float]) -> float:
    if not values:
        return 0.0
    mean_val = float(np.mean(values))
    if mean_val == 0:
        return 0.0
    return float(np.std(values) / mean_val)


def _detect_peaks(signal: np.ndarray, threshold: float, min_distance_frames: int) -> List[int]:
    if signal.size == 0:
        return []
    peaks = []
    last_peak = -min_distance_frames
    for idx in range(1, len(signal) - 1):
        if signal[idx] < threshold:
            continue
        if signal[idx] >= signal[idx - 1] and signal[idx] >= signal[idx + 1]:
            if idx - last_peak >= min_distance_frames:
                peaks.append(idx)
                last_peak = idx
    return peaks


def _merge_close_events(events: List[Dict], window: float) -> List[Dict]:
    if not events:
        return []
    events = sorted(events, key=lambda e: e["time"])
    merged = []
    current = [events[0]]
    for event in events[1:]:
        if event["time"] - current[-1]["time"] <= window:
            current.append(event)
        else:
            strongest = max(current, key=lambda e: e["strength"])
            merged.append(strongest)
            current = [event]
    strongest = max(current, key=lambda e: e["strength"])
    merged.append(strongest)
    return merged


def _limit_top_percent(events: List[Dict], start: float, end: float, percent: float) -> List[Dict]:
    window_events = [e for e in events if start <= e["time"] < end]
    if not window_events:
        return []
    keep_count = max(1, int(len(window_events) * percent))
    sorted_events = sorted(window_events, key=lambda e: e["strength"], reverse=True)
    return sorted_events[:keep_count]


def _dominant_in_window(events: List[Dict], window: float) -> List[Dict]:
    if not events:
        return []
    events = sorted(events, key=lambda e: e["time"])
    kept = []
    for event in events:
        start = event["time"] - window / 2
        end = event["time"] + window / 2
        window_events = [e for e in events if start <= e["time"] <= end]
        strongest = max(window_events, key=lambda e: e["strength"])
        if strongest["time"] == event["time"]:
            kept.append(event)
    return kept


class RhythmExtractor:
    def __init__(self, bpm: float):
        self.bpm = bpm

    def extract(self, audio_path: str) -> Dict:
        if not LIBROSA_AVAILABLE:
            return {"events": [], "confidence": 0.0}

        y, sr = librosa.load(audio_path, sr=None, mono=True, dtype="float32")
        hop_length = 512
        n_fft = 2048

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=1024))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)

        kick_mask = freqs <= 100
        snare_mask = (freqs >= 150) & (freqs <= 250)
        print(f"[EnhancedDBG] Маски: kick<=100 Hz | snare=150-250 Hz")

        kick_energy = S[kick_mask].sum(axis=0) if np.any(kick_mask) else np.zeros(S.shape[1])
        snare_energy = S[snare_mask].sum(axis=0) if np.any(snare_mask) else np.zeros(S.shape[1])

        if kick_energy.max() > 0:
            kick_energy = kick_energy / kick_energy.max()
        if snare_energy.max() > 0:
            snare_energy = snare_energy / snare_energy.max()

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        if onset_env.size == 0:
            return {"events": [], "confidence": 0.0}
        onset_env = onset_env / max(1e-6, onset_env.max())

        fps = sr / hop_length
        window = max(1, int(fps / 8))
        onset_env = _moving_average(onset_env, window)
        onset_env = np.where(onset_env >= 0.3, onset_env, 0.0)

        min_distance_frames = max(1, int(0.05 * fps))
        kick_thr = 0.35
        snare_thr = 0.35
        kick_peaks = _detect_peaks(kick_energy, kick_thr, min_distance_frames)
        snare_peaks = _detect_peaks(snare_energy, snare_thr, min_distance_frames)
        onset_peaks = _detect_peaks(onset_env, 0.7, min_distance_frames)
        print(f"[EnhancedDBG] Пики: kick={len(kick_peaks)} (thr={kick_thr:.2f}) | snare={len(snare_peaks)} (thr={snare_thr:.2f}) | onset={len(onset_peaks)} | min_dist_frames={min_distance_frames}")

        events = []
        for idx in kick_peaks:
            events.append({"time": float(times[idx]), "strength": float(kick_energy[idx])})
        for idx in snare_peaks:
            events.append({"time": float(times[idx]), "strength": float(snare_energy[idx])})
        for idx in onset_peaks:
            events.append({"time": float(times[idx]), "strength": float(onset_env[idx])})
        print(f"[EnhancedDBG] События до merge: {len(events)}")

        events = _merge_close_events(events, 0.05)
        print(f"[EnhancedDBG] События после merge: {len(events)}")

        strengths = [e["strength"] for e in events]
        strong_ratio = float(np.mean([1.0 if s >= 0.8 else 0.0 for s in strengths])) if strengths else 0.0
        top_count = max(1, int(len(strengths) * 0.2)) if strengths else 0
        top_mean = float(np.mean(sorted(strengths, reverse=True)[:top_count])) if strengths else 0.0
        confidence = min(1.0, 0.5 * strong_ratio + 0.5 * top_mean)
        print(f"[EnhancedDBG] Confidence={confidence:.2f} | strong_ratio={strong_ratio:.2f} | top_mean={top_mean:.2f} | events={len(events)}")

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = float(np.mean(spectral_centroid)) if spectral_centroid.size else 0.0

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1) if mfcc.size else np.zeros(13)
        mfcc_var = mfcc.var(axis=1) if mfcc.size else np.zeros(13)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_index = int(np.argmax(chroma.mean(axis=1))) if chroma.size else 0
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_name = key_names[key_index % 12] if key_names else "C"

        rhythm_spectrum = np.abs(np.fft.rfft(onset_env))
        if rhythm_spectrum.size > 1:
            rhythm_spectrum = rhythm_spectrum / max(1e-6, rhythm_spectrum.max())
        autocorr = np.correlate(rhythm_spectrum, rhythm_spectrum, mode="full")
        if autocorr.size > 0:
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / max(1e-6, autocorr.max())
        autocorr_peak = float(np.max(autocorr[1:])) if autocorr.size > 1 else 0.0

        return {
            "events": events,
            "confidence": confidence,
            "spectral_centroid": spectral_centroid_mean,
            "mfcc_mean": mfcc_mean,
            "mfcc_var": mfcc_var,
            "key": key_name,
            "autocorr_peak": autocorr_peak
        }


class GenrePatternMapper:
    def __init__(self, bpm: float, beat_start: float):
        self.bpm = bpm
        self.beat_start = beat_start
        self.beat_interval = 60.0 / bpm if bpm > 0 else 0.5

    def classify(
        self,
        track_info: Optional[Dict],
        mfcc_mean: np.ndarray,
        mfcc_var: np.ndarray,
        spectral_centroid: float,
        key: str,
        genres: Optional[List[str]]
    ) -> str:
        bpm = self.bpm
        centroid = spectral_centroid
        mfcc_energy = float(np.mean(np.abs(mfcc_mean))) if mfcc_mean.size else 0.0
        mfcc_spread = float(np.mean(mfcc_var)) if mfcc_var.size else 0.0
        genre_set = {g.strip().lower() for g in (genres or []) if isinstance(g, str)}
        primary = None
        if track_info and isinstance(track_info, dict):
            primary = track_info.get("primary_genre")
        if isinstance(primary, str):
            genre_set.add(primary.strip().lower())

        if "electronic" in genre_set:
            return "electronic"
        if "house" in genre_set:
            return "house"
        if "techno" in genre_set:
            return "techno"
        if "trance" in genre_set:
            return "trance"
        if "drum and bass" in genre_set or "dnb" in genre_set:
            return "drum and bass"
        if "funk" in genre_set:
            return "funk"
        if "jazz" in genre_set:
            return "jazz"
        if "latin" in genre_set:
            return "latin"

        if bpm >= 140 and centroid >= 2800 and mfcc_spread >= 20:
            return "drum and bass"
        if 118 <= bpm <= 135 and centroid >= 2600 and mfcc_energy >= 25:
            return "house"
        if 120 <= bpm <= 138 and centroid >= 3000:
            return "techno"
        if 125 <= bpm <= 140 and centroid >= 2400:
            return "trance"
        if 90 <= bpm <= 125 and mfcc_spread >= 15:
            return "funk"
        if mfcc_energy <= 20 and centroid <= 2200 and key in {"F", "Bb", "Eb", "Ab", "Db", "Gb", "C", "G"}:
            return "latin"
        return "default"

    def apply(self, events: List[Dict], autocorr_peak: float, density: float, interval_cv: float) -> List[Dict]:
        if not events:
            return events

        genre = self.classify(None, np.array([]), np.array([]), 0.0, "C", None)
        return self._apply_genre_patterns(events, genre, autocorr_peak, density, interval_cv)

    def apply_with_metadata(
        self,
        events: List[Dict],
        track_info: Optional[Dict],
        mfcc_mean: np.ndarray,
        mfcc_var: np.ndarray,
        spectral_centroid: float,
        key: str,
        genres: Optional[List[str]],
        autocorr_peak: float,
        density: float,
        interval_cv: float
    ) -> List[Dict]:
        genre = self.classify(track_info, mfcc_mean, mfcc_var, spectral_centroid, key, genres)
        return self._apply_genre_patterns(events, genre, autocorr_peak, density, interval_cv)

    def _apply_genre_patterns(
        self,
        events: List[Dict],
        genre: str,
        autocorr_peak: float,
        density: float,
        interval_cv: float
    ) -> List[Dict]:
        beat_interval = self.beat_interval
        measure_duration = beat_interval * 4
        start_time = min(e["time"] for e in events)
        end_time = max(e["time"] for e in events)
        filtered = []

        current_time = start_time
        while current_time <= end_time:
            measure_start = current_time
            measure_end = current_time + measure_duration
            measure_events = [e for e in events if measure_start <= e["time"] < measure_end]
            if not measure_events:
                current_time += measure_duration
                continue

            pattern_positions = None
            if genre in {"house", "techno", "trance", "electronic"}:
                pattern_positions = [0.0, 2.0]
            elif genre in {"drum and bass", "funk"}:
                if density >= 0.8 and interval_cv <= 0.15:
                    pattern_positions = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
                else:
                    pattern_positions = [0.0, 2.0]
            elif genre in {"latin", "jazz"}:
                if autocorr_peak >= 0.7:
                    pattern_positions = [0.5, 1.5, 2.5, 3.5]

            if not pattern_positions:
                filtered.extend(measure_events)
                current_time += measure_duration
                continue

            matched = 0
            for pos in pattern_positions:
                target_time = measure_start + pos * beat_interval
                if any(abs(e["time"] - target_time) <= 0.03 for e in measure_events):
                    matched += 1

            if matched / max(1, len(pattern_positions)) >= 0.75:
                for e in measure_events:
                    measure_pos = (e["time"] - measure_start) / beat_interval
                    if any(abs(measure_pos - pos) <= 0.15 for pos in pattern_positions):
                        filtered.append(e)
            else:
                filtered.extend(measure_events)

            current_time += measure_duration

        return filtered


class Quantizer:
    def __init__(self, bpm: float, beat_start: float, allow_eighths: bool = True):
        self.bpm = bpm
        self.beat_start = beat_start
        self.beat_interval = 60.0 / bpm if bpm > 0 else 0.5
        self.allow_eighths = allow_eighths

    def quantize(self, events: List[Dict], tolerance: float = 0.01) -> List[Dict]:
        if not events:
            return []
        beat_interval = self.beat_interval
        start_time = min(e["time"] for e in events)
        end_time = max(e["time"] for e in events)

        grid_start = min(self.beat_start, start_time)
        quarter_grid = np.arange(grid_start, end_time + beat_interval, beat_interval)
        grids = [quarter_grid]
        if self.allow_eighths:
            eighth_grid = np.arange(grid_start, end_time + beat_interval / 2, beat_interval / 2)
            grids.append(eighth_grid)

        quantized = []
        for event in events:
            t = event["time"]
            candidates = []
            for grid in grids:
                idx = int(np.argmin(np.abs(grid - t)))
                candidates.append(float(grid[idx]))

            strong_candidates = []
            for candidate in candidates:
                pos = (candidate - self.beat_start) / beat_interval
                measure_pos = pos % 4
                if abs(measure_pos - 0.0) <= 0.05 or abs(measure_pos - 2.0) <= 0.05:
                    strong_candidates.append(candidate)

            def best_match(cands: List[float]) -> float:
                diffs = [abs(c - t) for c in cands]
                idx = int(np.argmin(diffs))
                return cands[idx]

            snapped = None
            if strong_candidates:
                candidate = best_match(strong_candidates)
                if abs(candidate - t) <= tolerance:
                    snapped = candidate

            if snapped is None:
                candidate = best_match(candidates)
                snapped = candidate if abs(candidate - t) <= tolerance else t

            quantized.append({"time": float(snapped), "strength": event["strength"]})

        dedup = {}
        for e in quantized:
            key = round(e["time"], 3)
            if key not in dedup or dedup[key]["strength"] < e["strength"]:
                dedup[key] = e
        return list(dedup.values())


class DensityLimiter:
    def __init__(self, bpm: float, beat_start: float):
        self.bpm = bpm
        self.beat_start = beat_start
        self.beat_interval = 60.0 / bpm if bpm > 0 else 0.5

    def _measure_bounds(self, start_time: float, end_time: float) -> List[Tuple[float, float]]:
        measure_duration = self.beat_interval * 4
        bounds = []
        current = start_time
        while current <= end_time:
            bounds.append((current, current + measure_duration))
            current += measure_duration
        return bounds

    def _detect_drop(self, densities: List[float]) -> Optional[int]:
        if not densities:
            return None
        avg_density = float(np.mean(densities))
        if avg_density == 0:
            return None
        for idx, density in enumerate(densities):
            if density >= avg_density * 1.5:
                return idx
        return None

    def limit(self, events: List[Dict]) -> List[Dict]:
        if not events:
            return []
        events = sorted(events, key=lambda e: e["time"])
        start_time = events[0]["time"]
        end_time = events[-1]["time"]
        bounds = self._measure_bounds(start_time, end_time)

        densities = []
        for (m_start, m_end) in bounds:
            count = len([e for e in events if m_start <= e["time"] < m_end])
            densities.append(count)

        drop_index = self._detect_drop(densities)

        limited = []
        for i, (m_start, m_end) in enumerate(bounds):
            measure_events = [e for e in events if m_start <= e["time"] < m_end]
            if not measure_events:
                continue

            mean_strength = float(np.mean([e["strength"] for e in measure_events]))
            intense = mean_strength >= 0.85 or len(measure_events) > 4
            max_notes = 6 if intense else 4

            if drop_index is not None:
                if drop_index - 4 <= i < drop_index:
                    max_notes = int(np.ceil(max_notes * 1.25))
                if drop_index <= i < drop_index + 2:
                    max_notes = 6 if intense else 4

            beat_interval = self.beat_interval
            strong_times = [m_start, m_start + 2 * beat_interval]
            strong_events = []
            for e in measure_events:
                if any(abs(e["time"] - st) <= 0.03 for st in strong_times):
                    strong_events.append(e)
            remaining = [e for e in measure_events if e not in strong_events]

            if len(strong_events) >= max_notes:
                measure_events = strong_events[:max_notes]
            else:
                slots = max_notes - len(strong_events)
                remaining = sorted(remaining, key=lambda e: e["strength"], reverse=True)[:slots]
                measure_events = strong_events + remaining

            limited.extend(measure_events)

        limited = sorted(limited, key=lambda e: e["time"])
        smoothed = []
        measure_duration = self.beat_interval * 4
        for event in limited:
            if not smoothed:
                smoothed.append(event)
                continue
            interval = event["time"] - smoothed[-1]["time"]
            window_start = event["time"] - (measure_duration * 3)
            window_events = [e for e in smoothed if e["time"] >= window_start]
            window_intervals = [
                window_events[i]["time"] - window_events[i - 1]["time"]
                for i in range(1, len(window_events))
            ]
            avg_interval = float(np.mean(window_intervals)) if window_intervals else interval
            if avg_interval > 0 and abs(interval - avg_interval) / avg_interval > 0.2:
                continue
            smoothed.append(event)

        times = [e["time"] for e in smoothed]
        times = apply_temporal_filter(times, 0.12)
        strengths = {round(e["time"], 3): e["strength"] for e in smoothed}
        return [{"time": t, "strength": strengths.get(round(t, 3), 0.8)} for t in times]


def _compute_density(events: List[Dict], bpm: float) -> float:
    if not events or bpm <= 0:
        return 0.0
    beat_interval = 60.0 / bpm
    duration = max(beat_interval, events[-1]["time"] - events[0]["time"])
    total_beats = duration / beat_interval
    return float(len(events) / max(1.0, total_beats))


def _apply_top_percent_per_cycle(events: List[Dict], bpm: float) -> List[Dict]:
    if not events or bpm <= 0:
        return events
    beat_interval = 60.0 / bpm
    cycle_duration = beat_interval * 16
    start = min(e["time"] for e in events)
    end = max(e["time"] for e in events)
    filtered = []
    current = start
    while current <= end:
        window_events = _limit_top_percent(events, current, current + cycle_duration, 0.3)
        filtered.extend(window_events)
        current += cycle_duration
    return filtered


def _limit_to_basic_ratio(events: List[Dict], basic_count: int) -> List[Dict]:
    if basic_count <= 0 or not events:
        return events
    target = int(np.floor(basic_count * 0.7))
    if len(events) <= target:
        return events
    return sorted(events, key=lambda e: e["strength"], reverse=True)[:target]


def _normalize_events(events: List[Dict]) -> List[Dict]:
    unique = {}
    for e in events:
        key = round(e["time"], 3)
        if key not in unique or unique[key]["strength"] < e["strength"]:
            unique[key] = e
    return sorted(unique.values(), key=lambda e: e["time"])

def _measure_bounds(start_time: float, end_time: float, beat_interval: float) -> List[Tuple[float, float]]:
    measure_duration = beat_interval * 4
    bounds: List[Tuple[float, float]] = []
    current = start_time
    while current <= end_time:
        bounds.append((current, current + measure_duration))
        current += measure_duration
    return bounds

def _count_in_window(times: List[float], start: float, end: float) -> int:
    return sum(1 for t in times if start <= t < end)

def _has_near(t: float, existing: List[float], tol: float) -> bool:
    return any(abs(t - x) <= tol for x in existing)


def generate_drums_notes(
    song_path: str,
    bpm: float,
    lanes: int = 4,
    sync_tolerance: float = 0.2,
    use_madmom_beats: bool = True,
    use_stems: bool = True,
    track_info: Optional[Dict] = None,
    auto_identify_track: bool = False,
    use_filename_for_genres: bool = True,
    provided_genres: Optional[List[str]] = None,
    provided_primary_genre: Optional[str] = None,
    status_cb=None,
    cancel_cb=None
) -> Optional[List[Dict]]:
    print(f"🎮 Генерация барабанных нот (enhanced) для: {song_path} (BPM: {bpm})")

    if cancel_cb:
        cancel_cb()

    analysis = analyze_audio(
        song_path=song_path,
        bpm=bpm,
        use_stems=use_stems,
        auto_identify_track=auto_identify_track,
        use_filename_for_genres=use_filename_for_genres,
        track_info=track_info,
        stem_type="drums",
        cancel_cb=cancel_cb
    )
    if cancel_cb:
        cancel_cb()

    if not analysis or "bpm" not in analysis:
        print("[DrumGen-Enhanced] Fallback: отсутствуют данные анализа")
        return drum_generator_basic.generate_drums_notes(
            song_path,
            bpm,
            lanes=lanes,
            sync_tolerance=sync_tolerance,
            use_madmom_beats=use_madmom_beats,
            use_stems=use_stems,
            track_info=track_info,
            auto_identify_track=auto_identify_track,
            use_filename_for_genres=use_filename_for_genres,
            provided_genres=provided_genres,
            provided_primary_genre=provided_primary_genre,
            status_cb=status_cb,
            cancel_cb=cancel_cb
        )

    bpm = analysis.get("bpm", bpm)
    beats = np.array(analysis.get("beats", []))
    genre_params = analysis.get("genre_params", {})
    unique_genres = analysis.get("genres", [])
    track_info = analysis.get("track_info") or track_info or {}
    analysis_path = analysis.get("analysis_path", song_path)

    if provided_genres is not None:
        pg = [g.strip() for g in provided_genres if isinstance(g, str) and g.strip()]
        if pg:
            unique_genres = list({*unique_genres, *pg})
    if isinstance(provided_primary_genre, str) and provided_primary_genre.strip():
        track_info["primary_genre"] = provided_primary_genre.strip()

    basic_notes = drum_generator_basic.generate_drums_notes(
        song_path,
        bpm,
        lanes=lanes,
        sync_tolerance=sync_tolerance,
        use_madmom_beats=use_madmom_beats,
        use_stems=use_stems,
        track_info=track_info,
        auto_identify_track=auto_identify_track,
        use_filename_for_genres=use_filename_for_genres,
        provided_genres=provided_genres,
        provided_primary_genre=provided_primary_genre,
        status_cb=status_cb,
        cancel_cb=cancel_cb
    )
    base_times = [n["time"] for n in (basic_notes or []) if isinstance(n, dict) and "time" in n]
    if beats.size == 0:
        beats = np.arange(0.0, (max(base_times) if base_times else 180.0) + (60.0 / max(1.0, bpm)), 60.0 / max(1.0, bpm))
    beat_start = float(beats[0]) if beats.size else 0.0
    beat_interval = 60.0 / max(1.0, bpm)

    start_time = min(base_times) if base_times else beat_start
    end_time = max(base_times) if base_times else (beat_start + beat_interval * 64)
    bounds = _measure_bounds(start_time, end_time, beat_interval)

    mapper = GenrePatternMapper(bpm, beat_start)
    genre_label = None
    if isinstance(track_info, dict) and isinstance(track_info.get("primary_genre"), str) and track_info.get("primary_genre").strip():
        genre_label = track_info.get("primary_genre").strip().lower()
    elif isinstance(provided_primary_genre, str) and provided_primary_genre.strip():
        genre_label = provided_primary_genre.strip().lower()
    elif unique_genres:
        genre_label = str(unique_genres[0]).strip().lower()
    else:
        genre_label = "default"
    print(f"[DrumGen-Enhanced] Жанр для дополнений: {genre_label}")

    kick_times = analysis.get("kick_times", [])
    snare_times = analysis.get("snare_times", [])
    dominant_onsets = analysis.get("dominant_onsets", [])

    added: List[float] = []
    tol = 0.03
    genre_caps = {
        "pop": {"min": 2, "max": 4, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.15},
        "hyperpop": {"min": 2, "max": 4, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.15},
        "k-pop": {"min": 2, "max": 4, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.15},
        "j-pop": {"min": 2, "max": 4, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.15},
        "electronic": {"min": 4, "max": 6, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.10},
        "house": {"min": 4, "max": 6, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.10},
        "techno": {"min": 4, "max": 6, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.10},
        "trance": {"min": 4, "max": 6, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.10},
        "drum and bass": {"min": 4, "max": 8, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.12},
        "rap": {"min": 2, "max": 4, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.12},
        "r&b": {"min": 2, "max": 4, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.12},
        "rock": {"min": 2, "max": 5, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.10},
        "metal": {"min": 2, "max": 5, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.08},
        "hardcore": {"min": 2, "max": 5, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.08},
        "default": {"min": 3, "max": 5, "per_measure": 2, "per_measure_break": 3, "cap_ratio": 0.10},
    }
    caps = genre_caps.get(genre_label, genre_caps["default"])
    total_cap = int(len(base_times) * caps["cap_ratio"]) if base_times else 0
    added_total = 0
    for (m_start, m_end) in bounds:
        base_in_measure = [t for t in base_times if m_start <= t < m_end]
        energy = (
            _count_in_window(kick_times, m_start, m_end)
            + _count_in_window(snare_times, m_start, m_end)
            + _count_in_window(dominant_onsets, m_start, m_end)
        )
        intervals = [base_in_measure[i] - base_in_measure[i - 1] for i in range(1, len(base_in_measure))]
        cv = _coefficient_of_variation(intervals) if intervals else 0.0
        density = float(len(base_in_measure) / max(1.0, (m_end - m_start) / beat_interval))

        is_break = energy >= 6
        target_min = caps["min"]
        target_max = caps["max"]

        need_fill = (energy > 0) and (len(base_in_measure) < target_min or (is_break and len(base_in_measure) < target_max) or (density < 0.3 and cv > 1.0))
        if not need_fill:
            continue

        pattern_positions = None
        if genre_label in {"electronic", "house", "techno", "trance"}:
            pattern_positions = [0.0, 2.0]
            if is_break:
                pattern_positions += [0.5, 1.5, 2.5, 3.5]
        elif genre_label in {"drum and bass", "funk"}:
            pattern_positions = [0.0, 2.0]
            if is_break:
                pattern_positions += [0.5, 1.0, 1.5, 2.5, 3.0, 3.5]
        elif genre_label in {"pop", "hyperpop", "k-pop", "j-pop"}:
            pattern_positions = [0.0, 2.0]
            if is_break or len(base_in_measure) < target_min:
                pattern_positions += [0.5, 1.5, 2.5, 3.5]
        else:
            pattern_positions = [0.0, 2.0]

        proposed = []
        for pos in pattern_positions:
            tt = m_start + pos * beat_interval
            if not _has_near(tt, base_in_measure, tol) and not _has_near(tt, added, tol):
                proposed.append(tt)

        if proposed:
            if total_cap and added_total >= total_cap:
                continue
            per_measure_limit = caps["per_measure_break"] if is_break else caps["per_measure"]
            needed_to_min = max(0, target_min - len(base_in_measure))
            room_to_max = max(0, target_max - len(base_in_measure))
            limit = min(per_measure_limit, room_to_max)
            if needed_to_min > 0:
                limit = min(limit, needed_to_min)
            if total_cap:
                limit = min(limit, max(0, total_cap - added_total))
            if limit <= 0:
                continue
            keep = proposed[:limit]
            added.extend(keep)
            added_total += len(keep)

    all_times = sorted(set(base_times + added))
    all_events = [{"type": NoteType.DRUM, "time": t, "source": ("enhanced" if t in added else "basic")} for t in all_times]
    if status_cb:
        status_cb("Назначение линий...")
    if cancel_cb:
        cancel_cb()
    notes = assign_lanes_to_notes(all_events, lanes=lanes, song_offset=0.0)

    drum_count = len(notes)
    print(f"✅ Сгенерировано {drum_count} барабанных нот (enhanced-augment)")
    print(f"   - Дополнено: {len(added)} | Базовых: {len(base_times)}")
    print(f"   - Жанры: {unique_genres if unique_genres else 'не определены'}")
    print(f"   - BPM: {bpm}")
    return notes
