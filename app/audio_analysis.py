# app/audio_analysis.py
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import shutil
import bisect
import logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("TQDM_DISABLE", "1")
for _name in ["separator", "mdx_separator", "common_separator", "demucs_separator"]:
    try:
        logging.getLogger(_name).setLevel(logging.ERROR)
    except Exception:
        pass
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

MADMOM_AVAILABLE = False
RNNBeatProcessor = None
BeatTrackingProcessor = None

try:
    from audio_separator.separator import Separator
    AUDIO_SEPARATOR_AVAILABLE = True
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False

from .audio_separator import detect_kick_snare_with_essentia

try:
    from .genre_detector import detect_genres
    GENRE_DETECTION_AVAILABLE = True
except ImportError:
    GENRE_DETECTION_AVAILABLE = False

from .drum_utils import load_genre_configs, load_genre_aliases, get_genre_params


GENRE_CONFIGS = load_genre_configs()
GENRE_ALIAS_MAP = load_genre_aliases()
TEMP_UPLOADS_DIR = Path("temp_uploads")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"


def import_madmom() -> bool:
    global MADMOM_AVAILABLE, RNNBeatProcessor, BeatTrackingProcessor
    if MADMOM_AVAILABLE:
        return True
    try:
        import madmom
        from madmom.features.beats import RNNBeatProcessor as _RNNBeat
        from madmom.features.beats import BeatTrackingProcessor as _BeatTrack
        RNNBeatProcessor = _RNNBeat
        BeatTrackingProcessor = _BeatTrack
        MADMOM_AVAILABLE = True
        return True
    except Exception:
        MADMOM_AVAILABLE = False
        return False


def separate_stems(song_path: str, song_folder: Path, stem_type: str = "drums", cancel_cb: Optional[Callable[[], None]] = None) -> str:
    song_path = Path(song_path)
    splitter_folder = song_folder / "splitter"
    splitter_folder.mkdir(parents=True, exist_ok=True)

    expected_name = f"{song_path.stem}_{stem_type}.wav"
    output_path = splitter_folder / expected_name

    if cancel_cb:
        cancel_cb()
    if output_path.exists():
        print(f"[AudioAnalysis] Использую кешированный stem: {output_path.name}")
        return str(output_path)

    if cancel_cb:
        cancel_cb()
    candidates = [f for f in splitter_folder.glob("*.wav") if stem_type in f.name.lower()]
    preferred = None
    for f in candidates:
        if cancel_cb:
            cancel_cb()
        name = f.name.lower()
        if "no drums" in name or "(no drums)" in name or "no_drums" in name:
            continue
        preferred = f
        break
    if not preferred and candidates:
        preferred = candidates[0]
    if preferred:
        print(f"[AudioAnalysis] Найден локальный stem: {preferred.name} → {output_path.name}")
        shutil.copy2(preferred, output_path)
        return str(output_path)

    if not AUDIO_SEPARATOR_AVAILABLE:
        print("[AudioAnalysis] Separator недоступен — используем оригинальный файл")
        return str(song_path)

    def _try_separate(select_model: Optional[str] = None) -> Optional[str]:
        try:
            model_dir = str(LOCAL_MODELS_DIR) if LOCAL_MODELS_DIR.exists() else "/tmp/audio-separator-models/"
            if cancel_cb:
                cancel_cb()
            print(f"[AudioAnalysis] Запуск разделения: {song_path.name}")
            separator = Separator(
                output_dir=str(splitter_folder.resolve()),
                output_format="WAV",
                model_file_dir=model_dir
            )
            available_models = []
            try:
                available_models = separator.get_simplified_model_list()
            except Exception:
                available_models = []
            target_model = None
            for m in available_models:
                ml = m.lower()
                if stem_type in ml and ("kuielab" in ml or stem_type in ml):
                    target_model = m
                    break
            if not target_model:
                for m in available_models:
                    if "htdemucs" in m.lower():
                        target_model = m
                        break
            if not target_model and available_models:
                target_model = available_models[0]
                print(f"[AudioAnalysis] Fallback модель: {target_model}")
            if not target_model:
                print("[AudioAnalysis] Нет доступных моделей в separator — пропускаем разделение")
                return None
            if cancel_cb:
                cancel_cb()
            try:
                separator.load_model(target_model)
            except Exception as e:
                print(f"[AudioAnalysis] Ошибка загрузки модели: {e}")
                return None
            if cancel_cb:
                cancel_cb()
            import time as _time
            _t0 = _time.time()
            try:
                output_files = separator.separate(str(song_path))
                _dt = _time.time() - _t0
                print(f"[AudioAnalysis] Разделение завершено: {_dt:.1f}s, outputs: {len(output_files)}")
            except Exception as e:
                print(f"[AudioAnalysis] Ошибка separator.separate: {e}")
                output_files = []
            if cancel_cb:
                cancel_cb()
            norm_files_set = []
            for f in output_files:
                try:
                    p = Path(f)
                    chosen: Optional[Path] = None
                    if p.is_absolute() and p.exists():
                        chosen = p
                    else:
                        c1 = (splitter_folder / p).resolve()
                        if c1.exists():
                            chosen = c1
                        else:
                            c2 = (PROJECT_ROOT / p).resolve()
                            if c2.exists():
                                chosen = c2
                    if chosen and chosen.suffix.lower() == ".wav":
                        norm_files_set.append(str(chosen))
                except Exception:
                    continue
            if not norm_files_set:
                produced = [str(p.resolve()) for p in splitter_folder.rglob("*.wav")]
                norm_files_set = produced
            norm_files = norm_files_set
            def is_target(name: str) -> bool:
                n = name.lower()
                if stem_type == "drums":
                    return ("drum" in n or "percussion" in n) and not any(bad in n for bad in ["no drums", "(no drums)", "no_drums", "instrumental"])
                return (stem_type in n) and (f"no_{stem_type}" not in n) and ("(no " + stem_type + ")" not in n)
            candidates = [f for f in norm_files if is_target(f)]
            preferred_out = None
            for f in candidates:
                if cancel_cb:
                    cancel_cb()
                lf = f.lower()
                if "no drums" in lf or "(no drums)" in lf or "no_drums" in lf:
                    continue
                preferred_out = f
                break
            if not preferred_out and candidates:
                preferred_out = candidates[0]
            if not preferred_out and norm_files:
                preferred_out = norm_files[0]
                print(f"[AudioAnalysis] Подходящий drum-стем не найден по имени, берём первый файл: {Path(preferred_out).name}")
            if preferred_out and Path(preferred_out).exists():
                if cancel_cb:
                    cancel_cb()
                print(f"[AudioAnalysis] Выбран stem: {Path(preferred_out).name} → {output_path.name}")
                shutil.copy2(preferred_out, output_path)
                try:
                    if Path(output_path).exists():
                        print(f"[AudioAnalysis] Stem сохранён: {output_path.name}")
                except Exception:
                    pass
                for f in norm_files:
                    try:
                        if Path(f).exists() and Path(f) != output_path:
                            Path(f).unlink(missing_ok=True)
                    except Exception:
                        pass
                for aux in ["mdx_model_data.json", "vr_model_data.json", "download_checks.json"]:
                    try:
                        p = splitter_folder / aux
                        if p.exists():
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass
                return str(output_path)
            else:
                print("[AudioAnalysis] Не найден подходящий drum/percussion stem — используем оригинал")
            return None
        except Exception:
            return None
        finally:
            pass

    if cancel_cb:
        cancel_cb()
    path = _try_separate(select_model=None)
    if path:
        return path
    return str(song_path)


def extract_beats(audio_path: str, bpm: Optional[float] = None) -> np.ndarray:
    madmom_ready = import_madmom()
    beats = np.array([])

    if madmom_ready:
        try:
            proc = RNNBeatProcessor()
            act = proc(audio_path)
            tracker = BeatTrackingProcessor(fps=100)
            beats = np.array(tracker(act))
        except Exception:
            pass

    if len(beats) == 0 and LIBROSA_AVAILABLE:
        y, sr = librosa.load(audio_path, sr=None, mono=True, dtype='float32')
        try:
            _, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm, units='time')
        except:
            try:
                _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
            except:
                duration = len(y) / sr
                beats = np.arange(0, duration, 60.0 / (bpm or 120.0))

    return beats


def detect_drum_events(audio_path: str, bpm: float, genre_params: Optional[Dict] = None) -> Tuple[List[float], List[float]]:
    if not LIBROSA_AVAILABLE:
        return [], []

    y, sr = librosa.load(audio_path, sr=None, mono=True, dtype='float32')

    S_full = np.abs(librosa.stft(y, n_fft=2048, hop_length=512, win_length=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_mask = (freqs >= 50) & (freqs <= 200)
    S_kick = S_full[low_mask].sum(axis=0)

    mid_mask = (freqs > 200) & (freqs <= 5000)
    S_snare = S_full[mid_mask].sum(axis=0)

    times = librosa.frames_to_time(np.arange(len(S_kick)), sr=sr, hop_length=512, n_fft=2048)

    def detect_peaks_adaptive(energy, threshold_ratio=0.2, min_distance=0.05):
        from scipy.signal import find_peaks
        global_max = energy.max()
        threshold = global_max * threshold_ratio
        distance_frames = int(min_distance * sr / 512)
        peaks, _ = find_peaks(energy, prominence=threshold, distance=distance_frames)
        return times[peaks].tolist()

    kick_mult = (genre_params or {}).get('kick_sensitivity_multiplier', 1.0)
    snare_mult = (genre_params or {}).get('snare_sensitivity_multiplier', 1.0)

    kick_threshold = 0.20 / kick_mult
    snare_threshold = 0.25 / snare_mult

    kick_times = detect_peaks_adaptive(S_kick, threshold_ratio=kick_threshold)
    snare_times = detect_peaks_adaptive(S_snare, threshold_ratio=snare_threshold)

    sync_tolerance = genre_params.get('sync_tolerance', 0.1) if genre_params else 0.1
    subdivisions = genre_params.get('quantization_subdivisions', [4, 8, 16]) if genre_params else [4, 8, 16]

    kick_times = quantize_events_to_grid(kick_times, bpm, tolerance=sync_tolerance, subdivisions=subdivisions)
    snare_times = quantize_events_to_grid(snare_times, bpm, tolerance=sync_tolerance, subdivisions=subdivisions)

    return kick_times, snare_times


def extract_dominant_onsets(
    audio_path: str,
    bpm: Optional[float] = None,
    window_duration: Optional[float] = None,
    threshold_ratio: float = 0.15,
    genre_params: Optional[Dict] = None
) -> List[float]:
    if not LIBROSA_AVAILABLE:
        return []

    y, sr = librosa.load(audio_path, sr=None, mono=True, dtype='float32')
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0:
        return []

    onset_times = librosa.times_like(onset_env, sr=sr)
    global_max = float(onset_env.max())

    min_strength = global_max * threshold_ratio

    if window_duration is None:
        if bpm and bpm > 0:
            beat_interval = 60.0 / bpm
            window_duration = max(0.06, min(0.5, beat_interval * 0.5))
        else:
            window_duration = 0.2

    window_duration = max(0.05, float(window_duration))

    frame_times = onset_times
    dominant_onsets: List[float] = []
    window_start = float(frame_times[0]) if len(frame_times) else 0.0
    end_time = float(frame_times[-1]) if len(frame_times) else 0.0

    while window_start <= end_time:
        window_end = window_start + window_duration
        frame_indices = np.where((frame_times >= window_start) & (frame_times < window_end))[0]
        if frame_indices.size > 0:
            window_strengths = onset_env[frame_indices]
            peak_idx = frame_indices[int(np.argmax(window_strengths))]
            peak_strength = float(onset_env[peak_idx])
            if peak_strength >= min_strength:
                dominant_onsets.append(float(frame_times[peak_idx]))
        window_start = window_end

    min_event_distance = 0.05
    filtered_onsets = []
    for t in sorted(dominant_onsets):
        if not filtered_onsets or abs(t - filtered_onsets[-1]) > min_event_distance:
            filtered_onsets.append(t)

    if bpm:
        sync_tolerance = genre_params.get('sync_tolerance', 0.1) if genre_params else 0.1
        subdivisions = genre_params.get('quantization_subdivisions', [4, 8, 16]) if genre_params else [4, 8, 16]
        filtered_onsets = quantize_events_to_grid(filtered_onsets, bpm, tolerance=sync_tolerance, subdivisions=subdivisions)

    return filtered_onsets


def analyze_audio(
    song_path: str,
    bpm: Optional[float] = None,
    use_stems: bool = True,
    auto_identify_track: bool = False,
    use_filename_for_genres: bool = True,
    track_info: Optional[Dict] = None,
    stem_type: str = "drums",
    cancel_cb: Optional[Callable[[], None]] = None
) -> Dict:
    if cancel_cb:
        cancel_cb()
    base_name = Path(song_path).stem
    song_folder = TEMP_UPLOADS_DIR / base_name
    song_folder.mkdir(parents=True, exist_ok=True)

    original_file_path = song_folder / Path(song_path).name
    if not original_file_path.exists():
        shutil.copy2(song_path, original_file_path)

    analysis_path = str(original_file_path)


    all_genres = []
    if track_info and track_info.get('genres'):
        all_genres.extend(track_info['genres'])

    if not all_genres and GENRE_DETECTION_AVAILABLE:
        if cancel_cb:
            cancel_cb()
        genres = detect_genres("Unknown", "Unknown", audio_path=analysis_path)
        if genres:
            all_genres.extend(genres)

    unique_genres = list(set(g for g in all_genres if g and g.lower() != 'unknown'))
    genre_params = get_genre_params(unique_genres, GENRE_CONFIGS, GENRE_ALIAS_MAP)

    if bpm is None or bpm <= 0:
        if cancel_cb:
            cancel_cb()
        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(analysis_path, sr=None, mono=True, dtype='float32')
            try:
                _, bpm = librosa.beat.beat_track(y=y, sr=sr, units='time')
            except:
                bpm = 120.0
        else:
            bpm = 120.0

    if use_stems and AUDIO_SEPARATOR_AVAILABLE:
        if cancel_cb:
            cancel_cb()
        stem_path = separate_stems(str(original_file_path), song_folder, stem_type=stem_type, cancel_cb=cancel_cb)
        if stem_path != str(original_file_path):
            analysis_path = stem_path
            print(f"[AudioAnalysis] Для анализа выбран stem: {Path(analysis_path).name}")
        else:
            print("[AudioAnalysis] Stem не выбран — используем оригинальный аудиофайл")

    if cancel_cb:
        cancel_cb()
    beats = extract_beats(analysis_path, bpm)
    if cancel_cb:
        cancel_cb()
    dominant_onsets = extract_dominant_onsets(analysis_path, bpm=bpm, genre_params=genre_params)

    kick_times, snare_times = [], []
    if stem_type == "drums":
        if cancel_cb:
            cancel_cb()
        kick_times, snare_times = detect_drum_events(analysis_path, bpm=bpm, genre_params=genre_params)
        if cancel_cb:
            cancel_cb()

    return {
        "bpm": float(bpm),
        "beats": beats.tolist(),
        "kick_times": kick_times,
        "snare_times": snare_times,
        "dominant_onsets": dominant_onsets,
        "analysis_path": analysis_path,
        "original_path": str(original_file_path),
        "track_info": track_info,
        "genres": unique_genres,
        "genre_params": genre_params,
        "duration": len(librosa.load(analysis_path, sr=None)[0]) / librosa.load(analysis_path, sr=None)[1] if LIBROSA_AVAILABLE else 0.0
    }

def quantize_events_to_grid(events: List[float], bpm: float, tolerance: float = 0.1, subdivisions: List[int] = [4, 8, 16]) -> List[float]:
    from bisect import bisect_left

    beat_interval = 60.0 / bpm
    grids = []
    for div in subdivisions:
        step = beat_interval / div
        grid = np.arange(0.0, max(events) + beat_interval, step)
        grids.append(grid)

    quantized = []
    for t in events:
        best_snap = t
        min_diff = tolerance + 1

        for grid in grids:
            idx = bisect_left(grid, t)
            candidates = []
            if idx < len(grid):
                candidates.append(grid[idx])
            if idx > 0:
                candidates.append(grid[idx - 1])

            for candidate in candidates:
                diff = abs(candidate - t)
                if diff <= tolerance and diff < min_diff:
                    min_diff = diff
                    best_snap = candidate

        quantized.append(best_snap)

    return sorted(set(quantized))

def extract_drum_hits(
    song_path: str,
    bpm: Optional[float] = None,
    use_stems: bool = True,
    use_madmom_beats: bool = True,
    cancel_cb: Optional[Callable[[], None]] = None
) -> Dict[str, List[float]]:
    base_name = Path(song_path).stem
    temp_dir = TEMP_UPLOADS_DIR / base_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    local_path = temp_dir / Path(song_path).name
    if not local_path.exists():
        shutil.copy2(song_path, local_path)

    analysis_path = str(local_path)

    if use_stems and AUDIO_SEPARATOR_AVAILABLE:
        if cancel_cb:
            cancel_cb()
        stem_path = separate_stems(str(local_path), temp_dir, stem_type="drums", cancel_cb=cancel_cb)
        if stem_path != str(local_path):
            analysis_path = stem_path

    if cancel_cb:
        cancel_cb()
    beats = extract_beats(analysis_path, bpm)
    if cancel_cb:
        cancel_cb()
    dominant_onsets = extract_dominant_onsets(analysis_path, bpm=bpm)

    kick_times, snare_times = [], []
    if LIBROSA_AVAILABLE:
        if cancel_cb:
            cancel_cb()
        y, sr = librosa.load(analysis_path, sr=None, mono=True, dtype='float32')
        kick_times, snare_times = detect_kick_snare_with_essentia(y, sr, analysis_path)

    return {
        "beats": beats.tolist() if isinstance(beats, np.ndarray) else beats,
        "kick_times": kick_times,
        "snare_times": snare_times,
        "dominant_onsets": dominant_onsets
    }
