# app/audio_analysis.py
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import shutil
import bisect
import logging
import threading
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

try:
    from audio_separator.separator import Separator
    AUDIO_SEPARATOR_AVAILABLE = True
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False

from .drum_hit_detector import analyze_drum_hits
from .adtof_transcriber import drum_backend_name, is_adtof_available, transcribe_drum_stem
from .gpu_backend import resolve_torch_device_label, separator_hardware_options
from . import stem_memory_cache

try:
    from .genre_detector import detect_genres
    GENRE_DETECTION_AVAILABLE = True
except ImportError:
    GENRE_DETECTION_AVAILABLE = False

from .drum_utils import load_genre_configs, load_genre_aliases, get_genre_params


def _kuielab_mdx_segment_size(model_dir: Path, model_filename: str = "") -> int:
	"""kuielab_a_* MDX models use mdx_dim_t_set=9 (512). Default 256 skips ONNX+DirectML."""
	default = 512
	try:
		name = (model_filename or "kuielab_a_drums.onnx").strip()
		if not name.lower().endswith(".onnx"):
			name = f"{name}.onnx"
		onnx_path = model_dir / name
		if not onnx_path.is_file():
			# Prefer any local kuielab onnx for segment sizing.
			cands = sorted(model_dir.glob("kuielab_a_*.onnx"))
			onnx_path = cands[0] if cands else onnx_path
		data_path = model_dir / "mdx_model_data.json"
		if not onnx_path.is_file() or not data_path.is_file():
			return default
		import hashlib

		digest = hashlib.md5(onnx_path.read_bytes()).hexdigest()
		payload = json.loads(data_path.read_text(encoding="utf-8"))
		# UVR file is a flat {hash: meta} map (sometimes nested under mdx_model_data).
		table = payload.get("mdx_model_data", payload)
		if not isinstance(table, dict):
			return default
		entry = table.get(digest, {})
		if not isinstance(entry, dict):
			return default
		dim_t_set = int(entry.get("mdx_dim_t_set", 9))
		return int(2 ** dim_t_set)
	except Exception:
		return default


def _separator_mdx_params(model_dir: Path, model_filename: str = "") -> Dict:
	segment_size = _kuielab_mdx_segment_size(model_dir, model_filename)
	return {
		"hop_length": 1024,
		"segment_size": segment_size,
		"overlap": 0.25,
		"batch_size": 1,
		"enable_denoise": False,
	}


GENRE_CONFIGS = load_genre_configs()
GENRE_ALIAS_MAP = load_genre_aliases()
TEMP_UPLOADS_DIR = Path("temp_uploads")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"
_SEPARATION_LOCK = threading.RLock()
_WARM_SEPARATOR = None
_WARM_SEPARATOR_MODEL = ""
_WARM_SEPARATOR_DML: Optional[bool] = None
_WARM_OUTPUT_DIR = TEMP_UPLOADS_DIR / "_separator_warm"


def _stem_warm_enabled() -> bool:
    if os.getenv("RFALL_STEM_WARM", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return _stem_model_mode() not in ("htdemucs", "demucs")


def _beats_from_bpm_enabled() -> bool:
    return os.getenv("RFALL_BEATS_FROM_BPM", "1").strip().lower() not in ("0", "false", "no", "off")


def _build_beats_from_bpm(bpm: float, duration_sec: float) -> np.ndarray:
    if bpm <= 0 or duration_sec <= 0:
        return np.array([])
    interval = 60.0 / float(bpm)
    end = float(duration_sec) + interval * 0.25
    return np.arange(0.0, end, interval, dtype=np.float64)


def _stem_model_mode() -> str:
    """RFALL_STEM_MODEL: kuielab (default) | htdemucs | auto."""
    return os.getenv("RFALL_STEM_MODEL", "kuielab").strip().lower()


def _match_htdemucs_model(available_models: List[str]) -> Optional[str]:
    for m in available_models:
        if "htdemucs_ft" in m.lower():
            return m
    for m in available_models:
        if "htdemucs" in m.lower():
            return m
    return None


def _match_kuielab_model(available_models: List[str], stem_type: str) -> Optional[str]:
	"""Prefer kuielab_a_{drums|bass|…}.onnx; used for drums and bass stems."""
	stem = str(stem_type or "drums").strip().lower() or "drums"
	preferred = f"kuielab_a_{stem}"
	# Exact / prefix hits first (filename or display name from separator list).
	for m in available_models:
		ml = str(m).lower()
		if preferred in ml:
			return m
	for m in available_models:
		ml = str(m).lower()
		if "kuielab" in ml and stem in ml:
			return m
	# Local file present even if separator list is empty/odd.
	local = LOCAL_MODELS_DIR / f"{preferred}.onnx"
	if local.is_file():
		return f"{preferred}.onnx"
	return None


def _pick_stem_model_name(available_models: List[str], stem_type: str) -> Optional[str]:
    if not available_models:
        return None
    mode = _stem_model_mode()
    target_model: Optional[str] = None
    if mode in ("htdemucs", "demucs"):
        target_model = _match_htdemucs_model(available_models)
        if not target_model:
            target_model = _match_kuielab_model(available_models, stem_type)
    elif mode == "auto":
        target_model = _match_kuielab_model(available_models, stem_type)
        if not target_model:
            target_model = _match_htdemucs_model(available_models)
    else:
        # Default kuielab: kuielab_a_drums / kuielab_a_bass; htdemucs only if missing.
        target_model = _match_kuielab_model(available_models, stem_type)
        if not target_model:
            target_model = _match_htdemucs_model(available_models)
    if not target_model and available_models:
        target_model = available_models[0]
        print(f"[AudioAnalysis] Резервная модель: {target_model}")
    return target_model


def _is_stem_candidate(path: Path, stem_type: str) -> bool:
    if not path.exists() or path.suffix.lower() != ".wav":
        return False
    name = path.name.lower()
    if stem_type == "drums":
        if not ("drum" in name or "percussion" in name):
            return False
        if any(bad in name for bad in ["no drums", "(no drums)", "no_drums", "instrumental"]):
            return False
        return True
    return (stem_type in name) and (f"no_{stem_type}" not in name) and (f"(no {stem_type})" not in name)


def _resolve_disk_cached_stem(
    song_folder: Path,
    splitter_folder: Path,
    output_path: Path,
    stem_type: str,
) -> Optional[str]:
    if output_path.is_file():
        print(f"[AudioAnalysis] Использую кешированный стем: {output_path.name}")
        return str(output_path)
    recursive_candidates = [p for p in song_folder.rglob("*") if _is_stem_candidate(p, stem_type)]
    if recursive_candidates:
        recursive_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        preferred_cached = recursive_candidates[0]
        if preferred_cached.resolve() != output_path.resolve():
            print(f"[AudioAnalysis] Найден кешированный стем: {preferred_cached.name} → {output_path.name}")
            shutil.copy2(preferred_cached, output_path)
        else:
            print(f"[AudioAnalysis] Использую кешированный стем: {output_path.name}")
        return str(output_path)
    candidates = [p for p in splitter_folder.glob("*") if _is_stem_candidate(p, stem_type)]
    preferred: Optional[Path] = None
    for candidate in candidates:
        preferred = candidate
        break
    if preferred is None and candidates:
        preferred = candidates[0]
    if preferred is not None:
        print(f"[AudioAnalysis] Найден локальный стем: {preferred.name} → {output_path.name}")
        if preferred.resolve() != output_path.resolve():
            shutil.copy2(preferred, output_path)
        return str(output_path)
    return None


def _separator_single_stem_name(stem_type: str) -> str:
    mapping = {
        "drums": "Drums",
        "bass": "Bass",
        "vocals": "Vocals",
        "other": "Other",
    }
    kind = str(stem_type or "drums").strip().lower() or "drums"
    return mapping.get(kind, kind.capitalize())


def _create_separator(
	splitter_folder: Path,
	use_directml: bool,
	stem_type: str = "drums",
	model_filename: str = "",
) -> "Separator":
	model_dir = str(LOCAL_MODELS_DIR) if LOCAL_MODELS_DIR.exists() else "/tmp/audio-separator-models/"
	mdx_params = _separator_mdx_params(Path(model_dir), model_filename)
	return Separator(
		output_dir=str(splitter_folder.resolve()),
		output_format="WAV",
		model_file_dir=model_dir,
		use_directml=use_directml,
		output_single_stem=_separator_single_stem_name(stem_type),
		mdx_params=mdx_params,
	)


def _load_warm_separator(use_directml: bool) -> bool:
    global _WARM_SEPARATOR, _WARM_SEPARATOR_MODEL, _WARM_SEPARATOR_DML
    if not AUDIO_SEPARATOR_AVAILABLE or not _stem_warm_enabled():
        return False
    model_dir = str(LOCAL_MODELS_DIR) if LOCAL_MODELS_DIR.exists() else "/tmp/audio-separator-models/"
    _WARM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    separator = _create_separator(_WARM_OUTPUT_DIR, use_directml, "drums")
    available_models: List[str] = []
    try:
        available_models = separator.get_simplified_model_list()
    except Exception:
        available_models = []
    target_model = _pick_stem_model_name(available_models, "drums")
    if not target_model:
        print("[AudioAnalysis] Warm separator: нет доступных моделей")
        return False
    try:
        separator.load_model(target_model)
    except Exception as e:
        print(f"[AudioAnalysis] Warm separator: ошибка load_model: {e}")
        return False
    _WARM_SEPARATOR = separator
    _WARM_SEPARATOR_MODEL = target_model
    _WARM_SEPARATOR_DML = use_directml
    backend = "DirectML" if use_directml else "CPU"
    print(f"[AudioAnalysis] Warm separator ready ({backend}, model={target_model})")
    return True


def warm_stem_separator() -> bool:
    """Pre-load stem model at server startup (kuielab only when RFALL_STEM_MODEL allows)."""
    if not AUDIO_SEPARATOR_AVAILABLE or not _stem_warm_enabled():
        return False
    with _SEPARATION_LOCK:
        if _WARM_SEPARATOR is not None:
            return True
        gpu_opts = separator_hardware_options()
        use_dml = bool(gpu_opts.get("use_directml"))
        if _load_warm_separator(use_dml):
            return True
        if use_dml:
            print("[AudioAnalysis] Warm separator: DirectML failed, trying CPU...")
            return _load_warm_separator(False)
        return False


def _get_warm_separator(use_directml: bool):
    global _WARM_SEPARATOR, _WARM_SEPARATOR_MODEL, _WARM_SEPARATOR_DML
    if not _stem_warm_enabled():
        return None
    with _SEPARATION_LOCK:
        if _WARM_SEPARATOR is not None and _WARM_SEPARATOR_DML == use_directml:
            return _WARM_SEPARATOR
        if _WARM_SEPARATOR is None:
            _load_warm_separator(use_directml)
        if _WARM_SEPARATOR is not None and _WARM_SEPARATOR_DML == use_directml:
            return _WARM_SEPARATOR
    return None


def separate_stems(song_path: str, song_folder: Path, stem_type: str = "drums", cancel_cb: Optional[Callable[[], None]] = None) -> str:
    from app import song_storage

    song_path = Path(song_path)
    splitter_folder = song_folder / "splitter"
    splitter_folder.mkdir(parents=True, exist_ok=True)

    chart_id = song_folder.name if song_storage.is_hash_dir_name(song_folder.name) else ""
    if chart_id:
        expected_name = song_storage.stem_wav_name(chart_id, stem_type)
    else:
        expected_name = f"{song_path.stem}_{stem_type}.wav"
    output_path = splitter_folder / expected_name

    if cancel_cb:
        cancel_cb()

    disk_cached = _resolve_disk_cached_stem(song_folder, splitter_folder, output_path, stem_type)
    if disk_cached:
        stored = stem_memory_cache.store_cached_stem(str(song_path), disk_cached, stem_type)
        return stored or disk_cached

    cached_stem = stem_memory_cache.get_cached_stem(str(song_path), stem_type)
    if cached_stem:
        try:
            if not output_path.is_file():
                shutil.copy2(cached_stem, output_path)
                print(f"[AudioAnalysis] Стем из памяти записан на диск: {output_path.name}")
        except Exception as exc:
            print(f"[AudioAnalysis] Не удалось записать кэш-стем на диск: {exc}")
        print(f"[AudioAnalysis] Стем из памяти ({stem_type}, без повторной сепарации): {Path(cached_stem).name}")
        return str(output_path) if output_path.is_file() else cached_stem

    if not AUDIO_SEPARATOR_AVAILABLE:
        print("[AudioAnalysis] Сепаратор недоступен — используем исходный файл")
        return str(song_path)

    def _try_separate(use_directml: bool) -> Optional[str]:
        try:
            if cancel_cb:
                cancel_cb()
            backend = "DirectML" if use_directml else "CPU"
            warm = _get_warm_separator(use_directml) if stem_type == "drums" else None
            if warm is not None:
                separator = warm
                print(f"[AudioAnalysis] Запуск разделения: {song_path.name} ({backend}, warm model, {stem_type})")
            else:
                model_dir_path = LOCAL_MODELS_DIR if LOCAL_MODELS_DIR.exists() else Path("/tmp/audio-separator-models/")
                # Pick model first so MDX segment_size matches kuielab_a_bass / drums.
                probe = _create_separator(splitter_folder, use_directml, stem_type)
                available_models: List[str] = []
                try:
                    available_models = probe.get_simplified_model_list()
                except Exception:
                    available_models = []
                target_model = _pick_stem_model_name(available_models, stem_type)
                if not target_model:
                    print("[AudioAnalysis] В сепараторе нет доступных моделей — пропускаем разделение")
                    return None
                mdx_params = _separator_mdx_params(model_dir_path, str(target_model))
                print(
                    f"[AudioAnalysis] Запуск разделения: {song_path.name} ({backend}, {stem_type}, "
                    f"model={target_model}, mdx_segment_size={mdx_params['segment_size']})"
                )
                separator = _create_separator(
                    splitter_folder, use_directml, stem_type, model_filename=str(target_model)
                )
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
            search_dirs = [splitter_folder.resolve()]
            if warm is not None:
                search_dirs.append(_WARM_OUTPUT_DIR.resolve())
            norm_files_set = []
            for f in output_files:
                try:
                    p = Path(f)
                    chosen: Optional[Path] = None
                    if p.is_absolute() and p.exists():
                        chosen = p
                    else:
                        for base in search_dirs:
                            c1 = (base / p).resolve()
                            if c1.exists():
                                chosen = c1
                                break
                        if chosen is None:
                            c2 = (PROJECT_ROOT / p).resolve()
                            if c2.exists():
                                chosen = c2
                    if chosen and chosen.suffix.lower() == ".wav":
                        norm_files_set.append(str(chosen))
                except Exception:
                    continue
            if not norm_files_set:
                produced: List[str] = []
                for base in search_dirs:
                    produced.extend(str(p.resolve()) for p in base.rglob("*.wav"))
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
            if not preferred_out and norm_files and stem_type == "drums":
                preferred_out = norm_files[0]
                print(f"[AudioAnalysis] Подходящий drum-стем не найден по имени, берём первый файл: {Path(preferred_out).name}")
            if preferred_out and Path(preferred_out).exists():
                if cancel_cb:
                    cancel_cb()
                print(f"[AudioAnalysis] Выбран stem: {Path(preferred_out).name} → {output_path.name}")
                shutil.copy2(preferred_out, output_path)
                try:
                    if Path(output_path).exists():
                        print(f"[AudioAnalysis] Стем сохранён: {output_path.name}")
                except Exception:
                    pass
                for f in norm_files:
                    try:
                        fp = Path(f)
                        if fp.exists() and fp.resolve() != output_path.resolve():
                            fp.unlink(missing_ok=True)
                    except Exception:
                        pass
                if warm is not None and _WARM_OUTPUT_DIR.is_dir():
                    for wav in _WARM_OUTPUT_DIR.glob("*.wav"):
                        try:
                            wav.unlink(missing_ok=True)
                        except Exception:
                            pass
                for aux in ["mdx_model_data.json", "vr_model_data.json", "download_checks.json"]:
                    try:
                        p = splitter_folder / aux
                        if p.exists():
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass
                stored = stem_memory_cache.store_cached_stem(str(song_path), str(output_path), stem_type)
                if stored:
                    return stored
                return str(output_path)
            else:
                print(f"[AudioAnalysis] Не найден подходящий {stem_type}-стем — используем оригинал")
            return None
        except Exception:
            return None
        finally:
            pass

    if cancel_cb:
        cancel_cb()
    gpu_opts = separator_hardware_options()
    use_dml = bool(gpu_opts.get("use_directml"))
    with _SEPARATION_LOCK:
        disk_cached = _resolve_disk_cached_stem(song_folder, splitter_folder, output_path, stem_type)
        if disk_cached:
            stored = stem_memory_cache.store_cached_stem(str(song_path), disk_cached, stem_type)
            return stored or disk_cached
        path = _try_separate(use_directml=use_dml)
        if not path and use_dml:
            print("[AudioAnalysis] DirectML не удался — повтор на CPU...")
            path = _try_separate(use_directml=False)
    if path:
        return path
    return str(song_path)


def extract_beats(
    audio_path: str,
    bpm: Optional[float] = None,
    *,
    duration_sec: Optional[float] = None,
    prefer_bpm_grid: bool = False,
    y: Optional[np.ndarray] = None,
    sr: int = 0,
) -> np.ndarray:
    if prefer_bpm_grid and _beats_from_bpm_enabled() and bpm is not None and float(bpm) > 0:
        dur = duration_sec
        if dur is None or dur <= 0:
            if y is not None and len(y) > 0 and sr > 0:
                dur = float(len(y)) / float(sr)
            elif LIBROSA_AVAILABLE and audio_path:
                y_tmp, sr_tmp = librosa.load(audio_path, sr=None, mono=True, dtype="float32")
                dur = float(len(y_tmp)) / float(sr_tmp) if sr_tmp else 0.0
        if dur and dur > 0:
            beats = _build_beats_from_bpm(float(bpm), float(dur))
            if len(beats) > 0:
                print(f"[AudioAnalysis] Beat grid from BPM={float(bpm):.1f} ({len(beats)} beats)")
                return beats

    beats = np.array([])

    # Beats: BPM grid (above) → librosa. madmom removed (NC model weights).
    if len(beats) == 0 and LIBROSA_AVAILABLE:
        y_track = y
        sr_track = sr
        if y_track is None or sr_track <= 0:
            y_track, sr_track = librosa.load(audio_path, sr=None, mono=True, dtype="float32")
        try:
            _, beats = librosa.beat.beat_track(y=y_track, sr=sr_track, bpm=bpm, units="time")
        except Exception:
            try:
                _, beats = librosa.beat.beat_track(y=y_track, sr=sr_track, units="time")
            except Exception:
                duration = len(y_track) / sr_track if sr_track else 0.0
                beats = np.arange(0, duration, 60.0 / (bpm or 120.0))

    return beats


def extract_dominant_onsets(
    audio_path: str,
    bpm: Optional[float] = None,
    window_duration: Optional[float] = None,
    threshold_ratio: float = 0.15,
    genre_params: Optional[Dict] = None,
    y: Optional[np.ndarray] = None,
    sr: int = 0,
) -> List[float]:
    if not LIBROSA_AVAILABLE:
        return []

    if y is None or sr <= 0:
        y, sr = librosa.load(audio_path, sr=None, mono=True, dtype="float32")
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
    cancel_cb: Optional[Callable[[], None]] = None,
    chart_id: str = "",
) -> Dict:
    if cancel_cb:
        cancel_cb()
    from app import song_storage

    song_folder = song_storage.song_folder_for_audio_path(song_path, chart_id)
    song_folder.mkdir(parents=True, exist_ok=True)

    original_file_path = song_folder / Path(song_path).name
    if not original_file_path.exists():
        shutil.copy2(song_path, original_file_path)

    analysis_path = str(original_file_path)
    bpm_from_client = bpm is not None and float(bpm) > 0


    all_genres = []
    if track_info and track_info.get('genres'):
        all_genres.extend(track_info['genres'])

    if not all_genres and GENRE_DETECTION_AVAILABLE:
        if cancel_cb:
            cancel_cb()
        genres = detect_genres("Unknown", "Unknown", audio_path=analysis_path)
        if genres:
            all_genres.extend(genres)

    unique_genres = []
    for g in all_genres:
        if not g or g.lower() == 'unknown':
            continue
        if g not in unique_genres:
            unique_genres.append(g)
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
            print(f"[AudioAnalysis] Для анализа выбран стем: {Path(analysis_path).name}")
        else:
            print("[AudioAnalysis] Stem не выбран — используем оригинальный аудиофайл")

    stem_y: Optional[np.ndarray] = None
    stem_sr = 0
    duration_sec = 0.0
    dominant_onsets: List[float] = []
    kick_times, snare_times = [], []
    hat_times: List[float] = []
    classified_hits: List[Dict] = []

    if LIBROSA_AVAILABLE:
        if cancel_cb:
            cancel_cb()
        stem_y, stem_sr = librosa.load(analysis_path, sr=None, mono=True, dtype="float32")
        duration_sec = float(len(stem_y)) / float(stem_sr) if stem_sr else 0.0

    if cancel_cb:
        cancel_cb()
    beats = extract_beats(
        analysis_path,
        bpm,
        duration_sec=duration_sec,
        prefer_bpm_grid=bpm_from_client,
        y=stem_y,
        sr=stem_sr,
    )

    if stem_type == "drums" and LIBROSA_AVAILABLE and stem_y is not None:
        if cancel_cb:
            cancel_cb()
        backend = drum_backend_name()
        if backend == "adtof_fast" and is_adtof_available():
            try:
                hit_data = transcribe_drum_stem(
                    analysis_path, stem_y, stem_sr, genre_params=genre_params, cancel_cb=cancel_cb
                )
            except Exception as e:
                err = str(e).split("\n", 1)[0]
                if len(err) > 160:
                    err = err[:157] + "..."
                print(f"[AudioAnalysis] ADTOF failed ({err}), fallback heuristic")
                hit_data = analyze_drum_hits(stem_y, stem_sr, genre_params=genre_params)
        else:
            if backend == "adtof_fast" and not is_adtof_available():
                print("[AudioAnalysis] ADTOF not installed (pip adtof-pytorch), heuristic fallback")
            hit_data = analyze_drum_hits(stem_y, stem_sr, genre_params=genre_params)
        kick_times = hit_data["kick_times"]
        snare_times = hit_data["snare_times"]
        hat_times = hit_data.get("hat_times", [])
        classified_hits = hit_data.get("classified_hits", [])
        sync_tolerance = genre_params.get("sync_tolerance", 0.1) if genre_params else 0.1
        subdivisions = genre_params.get("quantization_subdivisions", [4, 8, 16]) if genre_params else [4, 8, 16]
        kick_times = quantize_events_to_grid(kick_times, bpm, tolerance=sync_tolerance, subdivisions=subdivisions)
        snare_times = quantize_events_to_grid(snare_times, bpm, tolerance=sync_tolerance, subdivisions=subdivisions)
        hat_times = quantize_events_to_grid(hat_times, bpm, tolerance=sync_tolerance, subdivisions=subdivisions)
        if classified_hits:
            q_classified = []
            for h in classified_hits:
                qt = quantize_events_to_grid([h["time"]], bpm, tolerance=sync_tolerance, subdivisions=subdivisions)
                if qt:
                    q_classified.append({**h, "time": qt[0]})
            classified_hits = q_classified
        if cancel_cb:
            cancel_cb()

    if classified_hits:
        dominant_onsets = sorted({float(h["time"]) for h in classified_hits})
    elif stem_y is not None and stem_sr > 0:
        dominant_onsets = extract_dominant_onsets(
            analysis_path,
            bpm=bpm,
            genre_params=genre_params,
            y=stem_y,
            sr=stem_sr,
        )

    return {
        "bpm": float(bpm),
        "beats": beats.tolist(),
        "kick_times": kick_times,
        "snare_times": snare_times,
        "hat_times": hat_times,
        "classified_hits": classified_hits,
        "dominant_onsets": dominant_onsets,
        "analysis_path": analysis_path,
        "original_path": str(original_file_path),
        "track_info": track_info,
        "genres": unique_genres,
        "genre_params": genre_params,
        "duration": duration_sec,
    }

def quantize_events_to_grid(events: List[float], bpm: float, tolerance: float = 0.1, subdivisions: List[int] = [4, 8, 16]) -> List[float]:
    from bisect import bisect_left

    if not events or len(events) == 0:
        return []
    beat_interval = 60.0 / max(1e-6, bpm)
    effective_tolerance = min(float(tolerance), beat_interval * 0.22)
    fast_gap_threshold = beat_interval * 0.40

    sorted_events = sorted(float(t) for t in events)
    fast_mask: List[bool] = []
    for i, t in enumerate(sorted_events):
        prev_gap = (t - sorted_events[i - 1]) if i > 0 else 1e9
        next_gap = (sorted_events[i + 1] - t) if i + 1 < len(sorted_events) else 1e9
        fast_mask.append(min(prev_gap, next_gap) <= fast_gap_threshold)

    grids = []
    for div in subdivisions:
        step = beat_interval / div
        grid = np.arange(0.0, max(sorted_events) + beat_interval, step)
        grids.append(grid)

    quantized = []
    for idx_event, t in enumerate(sorted_events):
        if fast_mask[idx_event]:
            quantized.append(t)
            continue
        best_snap = t
        min_diff = effective_tolerance + 1.0

        for grid in grids:
            idx = bisect_left(grid, t)
            candidates = []
            if idx < len(grid):
                candidates.append(grid[idx])
            if idx > 0:
                candidates.append(grid[idx - 1])

            for candidate in candidates:
                diff = abs(candidate - t)
                if diff <= effective_tolerance and diff < min_diff:
                    min_diff = diff
                    best_snap = candidate

        quantized.append(best_snap)

    return sorted(set(quantized))
