# app/audio_analysis.py
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

from .track_detector import REQUESTS_AVAILABLE, identify_track

if REQUESTS_AVAILABLE:
    import requests

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
    print("[AudioAnalysis] Audio-separator доступен")
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False
    print("[AudioAnalysis] Audio-separator недоступен")

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


def separate_stems(song_path: str, song_folder: Path, stem_type: str = "drums") -> str:
    song_path = Path(song_path)
    splitter_folder = song_folder / "splitter"
    splitter_folder.mkdir(parents=True, exist_ok=True)

    expected_name = f"{song_path.stem}_{stem_type}.wav"
    output_path = splitter_folder / expected_name

    if output_path.exists():
        return str(output_path)

    for f in splitter_folder.glob("*.wav"):
        if stem_type in f.name.lower():
            shutil.copy2(f, output_path)
            return str(output_path)

    if not AUDIO_SEPARATOR_AVAILABLE:
        return str(song_path)

    try:
        model_dir = "/tmp/audio-separator-models/"
        separator = Separator(
            output_dir=str(splitter_folder),
            output_format="WAV",
            model_file_dir=model_dir
        )

        available_models = separator.get_simplified_model_list()
        target_model = None

        for model in available_models:
            if stem_type in model.lower() and ('kuielab' in model.lower() or stem_type in model.lower()):
                target_model = model
                break
        if not target_model:
            for model in available_models:
                if 'htdemucs' in model.lower():
                    target_model = model
                    break

        if not target_model:
            return str(song_path)

        separator.load_model(target_model)
        output_files = separator.separate(str(song_path))

        for f in output_files:
            if stem_type in f.lower():
                src = splitter_folder / f
                if src.exists():
                    shutil.copy2(src, output_path)
                    return str(output_path)

        return str(song_path)

    except Exception:
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


def detect_drum_events(audio_path: str) -> Tuple[List[float], List[float]]:
    if not LIBROSA_AVAILABLE:
        return [], []

    y, sr = librosa.load(audio_path, sr=None, mono=True, dtype='float32')
    return detect_kick_snare_with_essentia(y, sr, audio_path)


def analyze_audio(
    song_path: str,
    bpm: Optional[float] = None,
    use_stems: bool = True,
    auto_identify_track: bool = False,
    use_filename_for_genres: bool = True,
    track_info: Optional[Dict] = None,
    stem_type: str = "drums"
) -> Dict:
    base_name = Path(song_path).stem
    song_folder = TEMP_UPLOADS_DIR / base_name
    song_folder.mkdir(parents=True, exist_ok=True)

    original_file_path = song_folder / Path(song_path).name
    if not original_file_path.exists():
        shutil.copy2(song_path, original_file_path)

    analysis_path = str(original_file_path)

    if not track_info and auto_identify_track:
        track_info = identify_track(song_path)

    all_genres = []
    if track_info and track_info.get('genres'):
        all_genres.extend(track_info['genres'])

    if use_filename_for_genres and not all_genres and GENRE_DETECTION_AVAILABLE:
        if track_info and track_info.get('success') and track_info.get('artist') != 'Unknown':
            genres = detect_genres(track_info['artist'], track_info['title'])
            if genres:
                all_genres.extend(genres)

    unique_genres = list(set(g for g in all_genres if g and g.lower() != 'unknown'))
    genre_params = get_genre_params(unique_genres, GENRE_CONFIGS, GENRE_ALIAS_MAP)

    if bpm is None or bpm <= 0:
        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(analysis_path, sr=None, mono=True, dtype='float32')
            try:
                _, bpm = librosa.beat.beat_track(y=y, sr=sr, units='time')
            except:
                bpm = 120.0
        else:
            bpm = 120.0

    if use_stems and AUDIO_SEPARATOR_AVAILABLE:
        stem_path = separate_stems(str(original_file_path), song_folder, stem_type=stem_type)
        if stem_path != str(original_file_path):
            analysis_path = stem_path

    beats = extract_beats(analysis_path, bpm)

    kick_times, snare_times = [], []
    if stem_type == "drums":
        kick_times, snare_times = detect_drum_events(analysis_path)

    return {
        "bpm": float(bpm),
        "beats": beats.tolist(),
        "kick_times": kick_times,
        "snare_times": snare_times,
        "analysis_path": analysis_path,
        "original_path": str(original_file_path),
        "track_info": track_info,
        "genres": unique_genres,
        "genre_params": genre_params,
        "duration": len(librosa.load(analysis_path, sr=None)[0]) / librosa.load(analysis_path, sr=None)[1] if LIBROSA_AVAILABLE else 0.0
    }
