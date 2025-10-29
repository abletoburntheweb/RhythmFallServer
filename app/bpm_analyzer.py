# app/bpm_analyzer.py
import os
import json
import numpy as np
from pathlib import Path

SONGS_CACHE_FILE = "data/songs_cache.json"


def load_songs_cache():
    if os.path.exists(SONGS_CACHE_FILE):
        try:
            with open(SONGS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return {}
    return {}


def save_songs_cache(cache):
    os.makedirs(os.path.dirname(SONGS_CACHE_FILE), exist_ok=True)
    with open(SONGS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_bpm_from_cache(song_path):
    cache = load_songs_cache()
    song_key = song_path
    if song_key in cache:
        return cache[song_key].get("bpm")

    filename = Path(song_path).name.lower()
    for key, info in cache.items():
        if Path(key).name.lower() == filename:
            return info.get("bpm")
    return None


def save_bpm_to_cache(song_path, bpm):
    cache = load_songs_cache()
    song_key = song_path
    if song_key not in cache:
        cache[song_key] = {
            "path": song_path,
            "title": Path(song_path).stem,
            "artist": "Неизвестен",
            "bpm": bpm,
            "year": "Н/Д",
            "duration": "Н/Д"
        }
    else:
        cache[song_key]["bpm"] = bpm
    save_songs_cache(cache)


BPM_CACHE = {Path(k).name.lower(): v.get("bpm") for k, v in load_songs_cache().items()}


def preprocess_audio_for_bpm(y, sr):
    try:
        import librosa
        if y.ndim > 1:
            y = librosa.to_mono(y)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = np.mean(spectral_centroids)
        coef = 0.8 if avg_centroid > 2000 else 0.97
        y = librosa.effects.preemphasis(y, coef=coef)
        y = librosa.util.normalize(y)
        y = librosa.effects.trim(y, top_db=40)[0]
        return y
    except:
        return y


def calculate_bpm(file_path, save_cache=True):
    fname = Path(file_path).name.lower()
    cached = get_bpm_from_cache(file_path)
    if cached is not None:
        return {"file": fname, "bpm": cached, "source": "cache"}

    try:
        import librosa
        from librosa.feature.rhythm import tempo as rhythm_tempo

        y, sr = librosa.load(file_path, sr=44100)
        y_processed = preprocess_audio_for_bpm(y, sr)

        onset_env = librosa.onset.onset_strength(y=y_processed, sr=sr)
        tempos = []

        tempo_configs = [
            (512, 4.0), (512, 8.0), (512, 2.0), (1024, 4.0),
            (256, 4.0), (256, 8.0), (1024, 2.0), (128, 4.0), (2048, 4.0)
        ]
        for hop, ac_size in tempo_configs:
            try:
                t = rhythm_tempo(onset_envelope=onset_env, sr=sr, hop_length=hop, ac_size=ac_size, max_tempo=300.0)
                tempos.append(t.item())
            except:
                continue

        try:
            t1, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempos.append(t1.item())
        except:
            pass
        try:
            t2, _ = librosa.beat.beat_track(y=y_processed, sr=sr)
            tempos.append(t2.item())
        except:
            pass

        tempos = [t for t in tempos if 20 <= t <= 300]
        bpm = int(round(np.median(tempos))) if tempos else 120

        if bpm < 60 and bpm * 2 <= 200:
            bpm *= 2
        if bpm > 200 and bpm / 2 >= 60:
            bpm = int(bpm / 2)
        bpm = max(60, min(200, bpm))

        if save_cache:
            save_bpm_to_cache(file_path, bpm)
            BPM_CACHE[fname] = bpm

        return {"file": fname, "bpm": bpm, "source": "calculated"}

    except ImportError:
        return {"file": fname, "bpm": None, "error": "librosa не установлена"}
    except Exception as e:
        return {"file": fname, "bpm": None, "error": str(e)}


def reset_cache():
    global BPM_CACHE
    cache = load_songs_cache()
    for key in cache:
        if "bpm" in cache[key]:
            del cache[key]["bpm"]
    save_songs_cache(cache)
    BPM_CACHE = {}
    return {"status": "ok", "message": "Кэш BPM сброшен"}
