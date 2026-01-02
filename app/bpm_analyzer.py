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
        try:
            y, sr = librosa.load(file_path, sr=22050)
        except Exception as load_error:
            print(f"[ERROR] Failed to load audio file {file_path}: {load_error}")
            return {"file": fname, "bpm": None, "error": f"Failed to load audio file: {str(load_error)}"}

        if y.ndim > 1:
            y = librosa.to_mono(y)
        y = librosa.util.normalize(y)

        y = librosa.effects.preemphasis(y, coef=0.97)

        tempos = []

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        configs = [
            (512, 4.0),
            (1024, 2.0),
            (256, 8.0),
            (512, 2.0),
            (1024, 4.0),
        ]

        for hop_length, ac_size in configs:
            try:
                tempo = librosa.beat.tempo(
                    y=y,
                    sr=sr,
                    hop_length=hop_length,
                    ac_size=ac_size,
                    max_tempo=300.0,
                    offset=0.0
                )
                tempos.extend([tempo[0]] * 2)
            except Exception as e:
                print(f"[WARNING] Tempo calculation failed for config {hop_length}, {ac_size}: {e}")
                continue

        try:
            tempo_alt, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            tempos.append(tempo_alt)
        except Exception as e:
            print(f"[WARNING] beat_track failed: {e}")

        try:
            tempo_ac = librosa.feature.tempo(
                y=y,
                sr=sr,
                hop_length=512,
                win_length=384
            )
            if tempo_ac is not None and len(tempo_ac) > 0:
                tempos.extend(tempo_ac.flatten())
        except Exception as e:
            print(f"[WARNING] feature.tempo failed: {e}")

        valid_tempos = [t for t in tempos if 40 <= t <= 250]

        if not valid_tempos:
            bpm = 120
        else:
            bpm = np.median(valid_tempos)

            if bpm < 60:
                bpm *= 2
            elif bpm > 180:
                bpm /= 2

            bpm = int(round(bpm))

        try:
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        except Exception as e:
            print(f"[WARNING] Feature calculation failed, using basic BPM: {e}")
        else:
            if zero_crossing_rate > 0.1:
                if 160 <= bpm <= 220:
                    bpm = int(bpm / 2)

        bpm = max(40, min(250, bpm))

        if save_cache:
            save_bpm_to_cache(file_path, bpm)

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
