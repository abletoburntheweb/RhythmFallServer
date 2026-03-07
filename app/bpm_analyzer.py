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

        hop_opts = [256, 512, 1024]
        ac_sizes = [2.0, 4.0, 8.0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        for h in hop_opts:
            for a in ac_sizes:
                try:
                    t = librosa.beat.tempo(y=y, sr=sr, hop_length=h, ac_size=a, max_tempo=300.0, offset=0.0)
                    tempos.extend([float(t[0])])
                except Exception as e:
                    pass
        try:
            from scipy.signal import find_peaks
            hop = 512
            onset_env_h = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
            acf = np.correlate(onset_env_h, onset_env_h, mode='full')[len(onset_env_h)-1:]
            lag_min = max(1, int((sr / hop) * (60.0 / 250.0)))
            lag_max = int((sr / hop) * (60.0 / 40.0))
            roi = acf[lag_min:lag_max] if lag_max > lag_min else acf
            pk, _ = find_peaks(roi, distance=max(1, int((sr / hop) * (60.0 / 300.0))))
            lags = (pk + lag_min) if lag_max > lag_min else pk
            bpms_acf = 60.0 / (lags * hop / sr + 1e-9)
            tempos.extend([float(v) for v in bpms_acf if 40.0 <= v <= 250.0])
        except Exception:
            pass

        try:
            t_bt, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            tempos.append(float(t_bt))
        except Exception as e:
            pass

        try:
            t_feat = librosa.feature.tempo(y=y, sr=sr, hop_length=512, win_length=384)
            if t_feat is not None and len(t_feat) > 0:
                tempos.extend([float(v) for v in t_feat.flatten()])
        except Exception as e:
            pass

        y_perc = librosa.effects.hpss(y, margin=(1.0, 5.0))[1]
        onset_env_p = librosa.onset.onset_strength(y=y_perc, sr=sr)
        peaks = librosa.util.peak_pick(onset_env_p, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=onset_env_p.max() * 0.1, wait=2)
        onset_times = librosa.times_like(onset_env_p, sr=sr)[peaks] if len(peaks) > 0 else librosa.times_like(onset_env_p, sr=sr)
        onset_times = onset_times if isinstance(onset_times, np.ndarray) else np.array(onset_times)
        onset_times = onset_times[(onset_times >= 0.0) & (onset_times <= (len(y) / sr))]
        onset_times = onset_times.tolist()
        def _grid_score(times, bpm):
            if bpm <= 0:
                return -1e9
            p = 60.0 / float(bpm)
            if len(times) == 0:
                return -1e9
            tt = np.array(times, dtype=np.float32)
            duration = float(len(y)) / float(sr)
            grid = np.arange(0.0, duration + p, p, dtype=np.float32)
            tol = max(0.02, 0.07 * p)
            dists = np.min(np.abs(tt[:, None] - grid[None, :]), axis=1)
            hits = float(np.sum(dists <= tol))
            return hits / max(1.0, float(len(tt)))
        def _penalty(bpm):
            if bpm < 60 or bpm > 220:
                return 0.92
            if 80 <= bpm <= 190:
                return 1.0
            return 0.97
        candidates = [t for t in tempos if 40 <= t <= 250]
        cand_set = set(int(round(c)) for c in candidates)
        cand_list = sorted(cand_set)
        scaled = []
        for c in cand_list:
            for s in [0.5, 2.0, 1.0, 1.5, (2.0/3.0)]:
                v = int(round(c * s))
                if 40 <= v <= 250:
                    scaled.append(v)
        scaled = sorted(set(scaled))
        if not scaled:
            bpm = 120
        else:
            scores = [(v, _grid_score(onset_times, v) * _penalty(v)) for v in scaled]
            scores.sort(key=lambda x: x[1], reverse=True)
            best_bpm, best_score = int(scores[0][0]), scores[0][1]
            raw_ints = [int(round(t)) for t in candidates if 40 <= t <= 250]
            try:
                from collections import Counter
                cluster = Counter(raw_ints)
                cluster_sorted = sorted(cluster.items(), key=lambda x: x[1], reverse=True)
                cluster_pick = cluster_sorted[0][0] if cluster_sorted else None
            except Exception:
                cluster_pick = None
            if cluster_pick is not None:
                fam = [cluster_pick]
                for s in [0.5, 2.0, 1.5, (2.0/3.0)]:
                    vv = int(round(cluster_pick * s))
                    if 40 <= vv <= 250:
                        fam.append(vv)
                fam = sorted(set(fam))
                fam_scores = [(v, _grid_score(onset_times, v) * _penalty(v)) for v in fam]
                fam_scores.sort(key=lambda x: x[1], reverse=True)
                cf_bpm, cf_score = int(fam_scores[0][0]), fam_scores[0][1]
                bpm = int(cf_bpm if cf_score >= best_score * 0.97 else best_bpm)
            else:
                bpm = int(best_bpm)

        try:
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        except Exception as e:
            print(f"[WARNING] Feature calculation failed, using basic BPM: {e}")
        else:
            if zero_crossing_rate > 0.1:
                if 160 <= bpm <= 220:
                    bpm = int(bpm / 2)
            family = [bpm]
            for s in [0.5, 2.0, 1.5, (2.0/3.0)]:
                vv = int(round(bpm * s))
                if 40 <= vv <= 250:
                    family.append(vv)
            family = sorted(set(family))
            fam_scores = [(v, _grid_score(onset_times, v) * _penalty(v)) for v in family]
            fam_scores.sort(key=lambda x: x[1], reverse=True)
            bpm = int(fam_scores[0][0])

        if bpm < 90:
            while bpm < 90:
                bpm *= 2
            bpm = int(round(bpm))
        elif bpm > 200:
            while bpm > 200:
                bpm /= 2.0
            bpm = int(round(bpm))
        bpm = max(90, min(200, bpm))

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
