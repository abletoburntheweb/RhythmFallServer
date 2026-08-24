# app/drum_hit_detector.py — Windows-native kick / snare / hat classification (librosa, no Essentia).
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

DrumClass = str  # kick | snare | hat | perc

KICK_FREQ = (40, 250)
SNARE_FREQ = (200, 5000)
HAT_FREQ = (5000, 14000)


def _remove_close(times: List[float], min_interval: float = 0.05) -> List[float]:
    if not times:
        return []
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] >= min_interval:
            out.append(t)
    return out


def _classify_frame(
    frame_idx: int,
    spectral_centroids: np.ndarray,
    zero_crossing_rate: np.ndarray,
    spectral_rolloffs: np.ndarray,
    kick_energy_norm: np.ndarray,
    snare_energy_norm: np.ndarray,
    hat_energy_norm: np.ndarray,
) -> Optional[DrumClass]:
    if frame_idx >= len(spectral_centroids):
        return None

    centroid = spectral_centroids[frame_idx]
    zcr = zero_crossing_rate[frame_idx]
    rolloff = spectral_rolloffs[frame_idx]
    kick_e = kick_energy_norm[frame_idx]
    snare_e = snare_energy_norm[frame_idx]
    hat_e = hat_energy_norm[frame_idx]

    centroid_kick = np.percentile(spectral_centroids, 30)
    centroid_snare = np.percentile(spectral_centroids, 70)
    zcr_kick = np.percentile(zero_crossing_rate, 35)
    zcr_snare = np.percentile(zero_crossing_rate, 65)
    rolloff_kick = np.percentile(spectral_rolloffs, 25)
    rolloff_snare = np.percentile(spectral_rolloffs, 75)

    kick_score = snare_score = hat_score = 0

    if centroid <= centroid_kick:
        kick_score += 2
    elif centroid >= centroid_snare:
        snare_score += 2
        if hat_e > snare_e * 0.45 and rolloff >= rolloff_snare:
            hat_score += 2
    else:
        kick_score += 1
        snare_score += 1

    if zcr <= zcr_kick:
        kick_score += 1
    elif zcr >= zcr_snare:
        snare_score += 1
        hat_score += 1

    if rolloff <= rolloff_kick:
        kick_score += 1
    elif rolloff >= rolloff_snare:
        snare_score += 1
        hat_score += 1

    if kick_e > snare_e and kick_e > hat_e:
        kick_score += 2
    elif snare_e > kick_e:
        snare_score += 1
    if hat_e > kick_e * 0.35 and hat_e >= snare_e * 0.25:
        hat_score += 2

    scores = {"kick": kick_score, "snare": snare_score, "hat": hat_score}
    best = max(scores, key=scores.get)
    if scores[best] < 2:
        return "perc"
    return best


def detect_classified_hits(
    y: np.ndarray,
    sr: int,
    *,
    onset_delta_ratio: float = 0.05,
    min_interval: float = 0.05,
    genre_params: Optional[Dict] = None,
) -> List[Dict]:
    """Find percussive onsets and label each as kick / snare / hat / perc."""
    if not LIBROSA_AVAILABLE or y is None or len(y) == 0:
        return []

    params = genre_params or {}
    kick_mult = float(params.get("kick_sensitivity_multiplier", 1.0))
    snare_mult = float(params.get("snare_sensitivity_multiplier", 1.0))
    delta_ratio = onset_delta_ratio / max(0.5, (kick_mult + snare_mult) * 0.5)

    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)
    if onset_env.size == 0 or float(onset_env.max()) <= 0:
        return []

    onset_times = librosa.times_like(onset_env, sr=sr)
    onset_frames = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=10,
        post_avg=10,
        delta=float(onset_env.max()) * delta_ratio,
        wait=2,
    )

    spectral_centroids = librosa.feature.spectral_centroid(y=y_percussive, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_percussive)[0]
    spectral_rolloffs = librosa.feature.spectral_rolloff(y=y_percussive, sr=sr, roll_percent=0.85)[0]

    S = np.abs(librosa.stft(y_percussive))
    freqs = librosa.fft_frequencies(sr=sr)
    total_energy = np.sum(S, axis=0) + 1e-8

    kick_mask = (freqs >= KICK_FREQ[0]) & (freqs <= KICK_FREQ[1])
    snare_mask = (freqs >= SNARE_FREQ[0]) & (freqs <= SNARE_FREQ[1])
    hat_mask = (freqs >= HAT_FREQ[0]) & (freqs <= HAT_FREQ[1])

    kick_energy_norm = np.sum(S[kick_mask, :], axis=0) / total_energy
    snare_energy_norm = np.sum(S[snare_mask, :], axis=0) / total_energy
    hat_energy_norm = np.sum(S[hat_mask, :], axis=0) / total_energy

    hits: List[Dict] = []
    for frame_idx in onset_frames:
        if frame_idx >= len(onset_times):
            continue
        drum = _classify_frame(
            frame_idx,
            spectral_centroids,
            zero_crossing_rate,
            spectral_rolloffs,
            kick_energy_norm,
            snare_energy_norm,
            hat_energy_norm,
        )
        if drum is None:
            continue
        hits.append({"time": float(onset_times[frame_idx]), "drum": drum, "strength": float(onset_env[frame_idx])})

    hits.sort(key=lambda h: h["time"])
    filtered: List[Dict] = []
    for h in hits:
        if not filtered or h["time"] - filtered[-1]["time"] >= min_interval:
            filtered.append(h)
        elif h["strength"] > filtered[-1]["strength"]:
            filtered[-1] = h

    return filtered


def split_by_class(classified_hits: List[Dict]) -> Tuple[List[float], List[float], List[float]]:
    kick, snare, hat = [], [], []
    for h in classified_hits:
        t = float(h["time"])
        drum = str(h.get("drum", "perc"))
        if drum == "kick":
            kick.append(t)
        elif drum == "snare":
            snare.append(t)
        elif drum == "hat":
            hat.append(t)
    return kick, snare, hat


def analyze_drum_hits(
    y: np.ndarray,
    sr: int,
    *,
    genre_params: Optional[Dict] = None,
) -> Dict:
    """Classify drum stem hits. Returns kick/snare/hat times plus classified_hits."""
    classified = detect_classified_hits(y, sr, genre_params=genre_params)
    kick_times, snare_times, hat_times = split_by_class(classified)

    return {
        "kick_times": kick_times,
        "snare_times": snare_times,
        "hat_times": hat_times,
        "classified_hits": classified,
    }


# When kick+hat (etc.) share a timestamp, prefer the playable core class.
_RESOLVE_CLASS_PRIORITY = {
    "kick": 0,
    "snare": 1,
    "tom": 2,
    "cymbal": 3,
    "hat": 4,
    "perc": 5,
}


def resolve_drum_at_time(
    t: float,
    classified_hits: List[Dict],
    tolerance: float = 0.06,
) -> Optional[DrumClass]:
    best: Optional[DrumClass] = None
    best_key: Optional[Tuple[float, int]] = None
    for h in classified_hits:
        d = abs(float(h["time"]) - t)
        if d > tolerance:
            continue
        drum = str(h.get("drum", "perc") or "perc").lower()
        key = (d, int(_RESOLVE_CLASS_PRIORITY.get(drum, 9)))
        if best_key is None or key < best_key:
            best_key = key
            best = drum
    return best
