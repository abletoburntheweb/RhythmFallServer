# app/audio_separator.py
import numpy as np
import librosa
import os

try:
    import essentia.standard as es

    ESSENTIA_AVAILABLE = True
    print("[AudioSeparator] Essentia доступна")
except ImportError:
    ESSENTIA_AVAILABLE = False
    print("[AudioSeparator] Essentia не установлена")


def detect_kick_snare_with_essentia(y, sr, audio_path: str):
    if not ESSENTIA_AVAILABLE:
        print("[Essentia] Essentia не установлена, используем librosa")
        return detect_kick_snare_original(y, sr)

    try:
        loader = es.MonoLoader(filename=audio_path, sampleRate=sr)
        audio = loader()

        onset_detector = es.OnsetDetection(method="hfc")
        frame_size = 1024
        hop_size = 512
        window = es.Windowing(type="hann")
        spectrum = es.Spectrum()
        frames = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)

        onset_times = []
        frame_index = 0
        for frame in frames:
            frame_windowed = window(frame)
            frame_spectrum = spectrum(frame_windowed)
            onset_strength = onset_detector(frame_spectrum, frame_spectrum)
            if onset_strength > 0.5:
                time = frame_index * hop_size / sr
                onset_times.append(time)
            frame_index += 1

        onset_times = sorted(set(onset_times))

        kick_times = []
        snare_times = []

        for time in onset_times:
            start_sample = int(max(0, (time - 0.05)) * sr)
            end_sample = int(min(len(audio), (time + 0.05) * sr))
            if end_sample - start_sample < 100:
                continue

            segment = audio[start_sample:end_sample]
            segment_fft = np.fft.fft(segment)
            freqs = np.fft.fftfreq(len(segment), 1 / sr)

            positive = freqs >= 0
            segment_fft = segment_fft[positive]
            freqs = freqs[positive]

            kick_mask = (freqs >= 60) & (freqs <= 200)
            snare_mask = (freqs >= 200) & (freqs <= 5000)
            sub_bass_mask = (freqs >= 40) & (freqs <= 100)

            kick_energy = np.sum(np.abs(segment_fft[kick_mask])) if np.any(kick_mask) else 0
            snare_energy = np.sum(np.abs(segment_fft[snare_mask])) if np.any(snare_mask) else 0
            sub_bass_energy = np.sum(np.abs(segment_fft[sub_bass_mask])) if np.any(sub_bass_mask) else 0

            if sub_bass_energy > kick_energy * 0.5 and kick_energy > snare_energy * 0.3:
                kick_times.append(time)
            elif snare_energy > kick_energy:
                snare_times.append(time)

        def remove_close(times, min_interval=0.05):
            if not times:
                return []
            filtered = [times[0]]
            for t in times[1:]:
                if t - filtered[-1] >= min_interval:
                    filtered.append(t)
            return filtered

        kick_times = remove_close(kick_times)
        snare_times = remove_close(snare_times)

        print(f"[Essentia] Найдено {len(kick_times)} kick и {len(snare_times)} snare")
        return kick_times, snare_times

    except Exception as e:
        print(f"[Essentia] Ошибка детекции: {e}")
        import traceback
        traceback.print_exc()
        return detect_kick_snare_original(y, sr)


def detect_kick_snare_original(y, sr) -> tuple[list, list]:
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))

        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)
        onset_times = librosa.times_like(onset_env, sr=sr)

        onset_frames = librosa.util.peak_pick(
            onset_env,
            pre_max=3, post_max=3,
            pre_avg=10, post_avg=10,
            delta=onset_env.max() * 0.05,
            wait=2
        )

        spectral_centroids = librosa.feature.spectral_centroid(y=y_percussive, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_percussive)[0]

        spectral_rolloffs = librosa.feature.spectral_rolloff(
            y=y_percussive, sr=sr, roll_percent=0.85
        )[0]

        S = np.abs(librosa.stft(y_percussive))
        freqs = librosa.fft_frequencies(sr=sr)

        KICK_FREQ_RANGE = (40, 250)
        SNARE_FREQ_RANGE = (200, 5000)

        kick_mask = (freqs >= KICK_FREQ_RANGE[0]) & (freqs <= KICK_FREQ_RANGE[1])
        kick_energy = np.sum(S[kick_mask, :], axis=0)

        snare_mask = (freqs >= SNARE_FREQ_RANGE[0]) & (freqs <= SNARE_FREQ_RANGE[1])
        snare_energy = np.sum(S[snare_mask, :], axis=0)

        total_energy = np.sum(S, axis=0)
        kick_energy_norm = kick_energy / (total_energy + 1e-8)
        snare_energy_norm = snare_energy / (total_energy + 1e-8)

        kick_times = []
        snare_times = []

        centroid_threshold_kick = np.percentile(spectral_centroids, 0.30)
        centroid_threshold_snare = np.percentile(spectral_centroids, 0.70)
        zcr_threshold_kick = np.percentile(zero_crossing_rate, 0.35)
        zcr_threshold_snare = np.percentile(zero_crossing_rate, 0.65)
        rolloff_threshold_kick = np.percentile(spectral_rolloffs, 0.25)
        rolloff_threshold_snare = np.percentile(spectral_rolloffs, 0.75)

        for frame_idx in onset_frames:
            if frame_idx >= len(spectral_centroids) or frame_idx >= len(freqs):
                continue

            time = onset_times[frame_idx]

            centroid = spectral_centroids[frame_idx]
            zcr = zero_crossing_rate[frame_idx]
            rolloff = spectral_rolloffs[frame_idx]
            kick_e = kick_energy_norm[frame_idx]
            snare_e = snare_energy_norm[frame_idx]

            kick_score = 0
            snare_score = 0

            if centroid <= centroid_threshold_kick:
                kick_score += 2
            elif centroid >= centroid_threshold_snare:
                snare_score += 2
            else:
                kick_score += 1
                snare_score += 1

            if zcr <= zcr_threshold_kick:
                kick_score += 1
            elif zcr >= zcr_threshold_snare:
                snare_score += 1

            if rolloff <= rolloff_threshold_kick:
                kick_score += 1
            elif rolloff >= rolloff_threshold_snare:
                snare_score += 1

            if kick_e > snare_e:
                kick_score += 1
            elif snare_e > kick_e:
                snare_score += 1

            if kick_score > snare_score and kick_score >= 2:
                kick_times.append(time)
            elif snare_score > kick_score and snare_score >= 2:
                snare_times.append(time)

        def remove_close_times(time_list, min_interval=0.05):
            if not time_list:
                return []
            sorted_times = sorted(set(time_list))
            filtered_times = [sorted_times[0]]
            for t in sorted_times[1:]:
                if t - filtered_times[-1] >= min_interval:
                    filtered_times.append(t)
            return filtered_times

        kick_times = remove_close_times(kick_times)
        snare_times = remove_close_times(snare_times)

        print(f"[AudioSeparator] Найдено {len(kick_times)} kick и {len(snare_times)} snare (после обработки)")

        return kick_times, snare_times
    except Exception as e:
        print(f"[AudioSeparator] Ошибка оригинальной детекции: {e}")
        import traceback
        traceback.print_exc()
        return [], []