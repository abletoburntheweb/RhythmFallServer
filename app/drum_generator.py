# app/drum_generator.py
import os
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

NOTES_DIR = Path("songs") / "notes"


def detect_kick_snare(y, sr) -> Tuple[List[float], List[float]]:
    try:
        import librosa

        y_harmonic, y_percussive = librosa.effects.hpss(y)

        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)
        onset_frames = librosa.util.peak_pick(
            onset_env,
            pre_max=3, post_max=3,
            pre_avg=10, post_avg=10,
            delta=onset_env.max() * 0.15,
            wait=1
        )

        spectral_centroids = librosa.feature.spectral_centroid(y=y_percussive, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_percussive)[0]

        kick_times, snare_times = [], []

        for i, frame in enumerate(onset_frames):
            if frame >= len(spectral_centroids):
                continue

            time = onset_times[frame]
            centroid = spectral_centroids[frame]
            zcr = zero_crossing_rate[frame]

            if centroid < np.percentile(spectral_centroids, 30) and zcr < np.percentile(zero_crossing_rate, 50):
                kick_times.append(time)
            elif centroid > np.percentile(spectral_centroids, 70) and zcr > np.percentile(zero_crossing_rate, 50):
                snare_times.append(time)
            else:
                S = np.abs(librosa.stft(y_percussive))
                freqs = librosa.fft_frequencies(sr=sr)

                if frame >= S.shape[1]:
                    continue

                spectrum = S[:, frame]
                kick_energy = np.sum(spectrum[(freqs >= 60) & (freqs <= 120)])
                snare_energy = np.sum(spectrum[(freqs >= 180) & (freqs <= 400)])

                total_energy = np.sum(spectrum)
                threshold = total_energy * 0.05

                if kick_energy > snare_energy and kick_energy > threshold:
                    kick_times.append(time)
                elif snare_energy >= kick_energy and snare_energy > threshold:
                    snare_times.append(time)

        return kick_times, snare_times
    except Exception as e:
        print(f"[DrumGen] Ошибка детекции kick/snare: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def generate_drums_notes(song_path: str, bpm: float, lanes: int = 4) -> Optional[List[Dict]]:
    print(f"🎧 Генерация барабанных нот для: {song_path} (BPM: {bpm})")

    if not bpm or bpm <= 0:
        print(f"Ошибка: некорректный BPM ({bpm})")
        return None

    try:
        import librosa

        print(f"[DrumGen] Загрузка аудио из: {song_path}")
        y, sr = librosa.load(song_path, sr=None, mono=True, dtype='float32')
        print(f"[DrumGen] Аудио загружено: длительность {len(y) / sr:.2f}с, частота {sr} Гц")

        kick_times, snare_times = detect_kick_snare(y, sr)
        print(f"[DrumGen] Найдено {len(kick_times)} kick и {len(snare_times)} snare (после обработки)")

        try:
            print(f"[DrumGen] Получение битов с BPM {bpm}...")
            _, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=float(bpm), units='time')
            print(f"[DrumGen] Найдено {len(beats)} битов для синхронизации")
        except Exception as beat_error:
            print(f"[DrumGen] Ошибка получения битов: {beat_error}")
            try:
                print("[DrumGen] Пробуем получить биты без BPM...")
                _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
                print(f"[DrumGen] Альтернативно найдено {len(beats)} битов")
            except:
                duration = len(y) / sr
                beat_interval = 60.0 / bpm
                beats = np.arange(0, duration, beat_interval)
                print(f"[DrumGen] Создано {len(beats)} битов вручную по BPM")

        def sync_to_beats(hit_times):
            if len(beats) == 0:
                print("[DrumGen] Нет битов для синхронизации, возвращаем как есть")
                return hit_times

            synced = []
            for t in hit_times:
                idx = np.argmin((beats - t) ** 2)
                beat_time = beats[idx]
                if abs(beat_time - t) <= 0.2:
                    synced.append(beat_time)

            unique_synced = []
            for t in sorted(synced):
                if not unique_synced or abs(t - unique_synced[-1]) > 0.01:
                    unique_synced.append(t)
            return unique_synced

        synced_kicks = sync_to_beats(kick_times)
        synced_snares = sync_to_beats(snare_times)

        print(f"[DrumGen] После синхронизации: {len(synced_kicks)} kick и {len(synced_snares)} snare")

        if len(synced_kicks) == 0 and len(synced_snares) == 0:
            print("[DrumGen] Нет нот после синхронизации, используем оригинальные времена")
            synced_kicks = kick_times
            synced_snares = snare_times

        song_offset = 0.0

        all_events = []

        for t in synced_kicks:
            all_events.append({
                "type": "KickNote",
                "time": t
            })

        for t in synced_snares:
            all_events.append({
                "type": "SnareNote",
                "time": t
            })

        all_events.sort(key=lambda x: x["time"])

        notes = []
        last_lane_usage = {}

        for event in all_events:
            adjusted_time = event["time"] + song_offset

            if adjusted_time <= 0:
                continue

            available_lanes = [lane for lane in range(lanes) if last_lane_usage.get(lane, -1) < adjusted_time]
            if not available_lanes:
                lane = min(range(lanes), key=lambda l: last_lane_usage.get(l, -1))
            else:
                lane = random.choice(available_lanes)

            last_lane_usage[lane] = adjusted_time

            notes.append({
                "type": event["type"],
                "lane": lane,
                "time": float(adjusted_time)
            })

        notes.sort(key=lambda x: x["time"])

        print(f"✅ Сгенерировано {len(notes)} барабанных нот для {Path(song_path).name}")
        print(f"   - Kicks: {len([n for n in notes if n['type'] == 'KickNote'])}")
        print(f"   - Snares: {len([n for n in notes if n['type'] == 'SnareNote'])}")

        if len(notes) == 0:
            print("[DrumGen] ВНИМАНИЕ: Сгенерировано 0 нот!")

        return notes

    except Exception as e:
        print(f"[DrumGen] Ошибка генерации барабанных нот: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_drums_notes(notes_data: List[Dict], song_path: str) -> bool:
    if not notes_data:
        print("[DrumGen] Нет данных нот для сохранения.")
        return False

    base_name = Path(song_path).stem
    song_folder = NOTES_DIR / base_name
    song_folder.mkdir(parents=True, exist_ok=True)

    notes_filename = f"{base_name}_drums.json"
    notes_path = song_folder / notes_filename

    try:
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            return obj

        notes_data_serializable = convert_types(notes_data)

        temp_path = notes_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(notes_data_serializable, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(notes_path)

        print(f"[DrumGen] Ноты (drums) сохранены в: {notes_path}")
        return True
    except Exception as e:
        print(f"[DrumGen] Ошибка сохранения нот в {notes_path}: {e}")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return False


def load_drums_notes(song_path: str) -> Optional[List[Dict]]:
    base_name = Path(song_path).stem
    song_folder = NOTES_DIR / base_name
    notes_filename = f"{base_name}_drums.json"
    notes_path = song_folder / notes_filename

    if not notes_path.exists():
        print(f"[DrumGen] Файл нот не найден: {notes_path}")
        return None

    try:
        with open(notes_path, 'r', encoding='utf-8') as f:
            notes_data = json.load(f)
        print(f"[DrumGen] Ноты (drums) загружены из: {notes_path}")
        return notes_data
    except Exception as e:
        print(f"[DrumGen] Ошибка загрузки нот из {notes_path}: {e}")
        return None