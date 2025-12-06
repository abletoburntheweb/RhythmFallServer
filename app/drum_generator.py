# app/routes.py
import os
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple


try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None  

NOTES_DIR = Path("songs") / "notes"

KICK_FREQ_RANGE = (60, 120)
SNARE_FREQ_RANGE = (200, 5000)
KICK_SPECTRAL_CENTROID_THRESHOLD_PERCENTILE = 25
SNARE_SPECTRAL_CENTROID_THRESHOLD_PERCENTILE = 75
KICK_ZCR_THRESHOLD_PERCENTILE = 40
SNARE_ZCR_THRESHOLD_PERCENTILE = 60
SPECTRAL_ROLLOFF_THRESHOLD_PERCENTILE = 85
KICK_ROLLOFF_THRESHOLD_PERCENTILE = 20
SNARE_ROLLOFF_THRESHOLD_PERCENTILE = 80


def detect_kick_snare(y, sr) -> Tuple[List[float], List[float]]:
    if not LIBROSA_AVAILABLE:
        print("[DrumGen] Ошибка: библиотека librosa не установлена.")
        return [], []

    try:

        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))

        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)

        onset_frames = librosa.util.peak_pick(
            onset_env,
            pre_max=5, post_max=5,  
            pre_avg=20, post_avg=20,  
            delta=onset_env.max() * 0.1,  
            wait=1
        )
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y_percussive, sr=sr)[0]

        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_percussive)[0]

        
        spectral_rolloffs = \
        librosa.feature.spectral_rolloff(y=y_percussive, sr=sr, roll_percent=SPECTRAL_ROLLOFF_THRESHOLD_PERCENTILE)[0]

        
        S = np.abs(librosa.stft(y_percussive))
        freqs = librosa.fft_frequencies(sr=sr)

        
        kick_mask = (freqs >= KICK_FREQ_RANGE[0]) & (freqs <= KICK_FREQ_RANGE[1])
        kick_energy = np.sum(S[kick_mask, :], axis=0)

        
        snare_mask = (freqs >= SNARE_FREQ_RANGE[0]) & (freqs <= SNARE_FREQ_RANGE[1])
        snare_energy = np.sum(S[snare_mask, :], axis=0)

        
        total_energy = np.sum(S, axis=0)
        kick_energy_norm = kick_energy / (total_energy + 1e-8)  
        snare_energy_norm = snare_energy / (total_energy + 1e-8)

        
        kick_times = []
        snare_times = []

        
        centroid_threshold_kick = np.percentile(spectral_centroids, KICK_SPECTRAL_CENTROID_THRESHOLD_PERCENTILE)
        centroid_threshold_snare = np.percentile(spectral_centroids, SNARE_SPECTRAL_CENTROID_THRESHOLD_PERCENTILE)
        zcr_threshold_kick = np.percentile(zero_crossing_rate, KICK_ZCR_THRESHOLD_PERCENTILE)
        zcr_threshold_snare = np.percentile(zero_crossing_rate, SNARE_ZCR_THRESHOLD_PERCENTILE)
        rolloff_threshold_kick = np.percentile(spectral_rolloffs, KICK_ROLLOFF_THRESHOLD_PERCENTILE)
        rolloff_threshold_snare = np.percentile(spectral_rolloffs, SNARE_ROLLOFF_THRESHOLD_PERCENTILE)

        for frame_idx in onset_frames:
            
            if frame_idx >= len(spectral_centroids) or frame_idx >= len(freqs):
                continue

            time = onset_times[frame_idx]

            
            centroid = spectral_centroids[frame_idx]
            zcr = zero_crossing_rate[frame_idx]
            rolloff = spectral_rolloffs[frame_idx]
            kick_e = kick_energy_norm[frame_idx]
            snare_e = snare_energy_norm[frame_idx]

            
            
            if centroid < centroid_threshold_kick and zcr < zcr_threshold_kick:
                kick_times.append(time)
                continue
            elif centroid > centroid_threshold_snare and zcr > zcr_threshold_snare:
                snare_times.append(time)
                continue

            
            if rolloff < rolloff_threshold_kick:
                kick_times.append(time)
                continue
            elif rolloff > rolloff_threshold_snare:
                snare_times.append(time)
                continue

            
            total_energy_frame = total_energy[frame_idx] if frame_idx < len(total_energy) else 1.0
            if total_energy_frame > 1e-8:  
                if kick_e > snare_e and kick_e > 0.05:  
                    kick_times.append(time)
                    continue
                elif snare_e >= kick_e and snare_e > 0.05:  
                    snare_times.append(time)
                    continue

            
            
            
            
            kick_score = 0
            snare_score = 0

            
            if centroid <= centroid_threshold_kick:
                kick_score += 1
            else:
                snare_score += 1

            if zcr <= zcr_threshold_kick:
                kick_score += 1
            else:
                snare_score += 1

            if rolloff <= rolloff_threshold_kick:
                kick_score += 1
            else:
                snare_score += 1

            if kick_e > snare_e:
                kick_score += 1
            else:
                snare_score += 1

            if kick_score > snare_score:
                kick_times.append(time)
            elif snare_score > kick_score:
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

        print(f"[DrumGen] Найдено {len(kick_times)} kick и {len(snare_times)} snare (после обработки)")

        return kick_times, snare_times
    except Exception as e:
        print(f"[DrumGen] Ошибка детекции kick/snare: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def generate_drums_notes(song_path: str, bpm: float, lanes: int = 4, sync_tolerance: float = 0.2) -> Optional[List[Dict]]: 
    print(f"🎧 Генерация барабанных нот для: {song_path} (BPM: {bpm})")

    if not bpm or bpm <= 0:
        print(f"Ошибка: некорректный BPM ({bpm})")
        return None

    try:
        if not LIBROSA_AVAILABLE:
            print("[DrumGen] Ошибка: библиотека librosa не установлена.")
            return None

        print(f"[DrumGen] Загрузка аудио из: {song_path}")
        y, sr = librosa.load(song_path, sr=None, mono=True, dtype='float32')
        print(f"[DrumGen] Аудио загружено: длительность {len(y) / sr:.2f}с, частота {sr} Гц")

        kick_times, snare_times = detect_kick_snare(y, sr)
        print(f"[DrumGen] После детекции: {len(kick_times)} kick и {len(snare_times)} snare")

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

        def sync_to_beats(hit_times, tolerance=0.2): 
            if len(beats) == 0:
                print("[DrumGen] Нет битов для синхронизации, возвращаем как есть")
                return hit_times

            synced = []
            for t in hit_times:
                idx = np.argmin(np.abs(beats - t)) 
                beat_time = beats[idx]
                if abs(beat_time - t) <= tolerance: 
                    synced.append(beat_time)

            
            unique_synced = []
            for t in sorted(synced):
                if not unique_synced or abs(t - unique_synced[-1]) > 0.01:
                    unique_synced.append(t)
            return unique_synced

        
        synced_kicks = sync_to_beats(kick_times, tolerance=sync_tolerance)
        synced_snares = sync_to_beats(snare_times, tolerance=sync_tolerance)

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