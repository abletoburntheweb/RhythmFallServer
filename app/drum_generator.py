# app/drum_generator.py
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


from .audio_separator import detect_kick_snare_with_essentia

NOTES_DIR = Path("songs") / "notes"

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


        kick_times, snare_times = detect_kick_snare_with_essentia(y, sr, song_path)
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