# app/drum_generator.py
import os
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

MADMOM_AVAILABLE = False
RNNBeatProcessor = None
BeatTrackingProcessor = None


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
        print("[DrumGen] madmom успешно импортирован (лениво) — готов для beat tracking")
        return True
    except Exception as e:
        print(f"[DrumGen] Не удалось импортировать madmom: {e}")
        MADMOM_AVAILABLE = False
        return False

from .audio_separator import detect_kick_snare_with_essentia

NOTES_DIR = Path("songs") / "notes"


def generate_drums_notes(
        song_path: str,
        bpm: float,
        lanes: int = 4,
        sync_tolerance: float = 0.2,
        use_madmom_beats: bool = True
) -> Optional[List[Dict]]:
    print(f"🎧 Генерация барабанных нот для: {song_path} (BPM: {bpm})")

    if not bpm or bpm <= 0:
        print("Ошибка: некорректный BPM")
        return None

    madmom_ready = False
    if use_madmom_beats:
        madmom_ready = import_madmom()

    beats = np.array([])

    if madmom_ready:
        print("[DrumGen] Используем madmom RNN для точного определения битов")
        try:
            proc = RNNBeatProcessor()
            act = proc(song_path)
            tracker = BeatTrackingProcessor(fps=100)
            beats = np.array(tracker(act))
            print(f"[Madmom] Найдено {len(beats)} битов")
        except Exception as e:
            print(f"[Madmom] Ошибка при beat tracking: {e}")
            beats = np.array([])

    if len(beats) == 0:
        print("[DrumGen] Fallback: используем librosa для определения битов")
        if not LIBROSA_AVAILABLE:
            print("[DrumGen] librosa недоступна — возвращаем None")
            return None
        y, sr = librosa.load(song_path, sr=None, mono=True, dtype='float32')
        try:
            _, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm, units='time')
            print(f"[Librosa] Найдено {len(beats)} битов (с заданным BPM)")
        except Exception:
            try:
                _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
                print(f"[Librosa] Найдено {len(beats)} битов (автоопределение BPM)")
            except Exception:
                duration = len(y) / sr
                beat_interval = 60.0 / bpm
                beats = np.arange(0, duration, beat_interval)
                print(f"[Librosa] Создано {len(beats)} битов вручную по BPM")

    print("[DrumGen] Детекция kick/snare через essentia")
    y, sr = librosa.load(song_path, sr=None, mono=True, dtype='float32')
    raw_kick_times, raw_snare_times = detect_kick_snare_with_essentia(y, sr, song_path)
    print(f"[Essentia] Сырые события: {len(raw_kick_times)} kick, {len(raw_snare_times)} snare")

    def sync_to_beats(hit_times: List[float]) -> List[float]:
        if len(beats) == 0 or not hit_times:
            return hit_times
        synced = []
        for t in hit_times:
            distances = np.abs(beats - t)
            min_dist = np.min(distances)
            if min_dist <= sync_tolerance:
                synced.append(float(beats[np.argmin(distances)]))
        unique = []
        for t in sorted(synced):
            if not unique or abs(t - unique[-1]) > 0.01:
                unique.append(t)
        return unique

    synced_kicks = sync_to_beats(raw_kick_times)
    synced_snares = sync_to_beats(raw_snare_times)

    print(f"[DrumGen] После синхронизации: {len(synced_kicks)} kick, {len(synced_snares)} snare")

    if len(synced_kicks) + len(synced_snares) == 0:
        print("[DrumGen] Нет нот после синхронизации — используем сырые времена")
        synced_kicks = raw_kick_times
        synced_snares = raw_snare_times

    all_events = []
    for t in synced_kicks:
        all_events.append({"type": "KickNote", "time": t})
    for t in synced_snares:
        all_events.append({"type": "SnareNote", "time": t})
    all_events.sort(key=lambda x: x["time"])

    notes = []
    last_lane_usage = {}
    song_offset = 0.0

    for event in all_events:
        adjusted_time = event["time"] + song_offset
        if adjusted_time <= 0:
            continue

        available_lanes = [lane for lane in range(lanes) if last_lane_usage.get(lane, -999) < adjusted_time]
        if available_lanes:
            lane = random.choice(available_lanes)
        else:
            lane = min(range(lanes), key=lambda l: last_lane_usage.get(l, -999))

        last_lane_usage[lane] = adjusted_time

        notes.append({
            "type": event["type"],
            "lane": lane,
            "time": float(adjusted_time)
        })

    notes.sort(key=lambda x: x["time"])

    kicks_count = len([n for n in notes if n["type"] == "KickNote"])
    snares_count = len([n for n in notes if n["type"] == "SnareNote"])

    print(f"✅ Сгенерировано {len(notes)} барабанных нот")
    print(f"   - Kick: {kicks_count} | Snare: {snares_count}")

    if len(notes) == 0:
        print("[DrumGen] ВНИМАНИЕ: Сгенерировано 0 нот!")

    return notes


def save_drums_notes(notes_data: List[Dict], song_path: str) -> bool:
    if not notes_data:
        print("[DrumGen] Нет данных нот для сохранения.")
        return False

    base_name = Path(song_path).stem
    song_folder = NOTES_DIR / base_name
    song_folder.mkdir(parents=True, exist_ok=True)

    notes_path = song_folder / f"{base_name}_drums.json"

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
                return {k: convert_types(v) for k, v in obj.items()}
            return obj

        serializable = convert_types(notes_data)

        temp_path = notes_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(notes_path)

        print(f"[DrumGen] Ноты сохранены в: {notes_path}")
        return True
    except Exception as e:
        print(f"[DrumGen] Ошибка сохранения нот: {e}")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return False

def load_drums_notes(song_path: str) -> Optional[List[Dict]]:
    base_name = Path(song_path).stem
    notes_path = NOTES_DIR / base_name / f"{base_name}_drums.json"

    if not notes_path.exists():
        print(f"[DrumGen] Файл нот не найден: {notes_path}")
        return None

    try:
        with open(notes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[DrumGen] Ноты загружены из: {notes_path}")
        return data
    except Exception as e:
        print(f"[DrumGen] Ошибка загрузки нот: {e}")
        return None