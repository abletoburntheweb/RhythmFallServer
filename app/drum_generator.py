# app/drum_generator.py
import os
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional
import tempfile

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


try:
    from audio_separator.separator import Separator

    AUDIO_SEPARATOR_AVAILABLE = True
    print("[DrumGen] Audio-separator доступен — будет использоваться для разделения стемов")
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False
    print("[DrumGen] Audio-separator не доступен — анализ на полном миксе")

from .audio_separator import detect_kick_snare_with_essentia

NOTES_DIR = Path("songs") / "notes"


def separate_drums_with_audiosep(song_path: str) -> str:
    song_path = Path(song_path)
    drums_path = song_path.parent / f"{song_path.stem}_drums.wav"

    if drums_path.exists():
        print(f"[AudioSep] Кэшированный drums-стем найден: {drums_path}")
        return str(drums_path)

    print("[AudioSep] Разделение через audio-separator...")
    try:
        separator = Separator(
            output_dir=str(song_path.parent),
            output_format="WAV"
        )

        target_model = None
        available_models = separator.get_simplified_model_list()

        for model in available_models:
            if 'drums' in model.lower() and ('kuielab' in model.lower() or 'drum' in model.lower()):
                target_model = model
                print(f"[AudioSep] Найдена специализированная drums-модель: {target_model}")
                break

        if not target_model:
            for model in available_models:
                if 'htdemucs' in model.lower():
                    target_model = model
                    print(f"[AudioSep] Найдена htdemucs-модель: {target_model}")
                    break

        if not target_model:
            print(f"[AudioSep] Ни одной подходящей модели не найдено")
            return str(song_path)

        print(f"[AudioSep] Загружаем модель: {target_model}")
        separator.load_model(target_model)

        output_files = separator.separate(str(song_path))

        print(f"[AudioSep] Output files returned: {output_files}")

        output_dir = Path(song_path.parent)
        drums_files = list(output_dir.glob(f"{song_path.stem}*(Drums)*.wav"))

        if drums_files:
            drums_file = drums_files[0]
            import shutil
            shutil.copy2(drums_file, drums_path)
            print(f"[AudioSep] Drums-стем успешно скопирован: {drums_path}")
            return str(drums_path)
        else:
            current_dir = Path(".")
            current_drums_files = list(current_dir.glob(f"*{song_path.stem}*(Drums)*.wav"))
            if current_drums_files:
                drums_file = current_drums_files[0]
                import shutil
                shutil.copy2(drums_file, drums_path)
                print(f"[AudioSep] Drums-стем найден в текущей директории и скопирован: {drums_path}")
                return str(drums_path)
            else:
                print("[AudioSep] Не удалось найти файл drums в output директории")
                all_created_files = list(output_dir.glob(f"{song_path.stem}*.*"))
                print(f"[AudioSep] Все созданные файлы: {[f.name for f in all_created_files]}")
                return str(song_path)

    except Exception as e:
        print(f"[AudioSep] Ошибка при разделении: {e}")
        import traceback
        traceback.print_exc()
        print("[AudioSep] Fallback на оригинальный файл")
        return str(song_path)


def generate_drums_notes(
        song_path: str,
        bpm: float,
        lanes: int = 4,
        sync_tolerance: float = 0.2,
        use_madmom_beats: bool = True,
        use_stems: bool = True
) -> Optional[List[Dict]]:
    print(f"🎧 Генерация барабанных нот для: {song_path} (BPM: {bpm})")

    if not bpm or bpm <= 0:
        print("Ошибка: некорректный BPM")
        return None

    analysis_path = song_path
    if use_stems and AUDIO_SEPARATOR_AVAILABLE:
        analysis_path = separate_drums_with_audiosep(song_path)
        if analysis_path != song_path:
            print(f"[DrumGen] Анализ проводится на изолированном drums-стеме: {analysis_path}")
            import os
            original_size = os.path.getsize(song_path)
            stem_size = os.path.getsize(analysis_path)
            print(f"[DrumGen] Оригинал: {original_size} байт, стем: {stem_size} байт")
            if original_size == stem_size:
                print("[DrumGen] ВНИМАНИЕ: Размеры файлов одинаковы - возможно, стем не был создан корректно")
        else:
            print("[DrumGen] Fallback: анализ на полном миксе (стем не был создан)")
    else:
        print("[DrumGen] Анализ на полном миксе (stems отключены или Audio-separator недоступен)")

    madmom_ready = False
    if use_madmom_beats:
        madmom_ready = import_madmom()

    beats = np.array([])

    if madmom_ready:
        print("[DrumGen] Используем madmom RNN для beat tracking")
        try:
            proc = RNNBeatProcessor()
            act = proc(analysis_path)
            tracker = BeatTrackingProcessor(fps=100)
            beats = np.array(tracker(act))
            print(f"[Madmom] Найдено {len(beats)} битов")
        except Exception as e:
            print(f"[Madmom] Ошибка beat tracking: {e}")

    if len(beats) == 0:
        print("[DrumGen] Fallback: librosa beat tracking")
        if not LIBROSA_AVAILABLE:
            return None
        y, sr = librosa.load(analysis_path, sr=None, mono=True, dtype='float32')
        try:
            _, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm, units='time')
            print(f"[Librosa] Найдено {len(beats)} битов (с BPM)")
        except:
            try:
                _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
                print(f"[Librosa] Найдено {len(beats)} битов (авто)")
            except:
                duration = len(y) / sr
                beats = np.arange(0, duration, 60.0 / bpm)
                print(f"[Librosa] Создано {len(beats)} битов вручную")

    print(f"[DrumGen] Детекция kick/snare через essentia на: {analysis_path}")
    y, sr = librosa.load(analysis_path, sr=None, mono=True, dtype='float32')
    raw_kick_times, raw_snare_times = detect_kick_snare_with_essentia(y, sr, analysis_path)
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
        print("[DrumGen] Нет нот после синхронизации — используем сырые")
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
    print(f"   - Использован файл: {analysis_path}")

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