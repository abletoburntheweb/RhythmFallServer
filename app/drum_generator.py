# app/drum_generator.py
import os
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
from .track_detector import REQUESTS_AVAILABLE

if REQUESTS_AVAILABLE:
    import requests

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
from .track_detector import identify_track

try:
    from .genre_detector import detect_genres

    GENRE_DETECTION_AVAILABLE = True
    print("[DrumGen] Genre detection доступен")
except ImportError:
    GENRE_DETECTION_AVAILABLE = False
    print("[DrumGen] Genre detection не установлен")

TEMP_UPLOADS_DIR = Path("temp_uploads")


def load_genre_configs():
    config_path = Path(__file__).parent / "genre_configs.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "default": {
                "kick_sensitivity_multiplier": 1.0,
                "snare_sensitivity_multiplier": 1.0,
                "pattern_complexity": "medium",
                "kick_priority": False,
                "sync_tolerance_multiplier": 1.0,
                "drum_start_window": 4.0,
                "drum_density_threshold": 0.5,
                "confidence_threshold": 0.3,
                "max_hits_per_second": 4,
                "min_note_distance": 0.05,
                "pattern_style": "groove"
            },
            "electronic": {
                "kick_sensitivity_multiplier": 1.2,
                "snare_sensitivity_multiplier": 1.1,
                "pattern_complexity": "high",
                "kick_priority": True,
                "sync_tolerance_multiplier": 0.8,
                "max_hits_per_second": 6,
                "min_note_distance": 0.04,
                "pattern_style": "precise"
            },
            "rock": {
                "kick_sensitivity_multiplier": 1.0,
                "snare_sensitivity_multiplier": 1.0,
                "pattern_complexity": "medium",
                "kick_priority": False,
                "sync_tolerance_multiplier": 1.0,
                "max_hits_per_second": 4,
                "min_note_distance": 0.05,
                "pattern_style": "groove"
            },
            "k-pop": {
                "kick_sensitivity_multiplier": 1.1,
                "snare_sensitivity_multiplier": 1.2,
                "pattern_complexity": "high",
                "kick_priority": False,
                "sync_tolerance_multiplier": 0.9,
                "max_hits_per_second": 5,
                "min_note_distance": 0.045,
                "pattern_style": "precise"
            }
        }


GENRE_CONFIGS = load_genre_configs()


def get_genre_params(genres: List[str]) -> Dict:
    if not genres:
        return GENRE_CONFIGS.get("default", {})

    genres_lower = [g.lower() for g in genres]

    for genre in genres_lower:
        if genre in GENRE_CONFIGS:
            print(f"[GenreParams] Используем параметры для жанра: {genre}")
            return GENRE_CONFIGS[genre]

    return GENRE_CONFIGS.get("default", {})


def detect_drum_section_start(times: List[float], window_duration: float = 2.0, threshold: float = 0.5) -> float:
    if len(times) < 2:
        return 0.0

    times = np.array(times)
    start_time = 0.0
    end_time = max(times)

    step = window_duration / 2
    current_time = start_time

    while current_time < end_time:
        window_start = current_time
        window_end = current_time + window_duration

        hits_in_window = sum(1 for t in times if window_start <= t < window_end)
        density = hits_in_window / window_duration

        if density >= threshold:
            print(f"[DrumStart] Найдено устойчивое начало ударных: {window_start:.2f}s (плотность: {density:.2f}/s)")
            return window_start

        current_time += step

    return 0.0


def apply_temporal_filter(events: List[float], min_distance: float = 0.05) -> List[float]:
    if not events:
        return events

    filtered = [events[0]]

    for event in events[1:]:
        if event - filtered[-1] >= min_distance:
            filtered.append(event)

    return filtered


def apply_groove_pattern(events: List[float], pattern_style: str = "groove", bpm: float = 120.0) -> List[float]:
    if not events:
        return events

    if pattern_style == "precise":

        return events
    elif pattern_style == "sparse":

        return events[::2]
    else:

        grid_step = 60.0 / bpm
        grooved_events = []

        for event in events:
            grid_position = round(event / (grid_step / 2)) * (grid_step / 2)

            groove_amount = 0.02
            offset = random.uniform(-groove_amount, groove_amount) * grid_step
            grooved_time = grid_position + offset
            grooved_events.append(max(0, grooved_time))

        return sorted(grooved_events)


def separate_drums_with_audiosep(song_path: str, song_folder: Path) -> str:
    song_path = Path(song_path)
    splitter_folder = song_folder / "splitter"
    splitter_folder.mkdir(parents=True, exist_ok=True)
    drums_path = splitter_folder / f"{song_path.stem}_drums.wav"

    if drums_path.exists():
        print(f"[AudioSep] Кэшированный drums-стем найден: {drums_path}")
        return str(drums_path)

    existing_files = list(splitter_folder.glob("*.wav"))
    if existing_files:
        for file in existing_files:
            if "drums" in file.name.lower() or "drum" in file.name.lower():
                print(f"[AudioSep] Кэшированный drums-стем найден (по названию): {file}")
                return str(file)
        print(f"[AudioSep] Файлы уже существуют в splitter (но не drums): {[f.name for f in existing_files]}")

    if not AUDIO_SEPARATOR_AVAILABLE:
        print("[AudioSep] Audio-separator недоступен, fallback на оригинальный файл")
        return str(song_path)

    print("[AudioSep] Разделение через audio-separator...")
    try:
        model_dir = "/tmp/audio-separator-models/"
        print(f"[AudioSep] Используем локальную модель из: {model_dir}")

        separator = Separator(
            output_dir=str(splitter_folder),
            output_format="WAV",
            model_file_dir=model_dir
        )

        print("[AudioSep] Получение списка моделей...")
        target_model = None
        try:
            available_models = separator.get_simplified_model_list()

        except Exception as e:
            if REQUESTS_AVAILABLE and isinstance(e, requests.exceptions.ConnectionError):
                print(f"[AudioSep] Ошибка соединения при получении списка моделей: {e}")
            else:
                print(f"[AudioSep] Ошибка при получении списка моделей: {e}")
            print("[AudioSep] Fallback на оригинальный файл из-за ошибки получения моделей")
            return str(song_path)

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
            print(f"[AudioSep] Ни одной подходящей модели не найдено. Доступные: {available_models}")
            print("[AudioSep] Fallback на оригинальный файл")
            return str(song_path)

        print(f"[AudioSep] Загружаем модель: {target_model}")
        separator.load_model(target_model)

        print(f"[AudioSep] Запуск разделения...")
        output_files = separator.separate(str(song_path))

        print(f"[AudioSep] Output files returned by separator: {output_files}")

        drums_files = list(splitter_folder.glob(f"{song_path.stem}*(Drums)*.wav"))
        if not drums_files:
            drums_files = [f for f in output_files if "Drums" in f or "drums" in f.lower()]
            if drums_files:
                drums_file = None
                for f in drums_files:
                    possible_path = splitter_folder / f
                    if possible_path.exists():
                        drums_file = possible_path
                        break
                if drums_file:
                    drums_files = [drums_file]
                else:
                    drums_files = []

        if drums_files:
            drums_file = drums_files[0]
            import shutil
            shutil.copy2(drums_file, drums_path)
            print(f"[AudioSep] Drums-стем успешно скопирован в кэш: {drums_path}")

            for created_file in output_files:
                created_path = splitter_folder / created_file
                if created_path.exists() and created_path != drums_path:
                    try:
                        os.remove(created_path)
                        print(f"[AudioSep] Временный файл удален: {created_path}")
                    except Exception as e:
                        print(f"[AudioSep] Ошибка при удаления временного файла {created_path}: {e}")

            return str(drums_path)
        else:
            print("[AudioSep] Не удалось найти файл drums в output директории после разделения")
            current_dir = Path(".")
            current_drums_files = list(current_dir.glob(f"*{song_path.stem}*(Drums)*.wav"))
            if current_drums_files:
                drums_file = current_drums_files[0]
                import shutil
                shutil.copy2(drums_file, drums_path)
                print(f"[AudioSep] Drums-стем найден в текущей директории и скопирован в кэш: {drums_path}")
                return str(drums_path)
            else:
                print("[AudioSep] Drums-стем не был создан успешно, fallback на оригинальный файл")
                return str(song_path)

    except Exception as e:
        if REQUESTS_AVAILABLE and isinstance(e, requests.exceptions.ConnectionError):
            print(f"[AudioSep] Сетевая ошибка при разделении: {e}")
        else:
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
        use_stems: bool = True,
        track_info: Optional[Dict] = None,
        auto_identify_track: bool = False,
        use_filename_for_genres: bool = True
) -> Optional[List[Dict]]:
    print(f"🎧 Генерация барабанных нот для: {song_path} (BPM: {bpm})")
    print(f"[DrumGen] Получен track_info: {track_info}")

    if not track_info and auto_identify_track:
        print(f"[DrumGen] Автоматическая идентификация трека для: {song_path}")
        track_info = identify_track(song_path)
        if track_info and track_info.get('success'):
            print(f"[DrumGen] Автоматически определен трек: {track_info['artist']} - {track_info['title']}")
            if track_info['genres']:
                print(f"[DrumGen] Жанры из аудио: {', '.join(track_info['genres'])}")
        else:
            print("[DrumGen] Не удалось автоматически идентифицировать трек")

    all_genres = []

    if track_info and track_info.get('genres'):
        all_genres.extend(track_info['genres'])
        print(f"[DrumGen] Жанры из переданного track_info: {track_info['genres']}")
    else:
        print("[DrumGen] track_info пуст или genres в track_info отсутствуют/пусты")

    if use_filename_for_genres and not all_genres:
        if GENRE_DETECTION_AVAILABLE:
            if track_info and track_info.get('success') and track_info.get('artist') != 'Unknown' and track_info.get(
                    'title') != 'Unknown':
                print(f"[MultiGenre] Получение жанров для: {track_info['artist']} - {track_info['title']}")
                filename_genres = detect_genres(track_info['artist'], track_info['title'])
                if filename_genres:
                    all_genres.extend(filename_genres)
                    print(f"[MultiGenre] Добавлены жанры из внешних источников: {filename_genres}")
                else:
                    print("[MultiGenre] Жанры из внешних источников не найдены")
            else:
                print("[MultiGenre] Трек не идентифицирован, пропускаем получение жанров")
        else:
            print("[MultiGenre] Genre detection недоступен, пропускаем получение жанров")

    unique_genres = list(set([g for g in all_genres if g and g.lower() != 'unknown']))

    genre_params = {}
    if unique_genres:
        genre_params = get_genre_params(unique_genres)
        print(f"[GenreParams] Применены параметры для жанра: {unique_genres[0] if unique_genres else 'default'}")
        print(f"[GenreParams] Параметры: {genre_params}")

        if 'sync_tolerance_multiplier' in genre_params:
            sync_tolerance *= genre_params['sync_tolerance_multiplier']
            print(f"[DrumGen] Sync tolerance изменен: {sync_tolerance:.2f}")
    else:
        print("[GenreParams] Уникальные жанры не определены, используются параметры по умолчанию.")

    if not bpm or bpm <= 0:
        print("Ошибка: некорректный BPM")
        return None

    base_name = Path(song_path).stem
    song_folder = TEMP_UPLOADS_DIR / base_name
    song_folder.mkdir(parents=True, exist_ok=True)

    original_file_path = song_folder / Path(song_path).name
    if not original_file_path.exists():
        import shutil
        shutil.copy2(song_path, original_file_path)
        print(f"[DrumGen] Оригинальный файл скопирован: {original_file_path}")

    analysis_path = str(original_file_path)
    drums_stem_path = None
    if use_stems and AUDIO_SEPARATOR_AVAILABLE:
        drums_stem_path = separate_drums_with_audiosep(str(original_file_path), song_folder)
        if drums_stem_path != str(original_file_path):
            analysis_path = drums_stem_path
            print(f"[DrumGen] Анализ проводится на изолированном drums-стеме: {analysis_path}")
            import os
            original_size = os.path.getsize(original_file_path)
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

    drum_start_window = genre_params.get('drum_start_window', 4.0)
    drum_density_threshold = genre_params.get('drum_density_threshold', 0.5)

    print(
        f"[DrumStart] Ищем начало ударных с параметрами: window={drum_start_window}s, threshold={drum_density_threshold}/s")

    all_raw_events = sorted(raw_kick_times + raw_snare_times)
    drum_section_start = detect_drum_section_start(all_raw_events, drum_start_window, drum_density_threshold)

    filtered_kicks = [t for t in raw_kick_times if t >= drum_section_start]
    filtered_snares = [t for t in raw_snare_times if t >= drum_section_start]

    print(
        f"[DrumStart] Отфильтровано до начала ударной секции: {len(raw_kick_times)}->{len(filtered_kicks)} kicks, {len(raw_snare_times)}->{len(filtered_snares)} snares")

    max_hits_per_second = genre_params.get('max_hits_per_second', 4)
    min_note_distance = genre_params.get('min_note_distance', 0.05)
    pattern_style = genre_params.get('pattern_style', 'groove')

    print(
        f"[GrooveFilter] Применяем ограничения: max_hits={max_hits_per_second}/s, min_distance={min_note_distance}s, style={pattern_style}")

    final_kicks = apply_temporal_filter(sorted(filtered_kicks), min_note_distance)
    final_snares = apply_temporal_filter(sorted(filtered_snares), min_note_distance)

    print(
        f"[TemporalFilter] После фильтрации расстояния: {len(filtered_kicks)}->{len(final_kicks)} kicks, {len(filtered_snares)}->{len(final_snares)} snares")

    print(f"[PatternApply] Применяем паттерн стиль: {pattern_style}")
    grooved_kicks = apply_groove_pattern(final_kicks, pattern_style, bpm)
    grooved_snares = apply_groove_pattern(final_snares, pattern_style, bpm)

    print(
        f"[PatternApplied] После применения паттерна: {len(final_kicks)}->{len(grooved_kicks)} kicks, {len(final_snares)}->{len(grooved_snares)} snares")

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

    synced_kicks = sync_to_beats(grooved_kicks)
    synced_snares = sync_to_beats(grooved_snares)

    print(f"[DrumGen] После синхронизации: {len(synced_kicks)} kick, {len(synced_snares)} snare")

    if len(synced_kicks) + len(synced_snares) == 0:
        print("[DrumGen] Нет нот после синхронизации — используем грув-паттерн")

        synced_kicks = grooved_kicks
        synced_snares = grooved_snares

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
    print(f"   - Жанры: {unique_genres if unique_genres else 'не определены'}")
    print(f"   - BPM: {bpm}, Style: {pattern_style}")

    if track_info and track_info.get('success'):
        notes.append({
            "type": "TrackInfo",
            "title": track_info['title'],
            "artist": track_info['artist'],
            "genres": unique_genres,
            "album": track_info['album'],
            "year": track_info['year'],
            "time": -1
        })

    if len(notes) == 0:
        print("[DrumGen] ВНИМАНИЕ: Сгенерировано 0 нот!")

    return notes


def save_drums_notes(notes_data: List[Dict], song_path: str) -> bool:
    if not notes_data:
        print("[DrumGen] Нет данных нот для сохранения.")
        return False

    base_name = Path(song_path).stem
    song_folder = TEMP_UPLOADS_DIR / base_name
    notes_folder = song_folder / "notes"
    notes_folder.mkdir(parents=True, exist_ok=True)

    notes_path = notes_folder / f"{base_name}_drums.json"

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

        filtered_notes = [note for note in serializable if note.get('type') != 'TrackInfo']

        temp_path = notes_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_notes, f, ensure_ascii=False, indent=4)
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
    notes_path = TEMP_UPLOADS_DIR / base_name / "notes" / f"{base_name}_drums.json"

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