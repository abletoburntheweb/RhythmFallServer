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
                "sync_tolerance_multiplier": 1.0
            }
        }


GENRE_CONFIGS = load_genre_configs()


def get_genre_params(genres: List[str]) -> Dict:
    if not genres:
        return GENRE_CONFIGS.get("default", {})

    genres_lower = [g.lower() for g in genres]

    for genre in genres_lower:
        if genre in GENRE_CONFIGS:
            return GENRE_CONFIGS[genre]

    return GENRE_CONFIGS.get("default", {})


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


def calculate_onset_strength(y: np.ndarray, sr: int, hop_length: int = 512):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    return times, onset_env


def find_amplitude_at_time(y: np.ndarray, sr: int, time: float, window_ms: float = 10.0):
    sample_idx = int(time * sr)
    window_samples = int(window_ms / 1000.0 * sr)
    start = max(0, sample_idx - window_samples // 2)
    end = min(len(y), sample_idx + window_samples // 2)
    if end > start:
        segment = y[start:end]
        rms = np.sqrt(np.mean(segment ** 2))
        return rms
    return 0.0


def apply_frequency_filter(y: np.ndarray, sr: int, low_freq: float, high_freq: float):
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist

    if low < 0.001:
        low = 0.001
    if high > 0.999:
        high = 0.999

    if low >= high:
        return y

    b, a = librosa.butter(N=4, Wn=[low, high], btype='band')
    return scipy.signal.filtfilt(b, a, y)


def apply_amplitude_filter(hit_times: List[float], y: np.ndarray, sr: int, percentile_threshold: float = 30.0) -> List[
    float]:
    if not hit_times:
        return []

    amplitudes = [find_amplitude_at_time(y, sr, time) for time in hit_times]
    if not amplitudes:
        return []

    threshold = np.percentile(amplitudes, percentile_threshold)
    filtered_times = [hit_times[i] for i in range(len(hit_times)) if amplitudes[i] >= threshold]

    return filtered_times


def apply_time_clustering(hit_times: List[float], y: np.ndarray, sr: int, cluster_window: float = 0.05) -> List[float]:
    if not hit_times:
        return []

    hit_times = sorted(hit_times)
    if not hit_times:
        return []

    clusters = []
    current_cluster = [hit_times[0]]

    for time in hit_times[1:]:
        if time - current_cluster[-1] <= cluster_window:
            current_cluster.append(time)
        else:
            
            cluster_amplitudes = [find_amplitude_at_time(y, sr, t) for t in current_cluster]
            best_time = current_cluster[np.argmax(cluster_amplitudes)]
            clusters.append(best_time)
            current_cluster = [time]

    if current_cluster:
        cluster_amplitudes = [find_amplitude_at_time(y, sr, t) for t in current_cluster]
        best_time = current_cluster[np.argmax(cluster_amplitudes)]
        clusters.append(best_time)

    return clusters


def remove_simultaneous_hits(kick_times: List[float], snare_times: List[float], min_separation: float = 0.02) -> tuple[
    List[float], List[float]]:
    filtered_kick = []
    filtered_snare = []

    all_kick = set(kick_times)
    all_snare = set(snare_times)

    for kick_time in kick_times:
        
        has_close_snare = any(abs(kick_time - snare_time) < min_separation for snare_time in snare_times)
        if not has_close_snare:
            filtered_kick.append(kick_time)

    for snare_time in snare_times:
        
        has_close_kick = any(abs(snare_time - kick_time) < min_separation for kick_time in kick_times)
        if not has_close_kick:
            filtered_snare.append(snare_time)

    return filtered_kick, filtered_snare


def limit_note_density(times: List[float], min_interval: float = 0.08, max_notes_per_interval: int = 1) -> List[float]:
    if not times:
        return []

    times = sorted(times)
    filtered = []
    i = 0

    while i < len(times):
        current_time = times[i]
        filtered.append(current_time)

        
        j = i + 1
        while j < len(times) and times[j] < current_time + min_interval:
            j += 1

        i = j

    return filtered


def limit_notes_per_beat(times: List[float], beats: np.ndarray, max_notes_per_beat: int = 1) -> List[float]:
    if not times or len(beats) == 0:
        return times

    
    beat_groups = {}
    for time in times:
        
        closest_beat_idx = np.argmin(np.abs(beats - time))
        closest_beat_time = beats[closest_beat_idx]

        if closest_beat_time not in beat_groups:
            beat_groups[closest_beat_time] = []
        beat_groups[closest_beat_time].append(time)

    
    filtered_times = []
    for beat_time, group_times in beat_groups.items():
        if len(group_times) <= max_notes_per_beat:
            filtered_times.extend(group_times)
        else:
            
            amplitudes = [find_amplitude_at_time(y, sr, t) for t in group_times]
            top_indices = np.argsort(amplitudes)[-max_notes_per_beat:]
            for idx in top_indices:
                filtered_times.append(group_times[idx])

    return sorted(filtered_times)


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

    global y, sr  

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
        print(f"[DrumGen] Жанры из аудио: {track_info['genres']}")

    if use_filename_for_genres and not all_genres:
        if GENRE_DETECTION_AVAILABLE:
            filename_genres = detect_genres(Path(song_path).name, track_info)
            if filename_genres:
                all_genres.extend(filename_genres)
                print(f"[DrumGen] Жанры из названия файла: {filename_genres}")

    unique_genres = list(set([g for g in all_genres if g and g.lower() != 'unknown']))

    genre_params = {}
    if unique_genres:
        genre_params = get_genre_params(unique_genres)
        print(f"[DrumGen] Применены параметры для жанра: {unique_genres[0]}")
        print(f"[DrumGen] Параметры: {genre_params}")

        if 'sync_tolerance_multiplier' in genre_params:
            sync_tolerance *= genre_params['sync_tolerance_multiplier']
            print(f"[DrumGen] Sync tolerance изменен: {sync_tolerance:.2f}")

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

    
    if raw_kick_times and len(beats) > 0:
        raw_kick_times = limit_notes_per_beat(raw_kick_times, beats, max_notes_per_beat=1)
        print(f"[BeatLimit] После ограничения по битам: {len(raw_kick_times)} kick")

    if raw_snare_times and len(beats) > 0:
        raw_snare_times = limit_notes_per_beat(raw_snare_times, beats, max_notes_per_beat=1)
        print(f"[BeatLimit] После ограничения по битам: {len(raw_snare_times)} snare")

    
    if raw_kick_times:
        raw_kick_times = apply_amplitude_filter(raw_kick_times, y, sr,
                                                percentile_threshold=25.0 * genre_params.get(
                                                    'kick_sensitivity_multiplier', 1.0))
        print(f"[AmplitudeFilter] После фильтрации по амплитуде: {len(raw_kick_times)} kick")

    if raw_snare_times:
        raw_snare_times = apply_amplitude_filter(raw_snare_times, y, sr,
                                                 percentile_threshold=25.0 * genre_params.get(
                                                     'snare_sensitivity_multiplier', 1.0))
        print(f"[AmplitudeFilter] После фильтрации по амплитуде: {len(raw_snare_times)} snare")

    
    if raw_kick_times:
        raw_kick_times = apply_time_clustering(raw_kick_times, y, sr, cluster_window=0.03)
        print(f"[TimeCluster] После кластеризации: {len(raw_kick_times)} kick")

    if raw_snare_times:
        raw_snare_times = apply_time_clustering(raw_snare_times, y, sr, cluster_window=0.03)
        print(f"[TimeCluster] После кластеризации: {len(raw_snare_times)} snare")

    
    if raw_kick_times:
        raw_kick_times = limit_note_density(raw_kick_times, min_interval=0.08)
        print(f"[DensityLimit] После ограничения плотности: {len(raw_kick_times)} kick")

    if raw_snare_times:
        raw_snare_times = limit_note_density(raw_snare_times, min_interval=0.08)
        print(f"[DensityLimit] После ограничения плотности: {len(raw_snare_times)} snare")

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

    
    synced_kicks, synced_snares = remove_simultaneous_hits(synced_kicks, synced_snares, min_separation=0.02)

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

        available_lanes = [lane for lane in range(lanes) if
                           last_lane_usage.get(lane, -999) < adjusted_time - 0.05]  
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