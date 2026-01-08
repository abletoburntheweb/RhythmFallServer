# app/track_detector.py
import os
import json
from pathlib import Path
from typing import Dict, Optional
import tempfile
import subprocess
import re
import difflib

try:
    import requests

    REQUESTS_AVAILABLE = True
    print("[TrackDetector] Requests доступен")
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[TrackDetector] Requests не установлен")

try:
    import musicbrainzngs

    MUSICBRAINZ_AVAILABLE = True
    print("[TrackDetector] MusicBrainz доступен")
except ImportError:
    MUSICBRAINZ_AVAILABLE = False
    print("[TrackDetector] MusicBrainz не установлен (pip install musicbrainzngs)")


def _load_config(config_path: str = None) -> Dict:
    if config_path is None:
        module_dir = Path(__file__).parent
        config_file = module_dir / "config.json"

        if config_file.exists():
            path_to_load = str(config_file)
            print(f"[TrackDetector] Используем конфиг из: {config_file}")
        else:
            print(f"[TrackDetector] Файл конфигурации {config_file} не найден. Используются значения по умолчанию.")
            return {}
    else:
        path_to_load = config_path
        if not Path(path_to_load).exists():
            print(f"[TrackDetector] Файл конфигурации {path_to_load} не найден. Используются значения по умолчанию.")
            return {}

    try:
        with open(path_to_load, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[TrackDetector] Ошибка парсинга JSON в {path_to_load}: {e}. Используются значения по умолчанию.")
        return {}
    except Exception as e:
        print(f"[TrackDetector] Ошибка при чтении {path_to_load}: {e}. Используются значения по умолчанию.")
        return {}


CONFIG = _load_config()
ACOUSTID_API_KEY = CONFIG.get("apis", {}).get("acoustid", {}).get("api_key")
ACOUSTID_BASE_URL = CONFIG.get("apis", {}).get("acoustid", {}).get("base_url", "https://api.acoustid.org/v2/")
MB_APP_NAME = CONFIG.get("apis", {}).get("musicbrainz", {}).get("app_name", "RhythmFall")
MB_VERSION = CONFIG.get("apis", {}).get("musicbrainz", {}).get("version", "1.0")
MB_CONTACT = CONFIG.get("apis", {}).get("musicbrainz", {}).get("contact", "abtw324@gmail.com")

if MUSICBRAINZ_AVAILABLE:
    musicbrainzngs.set_useragent(MB_APP_NAME, MB_VERSION, MB_CONTACT)


def fingerprint_audio(audio_path: str) -> tuple[Optional[float], Optional[str]]:
    try:
        result = subprocess.run(['fpcalc', '-length', '30', audio_path],
                                capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            duration = None
            fingerprint = None

            for line in lines:
                if line.startswith('DURATION='):
                    try:
                        duration = float(line.split('=')[1])
                    except ValueError:
                        continue
                elif line.startswith('FINGERPRINT='):
                    fingerprint = line.split('=')[1]

            if duration and fingerprint:
                return duration, fingerprint
            else:
                print("[TrackDetector] Не удалось извлечь duration или fingerprint")
                return None, None
        else:
            print(f"[TrackDetector] fpcalc ошибка: {result.stderr}")
            return None, None

    except subprocess.TimeoutExpired:
        print("[TrackDetector] Таймаут при генерации фингерпринта")
        return None, None
    except Exception as e:
        print(f"[TrackDetector] Ошибка при генерации фингерпринта: {e}")
        return None, None


def extract_artist_title_from_filename(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    parts = stem.split(' - ', 1)
    if len(parts) == 2:
        part1, part2 = parts[0].strip(), parts[1].strip()
        potential_artist = part1
        potential_title = part2
        return potential_artist, potential_title
    else:
        return "Unknown", stem


def find_closest_match_from_local_metadata(audio_path: str, local_metadata_path: str) -> Optional[Dict]:
    if not Path(local_metadata_path).exists():
        print(f"[TrackDetector] Локальный файл метаданных не найден: {local_metadata_path}")
        return None

    try:
        with open(local_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        print(f"[TrackDetector] Ошибка парсинга JSON в локальном файле метаданных: {local_metadata_path}")
        return None

    filename_no_ext = Path(audio_path).stem.lower()
    potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
    potential_artist_lower = potential_artist.lower()
    potential_title_lower = potential_title.lower()

    print(f"[TrackDetector] Ищем совпадения для: artist='{potential_artist}', title='{potential_title}' в {local_metadata_path}")

    known_artists = set()
    known_titles = set()
    lookup_dict = {}

    for path_key, meta in metadata.items():
        artist = meta.get('artist', 'Неизвестен').lower()
        title = meta.get('title', 'Без названия').lower()
        if artist != 'неизвестен' or title != 'без названия':
            known_artists.add(artist)
            known_titles.add(title)
            lookup_key = f"{artist} - {title}"
            if lookup_key not in lookup_dict:
                lookup_dict[lookup_key] = meta
            lookup_key_reverse = f"{title} - {artist}"
            if lookup_key_reverse not in lookup_dict:
                lookup_dict[lookup_key_reverse] = meta

    closest_artist = None
    closest_title = None

    if potential_artist != "Unknown":
        closest_artist_matches = difflib.get_close_matches(potential_artist_lower, known_artists, n=1, cutoff=0.3)
        if closest_artist_matches:
            closest_artist = closest_artist_matches[0]
            print(f"[TrackDetector] Найден ближайший артист: {closest_artist} (из {potential_artist_lower})")

    if potential_title != "Unknown":
        closest_title_matches = difflib.get_close_matches(potential_title_lower, known_titles, n=1, cutoff=0.3)
        if closest_title_matches:
            closest_title = closest_title_matches[0]
            print(f"[TrackDetector] Найдено ближайшее название: {closest_title} (из {potential_title_lower})")

    if closest_artist and closest_title:
        lookup_key1 = f"{closest_artist} - {closest_title}"
        lookup_key2 = f"{closest_title} - {closest_artist}"

        found_meta = lookup_dict.get(lookup_key1) or lookup_dict.get(lookup_key2)

        if found_meta:
            print(f"[TrackDetector] Найдено совпадение в локальных метаданных: {found_meta.get('artist')} - {found_meta.get('title')}")
            return {
                'title': found_meta.get('title', 'Unknown'),
                'artist': found_meta.get('artist', 'Unknown'),
                'album': found_meta.get('album', 'Unknown'),
                'year': found_meta.get('year', 'Unknown'),
                'genres': found_meta.get('genres', []),
                'acoustid_id': None,
                'score': 0.5,
                'duration': None,
                'success': True
            }

    print("[TrackDetector] Совпадений в локальных метаданных не найдено.")
    return None


def detect_track_by_audio(audio_path: str, local_metadata_path: Optional[str] = None) -> Optional[Dict]:
    if not REQUESTS_AVAILABLE:
        print("[TrackDetector] Requests не доступен")
        if local_metadata_path:
            result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
            if result:
                return result
        potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
        return {
            'title': potential_title,
            'artist': potential_artist,
            'album': 'Unknown',
            'year': 'Unknown',
            'genres': [],
            'acoustid_id': None,
            'score': 0.1,
            'duration': None,
            'success': True
        }

    if not ACOUSTID_API_KEY:
        print("[TrackDetector] AcoustID API ключ не предоставлен в конфиге")
        if local_metadata_path:
            result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
            if result:
                return result
        potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
        return {
            'title': potential_title,
            'artist': potential_artist,
            'album': 'Unknown',
            'year': 'Unknown',
            'genres': [],
            'acoustid_id': None,
            'score': 0.1,
            'duration': None,
            'success': True
        }

    try:
        print(f"[AcoustID] Анализ трека: {audio_path}")

        duration, fingerprint = fingerprint_audio(audio_path)

        if not fingerprint or not duration:
            print("[AcoustID] Не удалось получить фингерпринт")
            if local_metadata_path:
                result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
                if result:
                    return result
            potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
            return {
                'title': potential_title,
                'artist': potential_artist,
                'album': 'Unknown',
                'year': 'Unknown',
                'genres': [],
                'acoustid_id': None,
                'score': 0.1,
                'duration': None,
                'success': True
            }

        print(f"[AcoustID] Fingerprint получен (duration: {duration}s)")
        print(f"[AcoustID] Fingerprint length: {len(fingerprint) if fingerprint else 0}")
        print(f"[AcoustID] Duration type: {type(duration)}, value: {duration}")

        url = f"{ACOUSTID_BASE_URL}lookup"
        data = {
            'format': 'json',
            'client': ACOUSTID_API_KEY,
            'duration': int(duration),
            'fingerprint': fingerprint,
            'meta': 'recordings releasegroups releases tracks usermeta'
        }

        print(f"[AcoustID] Request to {url} with  {data}")

        response = requests.post(url, data=data, timeout=30)
        print(f"[AcoustID] Response status: {response.status_code}")
        response.raise_for_status()

        result = response.json()
        print(
            f"[AcoustID] API response: {{'results_count': {len(result.get('results', []))}, 'status': '{result.get('status')}'}}")

        if not result or 'results' not in result:
            print("[AcoustID] Нет результатов в ответе")
            if local_metadata_path:
                result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
                if result:
                    return result
            potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
            return {
                'title': potential_title,
                'artist': potential_artist,
                'album': 'Unknown',
                'year': 'Unknown',
                'genres': [],
                'acoustid_id': None,
                'score': 0.1,
                'duration': None,
                'success': True
            }

        if not result['results']:
            print("[AcoustID] Результаты AcoustID пусты")
            if local_metadata_path:
                result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
                if result:
                    return result
            potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
            return {
                'title': potential_title,
                'artist': potential_artist,
                'album': 'Unknown',
                'year': 'Unknown',
                'genres': [],
                'acoustid_id': None,
                'score': 0.1,
                'duration': None,
                'success': True
            }

        filename_no_ext = Path(audio_path).stem.lower()
        filename_words = set(re.split(r'[ -_]+', filename_no_ext.lower()))
        filename_words.discard('cutted')
        filename_words.discard('cut')

        potential_results = result['results']
        best_result = None
        best_combined_score = -1

        for res in potential_results:
            if 'recordings' in res and res['recordings']:
                recording = res['recordings'][0]

                artist_name = 'Unknown'
                title_name = 'Unknown'

                if 'artists' in recording and recording['artists']:
                    first_artist = recording['artists'][0]
                    if isinstance(first_artist, dict) and 'name' in first_artist:
                        artist_name = first_artist['name']
                    elif isinstance(first_artist, str):
                        artist_name = first_artist

                if 'title' in recording and recording['title']:
                    title_name = recording['title']

                result_text = f"{artist_name} {title_name}".lower()
                result_words = set(re.split(r'[ -_]+', result_text))

                match_count = len(filename_words.intersection(result_words))
                total_filename_words = len(filename_words)

                match_threshold = 0.5
                sufficient_match = total_filename_words > 0 and (match_count / total_filename_words) >= match_threshold

                current_score = res.get('score', 0)

                if sufficient_match:
                    match_factor = match_count / total_filename_words if total_filename_words > 0 else 0
                    combined_score = current_score + match_factor * 0.1

                    if combined_score > best_combined_score:
                        best_result = res
                        best_combined_score = combined_score
                        print(
                            f"[AcoustID] Найдено лучшее совпадение: {artist_name} - {title_name}, score: {current_score:.2f}, match_words: {match_count}/{total_filename_words}, combined: {combined_score:.3f}")

        if not best_result:
            print(
                "[AcoustID] Ни один результат не прошёл порог совпадения слов с именем файла. Выбираем по наивысшему score.")
            for res in potential_results:
                current_score = res.get('score', 0)
                if current_score > best_combined_score:
                    best_result = res
                    best_combined_score = current_score
            if best_result:
                print(f"[AcoustID] Выбран результат с наивысшим score: {best_result.get('score', 0):.2f}")
            else:
                print("[AcoustID] Ни один результат не найден.")
                if local_metadata_path:
                    result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
                    if result:
                        return result
                potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
                return {
                    'title': potential_title,
                    'artist': potential_artist,
                    'album': 'Unknown',
                    'year': 'Unknown',
                    'genres': [],
                    'acoustid_id': None,
                    'score': 0.1,
                    'duration': None,
                    'success': True
                }


        if 'recordings' not in best_result or not best_result['recordings']:
            print("[AcoustID] Нет информации о записи в лучшем результате")
            return None

        track_info = {
            'title': 'Unknown',
            'artist': 'Unknown',
            'album': 'Unknown',
            'year': None,
            'genres': [],
            'acoustid_id': best_result.get('id'),
            'score': best_result.get('score', 0),
            'duration': duration,
            'success': True
        }


        for recording in best_result['recordings']:
            if track_info['artist'] == 'Unknown' and 'artists' in recording and recording['artists']:
                first_artist = recording['artists'][0]
                if isinstance(first_artist, dict) and 'name' in first_artist:
                    track_info['artist'] = first_artist['name']
                elif isinstance(first_artist, str):
                    track_info['artist'] = first_artist
            if track_info['title'] == 'Unknown' and 'title' in recording and recording['title']:
                track_info['title'] = recording['title']

            if 'releasegroups' in recording and recording['releasegroups']:
                for releasegroup in recording['releasegroups']:
                    if track_info['artist'] == 'Unknown' and 'artists' in releasegroup and releasegroup['artists']:
                        first_artist = releasegroup['artists'][0]
                        if isinstance(first_artist, dict) and 'name' in first_artist:
                            track_info['artist'] = first_artist['name']
                        elif isinstance(first_artist, str):
                            track_info['artist'] = first_artist
                    if track_info['title'] == 'Unknown' and 'title' in releasegroup and releasegroup['title']:
                        track_info['title'] = releasegroup['title']
                    if 'releases' in releasegroup and releasegroup['releases'] and track_info['year'] is None:
                        first_release = releasegroup['releases'][0]
                        if 'date' in first_release and 'year' in first_release['date']:
                            track_info['year'] = first_release['date']['year']
                        if track_info['album'] == 'Unknown' and 'title' in first_release and first_release['title']:
                            track_info['album'] = first_release['title']

            if track_info['artist'] != 'Unknown' and track_info['title'] != 'Unknown':
                break

        for recording in best_result['recordings']:
            if 'tags' in recording:
                tags = []
                for tag in recording['tags']:
                    if 'name' in tag:
                        tags.append(tag['name'].lower())
                track_info['genres'] = tags[:10]
                break

        print(
            f"[AcoustID] Найден трек: {track_info['artist']} - {track_info['title']} (score: {track_info['score']:.2f})")
        if track_info['genres']:
            print(f"[AcoustID] Жанры: {', '.join(track_info['genres'])}")

        return track_info

    except requests.exceptions.HTTPError as e:
        print(f"[AcoustID] HTTP ошибка: {e}")
        print(f"[AcoustID] Response content: {e.response.text}")
        if local_metadata_path:
            result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
            if result:
                return result
        potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
        return {
            'title': potential_title,
            'artist': potential_artist,
            'album': 'Unknown',
            'year': 'Unknown',
            'genres': [],
            'acoustid_id': None,
            'score': 0.1,
            'duration': None,
            'success': True
        }
    except requests.exceptions.RequestException as e:
        print(f"[AcoustID] Ошибка HTTP запроса: {e}")
        if local_metadata_path:
            result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
            if result:
                return result
        potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
        return {
            'title': potential_title,
            'artist': potential_artist,
            'album': 'Unknown',
            'year': 'Unknown',
            'genres': [],
            'acoustid_id': None,
            'score': 0.1,
            'duration': None,
            'success': True
        }
    except Exception as e:
        print(f"[AcoustID] Ошибка при определении трека: {e}")
        import traceback
        traceback.print_exc()
        if local_metadata_path:
            result = find_closest_match_from_local_metadata(audio_path, local_metadata_path)
            if result:
                return result
        potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
        return {
            'title': potential_title,
            'artist': potential_artist,
            'album': 'Unknown',
            'year': 'Unknown',
            'genres': [],
            'acoustid_id': None,
            'score': 0.1,
            'duration': None,
            'success': True
        }


def identify_track(audio_path: str, local_metadata_path: Optional[str] = None) -> Dict:
    print(f"🔍 Идентификация трека: {audio_path}")

    result = {
        'title': 'Unknown',
        'artist': 'Unknown',
        'album': 'Unknown',
        'year': None,
        'genres': [],
        'primary_type': None,
        'secondary_types': [],
        'acoustid_id': None,
        'score': 0,
        'duration': None,
        'success': False
    }

    detection_result = detect_track_by_audio(audio_path, local_metadata_path)

    if detection_result:
        result.update(detection_result)
        result['success'] = detection_result.get('score', 0) > 0
    else:
        print("[TrackDetector] Не удалось идентифицировать трек (AcoustID и локальный поиск не дали результата)")
        potential_artist, potential_title = extract_artist_title_from_filename(audio_path)
        result['artist'] = potential_artist
        result['title'] = potential_title
        result['score'] = 0.1
        result['success'] = True

    result['genres'] = [g for g in result['genres'] if g and g != 'unknown']

    print(f"✅ Идентификация завершена: {result['artist']} - {result['title']}")
    if result['genres']:
        print(f"🎵 Жанры: {', '.join(result['genres'])}")

    return result
