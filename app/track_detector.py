# app/track_detector.py
import os
import json
from pathlib import Path
from typing import Dict, Optional
import tempfile
import subprocess

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

ACOUSTID_API_KEY = "0nUW6lXEvg"

if MUSICBRAINZ_AVAILABLE:
    musicbrainzngs.set_useragent("RhythmFall", "1.0", "abtw324@gmail.com")


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


def detect_track_by_audio(audio_path: str) -> Optional[Dict]:
    if not REQUESTS_AVAILABLE:
        print("[TrackDetector] Requests не доступен")
        return None

    try:
        print(f"[AcoustID] Анализ трека: {audio_path}")

        duration, fingerprint = fingerprint_audio(audio_path)

        if not fingerprint or not duration:
            print("[AcoustID] Не удалось получить фингерпринт")
            return None

        print(f"[AcoustID] Fingerprint получен (duration: {duration}s)")

        url = "https://api.acoustid.org/v2/lookup"
        data = {
            'format': 'json',
            'client': ACOUSTID_API_KEY,
            'duration': int(duration),
            'fingerprint': fingerprint,
            'meta': 'recordings releasegroups releases tracks usermeta'
        }

        response = requests.post(url, data=data, timeout=30)
        response.raise_for_status()

        result = response.json()

        if not result or 'results' not in result:
            print("[AcoustID] Нет результатов")
            return None

        if not result['results']:
            print("[AcoustID] Результаты пусты")
            return None

        best_result = result['results'][0]

        if 'recordings' not in best_result or not best_result['recordings']:
            print("[AcoustID] Нет информации о записи")
            return None

        recording = best_result['recordings'][0]

        track_info = {
            'title': recording.get('title', 'Unknown'),
            'artist': 'Unknown',
            'album': 'Unknown',
            'year': None,
            'genres': [],
            'acoustid_id': best_result.get('id'),
            'score': best_result.get('score', 0),
            'duration': duration,
            'success': True
        }

        if 'artists' in recording and recording['artists']:
            first_artist = recording['artists'][0]
            track_info['artist'] = first_artist.get('name', 'Unknown')

        if 'releases' in recording and recording['releases']:
            first_release = recording['releases'][0]
            track_info['album'] = first_release.get('title', 'Unknown')

            if 'date' in first_release and 'year' in first_release['date']:
                track_info['year'] = first_release['date']['year']

        if 'tags' in recording:
            tags = []
            for tag in recording['tags']:
                if 'name' in tag:
                    tags.append(tag['name'].lower())
            track_info['genres'] = tags[:10]

        print(
            f"[AcoustID] Найден трек: {track_info['artist']} - {track_info['title']} (score: {track_info['score']:.2f})")
        if track_info['genres']:
            print(f"[AcoustID] Жанры: {', '.join(track_info['genres'])}")

        return track_info

    except requests.exceptions.RequestException as e:
        print(f"[AcoustID] Ошибка HTTP запроса: {e}")
        return None
    except Exception as e:
        print(f"[AcoustID] Ошибка при определении трека: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_track_details_from_musicbrainz(acoustid_id: str) -> Optional[Dict]:
    if not MUSICBRAINZ_AVAILABLE:
        print("[MusicBrainz] MusicBrainz не доступен")
        return None

    try:
        print(f"[MusicBrainz] Запрос деталей для AcoustID: {acoustid_id}")

        recordings = musicbrainzngs.get_recordings_by_acoustid(acoustid_id)

        if not recordings or 'recordings' not in recordings:
            print("[MusicBrainz] Нет записей для этого AcoustID")
            return None

        if not recordings['recordings']:
            print("[MusicBrainz] Пустой список записей")
            return None

        recording = recordings['recordings'][0]

        details = {
            'title': recording.get('title', 'Unknown'),
            'artist': 'Unknown',
            'album': 'Unknown',
            'year': None,
            'genres': [],
            'primary_type': None,
            'secondary_types': []
        }

        if 'artist-credit' in recording:
            artist_credit = recording['artist-credit']
            if artist_credit and isinstance(artist_credit, list) and len(artist_credit) > 0:
                details['artist'] = artist_credit[0].get('artist', {}).get('name', 'Unknown')
            elif isinstance(artist_credit, str):
                details['artist'] = artist_credit
        elif 'artists' in recording and recording['artists']:
            details['artist'] = recording['artists'][0].get('name', 'Unknown')

        if 'releases' in recording and recording['releases']:
            first_release = recording['releases'][0]
            details['album'] = first_release.get('title', 'Unknown')

            if 'date' in first_release:
                date_info = first_release['date']
                if isinstance(date_info, dict) and 'year' in date_info:
                    details['year'] = date_info['year']
                elif isinstance(date_info, str) and len(date_info) >= 4:
                    try:
                        details['year'] = int(date_info[:4])
                    except ValueError:
                        pass

        if 'tags' in recording:
            tags = []
            for tag in recording['tags']:
                if 'name' in tag:
                    tags.append(tag['name'].lower())
            details['genres'] = tags[:10] 

        if 'release-group' in recording and recording['release-group']:
            rg = recording['release-group']
            if 'primary-type' in rg:
                details['primary_type'] = rg['primary-type']
            if 'secondary-types' in rg:
                details['secondary_types'] = rg['secondary-types']

        print(f"[MusicBrainz] Детали: {details['artist']} - {details['title']}")
        if details['genres']:
            print(f"[MusicBrainz] Жанры: {', '.join(details['genres'])}")

        return details

    except Exception as e:
        print(f"[MusicBrainz] Ошибка получения деталей: {e}")
        import traceback
        traceback.print_exc()
        return None


def identify_track(audio_path: str) -> Dict:
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

    acoustid_result = detect_track_by_audio(audio_path)

    if acoustid_result:
        result.update(acoustid_result)
        result['success'] = True

        if acoustid_result.get('acoustid_id'):
            mb_details = get_track_details_from_musicbrainz(acoustid_result['acoustid_id'])
            if mb_details:
                for key, value in mb_details.items():
                    if key not in ['acoustid_id', 'score', 'duration'] or not result.get(key):
                        if value and (not result.get(key) or result[key] == 'Unknown'):
                            result[key] = value
                        elif key == 'genres' and value:
                            result['genres'].extend(value)
                            result['genres'] = list(set(result['genres']))
    else:
        print("[TrackDetector] Не удалось идентифицировать трек")

    result['genres'] = [g for g in result['genres'] if g and g != 'unknown']

    print(f"✅ Идентификация завершена: {result['artist']} - {result['title']}")
    if result['genres']:
        print(f"🎵 Жанры: {', '.join(result['genres'])}")

    return result


def test_track_detection():
    test_file = "test_audio.mp3"
    if os.path.exists(test_file):
        info = identify_track(test_file)
        print(f"Тестовая информация: {json.dumps(info, indent=2, ensure_ascii=False)}")
    else:
        print("Тестовый файл не найден")


if __name__ == "__main__":
    test_track_detection()