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

ACOUSTID_API_KEY = "PUlkAEkjhm"

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
        print(f"[AcoustID] Fingerprint length: {len(fingerprint) if fingerprint else 0}")
        print(f"[AcoustID] Duration type: {type(duration)}, value: {duration}")

        url = "https://api.acoustid.org/v2/lookup"
        data = {
            'format': 'json',
            'client': ACOUSTID_API_KEY,
            'duration': int(duration),
            'fingerprint': fingerprint,
            'meta': 'recordings releasegroups releases tracks usermeta'
        }

        print(f"[AcoustID] Request data: {data}")

        response = requests.post(url, data=data, timeout=30)
        print(f"[AcoustID] Response status: {response.status_code}")
        print(f"[AcoustID] Response text: {response.text}")
        response.raise_for_status()

        result = response.json()
        print(f"[AcoustID] API response: {result}")

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

    except requests.exceptions.HTTPError as e:
        print(f"[AcoustID] HTTP ошибка: {e}")
        print(f"[AcoustID] Response content: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[AcoustID] Ошибка HTTP запроса: {e}")
        return None
    except Exception as e:
        print(f"[AcoustID] Ошибка при определении трека: {e}")
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