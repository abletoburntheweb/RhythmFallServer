import json

import requests
import time
from typing import List, Dict, Optional
import musicbrainzngs
from pathlib import Path


class MultiSourceGenreDetector:
    def __init__(self, lastfm_api_key: str = "9f31a0bf16c699522d76e746ea2b5b90"):
        self.lastfm_api_key = lastfm_api_key

        self.base_urls = {
            'lastfm': "http://ws.audioscrobbler.com/2.0/",
            'musicbrainz': "https://musicbrainz.org/ws/2/"
        }

        musicbrainzngs.set_useragent("RhythmFall", "1.0", "abtw324@gmail.com")

    def search_lastfm(self, artist: str, title: str) -> Optional[dict]:
        if not self.lastfm_api_key:
            print("[LastFM] API ключ не предоставлен")
            return None

        params = {
            'method': 'track.getInfo',
            'api_key': self.lastfm_api_key,
            'artist': artist,
            'track': title,
            'format': 'json'
        }

        try:
            response = requests.get(self.base_urls['lastfm'], params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                print(f"[LastFM] Ошибка: {data.get('message', 'Unknown error')}")
                return None

            return data.get('track')
        except Exception as e:
            print(f"[LastFM] Ошибка поиска: {e}")
            return None

    def get_lastfm_genres(self, artist: str, title: str) -> List[str]:
        track_data = self.search_lastfm(artist, title)
        if not track_data:
            return []

        genres = []

        if 'toptags' in track_data and 'tag' in track_data['toptags']:
            for tag in track_data['toptags']['tag']:
                if 'name' in tag:
                    genres.append(tag['name'].lower())

        if not genres and 'artist' in track_data:
            artist_name = track_data['artist'].get('name', artist)
            artist_genres = self.get_lastfm_artist_genres(artist_name)
            genres.extend(artist_genres)

        unique_genres = list(set(genres))[:10]
        print(f"[LastFM] Жанры: {unique_genres}")
        return unique_genres

    def get_lastfm_artist_genres(self, artist_name: str) -> List[str]:
        if not self.lastfm_api_key:
            return []

        params = {
            'method': 'artist.getTopTags',
            'api_key': self.lastfm_api_key,
            'artist': artist_name,
            'format': 'json'
        }

        try:
            response = requests.get(self.base_urls['lastfm'], params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'toptags' in data and 'tag' in data['toptags']:
                genres = []
                for tag in data['toptags']['tag']:
                    if 'name' in tag:
                        genres.append(tag['name'].lower())
                return genres[:10]

        except Exception as e:
            print(f"[LastFM] Ошибка получения тегов исполнителя: {e}")

        return []

    def search_musicbrainz(self, artist: str, title: str) -> Optional[Dict]:
        try:
            results = musicbrainzngs.search_recordings(query=f"{title} AND {artist}", limit=3)

            if 'recording-list' in results and results['recording-list']:
                return results['recording-list'][0]
            return None
        except Exception as e:
            print(f"[MusicBrainz] Ошибка поиска: {e}")
            return None

    def get_musicbrainz_genres(self, artist: str, title: str) -> List[str]:
        recording = self.search_musicbrainz(artist, title)
        if not recording:
            return []

        genres = []

        if 'artist-credit' in recording and recording['artist-credit']:
            artist_credit = recording['artist-credit'][0]
            if 'artist' in artist_credit:
                artist_mbid = artist_credit['artist']['id']

                try:
                    artist_data = musicbrainzngs.get_artist_by_id(artist_mbid, includes=['tags'])
                    if 'artist' in artist_data and 'tags' in artist_data['artist']:
                        for tag in artist_data['artist']['tags']:
                            genres.append(tag['name'].lower())
                except Exception as e:
                    print(f"[MusicBrainz] Ошибка получения тегов исполнителя: {e}")

        if 'tag-list' in recording:
            for tag in recording['tag-list']:
                if 'name' in tag:
                    genres.append(tag['name'].lower())

        unique_genres = list(set(genres))[:10]
        print(f"[MusicBrainz] Жанры: {unique_genres}")
        return unique_genres

    def detect_all_genres(self, artist: str, title: str) -> Dict[str, List[str]]:
        print(f"🔍 Поиск жанров для: {artist} - {title}")

        results = {}

        results['musicbrainz'] = self.get_musicbrainz_genres(artist, title)
        time.sleep(0.5)

        results['lastfm'] = self.get_lastfm_genres(artist, title)
        time.sleep(0.5)

        results['discogs'] = []

        all_genres = []
        for source, genres in results.items():
            all_genres.extend(genres)

        unique_genres = list(set(all_genres))

        print(f"📊 Итоговые жанры: {unique_genres}")
        print(f"📊 Жанры по источникам:")
        for source, genres in results.items():
            if genres:
                print(f"   {source.capitalize()}: {genres}")

        return {
            'all_genres': unique_genres,
            'by_source': results
        }


def load_music_genres() -> List[str]:
    try:
        config_path = Path(__file__).parent / "music_genres.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('music_genres', [])
    except FileNotFoundError:
        print(f"[GenreDetector] Файл music_genres.json не найден")
        return []
    except Exception as e:
        print(f"[GenreDetector] Ошибка загрузки жанров из JSON: {e}")
        return []


def detect_genres(artist: str, title: str) -> List[str]:
    music_genres = load_music_genres()

    detector = MultiSourceGenreDetector()
    results = detector.detect_all_genres(artist, title)

    filtered_genres = []
    for genre in results['all_genres']:
        if genre.lower() in music_genres:
            filtered_genres.append(genre.lower())

    if not filtered_genres:
        if results['by_source']['musicbrainz']:
            return results['by_source']['musicbrainz'][:3]
        for source in ['lastfm']:
            if results['by_source'][source]:
                return results['by_source'][source][:3]

    return filtered_genres[:5]