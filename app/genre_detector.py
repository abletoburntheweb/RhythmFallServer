# genre_detector.py
import requests
import base64
import time
from typing import Dict, List, Optional, Tuple


class SpotifyGenreDetector:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires_at = 0
        self.base_url = "https://api.spotify.com/v1"
        self.token_url = "https://accounts.spotify.com/api/token"

    def _get_access_token(self) -> Optional[str]:
        if self.access_token and time.time() < self.token_expires_at:
            return self.access_token


        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "grant_type": "client_credentials"
        }

        try:
            response = requests.post(self.token_url, headers=headers, data=data)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data["access_token"]

            self.token_expires_at = time.time() + token_data["expires_in"] - 300

            print(f"[Spotify] Получен новый access token")
            return self.access_token
        except requests.exceptions.RequestException as e:
            print(f"[Spotify] Ошибка получения токена: {e}")
            return None

    def search_track(self, artist: str, title: str) -> Optional[Dict]:
        token = self._get_access_token()
        if not token:
            return None

        headers = {
            "Authorization": f"Bearer {token}"
        }


        query = f"track:{title} artist:{artist}"
        params = {
            "q": query,
            "type": "track",
            "limit": 1
        }

        try:
            response = requests.get(f"{self.base_url}/search", headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            tracks = data.get("tracks", {}).get("items", [])

            if tracks:
                return tracks[0]
            else:
                print(f"[Spotify] Трек не найден: {artist} - {title}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[Spotify] Ошибка поиска трека: {e}")
            return None

    def get_artist_genres(self, artist_id: str) -> List[str]:
        token = self._get_access_token()
        if not token:
            return []

        headers = {
            "Authorization": f"Bearer {token}"
        }

        try:
            response = requests.get(f"{self.base_url}/artists/{artist_id}", headers=headers)
            response.raise_for_status()

            artist_data = response.json()
            return artist_data.get("genres", [])
        except requests.exceptions.RequestException as e:
            print(f"[Spotify] Ошибка получения жанров исполнителя: {e}")
            return []

    def get_track_genres(self, artist: str, title: str) -> List[str]:
        track_data = self.search_track(artist, title)
        if not track_data:
            return []


        artists = track_data.get("artists", [])
        if not artists:
            return []

        artist_id = artists[0].get("id")
        if not artist_id:
            return []


        genres = self.get_artist_genres(artist_id)
        print(f"[Spotify] Жанры для {artist} - {title}: {genres}")

        return genres



def detect_genres(artist: str, title: str) -> List[str]:
    detector = SpotifyGenreDetector(
        client_id="fc3d9f549993477091a8b1baa63766c5",
        client_secret="0d02fa1893944a5090362d4c2b74c838"
    )

    return detector.get_track_genres(artist, title)



if __name__ == "__main__":

    genres = detect_genres("The Beatles", "Hey Jude")
    print(f"Жанры: {genres}")