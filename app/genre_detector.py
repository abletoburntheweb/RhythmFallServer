import requests
import time
from typing import List, Dict, Optional
import musicbrainzngs


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


def detect_genres(artist: str, title: str) -> List[str]:
    music_genres = [
        'pop', 'rock', 'hip hop', 'rap', 'rnb', 'r&b', 'electronic', 'dance',
        'techno', 'house', 'trance', 'dubstep', 'trap', 'reggae', 'ska',
        'punk', 'metal', 'heavy metal', 'death metal', 'black metal', 'folk',
        'country', 'jazz', 'blues', 'soul', 'funk', 'disco', 'latin', 'reggaeton',
        'k-pop', 'kpop', 'korean pop', 'j-pop', 'jpop', 'anime', 'ost', 'soundtrack',
        'classical', 'ambient', 'chillout', 'lofi', 'indie', 'alternative', 'punk rock',
        'grunge', 'synthwave', 'retro', 'synthpop', 'new wave', 'industrial',
        'gothic', 'darkwave', 'breakbeat', 'drum and bass', 'dnb', 'hardcore',
        'hardstyle', 'trance', 'progressive house', 'deep house', 'minimal techno',
        'acid jazz', 'bossa nova', 'salsa', 'merengue', 'cumbia', 'flamenco',
        'gospel', 'guitar', 'piano', 'orchestral', 'symphonic', 'choir', 'acoustic',
        'experimental', 'avant-garde', 'noise', 'grime', 'uk garage', '2-step',
        'downtempo', 'trip hop', 'nu jazz', 'broken beat', 'jungle', 'ragga jungle',
        'drone', 'post-rock', 'math rock', 'emo', 'post-hardcore', 'screamo',
        'metalcore', 'deathcore', 'mathcore', 'southern rock', 'blues rock',
        'psychedelic', 'stoner rock', 'space rock', 'krautrock', 'shoegaze',
        'dream pop', 'art rock', 'prog rock', 'prog metal', 'symphonic metal',
        'power metal', 'folk metal', 'celtic', 'bluegrass', 'country rock',
        'surf rock', 'garage rock', 'psychobilly', 'rockabilly', 'swing',
        'big band', 'bebop', 'free jazz', 'fusion', 'smooth jazz', 'acid jazz',
        'afrobeat', 'afropop', 'world music', 'ethnic', 'tribal', 'medieval',
        'renaissance', 'baroque', 'classical period', 'romantic era', 'modern classical',
        'contemporary classical', 'minimalism', 'serialism', 'electro', 'electronica',
        'idm', 'glitch', 'chip tune', 'video game music', 'chiptune', 'synthesizer',
        'vocal', 'a cappella', 'barbershop', 'doom metal', 'sludge metal', 'stoner metal',
        'folk punk', 'anti-folk', 'neofolk', 'dark folk', 'neoclassical', 'dark ambient',
        'ritual ambient', 'martial industrial', 'power electronics', 'noise rock',
        'post-punk', 'new romantic', 'coldwave', 'ethereal wave', 'dark wave',
        'ebm', 'aggrotech', 'futurepop', 'synthpop', 'indietronica', 'skate punk',
        'thrash metal', 'speed metal', 'blackened death metal', 'symphonic black metal',
        'melodic death metal', 'technical death metal', 'brutal death metal',
        'groove metal', 'nu metal', 'rap rock', 'rap metal', 'alternative metal',
        'funk metal', 'rapcore', 'crunk', 'crunkcore', 'hyphy', 'snap music',
        'bounce music', 'trap music', 'drill', 'cloud rap', 'chillwave', 'witch house',
        'dub', 'dub techno', 'minimal', 'microhouse', 'tech house', 'deep house',
        'future house', 'tropical house', 'melodic dubstep', 'brostep', 'riddim',
        'chillstep', 'wonky', 'footwork', 'juke', 'ghetto house', 'gabber',
        'happy hardcore', 'uk hard house', 'breakcore', 'nintendocore', 'nerdcore',
        'chiptune', 'bitpop', 'game boy music', 'nintendo', 'sega', 'playstation',
        'xbox', 'arcade', 'retro gaming', 'anime rock', 'j-rock', 'visual kei',
        'japanese rock', 'korean rock', 'c-pop', 'mandopop', 'cantopop', 'hk-pop',
        'manilla sound', 'pinoy rock', 'pinoy pop', 'filmi', 'bollywood', 'tollywood',
        'punjabi', 'desi', 'bhangra', 'qawwali', 'sufi', 'arabic', 'persian',
        'turkish', 'greek', 'balkan', 'klezmer', 'roma', 'flamenco', 'andalusian',
        'north african', 'west african', 'afrobeat', 'afrofunk', 'afropop', 'highlife',
        'juju', 'makossa', 'soukous', 'raï', 'chaabi', 'malouf', 'tarab', 'fado',
        'bossa nova', 'samba', 'forró', 'frevo', 'maracatu', 'axé', 'carimbó',
        'candombe', 'milonga', 'tango', 'nuevo tango', 'bandoneón', 'chamamé',
        'cumbia', 'vallenato', 'porro', 'guaracha', 'son', 'mariachi', 'ranchera',
        'norteño', 'banda', 'corrido', 'bolero', 'trova', 'salsa', 'salsa romántica',
        'timba', 'merengue', 'bachata', 'reggaeton', 'dembow', 'latin trap', 'urbano',
        'regional mexican', 'tejano', 'conjunto', 'chicago house', 'detroit techno',
        'acid house', 'rave', 'breakbeat', 'big beat', 'breaks', 'nu breaks',
        'breakcore', 'russian breakcore', 'norsk bølge', 'kwaito', 'shangaan electro',
        'bassline', 'speed garage', '2-step garage', 'grime', 'bashment', 'ragga',
        'uk funky', 'electro swing', 'gypsy jazz', 'hot club jazz', 'western swing',
        'cowboy', 'cowpunk', 'alt-country', 'country rap', 'country rock', 'americana',
        'blue-eyed soul', 'northern soul', 'motown', 'urban contemporary', 'quiet storm',
        'contemporary r&b', 'neo soul', 'gospel rap', 'christian hip hop', 'ccm',
        'worship', 'praise', 'christian rock', 'christian metal', 'christian punk',
        'christian alternative', 'christian pop', 'gospel blues', 'spirituals',
        'contemporary christian', 'christian ska', 'christian reggae', 'christian punk',
        'christian hardcore', 'christian metalcore', 'christian deathcore'
    ]

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

    return filtered_genres[:5]  # Максимум 5 подходящих жанров


if __name__ == "__main__":
    genres = detect_genres("Stray Kids", "Divine")
    print(f"Финальные жанры: {genres}")