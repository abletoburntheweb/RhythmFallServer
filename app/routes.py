# app/routes.py
from flask import Blueprint, request, jsonify
import os
import time
from pathlib import Path
import app.bpm_analyzer as bpm_analyzer
from . import drum_generator_basic
from . import drum_generator_enhanced
from .track_detector import identify_track

try:
    from .genre_detector import detect_genres

    GENRE_DETECTION_AVAILABLE = True
    print("[Routes] Genre detection доступен")
except ImportError:
    GENRE_DETECTION_AVAILABLE = False
    print("[Routes] Genre detection не установлен")

bp = Blueprint("main", __name__)

os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("songs", exist_ok=True)


@bp.route("/")
def home():
    return jsonify({
        "message": "RhythmFallServer is running",
        "endpoints": {
            "analyze_bpm": "POST /analyze_bpm - Analyze BPM from audio",
            "generate_drums": "POST /generate_drums - Generate drum notes",
            "identify_track": "POST /identify_track - Identify track by audio",
            "get_genres_manual": "POST /get_genres_manual - Get genres for manually entered artist/title",
            "list_songs": "GET /songs - List available songs",
            "health": "GET /health - Health check"
        }
    })


@bp.route("/analyze_bpm", methods=["POST"])
def analyze_bpm():
    print("DEBUG: Content-Type:", request.content_type)
    print("DEBUG: Files keys:", list(request.files.keys()))
    print("DEBUG: Form keys:", list(request.form.keys()))

    temp_path = None
    try:
        if "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ").rstrip()
            temp_path = os.path.join("temp_uploads", f"bpm_{int(time.time())}_{safe_filename}")
            file.save(temp_path)
            print(f"[INFO] File received via multipart: {temp_path}")

        elif request.data:
            temp_path = os.path.join("temp_uploads", f"bpm_{int(time.time())}_uploaded_audio.mp3")
            with open(temp_path, "wb") as f:
                f.write(request.data)
            print(f"[INFO] Raw audio data saved: {temp_path}")

        else:
            print("[ERROR] No audio data found in request")
            return jsonify({"error": "No audio file provided in the request"}), 400

        result = bpm_analyzer.calculate_bpm(temp_path, save_cache=False)

        if result.get("bpm") is not None:
            print(f"[SUCCESS] BPM calculated: {result['bpm']}")
            return jsonify({
                "bpm": result["bpm"],
                "filename": os.path.basename(temp_path),
                "status": "success"
            })
        elif "error" in result:
            error_msg = result.get("error", "Unknown error during BPM analysis")
            print(f"[ERROR] BPM analysis failed: {error_msg}")
            return jsonify({"error": error_msg}), 500
        else:
            print("[ERROR] BPM analysis returned unexpected result format")
            return jsonify({"error": "BPM analysis returned unexpected result format"}), 500

    except Exception as e:
        print(f"[EXCEPTION] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"[CLEANUP] Temporary file removed: {temp_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file: {e}")


@bp.route("/identify_track", methods=["POST"])
def identify_track_endpoint():
    print("DEBUG: /identify_track request received")
    print(f"DEBUG: Content-Type: {request.content_type}")
    print(f"DEBUG: Headers: {dict(request.headers)}")

    temp_path = None
    try:
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data received"}), 400

        filename = request.headers.get("X-Filename", "uploaded_audio.mp3")

        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ").rstrip()
        if not safe_filename:
            safe_filename = f"track_{int(time.time())}.mp3"

        temp_path = os.path.join("temp_uploads", safe_filename)
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        print(f"[TrackDetector] Processing {temp_path}")

        track_info = identify_track(temp_path)

        if not track_info.get('success'):
            print("[TrackDetector] Track identification failed")
            return jsonify({
                "error": "Could not identify track from audio",
                "status": "not_found"
            }), 404

        print(f"[TrackDetector] Successfully identified: {track_info['artist']} - {track_info['title']}")

        if GENRE_DETECTION_AVAILABLE and track_info.get('artist') != 'Unknown' and track_info.get('title') != 'Unknown':
            spotify_genres = detect_genres(track_info['artist'], track_info['title'])
            if spotify_genres:
                track_info['genres'].extend(spotify_genres)
                track_info['genres'] = list(set(track_info['genres']))
                print(f"[Spotify] Добавлены жанры: {spotify_genres}")

        return jsonify({
            "track_info": track_info,
            "status": "success"
        })

    except Exception as e:
        print(f"[TrackDetector] Exception in identify_track: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"[CLEANUP] Temporary file removed: {temp_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file: {e}")


@bp.route("/get_genres_manual", methods=["POST"])
def get_genres_manual():
    try:
        data = request.get_json(force=True)
        artist = data.get("artist")
        title = data.get("title")

        if not artist or not title:
            return jsonify({"error": "Both 'artist' and 'title' are required."}), 400

        print(f"[ManualGenreDetect] Getting genres for: {artist} - {title}")

        detected_genres = []
        if GENRE_DETECTION_AVAILABLE:
            detected_genres = detect_genres(artist, title)
        else:
            print("[ManualGenreDetect] Genre detection not available.")

        print(f"[ManualGenreDetect] Detected genres: {detected_genres}")

        return jsonify({
            "artist": artist,
            "title": title,
            "genres": detected_genres,
            "status": "success"
        })

    except Exception as e:
        print(f"[ManualGenreDetect] Exception: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@bp.route("/generate_drums", methods=["POST"])
def generate_drums():
    print("DEBUG: /generate_drums request received")
    print(f"DEBUG: Content-Type: {request.content_type}")
    print(f"DEBUG: Headers: {dict(request.headers)}")

    temp_path = None
    temp_track_path = None
    try:
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data received"}), 400

        bpm = request.headers.get("X-BPM")
        instrument_type = request.headers.get("X-Instrument", "drums")
        filename = request.headers.get("X-Filename", "uploaded_audio.mp3")

        drum_mode = request.headers.get("X-Drum-Mode", "basic").lower()
        if drum_mode not in ["basic", "enhanced"]:
            return jsonify({"error": "X-Drum-Mode must be 'basic' or 'enhanced'"}), 400

        generator = drum_generator_basic if drum_mode == "basic" else drum_generator_enhanced

        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ").rstrip()
        if not safe_filename:
            safe_filename = f"audio_{int(time.time())}.mp3"

        try:
            lanes = int(request.headers.get("X-Lanes", "4"))
            if lanes <= 0 or lanes > 8:
                raise ValueError("Lanes must be between 1 and 8")
        except ValueError:
            return jsonify({"error": "Invalid X-Lanes value, must be a positive integer (1-8)"}), 400

        try:
            sync_tolerance = float(request.headers.get("X-Sync-Tolerance", "0.2"))
            if sync_tolerance <= 0 or sync_tolerance > 1.0:
                raise ValueError("Sync tolerance must be between 0.01 and 1.0")
        except ValueError:
            return jsonify({"error": "Invalid X-Sync-Tolerance value, must be a positive number (0.01-1.0)"}), 400

        use_madmom_beats = request.headers.get("X-Use-Madmom", "true").lower() == "true"
        use_stems = True
        use_filename_for_genres = request.headers.get("X-Use-Filename-Genres", "true").lower() == "true"

        manual_artist = request.headers.get("X-Artist-Manual")
        manual_title = request.headers.get("X-Title-Manual")

        track_info = None
        if manual_artist and manual_title:
            print(f"[DrumGen] Используем введённые пользователем данные для генерации: {manual_artist} - {manual_title}")
            track_info = {
                'title': manual_title,
                'artist': manual_artist,
                'album': 'Unknown',
                'year': 'Unknown',
                'genres': [],
                'primary_type': None,
                'secondary_types': [],
                'acoustid_id': None,
                'score': 1.0,
                'duration': None,
                'success': True
            }
            if GENRE_DETECTION_AVAILABLE:
                print(f"[DrumGen] Запрашиваем жанры для введённых данных: {manual_artist} - {manual_title}")
                try:
                    detected_genres = detect_genres(manual_artist, manual_title)
                    if detected_genres:
                        track_info['genres'] = detected_genres
                        print(f"[DrumGen] Получены жанры для введённых данных: {detected_genres}")
                    else:
                        print(f"[DrumGen] Жанры для введённых данных не найдены.")
                except Exception as e:
                    print(f"[DrumGen] Ошибка при получении жанров для введённых данных: {e}")
        elif request.headers.get("X-Identify-Track", "false").lower() == "true":
            print("[DrumGen] Идентификация трека для генерации на основе аудио...")
            temp_track_path = os.path.join("temp_uploads", f"track_{int(time.time())}_{safe_filename}")
            with open(temp_track_path, "wb") as f:
                f.write(audio_data)

            track_info = identify_track(temp_track_path)

            if track_info and not track_info.get('success'):
                print(f"[DrumGen] Идентификация не удалась (success=False). Требуется ручной ввод artist/title.")
                if os.path.exists(temp_track_path):
                    os.remove(temp_track_path)
                    print(f"[CLEANUP] Временный файл идентификации удалён: {temp_track_path}")
                    temp_track_path = None

                return jsonify({
                    "error": "Track identification failed. Please provide artist and title manually.",
                    "requires_manual_input": True,
                    "fallback_artist": track_info.get('artist', 'Unknown'),
                    "fallback_title": track_info.get('title', 'Unknown'),
                    "status": "requires_manual_input"
                }), 200

            if track_info and track_info.get('success'):
                print(f"[DrumGen] Успешно идентифицирован трек: {track_info['artist']} - {track_info['title']}")
                if track_info['genres']:
                    print(f"[DrumGen] Жанры из аудио: {', '.join(track_info['genres'])}")

                if GENRE_DETECTION_AVAILABLE and track_info.get('artist') != 'Unknown' and track_info.get('title') != 'Unknown':
                    spotify_genres = detect_genres(track_info['artist'], track_info['title'])
                    if spotify_genres:
                        original_genres = track_info['genres'][:]
                        track_info['genres'].extend(spotify_genres)
                        track_info['genres'] = list(set(track_info['genres']))
                        if set(track_info['genres']) != set(original_genres):
                            print(f"[Spotify] Добавлены жанры: {spotify_genres}")
            else:
                print("[DrumGen] Идентификация трека вернула None или неопределённые данные.")

            if os.path.exists(temp_track_path):
                os.remove(temp_track_path)
                print(f"[CLEANUP] Временный файл идентификации удалён: {temp_track_path}")
                temp_track_path = None


        if bpm:
            try:
                bpm = float(bpm)
                if bpm <= 0 or bpm > 300:
                    raise ValueError("BPM must be between 1 and 300")
            except ValueError:
                return jsonify({"error": "Invalid BPM value, must be a number between 1 and 300"}), 400
        else:
            print("[DrumGen] BPM не предоставлен в заголовках, производим расчёт...")
            temp_bpm_path = os.path.join("temp_uploads", f"temp_bpm_{int(time.time())}.mp3")
            with open(temp_bpm_path, "wb") as f:
                f.write(audio_data)

            bpm_result = bpm_analyzer.calculate_bpm(temp_bpm_path, save_cache=False)

            if os.path.exists(temp_bpm_path):
                os.remove(temp_bpm_path)

            if bpm_result.get("bpm") is None:
                error_msg = bpm_result.get("error", "Failed to calculate BPM")
                return jsonify({"error": f"Could not determine BPM: {error_msg}"}), 500

            bpm = bpm_result["bpm"]
            print(f"[DrumGen] Рассчитанный BPM: {bpm}")

        temp_path = os.path.join("temp_uploads", safe_filename)
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        print(f"[DrumGen] Обработка {temp_path}")
        print(f"  BPM: {bpm}, Lanes: {lanes}, Sync Tolerance: {sync_tolerance}")
        print(f"  Use Madmom: {use_madmom_beats}, Use Stems: {use_stems}")
        print(f"  Use Filename Genres: {use_filename_for_genres}")
        print(f"  Mode: {drum_mode}")

        notes = generator.generate_drums_notes(
            temp_path,
            bpm,
            lanes=lanes,
            sync_tolerance=sync_tolerance,
            use_madmom_beats=use_madmom_beats,
            use_stems=use_stems,
            track_info=track_info,
            auto_identify_track=False,
            use_filename_for_genres=use_filename_for_genres
        )

        if not notes:
            return jsonify({"error": "Failed to generate drum notes (no notes generated)"}), 500

        generator.save_drums_notes(notes, temp_path, mode=drum_mode)

        print(f"[DrumGen] Успешно сгенерировано {len(notes)} барабанных нот ({drum_mode})")

        drum_count = len([n for n in notes if n["type"] == "DrumNote"])

        response_data = {
            "notes": notes,
            "bpm": bpm,
            "lanes": lanes,
            "instrument_type": instrument_type,
            "mode": drum_mode,
            "statistics": {
                "total_notes": len(notes),
                "drum_notes": drum_count
            },
            "status": "success"
        }

        if track_info and track_info.get('success'):
            response_data['track_info'] = {
                'title': track_info['title'],
                'artist': track_info['artist'],
                'album': track_info['album'],
                'year': track_info['year'],
                'genres': track_info['genres']
            }

        return jsonify(response_data)

    except Exception as e:
        print(f"[DrumGen] Исключение в generate_drums: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"[CLEANUP] Временный файл загрузки удалён: {temp_path}")
            except Exception as e:
                print(f"[WARNING] Не удалось удалить временный файл загрузки: {e}")
        if temp_track_path and os.path.exists(temp_track_path):
            try:
                os.remove(temp_track_path)
                print(f"[CLEANUP] Временный файл идентификации удалён в finally: {temp_track_path}")
            except Exception as e:
                print(f"[WARNING] Не удалось удалить временный файл идентификации в finally: {e}")


@bp.route("/generate_notes", methods=["POST"])
def generate_notes():
    print("DEBUG: /generate_notes request received")

    instrument_type = request.headers.get("X-Instrument", "drums")

    if instrument_type.lower() == "drums":
        return generate_drums()

    return jsonify({
        "error": f"Instrument '{instrument_type}' is not supported yet. Use /generate_drums for drums.",
        "status": "error"
    }), 400


@bp.route("/songs", methods=["GET"])
def list_songs():
    songs_dir = "songs"
    try:
        if not os.path.exists(songs_dir):
            return jsonify({"songs": [], "message": "Songs directory is empty"})

        files = os.listdir(songs_dir)
        songs = []

        for file in files:
            file_path = os.path.join(songs_dir, file)
            if os.path.isfile(file_path) and file.lower().endswith((".mp3", ".wav", ".ogg", ".flac")):
                size = os.path.getsize(file_path)
                size_mb = round(size / (1024 * 1024), 2)

                songs.append({
                    "name": file,
                    "size_mb": size_mb,
                    "path": file_path
                })

        return jsonify({
            "songs": songs,
            "count": len(songs),
            "status": "success"
        })
    except Exception as e:
        print(f"[ERROR] Failed to list songs: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "endpoints": ["/", "/analyze_bpm", "/generate_drums", "/identify_track", "/get_genres_manual", "/songs", "/health"]
    })