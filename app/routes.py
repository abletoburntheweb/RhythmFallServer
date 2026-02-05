# app/routes.py

from flask import Blueprint, request, jsonify
import os
import time
import json
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

    temp_path = None
    try:
        if "audio_file" not in request.files:
            return jsonify({"error": "Missing 'audio_file' in multipart form"}), 400

        if "metadata" not in request.form:
            return jsonify({"error": "Missing 'metadata' in multipart form"}), 400

        audio_file = request.files["audio_file"]
        metadata_json = request.form["metadata"]

        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON in 'metadata': {str(e)}"}), 400

        original_filename = metadata.get("original_filename", "uploaded_audio.mp3")
        bpm = metadata.get("bpm")
        lanes = metadata.get("lanes", 4)
        instrument_type = metadata.get("instrument_type", "drums")
        sync_tolerance = metadata.get("sync_tolerance", 0.2)
        generation_mode = metadata.get("generation_mode", "basic")
        auto_identify_track = metadata.get("auto_identify_track", False)
        manual_artist = metadata.get("manual_artist", "")
        manual_title = metadata.get("manual_title", "")
        genres = metadata.get("genres")
        primary_genre = metadata.get("primary_genre")

        if not isinstance(lanes, int) or not (1 <= lanes <= 8):
            return jsonify({"error": "Invalid 'lanes': must be integer 1-8"}), 400
        if not isinstance(sync_tolerance, (int, float)) or not (0.01 <= sync_tolerance <= 1.0):
            return jsonify({"error": "Invalid 'sync_tolerance': must be float 0.01-1.0"}), 400
        if generation_mode not in ["basic", "enhanced"]:
            return jsonify({"error": "'generation_mode' must be 'basic' or 'enhanced'"}), 400
        if genres is not None and not isinstance(genres, list):
            return jsonify({"error": "'genres' must be a list of strings"}), 400
        if primary_genre is not None and not isinstance(primary_genre, str):
            return jsonify({"error": "'primary_genre' must be a string"}), 400

        drum_mode = generation_mode
        generator = drum_generator_basic if drum_mode == "basic" else drum_generator_enhanced

        safe_filename = "".join(c for c in original_filename if c.isalnum() or c in "._- ").rstrip()
        if not safe_filename:
            safe_filename = f"audio_{int(time.time())}.mp3"

        temp_path = os.path.join("temp_uploads", safe_filename)
        audio_file.save(temp_path)
        print(f"[DrumGen] Audio saved: {temp_path}")
        print(f"[DrumGen] Original filename (Unicode-safe): {original_filename}")

        if bpm is None:
            print("[DrumGen] BPM not provided, calculating...")
            bpm_result = bpm_analyzer.calculate_bpm(temp_path, save_cache=False)
            if bpm_result.get("bpm") is None:
                error_msg = bpm_result.get("error", "Failed to calculate BPM")
                return jsonify({"error": f"Could not determine BPM: {error_msg}"}), 500
            bpm = bpm_result["bpm"]
            print(f"[DrumGen] Calculated BPM: {bpm}")
        else:
            try:
                bpm = float(bpm)
                if not (1 <= bpm <= 300):
                    raise ValueError("BPM out of range")
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid 'bpm': must be number 1-300"}), 400

        track_info = None

        if manual_artist and manual_title:
            print(f"[DrumGen] Using manual artist/title: {manual_artist} - {manual_title}")
            is_unknown = (
                manual_artist.strip().lower() == "unknown" and
                manual_title.strip().lower() == "unknown"
            )
            if not is_unknown and GENRE_DETECTION_AVAILABLE:
                try:
                    detected_genres = detect_genres(manual_artist, manual_title)
                    track_info = {
                        'title': manual_title,
                        'artist': manual_artist,
                        'genres': detected_genres or [],
                        'success': True
                    }
                    print(f"[DrumGen] Genres from manual input: {detected_genres}")
                except Exception as e:
                    print(f"[DrumGen] Genre detection failed: {e}")
                    track_info = {
                        'title': manual_title,
                        'artist': manual_artist,
                        'genres': [],
                        'success': True
                    }
        elif auto_identify_track:
            print("[DrumGen] Auto-identifying track from audio...")
            track_info = identify_track(temp_path)
            if track_info and track_info.get('success'):
                print(f"[DrumGen] Identified: {track_info['artist']} - {track_info['title']}")
                if GENRE_DETECTION_AVAILABLE and track_info.get('artist') != 'Unknown' and track_info.get('title') != 'Unknown':
                    try:
                        spotify_genres = detect_genres(track_info['artist'], track_info['title'])
                        if spotify_genres:
                            original = track_info['genres'][:]
                            track_info['genres'] = list(set(original + spotify_genres))
                            if set(track_info['genres']) != set(original):
                                print(f"[Spotify] Added genres: {spotify_genres}")
                    except Exception as e:
                        print(f"[Spotify] Failed: {e}")
            else:
                print("[DrumGen] Auto-identification failed — proceeding without track info")
                track_info = None
        else:
            print("[DrumGen] No track identification requested")
            track_info = None

        normalized_genres = [g for g in (genres or []) if isinstance(g, str) and g.strip()]
        provided_genres = normalized_genres if normalized_genres else None
        normalized_primary_genre = (
            primary_genre
            if primary_genre and primary_genre.strip().lower() != "unknown"
            else None
        )

        print(f"[DrumGen] Generating notes | BPM: {bpm}, Lanes: {lanes}, Mode: {drum_mode}")
        notes = generator.generate_drums_notes(
            temp_path,
            bpm,
            lanes=lanes,
            sync_tolerance=sync_tolerance,
            use_madmom_beats=True,
            use_stems=True,
            track_info=track_info,
            auto_identify_track=False,
            use_filename_for_genres=(manual_artist and manual_title and not (manual_artist.lower() == "unknown" and manual_title.lower() == "unknown")),
            provided_genres=provided_genres,
            provided_primary_genre=normalized_primary_genre
        )

        if not notes:
            return jsonify({"error": "No drum notes generated"}), 500

        generator.save_drums_notes(notes, temp_path, mode=drum_mode)

        drum_count = len([n for n in notes if n.get("type") == "DrumNote"])
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

        final_genres = provided_genres if provided_genres is not None else (track_info.get("genres") if track_info else [])
        final_primary = normalized_primary_genre or (track_info.get("primary_genre") if track_info else (final_genres[0] if final_genres else "groove"))

        response_data['track_info'] = {
            'title': manual_title or (track_info.get('title') if track_info else 'Unknown'),
            'artist': manual_artist or (track_info.get('artist') if track_info else 'Unknown'),
            'genres': final_genres,
            'primary_genre': final_primary
        }

        print(f"[DrumGen] Successfully generated {len(notes)} notes ({drum_mode})")
        print(f"   - Жанры: {', '.join(final_genres) if final_genres else 'не определены'}")
        return jsonify(response_data)

    except Exception as e:
        print(f"[DrumGen] Exception: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"[CLEANUP] Removed temp file: {temp_path}")
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file: {e}")


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
