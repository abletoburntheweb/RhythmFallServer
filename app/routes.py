# app/routes.py
from flask import Blueprint, request, jsonify
import tempfile
import os
import base64
import time
import app.bpm_analyzer as bpm_analyzer
from . import drum_generator

bp = Blueprint("main", __name__)


@bp.route("/")
def home():
    return jsonify({"message": "RhythmFallServer is running"})


@bp.route("/analyze_bpm", methods=["POST"])
def analyze_bpm():
    print("DEBUG: Content-Type:", request.content_type)
    print("DEBUG: Files keys:", list(request.files.keys()))
    print("DEBUG: Form keys:", list(request.form.keys()))

    os.makedirs("temp_uploads", exist_ok=True)

    temp_path = None

    try:
        if "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            temp_path = os.path.join("temp_uploads", file.filename)
            file.save(temp_path)
            print(f"[INFO] File received via multipart: {temp_path}")

        elif request.data:
            temp_path = os.path.join("temp_uploads", "uploaded_audio.mp3")
            with open(temp_path, "wb") as f:
                f.write(request.data)
            print(f"[INFO] Raw audio data saved: {temp_path}")

        else:
            print("[ERROR] No audio data found in request")
            return jsonify({"error": "No audio file provided in the request"}), 400

        result = bpm_analyzer.calculate_bpm(temp_path, save_cache=False)

        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[CLEANUP] Temporary file removed: {temp_path}")

        if result.get("bpm") is not None:
            print(f"[SUCCESS] BPM calculated for {temp_path}: {result['bpm']}")
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
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@bp.route("/generate_drums", methods=["POST"])
def generate_drums():
    print("DEBUG: /generate_drums request received")
    print(f"DEBUG: Content-Type: {request.content_type}")
    print(f"DEBUG: Headers: {dict(request.headers)}")

    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = None

    try:
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data received"}), 400

        bpm = request.headers.get("X-BPM")
        instrument_type = request.headers.get("X-Instrument", "drums")
        filename = request.headers.get("X-Filename", "uploaded_audio.mp3")

        try:
            lanes = int(request.headers.get("X-Lanes", "4"))
            if lanes <= 0:
                raise ValueError("Lanes must be positive")
        except ValueError:
            return jsonify({"error": "Invalid X-Lanes value, must be a positive integer"}), 400

        try:
            sync_tolerance = float(request.headers.get("X-Sync-Tolerance", "0.2"))
            if sync_tolerance <= 0:
                raise ValueError("Sync tolerance must be positive")
        except ValueError:
            return jsonify({"error": "Invalid X-Sync-Tolerance value, must be a positive number"}), 400

        if bpm:
            try:
                bpm = float(bpm)
            except ValueError:
                return jsonify({"error": "Invalid BPM value"}), 400
        else:
            print("[DrumGen] BPM not provided in headers, calculating...")
            temp_path = os.path.join("temp_uploads", f"temp_for_bpm_{int(time.time())}.mp3")
            with open(temp_path, "wb") as f:
                f.write(audio_data)

            bpm_result = bpm_analyzer.calculate_bpm(temp_path, save_cache=False)
            if bpm_result.get("bpm") is None:
                error_msg = bpm_result.get("error", "Failed to calculate BPM")
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                return jsonify({"error": f"Could not determine BPM: {error_msg}"}), 500
            bpm = bpm_result["bpm"]
            print(f"[DrumGen] Calculated BPM: {bpm}")

        temp_path = os.path.join("temp_uploads", filename)
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        print(f"[DrumGen] Processing {temp_path} with BPM: {bpm}, instrument: {instrument_type}, lanes: {lanes}, sync_tolerance: {sync_tolerance}")


        notes = drum_generator.generate_drums_notes(temp_path, bpm, lanes=lanes, sync_tolerance=sync_tolerance)

        if not notes:
            return jsonify({"error": "Failed to generate drum notes"}), 500

        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[CLEANUP] Temporary file removed: {temp_path}")

        print(f"[DrumGen] Successfully generated {len(notes)} drum notes")
        return jsonify({
            "notes": notes,
            "bpm": bpm,
            "lanes": lanes,
            "instrument_type": instrument_type,
            "status": "success"
        })

    except Exception as e:
        print(f"[DrumGen] Exception in generate_drums: {e}")
        import traceback
        traceback.print_exc()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({"error": str(e)}), 500


@bp.route("/generate_notes", methods=["POST"])
def generate_notes():
    print("DEBUG: /generate_notes request received")

    return jsonify({
        "error": "Only drums instrument is currently supported. Use /generate_drums endpoint.",
        "status": "error"
    }), 400


@bp.route("/songs", methods=["GET"])
def list_songs():
    songs_dir = "songs"
    try:
        files = os.listdir(songs_dir)
        songs = [f for f in files if f.lower().endswith((".mp3", ".wav"))]
        return jsonify({"songs": songs})
    except FileNotFoundError:
        return jsonify({"error": "Songs directory not found"}), 404