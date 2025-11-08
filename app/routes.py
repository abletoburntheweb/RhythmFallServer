# app/routes.py
from flask import Blueprint, request, jsonify
import tempfile
import os
import base64
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

        result = bpm_analyzer.calculate_bpm(temp_path, save_cache=True)

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
    """Генерация барабанных нот для песни"""
    print("DEBUG: /generate_drums request received")

    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = None

    try:
        if "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            temp_path = os.path.join("temp_uploads", file.filename)
            file.save(temp_path)
            print(f"[INFO] Audio file received: {temp_path}")
        elif request.data:
            temp_path = os.path.join("temp_uploads", "uploaded_audio.mp3")
            with open(temp_path, "wb") as f:
                f.write(request.data)
            print(f"[INFO] Raw audio data received: {temp_path}")
        else:
            return jsonify({"error": "No audio file provided"}), 400

        bpm = request.form.get("bpm")
        if bpm:
            try:
                bpm = float(bpm)
            except ValueError:
                return jsonify({"error": "Invalid BPM value"}), 400
        else:
            print("[DrumGen] BPM not provided, calculating...")
            bpm_result = bpm_analyzer.calculate_bpm(temp_path, save_cache=False)
            if bpm_result.get("bpm") is None:
                error_msg = bpm_result.get("error", "Failed to calculate BPM")
                return jsonify({"error": f"Could not determine BPM: {error_msg}"}), 500
            bpm = bpm_result["bpm"]
            print(f"[DrumGen] Calculated BPM: {bpm}")

        lanes = request.form.get("lanes", 4)
        try:
            lanes = int(lanes)
        except ValueError:
            lanes = 4

        print(f"[DrumGen] Generating drums for {temp_path} with BPM: {bpm}, lanes: {lanes}")

        notes = drum_generator.generate_drums_notes(temp_path, bpm, lanes)

        if not notes:
            return jsonify({"error": "Failed to generate drum notes"}), 500

        drum_generator.save_drums_notes(notes, temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[CLEANUP] Temporary file removed: {temp_path}")

        print(f"[DrumGen] Successfully generated {len(notes)} drum notes")
        return jsonify({
            "notes": notes,
            "bpm": bpm,
            "lanes": lanes,
            "instrument_type": "drums",
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