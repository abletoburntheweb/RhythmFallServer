# app/routes.py
from flask import Blueprint, request, jsonify
import tempfile
import os
import base64
import app.bpm_analyzer as bpm_analyzer

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
        else:
            error_msg = result.get("error", "BPM analysis failed")
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        print(f"[EXCEPTION] {e}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@bp.route("/songs", methods=["GET"])
def list_songs():
    songs_dir = "songs"
    try:
        files = os.listdir(songs_dir)
        songs = [f for f in files if f.lower().endswith((".mp3", ".wav"))]
        return jsonify({"songs": songs})
    except FileNotFoundError:
        return jsonify({"error": "Songs directory not found"}), 404
