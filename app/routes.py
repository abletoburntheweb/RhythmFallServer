# app/routes.py

from flask import Blueprint, request, jsonify
import os
import time
import json
from pathlib import Path
import shutil
import re
import app.bpm_analyzer as bpm_analyzer
from . import drum_generator
from .drum_utils import assign_lanes_to_notes

try:
    from .genre_detector import detect_genres

    GENRE_DETECTION_AVAILABLE = True
    print("[Routes] Определение жанров доступно")
except ImportError:
    GENRE_DETECTION_AVAILABLE = False
    print("[Routes] Определение жанров недоступно")

bp = Blueprint("main", __name__)

os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("songs", exist_ok=True)

TASK_PROGRESS = {}
TASK_CANCELLED = set()
TASK_CONTEXT = {}
DEBUG_HTTP = os.getenv("RF_DEBUG_HTTP", "0") == "1"
DRUMGEN_VERBOSE = os.getenv("RF_VERBOSE_DRUMGEN", "0") == "1"


def _debug_log(*args):
    if DEBUG_HTTP:
        print(*args)


def _reroll_notes_to_lanes(notes: list, target_lanes: int) -> list:
    if not notes:
        return []
    if target_lanes <= 1:
        return [dict(n) for n in notes if isinstance(n, dict)]
    stripped = []
    for n in notes:
        if not isinstance(n, dict):
            continue
        m = dict(n)
        if "lane" in m:
            m.pop("lane")
        stripped.append(m)
    return assign_lanes_to_notes(stripped, lanes=target_lanes, song_offset=0.0)


def _extract_artist_title_from_filename(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    parts = stem.split(' - ', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "Unknown", stem

def _report_status(task_id: str, status_text: str):
    if not task_id:
        return
    lst = TASK_PROGRESS.get(task_id)
    if lst is None:
        lst = []
        TASK_PROGRESS[task_id] = lst
    lst.append(status_text)

def _register_task_context(task_id: str, temp_path: str):
    if not task_id or not temp_path:
        return
    base_name = Path(temp_path).stem
    song_folder = Path("temp_uploads") / base_name
    TASK_CONTEXT[task_id] = {
        "temp_path": temp_path,
        "song_folder": str(song_folder)
    }
def _normalize_filename(name: str, default_ext: str = ".mp3") -> str:
    s = "".join(c for c in name if c.isalnum() or c in "._- ")
    s = re.sub(r"\s+", " ", s).strip()
    base, ext = os.path.splitext(s)
    if not ext:
        ext = default_ext
    base = re.sub(r"\.{2,}", ".", base)
    base = base.lstrip(". ").strip()
    if not base:
        base = f"audio_{int(time.time())}"
    return f"{base}{ext}"

def _mark_cancelled(task_id: str):
    if not task_id:
        return
    TASK_CANCELLED.add(task_id)
    _report_status(task_id, "Отмена запрошена")
    _report_status(task_id, "Отменено пользователем")

def _is_cancelled(task_id: str) -> bool:
    if not task_id:
        return False
    return task_id in TASK_CANCELLED

def _cleanup_task(task_id: str):
    ctx = TASK_CONTEXT.get(task_id, {})
    temp_path = ctx.get("temp_path")
    song_folder = ctx.get("song_folder")
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print(f"[Очистка] Удалён временный файл: {temp_path}")
        except Exception as e:
            print(f"[Предупреждение] Не удалось удалить временный файл: {e}")
    if song_folder and os.path.isdir(song_folder):
        try:
            shutil.rmtree(song_folder, ignore_errors=True)
            print(f"[Очистка] Удалена папка песни: {song_folder}")
        except Exception as e:
            print(f"[Предупреждение] Не удалось удалить папку песни: {e}")
    TASK_CONTEXT.pop(task_id, None)
    TASK_CANCELLED.discard(task_id)

def _check_cancel(task_id: str):
    if _is_cancelled(task_id):
        raise RuntimeError("__CANCELLED__")

@bp.route("/task_status", methods=["GET"])
def task_status():
    task_id = request.args.get("task_id", "")
    if not task_id:
        return jsonify({"error": "task_id required"}), 400
    statuses = TASK_PROGRESS.get(task_id, [])
    return jsonify({"task_id": task_id, "statuses": statuses, "status": "ok"})

@bp.route("/cancel_task", methods=["POST", "GET"])
def cancel_task():
    task_id = None
    if request.method == "GET":
        task_id = request.args.get("task_id", "")
    else:
        try:
            data = request.get_json(force=True, silent=True) or {}
        except Exception:
            data = {}
        task_id = (data.get("task_id") or request.args.get("task_id", "")).strip()
    if not task_id:
        return jsonify({"error": "task_id required"}), 400
    _mark_cancelled(task_id)
    return jsonify({"task_id": task_id, "status": "cancel_requested"})

@bp.route("/")
def home():
    return jsonify({
        "message": "RhythmFallServer is running",
        "endpoints": {
            "analyze_bpm": "POST /analyze_bpm - Analyze BPM from audio",
            "generate_drums": "POST /generate_drums - Generate drum notes",
            "list_songs": "GET /songs - List available songs",
            "health": "GET /health - Health check"
        }
    })


@bp.route("/analyze_bpm", methods=["POST"])
def analyze_bpm():
    _debug_log("DEBUG /analyze_bpm Content-Type:", request.content_type)
    _debug_log("DEBUG /analyze_bpm Files keys:", list(request.files.keys()))
    _debug_log("DEBUG /analyze_bpm Form keys:", list(request.form.keys()))

    temp_path = None
    try:
        if "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            safe_filename = _normalize_filename(file.filename, default_ext=".mp3")
            temp_path = os.path.join("temp_uploads", f"bpm_{int(time.time())}_{safe_filename}")
            file.save(temp_path)
            print(f"[BPM] Файл получен через multipart: {temp_path}")

        elif request.data:
            temp_path = os.path.join("temp_uploads", f"bpm_{int(time.time())}_uploaded_audio.mp3")
            with open(temp_path, "wb") as f:
                f.write(request.data)
            print(f"[BPM] Сохранены сырые аудиоданные: {temp_path}")

        else:
            print("[Ошибка] В запросе не найдены аудиоданные")
            return jsonify({"error": "No audio file provided in the request"}), 400

        result = bpm_analyzer.calculate_bpm(temp_path, save_cache=False)

        if result.get("bpm") is not None:
            print(f"[BPM] Успешно рассчитан BPM: {result['bpm']}")
            return jsonify({
                "bpm": result["bpm"],
                "filename": os.path.basename(temp_path),
                "status": "success"
            })
        elif "error" in result:
            error_msg = result.get("error", "Unknown error during BPM analysis")
            print(f"[Ошибка] Анализ BPM завершился с ошибкой: {error_msg}")
            return jsonify({"error": error_msg}), 500
        else:
            print("[Ошибка] Анализ BPM вернул неожиданный формат результата")
            return jsonify({"error": "BPM analysis returned unexpected result format"}), 500

    except Exception as e:
        print(f"[Исключение] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"[Очистка] Удалён временный файл: {temp_path}")
            except Exception as e:
                print(f"[Предупреждение] Не удалось удалить временный файл: {e}")


@bp.route("/generate_drums", methods=["POST"])
def generate_drums():
    _debug_log("DEBUG /generate_drums: запрос получен")
    _debug_log(f"DEBUG /generate_drums Content-Type: {request.content_type}")

    temp_path = None
    task_id = request.headers.get("X-Task-Id", "")
    try:
        _check_cancel(task_id)
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
        progress_delay_seconds = float(metadata.get("progress_delay_seconds", 0.0))
        genres = metadata.get("genres")
        primary_genre = metadata.get("primary_genre")
        use_stems = bool(metadata.get("use_stems", True))
        fill    = metadata.get("fill")
        groove  = metadata.get("groove")
        density = metadata.get("density")
        grid_snap_strength = metadata.get("grid_snap_strength")
        accent_strong_beats = metadata.get("accent_strong_beats")
        genre_template_strength = metadata.get("genre_template_strength")
        if fill is not None:
            try:
                fill = int(fill)
                fill = max(0, min(100, fill))
            except (ValueError, TypeError):
                fill = None
        if groove is not None:
            try:
                groove = int(groove)
                groove = max(0, min(100, groove))
            except (ValueError, TypeError):
                groove = None
        if density is not None:
            try:
                density = int(density)
                density = max(0, min(100, density))
            except (ValueError, TypeError):
                density = None
        if grid_snap_strength is not None:
            try:
                grid_snap_strength = int(grid_snap_strength)
                grid_snap_strength = max(0, min(100, grid_snap_strength))
            except (ValueError, TypeError):
                grid_snap_strength = None
        if accent_strong_beats is not None:
            accent_strong_beats = bool(accent_strong_beats)
        if genre_template_strength is not None:
            try:
                genre_template_strength = int(genre_template_strength)
                genre_template_strength = max(0, min(100, genre_template_strength))
            except (ValueError, TypeError):
                genre_template_strength = None

        valid_modes = {"minimal", "basic", "enhanced", "natural", "custom"}
        if not isinstance(lanes, int) or not (1 <= lanes <= 8):
            return jsonify({"error": "Invalid 'lanes': must be integer 1-8"}), 400
        if not isinstance(sync_tolerance, (int, float)) or not (0.01 <= sync_tolerance <= 1.0):
            return jsonify({"error": "Invalid 'sync_tolerance': must be float 0.01-1.0"}), 400
        if generation_mode not in valid_modes:
            return jsonify({"error": f"'generation_mode' must be one of {sorted(valid_modes)}"}), 400
        if genres is not None and not isinstance(genres, list):
            return jsonify({"error": "'genres' must be a list of strings"}), 400
        if primary_genre is not None and not isinstance(primary_genre, str):
            return jsonify({"error": "'primary_genre' must be a string"}), 400

        safe_filename = _normalize_filename(original_filename, default_ext=".mp3")

        temp_path = os.path.join("temp_uploads", safe_filename)
        audio_file.save(temp_path)
        print(f"[DrumGen] Аудио сохранено: {temp_path}")
        print(f"[DrumGen] Исходное имя файла (Unicode-safe): {original_filename}")
        _register_task_context(task_id, temp_path)
        _check_cancel(task_id)

        if bpm is None:
            print("[DrumGen] BPM не передан, выполняется расчёт...")
            _check_cancel(task_id)
            bpm_result = bpm_analyzer.calculate_bpm(temp_path, save_cache=False, cancel_cb=lambda: _check_cancel(task_id))
            if bpm_result.get("bpm") is None:
                error_msg = bpm_result.get("error", "Failed to calculate BPM")
                return jsonify({"error": f"Could not determine BPM: {error_msg}"}), 500
            bpm = bpm_result["bpm"]
            print(f"[DrumGen] Рассчитанный BPM: {bpm}")
        else:
            try:
                bpm = float(bpm)
                if not (1 <= bpm <= 300):
                    raise ValueError("BPM out of range")
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid 'bpm': must be number 1-300"}), 400

        normalized_genres = [g for g in (genres or []) if isinstance(g, str) and g.strip()]
        provided_genres = normalized_genres if normalized_genres else None
        normalized_primary_genre = (
            primary_genre
            if primary_genre and primary_genre.strip().lower() != "unknown"
            else None
        )
        use_auto_identify = False

        track_info = None
        print("[DrumGen] Используем только аудио-модель жанров (без сетевых источников)")
        artist_guess, title_guess = _extract_artist_title_from_filename(original_filename)
        track_info = {
            "artist": artist_guess or "Unknown",
            "title": title_guess or "Unknown",
            "genres": [],
            "primary_genre": "",
            "success": False
        }
        _check_cancel(task_id)

        _report_status(task_id, "Идентификация трека...")
        _report_status(task_id, "Определение жанров...")
        if progress_delay_seconds > 0:
            time.sleep(progress_delay_seconds)
        print(f"[DrumGen] Генерация нот | BPM: {bpm}, Линии: {lanes}, Режим: {generation_mode}")
        print(f"[DrumGen] Использовать стемы: {'да' if use_stems else 'нет'}")
        print(f"[DrumGen] Параметры | fill={fill} groove={groove} density={density} grid_snap_strength={grid_snap_strength} accent_strong_beats={accent_strong_beats} genre_template_strength={genre_template_strength}")
        effective_primary = normalized_primary_genre
        if effective_primary is None and provided_genres is None and GENRE_DETECTION_AVAILABLE:
            try:
                detected = detect_genres("Unknown", "Unknown", audio_path=temp_path) or []
                track_info['genres'] = detected
                if detected:
                    effective_primary = detected[0]
                print(f"[DrumGen] Жанры по аудио: {detected}")
            except Exception as e:
                print(f"[DrumGen] Ошибка определения жанров: {e}")
        if provided_genres is not None:
            track_info["genres"] = list(provided_genres)
        if effective_primary is not None:
            track_info["primary_genre"] = effective_primary
        print(f"[DrumGen] Жанр для генерации: {effective_primary or 'groove'}")
        _report_status(task_id, "Разделение на стемы...")
        _check_cancel(task_id)
        notes_variants = {}
        lanes_set = [3, 4, 5]
        _report_status(task_id, "Детекция ударных...")
        print(f"[DrumGen] Вариант линий старт: {lanes} (первичная генерация)")
        primary_notes = drum_generator.generate_drums_notes(
            temp_path,
            bpm,
            lanes=lanes,
            sync_tolerance=sync_tolerance,
            use_madmom_beats=True,
            use_stems=use_stems,
            generation_mode=generation_mode,
            fill=fill,
            groove=groove,
            density=density,
            grid_snap_strength=grid_snap_strength,
            accent_strong_beats=accent_strong_beats,
            genre_template_strength=genre_template_strength,
            track_info=track_info,
            auto_identify_track=False,
            use_filename_for_genres=False,
            provided_genres=provided_genres,
            provided_primary_genre=effective_primary,
            verbose=DRUMGEN_VERBOSE,
            status_cb=lambda s: _report_status(task_id, s),
            cancel_cb=lambda: _check_cancel(task_id)
        )
        _check_cancel(task_id)
        if not primary_notes or len(primary_notes) == 0:
            return jsonify({"error": f"No drum notes generated for lanes={lanes}"}), 500
        notes_variants[str(lanes)] = primary_notes
        print(f"[DrumGen] Вариант линий готов: {lanes}, нот={len(primary_notes)}")

        for L in lanes_set:
            if L == lanes:
                continue
            rerolled_notes = _reroll_notes_to_lanes(primary_notes, L)
            notes_variants[str(L)] = rerolled_notes
            print(f"[DrumGen] Вариант линий перераспределён: {L}, нот={len(rerolled_notes)}")

        if not notes_variants:
            return jsonify({"error": "No drum notes generated for any lanes"}), 500
        chosen_notes = notes_variants.get(str(lanes)) or notes_variants.get("4") or next(iter(notes_variants.values()))

        _report_status(task_id, "Сохранение нот...")
        _check_cancel(task_id)
        drum_generator.save_drums_notes(chosen_notes, temp_path, mode=generation_mode)
        _check_cancel(task_id)

        drum_count = len([n for n in chosen_notes if n.get("type") == "DrumNote"])
        response_data = {
            "notes": chosen_notes,
            "notes_variants": notes_variants,
            "bpm": bpm,
            "lanes": lanes,
            "instrument_type": instrument_type,
            "mode": generation_mode,
            "statistics": {
                "total_notes": len(chosen_notes),
                "drum_notes": drum_count
            },
            "status": "success"
        }

        final_genres = provided_genres if provided_genres is not None else (track_info.get("genres") if track_info else [])
        final_primary = effective_primary or (track_info.get("primary_genre") if track_info else "")
        genres_source = "server" if use_auto_identify else "client"

        response_data['track_info'] = {
            'title': (track_info.get('title') if track_info else 'Unknown'),
            'artist': (track_info.get('artist') if track_info else 'Unknown'),
            'genres': final_genres,
            'primary_genre': final_primary,
            'genres_source': genres_source
        }
        print(f"[DrumGen] Успешно сгенерировано {len(chosen_notes)} нот ({generation_mode})")
        print(f"   - Жанры: {', '.join(final_genres) if final_genres else 'не определены'}")
        print(f"   - Источник жанров: {genres_source} | primary: {final_primary or 'не задан'}")
        _report_status(task_id, "Формирование ответа...")
        return jsonify(response_data)

    except RuntimeError as e:
        if str(e) == "__CANCELLED__":
            print(f"[DrumGen] Задача отменена: {task_id}")
            _report_status(task_id, "Отменено пользователем")
            _cleanup_task(task_id)
            return jsonify({"status": "cancelled_by_user", "message": "Отменено пользователем", "task_id": task_id}), 200
        raise
    except Exception as e:
        print(f"[DrumGen] Исключение: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if _is_cancelled(task_id):
            _cleanup_task(task_id)
        else:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"[Очистка] Удалён временный файл: {temp_path}")
                except Exception as e:
                    print(f"[Предупреждение] Не удалось удалить временный файл: {e}")


@bp.route("/debug_genres_audio", methods=["POST"])
def debug_genres_audio():
    temp_path = None
    try:
        if "audio_file" in request.files:
            file = request.files["audio_file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ").rstrip()
            temp_path = os.path.join("temp_uploads", f"genres_{int(time.time())}_{safe_filename}")
            file.save(temp_path)
        elif request.data:
            temp_path = os.path.join("temp_uploads", f"genres_{int(time.time())}_uploaded_audio.mp3")
            with open(temp_path, "wb") as f:
                f.write(request.data)
        else:
            return jsonify({"error": "No audio file provided in the request"}), 400
        audio_available = False
        diagnostics = {}
        try:
            from .genre_detector import is_discogs400_available, _default_model_dir, _resolve_embedding_pb
            md = _default_model_dir()
            head_pb = md / f"{md.name}.pb"
            head_onnx = md / f"{md.name}.onnx"
            labels_json = md / f"{md.name}.json"
            emb_pb = _resolve_embedding_pb(md)
            audio_available = bool(is_discogs400_available())
            diagnostics = {
                "model_dir": str(md),
                "head_pb": head_pb.exists(),
                "head_onnx": head_onnx.exists(),
                "labels_json": labels_json.exists(),
                "embedding_pb": bool(emb_pb and (emb_pb.exists() if hasattr(emb_pb, 'exists') else True))
            }
        except Exception:
            audio_available = False
        try:
            from .genre_detector import MultiSourceGenreDetector
            det = MultiSourceGenreDetector()
            res = det.detect_all_genres("Unknown", "Unknown", audio_path=temp_path)
            try:
                from .genre_detector import classify_discogs400
                raw_top = classify_discogs400(temp_path, top_k=10)
            except Exception:
                raw_top = []
        except Exception as e:
            return jsonify({"audio_model_available": audio_available, "diagnostics": diagnostics, "error": str(e)}), 500
        def _pretty_label(lbl: str) -> str:
            try:
                a, b = lbl.split('---', 1)
                return f"{a.strip().title()}: {b.strip().title()}"
            except Exception:
                return lbl.replace('---', ' — ').title()
        top_pretty = [{"label": _pretty_label(l), "score": float(s) if isinstance(s, (int, float)) else s, "raw": l} for (l, s) in raw_top]
        audio_discogs = res.get("by_source", {}).get("audio_discogs400", [])
        audio_discogs_pretty = [_pretty_label(x) for x in audio_discogs]
        return jsonify({
            "audio_model_available": audio_available,
            "diagnostics": diagnostics,
            "by_source": res.get("by_source", {}),
            "all_genres": res.get("all_genres", []),
            "discogs400_top": raw_top,
            "discogs400_top_pretty": top_pretty,
            "audio_discogs400_pretty": audio_discogs_pretty
        })
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@bp.route("/generate_notes", methods=["POST"])
def generate_notes():
    _debug_log("DEBUG /generate_notes: запрос получен")

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
        print(f"[Ошибка] Не удалось получить список песен: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "endpoints": ["/", "/analyze_bpm", "/generate_drums", "/cancel_task", "/songs", "/health"]
    })
