# app/__init__.py
from flask import Flask
import os


def create_app():
    app = Flask(__name__)

    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

    from app.routes import bp
    app.register_blueprint(bp)

    try:
        os.makedirs("models", exist_ok=True)
    except Exception:
        pass
    try:
        from .genre_detector import (
            _default_effnet_model_dir,
            genre_backend_name,
            is_discogs400_available,
            is_effnet_onnx_available,
        )
        backend = genre_backend_name()
        if backend == "effnet-onnx":
            eff = _default_effnet_model_dir()
            print(f"[Startup] Модель жанров доступна (EffNet ONNX): {eff.resolve()}")
        elif not is_effnet_onnx_available():
            print(f"[Startup] Жанры: положите discogs-effnet-bsdynamic-1.onnx + .json в {_default_effnet_model_dir()}")
        if not is_discogs400_available():
            print("[Startup] Авто-жанры недоступны (нет EffNet ONNX)")
    except Exception as e:
        print(f"[Startup] Ошибка проверки модели жанров: {e}")
    try:
        from .bpm_analyzer import (
            _tempocnn_pip_available,
            _tempocnn_pip_model_name,
        )
        if _tempocnn_pip_available():
            print(
                f"[Startup] BPM: tempocnn OK ({_tempocnn_pip_model_name()})"
            )
        else:
            print("[Startup] BPM: pip install tempocnn (или fallback librosa)")
    except Exception as e:
        print(f"[Startup] Ошибка проверки tempo-модели: {e}")
    try:
        from .adtof_transcriber import drum_backend_name, is_adtof_available

        if is_adtof_available():
            print(f"[Startup] Drum hits: ADTOF fast-pick (RFALL_DRUM_BACKEND={drum_backend_name()})")
        else:
            print(f"[Startup] Drum hits: librosa heuristic (install adtof-pytorch for ADTOF)")
    except Exception as e:
        print(f"[Startup] Drum backend: {e}")
    try:
        from .bass_transcriber import bass_backend_name, is_basic_pitch_available

        backend = bass_backend_name()
        if backend == "basic_pitch" and is_basic_pitch_available():
            print(f"[Startup] Bass transcription: Basic Pitch (RFALL_BASS_BACKEND={backend})")
        elif backend == "basic_pitch":
            print("[Startup] Bass transcription: heuristic (pip install basic-pitch --no-deps)")
        else:
            print(f"[Startup] Bass transcription: heuristic (RFALL_BASS_BACKEND={backend})")
    except Exception as e:
        print(f"[Startup] Bass backend: {e}")
    try:
        from .gpu_backend import startup_gpu_message
        from . import stem_memory_cache

        print(startup_gpu_message())
        if stem_memory_cache.is_enabled():
            keep = os.getenv("RFALL_KEEP_TEMP_UPLOADS", "1").strip().lower() in ("1", "true", "yes", "on")
            suffix = "; temp_uploads сохраняются (RFALL_KEEP_TEMP_UPLOADS=1)" if keep else "; temp_uploads очищается после каждой генерации"
            print(
                f"[Startup] Стемы: память + TTL {stem_memory_cache.ttl_seconds()}s "
                f"(RFALL_STEM_CACHE_TTL{suffix})"
            )
        else:
            print("[Startup] Стемы: память отключена (RFALL_STEM_CACHE_TTL=0)")
    except Exception as e:
        print(f"[Startup] GPU / stem cache: {e}")
    try:
        from .audio_analysis import warm_stem_separator, _stem_model_mode

        if warm_stem_separator():
            print("[Startup] Stem separator: kuielab warm-loaded (RFALL_STEM_WARM=1)")
        elif os.getenv("RFALL_STEM_WARM", "1").strip().lower() not in ("0", "false", "no", "off"):
            mode = _stem_model_mode()
            if mode in ("htdemucs", "demucs"):
                print("[Startup] Stem separator: htdemucs (lazy load on first job)")
            else:
                print("[Startup] Stem separator: lazy kuielab warm on first separation")
        else:
            print("[Startup] Stem separator: warm load disabled (RFALL_STEM_WARM=0)")
        print(f"[Startup] Stem model: RFALL_STEM_MODEL={_stem_model_mode()}")
    except Exception as e:
        print(f"[Startup] Stem separator warm: {e}")

    return app
