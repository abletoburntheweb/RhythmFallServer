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
        from .genre_detector import is_discogs400_available, _default_model_dir
        md_ok = bool(is_discogs400_available())
        if not md_ok:
            md = _default_model_dir()
            print(f"[Startup] Genre model not available; expected at: {md}")
            print("[Startup] Place models under 'models/' or set RF_DISCOGS400_DIR/RF_MAEST_EMBED_PB")
        else:
            print("[Startup] Genre model available")
    except Exception as e:
        print(f"[Startup] Genre model check failed: {e}")
    try:
        from .bpm_analyzer import _tempcnn_model_path
        tp = _tempcnn_model_path()
        if tp:
            print("[Startup] TempoCNN model found")
        else:
            print("[Startup] TempoCNN model not found; set RF_TEMPOCNN_DIR or place under 'models/'")
    except Exception as e:
        print(f"[Startup] Tempo model check failed: {e}")

    return app
