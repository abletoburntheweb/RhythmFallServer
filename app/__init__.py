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
            print(f"[Startup] Модель жанров недоступна; ожидается в: {md}")
            print("[Startup] Положите модели в 'models/' или задайте RF_DISCOGS400_DIR/RF_MAEST_EMBED_PB")
        else:
            print("[Startup] Модель жанров доступна")
    except Exception as e:
        print(f"[Startup] Ошибка проверки модели жанров: {e}")
    try:
        from .bpm_analyzer import _tempcnn_model_path
        tp = _tempcnn_model_path()
        if tp:
            print("[Startup] Модель TempoCNN найдена")
        else:
            print("[Startup] Модель TempoCNN не найдена; задайте RF_TEMPOCNN_DIR или положите её в 'models/'")
    except Exception as e:
        print(f"[Startup] Ошибка проверки tempo-модели: {e}")

    return app
