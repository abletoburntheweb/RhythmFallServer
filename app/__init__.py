# app/__init__.py
from flask import Flask


def create_app():
    app = Flask(__name__)

    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

    from app.routes import bp
    app.register_blueprint(bp)

    return app