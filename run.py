# run.py
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    bind_host = os.environ.get("RF_BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("RF_BIND_PORT", "5000"))
    print(f"[RhythmFallServer] http://{bind_host}:{port}/  (RF_BIND_HOST / RF_BIND_PORT при необходимости)")
    app.run(debug=True, port=port, host=bind_host)