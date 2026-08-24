# RhythmFallServer

Local Flask server for [RhythmFall](https://github.com/abletoburntheweb/RhythmFall): analyzes audio and returns automatically generated charts. Runs on your machine next to the Godot client (or inside the Windows game install as a bundled worker).

Languages: English | [Русский](./README.ru.md)

**Client:** https://github.com/abletoburntheweb/RhythmFall  
**Models:** not committed. Download from [Releases](https://github.com/abletoburntheweb/RhythmFallServer/releases) or run the client’s `worker\download_effnet_models.ps1`, then place files under `models/` (e.g. EffNet / Discogs paths expected by the app).

## What It Is

- Lightweight local HTTP API (Flask) for BPM, drum charts, bass charts (beta), genre hints, and Rhythm DNA sidecars
- Writes human-readable **`.rf`** charts (RFC v1) and optional **`.rfd`** DNA passports for drums
- Fully local — audio is not uploaded to external services
- On Windows, the RhythmFall client can auto-start this server via `RhythmFallServer.exe` + a prepared `.venv`

## How It Works

1. Client uploads (or points at) an audio file and metadata: instrument, **goal** (`original` / `arcade`), difficulty tier for Arcade, lanes, optional sliders.
2. Server estimates tempo (TempoCNN / fallbacks), optionally separates stems (Demucs / audio-separator), detects hits (**ADTOF** by default, heuristic fallback).
3. Goal policies + genre profiles shape density and lane routing; Arcade uses an ergonomic lane router.
4. Response includes chart payload / paths; drum generation also bakes `.rfd` next to the `.rf`.

**Chart stems**

| Goal | Files (examples) |
| --- | --- |
| **Original** | `drums_original.rf`, `bass_original.rf` (single documentary bake; legacy `*_original_standard` still accepted) |
| **Arcade** | `drums_arcade_relaxed.rf` / `_standard` / `_dense` (Easy / Medium / Hard), same pattern for bass |

## Quick Start (Windows — recommended)

Use the scripts from the **client** repo (`worker/`), with this server checkout as `RhythmFallServer/` or `RhythmFallServer-main/` next to `worker/`:

```powershell
powershell -ExecutionPolicy Bypass -File worker\install_windows_server.ps1 -Gpu auto
powershell -ExecutionPolicy Bypass -File worker\download_effnet_models.ps1
powershell -ExecutionPolicy Bypass -File worker\build_server_launcher.ps1
```

- `-Gpu auto|nvidia|amd|cpu` — chooses CUDA / DirectML / CPU **when building the venv** (not asked by the game installer).
- Requires **Python 3.9–3.11** and **ffmpeg** on PATH for stems.
- Creates `RhythmFallServer\.venv` and `worker\windows_python.path`.
- Run `RhythmFallServer.exe` or `python run.py` — http://127.0.0.1:5000 should report the server is up.

Manual venv (any OS):

```bash
python -m venv .venv
# activate, then:
pip install -r requirements.txt
python run.py
```

## Endpoints (summary)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` / `/health` | Liveness + endpoint list |
| POST | `/analyze_bpm` | Tempo from audio |
| POST | `/generate_drums` | Drum `.rf` (+ `.rfd`) |
| POST | `/generate_bass` | Bass `.rf` (beta; no `.rfd`) |
| GET | `/task_status`, `/task_result` | Async job polling |
| POST | `/cancel_task` | Cancel a job |
| GET/POST | `/rhythm_dna`, `/rhythm_dna_sidecar` | DNA report / sidecar |
| GET/POST | `/storage_usage`, `/storage_reclaim` | Temp upload cleanup |

Exact request fields are driven by the RhythmFall client (`generation_api_client.gd`).

## Notes

- Default bind: `localhost:5000`. Stop with Ctrl+C (or close the launcher console).
- No cloud upload — processing stays on the device.
- First stem separation is slower; later runs reuse `temp_uploads/` cache.
- GPU packages are optional; CPU works with longer stem times.
- WSL is deprecated for the Windows release path — use the native venv.

## Dependencies

- Core: Python **3.9–3.11**, Flask, librosa, NumPy 1.x (see `requirements.txt`)
- Windows stack extras (via install script): TempoCNN, EffNet/ONNX, Demucs / audio-separator, optional madmom, **ADTOF**
- System: **ffmpeg** for stem workflows

## Genre models (optional)

- Discogs / EffNet-style heads if present under `models/`
- Env overrides may include `RF_DISCOGS400_DIR`, `RF_MAEST_EMBED_PB` (see app code / docs)
- Drum backend: `RFALL_DRUM_BACKEND=adtof_fast` (default after install) or `heuristic`

## Platform Notes

- **Windows** is the primary release target for the bundled client worker.
- Shipping inside the game: copy `.venv`, `app/`, `models/`, `run.py`, and `RhythmFallServer.exe` next to `RhythmFall.exe` — see client `RFALL/BUILD.txt`.
