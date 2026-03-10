# RhythmFallServer

Local Flask server for RhythmFall that analyzes audio and returns automatically generated notes. Designed to run on your machine alongside the Godot client.

Languages: English | [Русский](./README.ru.md)

Note: This repository contains the server/backend for RhythmFall. For the Godot client, use the client repository: https://github.com/abletoburntheweb/RhythmFall

## What It Is
- Lightweight local HTTP server (Flask) handling audio analysis and chart generation.
- Provides endpoints for BPM detection and drum‑focused note generation.
- Works entirely on localhost; no audio is uploaded to external services.

## How It Works
- Receives an audio file via HTTP.
- Estimates tempo, optionally splits stems, detects drum events, applies genre‑aware patterns.
- Returns a JSON chart and simple statistics back to the client.

## Quick Start
1) Create a virtual environment
```bash
python -m venv .venv
```
2) Activate it
- Windows (cmd):
  ```cmd
  .venv\Scripts\activate
  ```
- Windows (PowerShell):
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
3) Install dependencies
```bash
pip install -r requirements.txt
```
4) Run the server
```bash
python run.py
```
Open http://127.0.0.1:5000 — you should see: {"message": "RhythmFallServer is running"}

## Endpoints (summary)
- POST /analyze_bpm — accepts an audio file; returns { "bpm": number }
- POST /generate_drums — accepts an audio file and metadata; returns notes JSON

## Notes
- Runs on localhost:5000 by default; stop with Ctrl+C.
- No cloud upload — processing stays on your device.

## Dependencies
- Core: Python 3.10+, Flask, librosa (see requirements.txt).
- Optional (improves quality/timing):
  - demucs — stems separation for cleaner drum detection (requires PyTorch; ffmpeg recommended).
  - madmom — beat tracking for more stable alignment.
  - Essentia (TempoCNN) — robust tempo estimation and audio feature extraction.
  - onnxruntime — run ONNX heads for genre classification.

## Genre Classification Models (optional)
- Discogs400 head and labels (+ MAEST embedding) are supported if present.
- Expected files inside a model directory (default: models/genre_discogs400-discogs-maest-10s-pw-1):
  - <model>.pb or <model>.onnx (classification head),
  - <model>.json (labels/metadata),
  - MAEST embedding graph .pb (can be placed in model dir or sibling “maest” dir).
- Environment variables:
  - RF_DISCOGS400_DIR — path to the model directory.
  - RF_MAEST_EMBED_PB — override path to the MAEST embedding graph.

## Platform Notes
- Primarily tested on Ubuntu; on Windows some optional packages may require extra setup.
- The server works without optional extras; they are detected at runtime and used when available.
