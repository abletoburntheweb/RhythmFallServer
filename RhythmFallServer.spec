# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for RhythmFallServer
# Build with:  pyinstaller RhythmFallServer.spec

import sys
from pathlib import Path

block_cipher = None

# ---------------------------------------------------------------------------
# Hidden imports needed because PyInstaller misses dynamic imports
# ---------------------------------------------------------------------------
HIDDEN_IMPORTS = [
    # Flask / Werkzeug internals
    "flask",
    "werkzeug",
    "werkzeug.serving",
    "werkzeug.debug",
    # Audio
    "librosa",
    "librosa.core",
    "librosa.feature",
    "librosa.beat",
    "soundfile",
    "scipy",
    "scipy.signal",
    "scipy.io",
    "scipy.io.wavfile",
    "numpy",
    # ML / inference
    "onnxruntime",
    "onnxruntime.capi",
    # demucs
    "demucs",
    "demucs.pretrained",
    "demucs.apply",
    # audio-separator
    "audio_separator",
    # madmom
    "madmom",
    "madmom.audio",
    "madmom.audio.signal",
    "madmom.features",
    "madmom.features.beats",
    "madmom.ml",
    "madmom.ml.nn",
    # essentia (optional; comment out if not installed)
    "essentia",
    "essentia.standard",
    # app modules
    "app",
    "app.routes",
    "app.bpm_analyzer",
    "app.drum_generator",
    "app.drum_utils",
    "app.generation_presets",
    "app.genre_detector",
    "app.audio_analysis",
    "app.audio_separator",
    "app.note_types",
    "app.shutdown",
]

# ---------------------------------------------------------------------------
# Data files bundled alongside the exe
# ---------------------------------------------------------------------------
DATAS = [
    # JSON config files
    ("app/config.json",        "app"),
    ("app/genre_configs.json", "app"),
    ("app/genre_aliases.json", "app"),
    # models/ directory (place ML models here before building)
    ("models",                 "models"),
]

a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RhythmFallServer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,       # no console window — runs silently in background
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RhythmFallServer",
)
