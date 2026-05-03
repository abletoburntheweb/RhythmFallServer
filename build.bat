@echo off
:: ============================================================
:: Build RhythmFallServer.exe with PyInstaller
:: Run this from the repo root: build.bat
:: ============================================================

echo [Build] Installing / upgrading PyInstaller...
pip install --upgrade pyinstaller

echo.
echo [Build] Running PyInstaller...
pyinstaller RhythmFallServer.spec --noconfirm

echo.
if exist "dist\RhythmFallServer\RhythmFallServer.exe" (
    echo [Build] SUCCESS!
    echo Output: dist\RhythmFallServer\
    echo.
    echo Copy the entire dist\RhythmFallServer\ folder next to your game client.
    echo Also copy client_launcher.py into your game client project.
) else (
    echo [Build] FAILED — RhythmFallServer.exe not found in dist\
    echo Check the output above for errors.
    exit /b 1
)

pause
