# RhythmFallServer

Локальный Flask-сервер для [RhythmFall](https://github.com/abletoburntheweb/RhythmFall): анализирует аудио и возвращает автоматически сгенерированные чарты. Работает на том же ПК, что и клиент Godot (или внутри Windows-установки игры как bundled worker).

Языки: [English](./README.md) | Русский

**Клиент:** https://github.com/abletoburntheweb/RhythmFall  
**Модели:** в репозиторий не входят. Скачайте из [Releases](https://github.com/abletoburntheweb/RhythmFallServer/releases) или запустите `worker\download_effnet_models.ps1` из клиента и положите файлы в `models/`.

## Что это

- Локальный HTTP API (Flask) для BPM, ударных чартов, баса (бета), жанровых подсказок и sidecar Rhythm DNA
- Пишет читаемые **`.rf`** (RFC v1) и опционально **`.rfd`** для ударных
- Полностью локально — аудио не уходит во внешние сервисы
- На Windows клиент может поднимать сервер через `RhythmFallServer.exe` + готовый `.venv`

## Как это работает

1. Клиент передаёт аудио и метаданные: инструмент, **цель** (`original` / `arcade`), сложность для Аркады, дорожки, опциональные слайдеры.
2. Сервер оценивает темп (TempoCNN / fallback), опционально делит стемы (Demucs / audio-separator), детектирует удары (**ADTOF** по умолчанию, эвристика — fallback).
3. Политики цели + жанровые профили задают плотность и раскладку по линиям; для Arcade — эргономичный lane router.
4. В ответе — чарт / пути; для ударных рядом с `.rf` печётся `.rfd`.

**Имена стемов**

| Цель | Файлы (примеры) |
| --- | --- |
| **Оригинал** | `drums_original.rf`, `bass_original.rf` (один документальный bake; legacy `*_original_standard` принимается) |
| **Аркада** | `drums_arcade_relaxed.rf` / `_standard` / `_dense` (Лёгкая / Средняя / Сложная), аналогично для баса |

## Быстрый старт (Windows — рекомендуется)

Скрипты из репозитория **клиента** (`worker/`), этот сервер — как `RhythmFallServer/` или `RhythmFallServer-main/` рядом с `worker/`:

```powershell
powershell -ExecutionPolicy Bypass -File worker\install_windows_server.ps1 -Gpu auto
powershell -ExecutionPolicy Bypass -File worker\download_effnet_models.ps1
powershell -ExecutionPolicy Bypass -File worker\build_server_launcher.ps1
```

- `-Gpu auto|nvidia|amd|cpu` — выбор CUDA / DirectML / CPU **при сборке venv** (установщик игры про GPU не спрашивает).
- Нужны **Python 3.9–3.11** и **ffmpeg** в PATH для стемов.
- Создаёт `RhythmFallServer\.venv` и `worker\windows_python.path`.
- Запуск: `RhythmFallServer.exe` или `python run.py` — http://127.0.0.1:5000 должен отвечать.

Ручной venv (любая ОС):

```bash
python -m venv .venv
# активировать, затем:
pip install -r requirements.txt
python run.py
```

## Эндпоинты (кратко)

| Метод | Путь | Назначение |
| --- | --- | --- |
| GET | `/` / `/health` | Жив ли сервер + список путей |
| POST | `/analyze_bpm` | Темп из аудио |
| POST | `/generate_drums` | Ударные `.rf` (+ `.rfd`) |
| POST | `/generate_bass` | Бас `.rf` (бета; без `.rfd`) |
| GET | `/task_status`, `/task_result` | Статус асинхронной задачи |
| POST | `/cancel_task` | Отмена |
| GET/POST | `/rhythm_dna`, `/rhythm_dna_sidecar` | Отчёт / sidecar DNA |
| GET/POST | `/storage_usage`, `/storage_reclaim` | Очистка temp |

Точные поля запросов задаёт клиент RhythmFall (`generation_api_client.gd`).

## Примечания

- По умолчанию: `localhost:5000`. Остановка — Ctrl+C (или закрыть консоль launcher).
- Без облака — обработка на устройстве.
- Первое разделение стемов дольше; дальше — кэш в `temp_uploads/`.
- GPU-пакеты опциональны; на CPU стемы просто медленнее.
- WSL для релизного Windows-пути устарел — используйте native venv.

## Зависимости

- База: Python **3.9–3.11**, Flask, librosa, NumPy 1.x (см. `requirements.txt`)
- Windows-стек через install-скрипт: TempoCNN, EffNet/ONNX, Demucs / audio-separator, опционально madmom, **ADTOF**
- Система: **ffmpeg** для стемов

## Модели жанров (опционально)

- Discogs / EffNet-подобные головы в `models/`, если файлы на месте
- Переменные вроде `RF_DISCOGS400_DIR`, `RF_MAEST_EMBED_PB` — см. код / docs
- Бэкенд ударных: `RFALL_DRUM_BACKEND=adtof_fast` (после install) или `heuristic`

## Платформы

- **Windows** — основной целевой путь для bundled worker в клиенте.
- Упаковка в игру: скопировать `.venv`, `app/`, `models/`, `run.py` и `RhythmFallServer.exe` рядом с `RhythmFall.exe` — см. `RFALL/BUILD.txt` в клиенте.
