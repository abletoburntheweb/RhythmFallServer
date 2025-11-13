# app/drum_generator.py
import os
import json
import numpy as np
import random
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

NOTES_DIR = Path("songs") / "notes"
SPLITTER_CACHE_DIR = Path("temp") / "demucs_cache"

def separate_audio_with_demucs(wav_path: str) -> Tuple[Optional[str], Optional[str]]:
    song_path_obj = Path(wav_path)
    song_name = song_path_obj.stem
    final_cache_dir = SPLITTER_CACHE_DIR / song_name

    final_no_vocals_path = final_cache_dir / "no_vocals.wav"
    final_vocals_path = final_cache_dir / "vocals.wav"

    if final_no_vocals_path.exists() and final_vocals_path.exists():
        print(f"[DrumGen] Используем кэшированные дорожки: {final_cache_dir}")
        return str(final_no_vocals_path), str(final_vocals_path)

    try:
        print(f"[DrumGen] Запускаю Demucs для {wav_path}...")

        cmd = ["demucs", "-n", "htdemucs", "--two-stems", "vocals",
               "-o", str(SPLITTER_CACHE_DIR), wav_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[DrumGen] Demucs завершён для {song_name}")

        demucs_output_subdir = SPLITTER_CACHE_DIR / "htdemucs" / song_name
        original_no_vocals = demucs_output_subdir / "no_vocals.wav"
        original_vocals = demucs_output_subdir / "vocals.wav"

        if original_no_vocals.exists():
            final_cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(original_no_vocals), str(final_no_vocals_path))
            if original_vocals.exists():
                shutil.move(str(original_vocals), str(final_vocals_path))
            print(f"[DrumGen] Файлы перемещены в: {final_cache_dir}")

            try:
                if demucs_output_subdir.exists():
                    shutil.rmtree(demucs_output_subdir)
                demucs_parent = SPLITTER_CACHE_DIR / "htdemucs"
                if demucs_parent.exists() and not any(demucs_parent.iterdir()):
                    demucs_parent.rmdir()
            except:
                pass

            return str(final_no_vocals_path), str(final_vocals_path)
        else:
            print(f"[DrumGen] Ошибка: Demucs не создал no_vocals в {demucs_output_subdir}")
            return None, None

    except subprocess.CalledProcessError as e:
        print(f"[DrumGen] Ошибка Demucs: {e}")
        print(f"[DrumGen] stderr: {e.stderr}")
        return None, None
    except FileNotFoundError:
        print("[DrumGen] Demucs не установлен. Используем оригинальный файл без разделения.")
        return str(wav_path), None
    except Exception as e:
        print(f"[DrumGen] Ошибка разделения Demucs: {e}")
        return None, None


def detect_kick_snare(y, sr) -> Tuple[List[float], List[float]]:
    try:
        import librosa

        y_harmonic, y_percussive = librosa.effects.hpss(y)

        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)
        onset_frames = librosa.util.peak_pick(
            onset_env,
            pre_max=3, post_max=3,
            pre_avg=10, post_avg=10,
            delta=onset_env.max() * 0.15,
            wait=1
        )

        spectral_centroids = librosa.feature.spectral_centroid(y=y_percussive, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_percussive)[0]

        kick_times, snare_times = [], []

        for i, frame in enumerate(onset_frames):
            if frame >= len(spectral_centroids):
                continue

            time = onset_times[frame]
            centroid = spectral_centroids[frame]
            zcr = zero_crossing_rate[frame]

            if centroid < np.percentile(spectral_centroids, 30) and zcr < np.percentile(zero_crossing_rate, 50):
                kick_times.append(time)
            elif centroid > np.percentile(spectral_centroids, 70) and zcr > np.percentile(zero_crossing_rate, 50):
                snare_times.append(time)
            else:
                S = np.abs(librosa.stft(y_percussive))
                freqs = librosa.fft_frequencies(sr=sr)

                if frame >= S.shape[1]:
                    continue

                spectrum = S[:, frame]
                kick_energy = np.sum(spectrum[(freqs >= 60) & (freqs <= 120)])
                snare_energy = np.sum(spectrum[(freqs >= 180) & (freqs <= 400)])

                if kick_energy > snare_energy:
                    kick_times.append(time)
                else:
                    snare_times.append(time)

        return kick_times, snare_times
    except Exception as e:
        print(f"[DrumGen] Ошибка детекции kick/snare: {e}")
        return [], []


def generate_drums_notes(song_path: str, bpm: float, lanes: int = 4) -> Optional[List[Dict]]:
    print(f"🎧 Генерация барабанных нот для: {song_path} (BPM: {bpm})")

    if not bpm or bpm <= 0:
        print(f"Ошибка: некорректный BPM ({bpm})")
        return None

    try:
        import librosa

        accompaniment_path, vocals_path = separate_audio_with_demucs(song_path)
        if not accompaniment_path:
            print("[DrumGen] Не удалось получить инструментальную дорожку")
            return None

        y, sr = librosa.load(accompaniment_path, sr=None, mono=True, dtype='float32')

        kick_times, snare_times = detect_kick_snare(y, sr)
        print(f"[DrumGen] Найдено {len(kick_times)} kick и {len(snare_times)} snare (после Demucs)")

        _, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=float(bpm), units='time')

        def sync_to_beats(hit_times):
            synced = []
            for t in hit_times:
                idx = np.argmin((beats - t) ** 2)
                beat_time = beats[idx]
                if abs(beat_time - t) <= 0.25:
                    synced.append(beat_time)
            return synced

        synced_kicks = sync_to_beats(kick_times)
        synced_snares = sync_to_beats(snare_times)

        song_offset = 0.0

        all_hits = []
        for t in synced_kicks:
            all_hits.append({
                "type": "KickNote",
                "time": t
            })

        for t in synced_snares:
            all_hits.append({
                "type": "SnareNote",
                "time": t
            })

        all_hits.sort(key=lambda x: x["time"])

        filtered_hits = []
        last_time = -1.0

        for hit in all_hits:
            if abs(hit["time"] - last_time) > 0.05:
                filtered_hits.append(hit)
                last_time = hit["time"]
            else:
                if filtered_hits and abs(filtered_hits[-1]["time"] - hit["time"]) < 0.05:
                    if filtered_hits[-1]["type"] != hit["type"]:
                        pass
                else:
                    filtered_hits.append(hit)
                last_time = hit["time"]

        notes = []
        last_lane_usage = {}

        for hit in filtered_hits:
            adjusted_time = hit["time"] + song_offset

            if adjusted_time <= 0:
                continue

            available_lanes = [lane for lane in range(lanes) if last_lane_usage.get(lane, -1) < adjusted_time]
            if not available_lanes:
                lane = min(range(lanes), key=lambda l: last_lane_usage.get(l, -1))
            else:
                lane = random.choice(available_lanes)

            last_lane_usage[lane] = adjusted_time

            notes.append({
                "type": hit["type"],
                "lane": lane,
                "time": float(adjusted_time)
            })

        notes.sort(key=lambda x: x["time"])

        print(f"✅ Сгенерировано {len(notes)} барабанных нот для {Path(song_path).name}")
        return notes

    except Exception as e:
        print(f"[DrumGen] Ошибка генерации барабанных нот: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_drums_notes(notes_data: List[Dict], song_path: str) -> bool:
    if not notes_data:
        print("[DrumGen] Нет данных нот для сохранения.")
        return False

    base_name = Path(song_path).stem
    song_folder = NOTES_DIR / base_name
    song_folder.mkdir(parents=True, exist_ok=True)

    notes_filename = f"{base_name}_drums.json"
    notes_path = song_folder / notes_filename

    try:
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            return obj

        notes_data_serializable = convert_types(notes_data)

        temp_path = notes_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(notes_data_serializable, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(notes_path)

        print(f"[DrumGen] Ноты (drums) сохранены в: {notes_path}")
        return True
    except Exception as e:
        print(f"[DrumGen] Ошибка сохранения нот в {notes_path}: {e}")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return False


def load_drums_notes(song_path: str) -> Optional[List[Dict]]:
    base_name = Path(song_path).stem
    song_folder = NOTES_DIR / base_name
    notes_filename = f"{base_name}_drums.json"
    notes_path = song_folder / notes_filename

    if not notes_path.exists():
        print(f"[DrumGen] Файл нот не найден: {notes_path}")
        return None

    try:
        with open(notes_path, 'r', encoding='utf-8') as f:
            notes_data = json.load(f)
        print(f"[DrumGen] Ноты (drums) загружены из: {notes_path}")
        return notes_data
    except Exception as e:
        print(f"[DrumGen] Ошибка загрузки нот из {notes_path}: {e}")
        return None