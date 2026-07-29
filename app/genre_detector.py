# app/genre_detector.py
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

_EFFNET_ORT_SESSION = None
_EFFNET_ONNX_PATH: Optional[Path] = None
_EFFNET_PATCH_SIZE = 128
_EFFNET_PATCH_HOP = 62
_EFFNET_N_MELS = 96




def _default_effnet_model_dir() -> Path:
    env_p = os.environ.get("RF_EFFNET_DIR")
    if env_p:
        try:
            return Path(env_p)
        except Exception:
            pass
    return Path("models/discogs-effnet")


def _resolve_effnet_onnx(model_dir: Optional[Path] = None) -> Optional[Path]:
    md = Path(model_dir) if model_dir else _default_effnet_model_dir()
    env_p = os.environ.get("RF_EFFNET_ONNX")
    if env_p and Path(env_p).exists():
        return Path(env_p)
    for name in ("discogs-effnet-bsdynamic-1.onnx", "discogs-effnet-bs64-1.onnx"):
        p = md / name
        if p.exists():
            return p
    matches = sorted(md.glob("*.onnx"))
    return matches[0] if matches else None


def _resolve_effnet_json(model_dir: Optional[Path] = None) -> Optional[Path]:
    md = Path(model_dir) if model_dir else _default_effnet_model_dir()
    env_p = os.environ.get("RF_EFFNET_JSON")
    if env_p and Path(env_p).exists():
        return Path(env_p)
    onnx_path = _resolve_effnet_onnx(md)
    if onnx_path is not None:
        sibling = onnx_path.with_suffix(".json")
        if sibling.exists():
            return sibling
    for name in ("discogs-effnet-bsdynamic-1.json", "discogs-effnet-bs64-1.json"):
        p = md / name
        if p.exists():
            return p
    matches = sorted(md.glob("*.json"))
    return matches[0] if matches else None


def _load_effnet_labels(model_dir: Optional[Path] = None) -> List[str]:
    labels_path = _resolve_effnet_json(model_dir)
    if labels_path is None:
        return []
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes") or data.get("labels") or []
        return [str(c).strip().lower() for c in classes]
    except Exception:
        return []


def is_effnet_onnx_available(model_dir: Optional[Path] = None) -> bool:
    if not ORT_AVAILABLE:
        return False
    md = Path(model_dir) if model_dir else _default_effnet_model_dir()
    return _resolve_effnet_onnx(md) is not None and bool(_load_effnet_labels(md))


def genre_backend_name() -> str:
    if is_effnet_onnx_available():
        return "effnet-onnx"
    return "none"


def is_discogs400_available(model_dir: Optional[Path] = None) -> bool:
    return is_effnet_onnx_available(model_dir)


def _effnet_melspectrogram(audio_path: str) -> np.ndarray:
    import librosa

    y, _sr = librosa.load(audio_path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=16000,
        n_fft=512,
        hop_length=256,
        n_mels=_EFFNET_N_MELS,
    )
    mel = np.log10(10000.0 * mel + 1.0, dtype=np.float64)
    mel = mel.T.astype(np.float32)
    if mel.shape[0] < _EFFNET_PATCH_SIZE:
        pad = _EFFNET_PATCH_SIZE - mel.shape[0]
        mel = np.pad(mel, ((0, pad), (0, 0)), mode="constant")
    return mel


def _effnet_patches(mel: np.ndarray) -> np.ndarray:
    if mel.shape[0] < _EFFNET_PATCH_SIZE:
        pad = _EFFNET_PATCH_SIZE - mel.shape[0]
        mel = np.pad(mel, ((0, pad), (0, 0)), mode="constant")
    patches: list[np.ndarray] = []
    for start in range(0, mel.shape[0] - _EFFNET_PATCH_SIZE + 1, _EFFNET_PATCH_HOP):
        patches.append(mel[start : start + _EFFNET_PATCH_SIZE])
    if not patches:
        patches.append(mel[: _EFFNET_PATCH_SIZE])
    return np.stack(patches, axis=0)


def _get_effnet_session(onnx_path: Path):
    global _EFFNET_ORT_SESSION, _EFFNET_ONNX_PATH
    if _EFFNET_ORT_SESSION is None or _EFFNET_ONNX_PATH != onnx_path:
        _EFFNET_ONNX_PATH = onnx_path
        _EFFNET_ORT_SESSION = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
    return _EFFNET_ORT_SESSION


def classify_effnet_discogs_onnx(
    audio_path: str, top_k: int = 5, model_dir: Optional[Path] = None
) -> List[Tuple[str, float]]:
    md = Path(model_dir) if model_dir else _default_effnet_model_dir()
    if not is_effnet_onnx_available(md):
        print("[EffNet] Модель недоступна (onnx + json + onnxruntime)")
        return []
    labels = _load_effnet_labels(md)
    onnx_path = _resolve_effnet_onnx(md)
    if onnx_path is None or not labels:
        print("[EffNet] Не найдены onnx/json")
        return []
    try:
        mel = _effnet_melspectrogram(audio_path)
        patches = _effnet_patches(mel)
        sess = _get_effnet_session(onnx_path)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: patches})
        preds = np.asarray(outputs[0], dtype=np.float32)
        if preds.ndim == 2:
            scores = preds.mean(axis=0)
        else:
            scores = np.squeeze(preds)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(scores)[::-1][: max(1, int(top_k))]
        out: List[Tuple[str, float]] = []
        for i in idx:
            if 0 <= int(i) < len(labels):
                out.append((labels[int(i)], float(scores[int(i)])))
        if out:
            print(f"[EffNet] top: {out[0][0]} ({out[0][1]:.3f})")
        return out
    except Exception as e:
        print(f"[EffNet] Ошибка инференса: {e}")
        return []


def classify_discogs400(audio_path: str, top_k: int = 5, model_dir: Optional[Path] = None) -> List[Tuple[str, float]]:
    if is_effnet_onnx_available():
        return classify_effnet_discogs_onnx(audio_path, top_k=top_k)
    print("[Discogs400] Модель недоступна")
    return []


class MultiSourceGenreDetector:
    def __init__(self, config_path: str = None):
        if config_path is None:
            module_dir = Path(__file__).parent
            self.config_path = module_dir / "config.json"
        else:
            self.config_path = Path(config_path)
        self.config = self._load_config()
    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[GenreDetector] Файл конфигурации {self.config_path} не найден. Используются значения по умолчанию.")
            return {}
        except json.JSONDecodeError as e:
            print(f"[GenreDetector] Ошибка парсинга JSON в {self.config_path}: {e}. Используются значения по умолчанию.")
            return {}
    def detect_all_genres(self, artist: str, title: str, audio_path: Optional[str] = None) -> Dict[str, List[str]]:
        print(f"[GenreDetector] Поиск жанров для: {artist} - {title}")
        results = {}
        if audio_path and is_discogs400_available():
            audio_preds = classify_discogs400(audio_path, top_k=5)
            audio_labels = [label for label, prob in audio_preds]
        else:
            audio_labels = []
        results['audio_discogs400'] = audio_labels
        unique_genres = list(dict.fromkeys(audio_labels))
        print(f"[GenreDetector] Итоговые жанры: {unique_genres}")
        if audio_labels:
            print("[GenreDetector] Жанры по источникам:")
            print(f"   audio_discogs400: {audio_labels}")
        return {'all_genres': unique_genres, 'by_source': results}
def detect_genres(artist: str, title: str, audio_path: Optional[str] = None) -> List[str]:
    detector = MultiSourceGenreDetector()
    results = detector.detect_all_genres(artist, title, audio_path=audio_path)
    alias_map = _GENRE_ALIAS_MAP if isinstance(_GENRE_ALIAS_MAP, dict) else {}
    canonical_keys = set(_GENRE_CONFIGS.keys()) if isinstance(_GENRE_CONFIGS, dict) else set()
    mapped: List[str] = []
    seen = set()
    def norm(s: str) -> str:
        x = str(s).strip().lower()
        x = x.replace("—", "-").replace("_", " ").replace("  ", " ")
        x = x.replace(" - ", "-").replace("-", "-")
        x = x.replace("/", " / ")
        x = " ".join(x.split())
        return x
    def candidates(label: str) -> List[str]:
        k = norm(label)
        cands = [k]
        if '---' in k:
            parent, child = k.split('---', 1)
            child = child.strip()
            cands.append(child)
            cands.append(child.replace('-', ' '))
            cands.append(child.replace(' / ', ' '))
        cands.append(k.replace('---', ' '))
        cands.append(k.replace('---', ' ').replace('-', ' '))
        return list(dict.fromkeys([c.strip() for c in cands if c.strip()]))
    for raw in results.get('all_genres', []):
        found = None
        for cand in candidates(raw):
            if cand in canonical_keys:
                found = cand
                break
            if cand in alias_map:
                tgt = alias_map[cand]
                if tgt in canonical_keys:
                    found = tgt
                    break
            if '---' in cand:
                try:
                    _, sub = cand.split('---', 1)
                    sub_norm = sub.strip()
                    if sub_norm in canonical_keys:
                        found = sub_norm
                        break
                    sub_space = sub_norm.replace('-', ' ')
                    if sub_space in alias_map and alias_map[sub_space] in canonical_keys:
                        found = alias_map[sub_space]
                        break
                except Exception:
                    pass
        if found and found not in seen:
            mapped.append(found)
            seen.add(found)
    return mapped[:5]
from .drum_utils import load_genre_configs, load_genre_aliases, get_genre_params
_GENRE_CONFIGS = load_genre_configs()
_GENRE_ALIAS_MAP = load_genre_aliases()


def _norm_label(s: str) -> str:
    x = str(s).strip().lower()
    x = x.replace("—", "-").replace("_", " ").replace("  ", " ")
    x = x.replace(" - ", "-").replace("-", "-")
    x = x.replace("/", " / ")
    x = " ".join(x.split())
    return x


def _label_candidates(label: str) -> List[str]:
    k = _norm_label(label)
    cands = [k]
    if '---' in k:
        parent, child = k.split('---', 1)
        child = child.strip()
        cands.append(child)
        cands.append(child.replace('-', ' '))
        cands.append(child.replace(' / ', ' '))
    cands.append(k.replace('---', ' '))
    cands.append(k.replace('---', ' ').replace('-', ' '))
    return list(dict.fromkeys([c.strip() for c in cands if c.strip()]))


def _map_raw_label_to_canonical(raw_label: str) -> Optional[str]:
    alias_map = _GENRE_ALIAS_MAP if isinstance(_GENRE_ALIAS_MAP, dict) else {}
    canonical_keys = set(_GENRE_CONFIGS.keys()) if isinstance(_GENRE_CONFIGS, dict) else set()
    for cand in _label_candidates(raw_label):
        if cand in canonical_keys:
            return cand
        if cand in alias_map:
            tgt = alias_map[cand]
            if tgt in canonical_keys:
                return tgt
        if '---' in cand:
            try:
                _, sub = cand.split('---', 1)
                sub_norm = sub.strip()
                if sub_norm in canonical_keys:
                    return sub_norm
                sub_space = sub_norm.replace('-', ' ')
                if sub_space in alias_map and alias_map[sub_space] in canonical_keys:
                    return alias_map[sub_space]
            except Exception:
                pass
    return None


def detect_genre_predictions(audio_path: Optional[str], top_k: int = 5) -> List[Dict]:
    if not audio_path or not is_discogs400_available():
        return []
    raw_preds = classify_discogs400(audio_path, top_k=max(30, top_k * 6))
    by_canon: Dict[str, Dict] = {}
    for raw_label, score in raw_preds:
        canon = _map_raw_label_to_canonical(raw_label)
        if not canon:
            continue
        prev = by_canon.get(canon)
        score_f = float(score)
        if prev is None or score_f > float(prev.get("score", 0.0)):
            by_canon[canon] = {
                "id": canon,
                "score": score_f,
                "raw": str(raw_label),
            }
    items = sorted(by_canon.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    total = sum(max(0.0, x["score"]) for x in items) or 1.0
    for x in items:
        x["percent"] = round(100.0 * max(0.0, x["score"]) / total, 1)
    return items


def get_genre_config(genre_name: str) -> dict:
    key = genre_name.strip().lower() if isinstance(genre_name, str) else "groove"
    if key in _GENRE_CONFIGS:
        return _GENRE_CONFIGS[key]
    if key in _GENRE_ALIAS_MAP:
        canonical = _GENRE_ALIAS_MAP[key]
        return _GENRE_CONFIGS.get(canonical, _GENRE_CONFIGS.get("groove", {}))
    print(f"[GenreDetector] Жанр '{genre_name}' не найден, используем 'groove'")
    return _GENRE_CONFIGS.get("groove", {
        "min_note_distance": 0.05,
        "drum_start_window": 4.0,
        "drum_density_threshold": 0.5,
        "sync_tolerance_multiplier": 1.0
    })
