# app/genre_detector.py
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
import numpy as np
try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except Exception:
    ESSENTIA_AVAILABLE = False
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False
def _default_model_dir() -> Path:
    env_p = os.environ.get("RF_DISCOGS400_DIR")
    if env_p:
        try:
            p = Path(env_p)
            return p
        except Exception:
            pass
    return Path("models/genre_discogs400-discogs-maest-10s-pw-1")
def _load_labels(model_dir: Path) -> List[str]:
    labels_path = model_dir / f"{model_dir.name}.json"
    if not labels_path.exists():
        return []
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes") or data.get("labels") or []
        return [str(c).strip().lower() for c in classes]
    except Exception:
        return []
def _resolve_embedding_pb(model_dir: Path) -> Optional[Path]:
    labels_path = model_dir / f"{model_dir.name}.json"
    if not labels_path.exists():
        return None
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_name = None
        inf = data.get("inference") or {}
        emb = inf.get("embedding_model") or {}
        model_name = emb.get("model_name")
        env_p = os.environ.get("RF_MAEST_EMBED_PB")
        if env_p and Path(env_p).exists():
            return Path(env_p)
        if model_name:
            cand1 = model_dir / f"{model_name}.pb"
            if cand1.exists():
                return cand1
            cand2 = model_dir.parent / "maest" / f"{model_name}.pb"
            if cand2.exists():
                return cand2
        parent = model_dir.parent
        for p in [parent / "discogs-maest-10s-pw-2.pb", model_dir / "discogs-maest-10s-pw-2.pb"]:
            if p.exists():
                return p
    except Exception:
        return None
    return None
def is_discogs400_available(model_dir: Optional[Path] = None) -> bool:
    md = Path(model_dir) if model_dir else _default_model_dir()
    pb = md / f"{md.name}.pb"
    onnx_path = md / f"{md.name}.onnx"
    js = md / f"{md.name}.json"
    emb = _resolve_embedding_pb(md)
    head_ok = pb.exists() or onnx_path.exists()
    return ESSENTIA_AVAILABLE and head_ok and js.exists() and emb is not None
def classify_discogs400(audio_path: str, top_k: int = 5, model_dir: Optional[Path] = None) -> List[Tuple[str, float]]:
    md = Path(model_dir) if model_dir else _default_model_dir()
    if not is_discogs400_available(md):
        print("[Discogs400] Model not available")
        return []
    labels = _load_labels(md)
    if not labels:
        print("[Discogs400] Labels not loaded")
        return []
    head_pb = md / f"{md.name}.pb"
    head_onnx = md / f"{md.name}.onnx"
    emb_pb = _resolve_embedding_pb(md)
    try:
        loader = es.MonoLoader(filename=audio_path, sampleRate=16000)
        audio = loader()
        if emb_pb is None or (not head_pb.exists() and not head_onnx.exists()):
            print("[Discogs400] Missing head or embedder graph")
            return []
        try:
            embedder = es.TensorflowPredictMAEST(graphFilename=str(emb_pb))
        except Exception:
            print("[Discogs400] Failed to init MAEST embedder")
            return []
        try:
            head = None
            try:
                head = es.TensorflowPredict(graphFilename=str(head_pb))
            except Exception:
                pass
            if head is None:
                try:
                    head = es.TensorflowPredict(graphFilename=str(head_pb), input="embeddings")
                except Exception:
                    pass
            if head is None:
                try:
                    head = es.TensorflowPredict(graphFilename=str(head_pb), output="PartitionedCall/Identity_1")
                except Exception:
                    pass
            if head is None:
                try:
                    head = es.TensorflowPredict(graphFilename=str(head_pb), input="embeddings", output="PartitionedCall/Identity_1")
                except Exception:
                    pass
            tf_head_failed = head is None
        except Exception:
            tf_head_failed = True
        try:
            emb = embedder(audio)
        except Exception:
            print("[Discogs400] Embedding extraction failed")
            return []
        emb_arr = np.asarray(emb, dtype=np.float32)
        try:
            print(f"[Discogs400] Raw embedding shape: {emb_arr.shape}")
        except Exception:
            pass
        if emb_arr.shape[-1] == 400:
            try:
                y = emb_arr
                while y.ndim > 2:
                    y = y.mean(axis=0)
                if y.ndim == 2 and y.shape[0] > 1:
                    y = y.mean(axis=0)
                y = np.squeeze(y)
                y = y.astype(np.float32)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                idx = np.argsort(y)[::-1][:max(1, int(top_k))]
                out2: List[Tuple[str, float]] = []
                for i in idx:
                    if 0 <= int(i) < len(labels):
                        out2.append((labels[int(i)], float(y[int(i)])))
                if not out2:
                    print("[Discogs400] No top-k predictions produced (direct)")
                return out2
            except Exception:
                print("[Discogs400] Direct predictions path failed")
                return []
        if emb_arr.ndim == 2:
            emb_arr = np.expand_dims(emb_arr, 0)
        if emb_arr.shape[1] != 560:
            T = emb_arr.shape[1]
            if T > 560:
                start = (T - 560) // 2
                emb_arr = emb_arr[:, start:start + 560, :]
            else:
                pad_before = (560 - T) // 2
                pad_after = 560 - T - pad_before
                emb_arr = np.pad(emb_arr, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
        if emb_arr.shape[2] != 768:
            print(f"[Discogs400] Unexpected embedding dim: {emb_arr.shape}. Expected (*, 560, 768)")
            return []
        try:
            print(f"[Discogs400] Prepared embedding shape: {emb_arr.shape}")
        except Exception:
            pass
        pred = None
        if not tf_head_failed and head is not None:
            try:
                pred = head(emb_arr)
            except Exception:
                print("[Discogs400] Head inference failed")
                pred = None
        if pred is None and ORT_AVAILABLE:
            if head_onnx.exists():
                try:
                    sess = ort.InferenceSession(str(head_onnx), providers=["CPUExecutionProvider"])
                    inputs = sess.get_inputs()
                    input_name = inputs[0].name if inputs else "embeddings"
                    run_input: Dict[str, np.ndarray] = {}
                    run_input[input_name] = emb_arr.astype(np.float32)
                    output_names = [o.name for o in sess.get_outputs()]
                    if "activations" in output_names:
                        outputs = sess.run(["activations"], run_input)
                        pred = outputs[0]
                    else:
                        outputs = sess.run(None, run_input)
                        if isinstance(outputs, list) and len(outputs) >= 2:
                            pred = outputs[1]
                        elif isinstance(outputs, list) and len(outputs) >= 1:
                            pred = outputs[0]
                except Exception as e:
                    print(f"[Discogs400] ONNX head inference failed: {e}")
                    pred = None
        if pred is None:
            print("[Discogs400] Failed to init head graph")
            return []
        if isinstance(pred, list) or isinstance(pred, tuple):
            if len(pred) >= 2:
                y = np.array(pred[1]).astype(np.float32)
            else:
                y = np.array(pred[0]).astype(np.float32)
        else:
            y = np.array(pred).astype(np.float32)
        if y.ndim == 2:
            scores = y.mean(axis=0)
        elif y.ndim == 1:
            scores = y
        else:
            scores = np.squeeze(y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(scores)[::-1][:max(1, int(top_k))]
        out: List[Tuple[str, float]] = []
        for i in idx:
            if 0 <= int(i) < len(labels):
                out.append((labels[int(i)], float(scores[int(i)])))
        if not out:
            print("[Discogs400] No top-k predictions produced")
        return out
    except Exception:
        print("[Discogs400] Unexpected error during classification")
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
        print(f"🔍 Поиск жанров для: {artist} - {title}")
        results = {}
        if audio_path and is_discogs400_available():
            audio_preds = classify_discogs400(audio_path, top_k=5)
            audio_labels = [label for label, prob in audio_preds]
        else:
            audio_labels = []
        results['audio_discogs400'] = audio_labels
        unique_genres = list(set(audio_labels))
        print(f"📊 Итоговые жанры: {unique_genres}")
        if audio_labels:
            print(f"📊 Жанры по источникам:")
            print(f"   Audio_discogs400: {audio_labels}")
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
def get_genre_config(genre_name: str) -> dict:
    key = genre_name.strip().lower() if isinstance(genre_name, str) else "groove"
    if key in _GENRE_CONFIGS:
        return _GENRE_CONFIGS[key]
    if key in _GENRE_ALIAS_MAP:
        canonical = _GENRE_ALIAS_MAP[key]
        return _GENRE_CONFIGS.get(canonical, _GENRE_CONFIGS.get("groove", {}))
    print(f"[GenreDetector] Жанр '{genre_name}' не найден, используем 'groove'")
    return _GENRE_CONFIGS.get("groove", {
        "pattern_style": "groove",
        "min_note_distance": 0.05,
        "drum_start_window": 4.0,
        "drum_density_threshold": 0.5,
        "sync_tolerance_multiplier": 1.0
    })
