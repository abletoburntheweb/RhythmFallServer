"""GPU backend selection for audio-separator (CUDA / DirectML / CPU) and PyTorch (ADTOF)."""
from __future__ import annotations

import importlib.util
import os
from typing import Dict, Union

try:
    import torch

    _TorchDevice = torch.device
except Exception:
    torch = None  # type: ignore
    _TorchDevice = object


def _env_mode() -> str:
    return os.environ.get("RFALL_GPU", "auto").strip().lower()


def separator_hardware_options() -> Dict[str, object]:
    mode = _env_mode()
    if mode in ("0", "false", "no", "off", "cpu"):
        return {"use_directml": False, "label": "cpu"}

    if mode in ("directml", "dml", "amd", "intel", "amd/intel"):
        return {"use_directml": True, "label": "directml"}

    if mode in ("cuda", "nvidia"):
        return {"use_directml": False, "label": "cuda"}

    try:
        import torch as _torch

        if _torch.cuda.is_available():
            return {"use_directml": False, "label": "cuda"}
    except Exception:
        pass

    if importlib.util.find_spec("torch_directml") is not None:
        return {"use_directml": True, "label": "directml"}

    return {"use_directml": False, "label": "cpu"}


def resolve_torch_device_label() -> str:
    """PyTorch inference device for generic models: cpu | cuda | directml."""
    mode = _env_mode()
    if mode in ("0", "false", "no", "off", "cpu"):
        return "cpu"

    if mode in ("cuda", "nvidia"):
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    if mode in ("directml", "dml", "amd", "intel", "amd/intel"):
        if importlib.util.find_spec("torch_directml") is not None:
            return "directml"
        return "cpu"

    try:
        import torch as _torch

        if _torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    if importlib.util.find_spec("torch_directml") is not None:
        return "directml"

    return "cpu"


def resolve_torch_device() -> _TorchDevice:
    if torch is None:
        raise RuntimeError("torch is not installed")
    label = resolve_torch_device_label()
    if label == "cuda":
        return torch.device("cuda")
    if label == "directml":
        import torch_directml

        return torch_directml.device()
    return torch.device("cpu")


def resolve_adtof_torch_device_label() -> str:
    """ADTOF Frame-RNN uses GRU — DirectML cannot run fused GRU; CUDA or CPU only."""
    mode = _env_mode()
    if mode in ("0", "false", "no", "off", "cpu"):
        return "cpu"
    if mode in ("cuda", "nvidia"):
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    # directml / amd / auto on AMD: stems may use DML, ADTOF stays CPU
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_adtof_torch_device() -> _TorchDevice:
    if torch is None:
        raise RuntimeError("torch is not installed")
    if resolve_adtof_torch_device_label() == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def startup_gpu_message() -> str:
    mode = _env_mode()
    stem_label = str(separator_hardware_options().get("label", "cpu"))
    adtof_label = resolve_adtof_torch_device_label()

    stem_detail = stem_label
    if stem_label == "cuda":
        try:
            import torch as _torch

            name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "CUDA"
            stem_detail = f"CUDA ({name})"
        except Exception:
            stem_detail = "CUDA"
    elif stem_label == "directml":
        stem_detail = "DirectML (AMD/Intel ONNX)"

    adtof_detail = adtof_label
    if adtof_label == "cuda":
        adtof_detail = "CUDA"
    elif stem_label == "directml" and adtof_label == "cpu":
        adtof_detail = "CPU (GRU not supported on DirectML)"

    return (
        f"[Startup] GPU: stems={stem_detail}, ADTOF={adtof_detail}, RFALL_GPU={mode}"
    )
