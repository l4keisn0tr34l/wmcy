"""Shared PyTorch device selection and portable checkpoint helpers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if value == "cuda" and not torch.cuda.is_available():
        detail = "PyTorch was built without CUDA" if torch.version.cuda is None else "CUDA runtime/device unavailable"
        raise RuntimeError(f"CUDA requested but unavailable: {detail}")
    return torch.device(value)


def device_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "device": str(device), "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        summary.update({
            "device_name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": properties.total_memory,
        })
    return summary


def cpu_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach checkpoint tensors to CPU so CUDA is never required to load them."""
    return {name: value.detach().cpu() for name, value in state.items()}
