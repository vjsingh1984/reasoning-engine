"""
Device abstraction for CUDA (NVIDIA) and ROCm (AMD) GPU support.

Provides unified device selection, auto-detection, and helper utilities.
PyTorch's torch.cuda API works transparently for both NVIDIA CUDA and AMD ROCm,
so the common code path is shared — only installation differs.
"""

import torch
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the compute device based on user preference or auto-detection.

    Priority: explicit choice > CUDA/ROCm GPU > CPU

    Args:
        device: One of "cuda", "rocm", "cpu", or None for auto-detect.
                "rocm" is an alias for "cuda" since ROCm uses PyTorch's CUDA API.

    Returns:
        torch.device instance
    """
    if device is not None:
        device = device.lower()
        if device in ("cuda", "rocm"):
            if not torch.cuda.is_available():
                backend = "ROCm" if device == "rocm" else "CUDA"
                raise RuntimeError(
                    f"{backend} requested but torch.cuda.is_available() is False. "
                    f"Install the correct PyTorch build for your GPU."
                )
            return torch.device("cuda")
        elif device == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device: {device!r}. Use 'cuda', 'rocm', or 'cpu'.")

    # Auto-detect
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """
    Return diagnostic info about the selected device.

    Args:
        device: torch.device to inspect

    Returns:
        Dictionary with device details
    """
    info = {"device": str(device), "type": device.type}

    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "gpu_count": torch.cuda.device_count(),
            # ROCm reports via the same API; the name typically contains "gfx" or "Radeon"
            "backend": "ROCm" if "gfx" in props.name.lower() or "radeon" in props.name.lower() else "CUDA",
        })
    else:
        info["backend"] = "CPU"

    return info


def print_device_info(device: torch.device) -> None:
    """Print a summary of the selected device."""
    info = get_device_info(device)
    print(f"Device: {info['device']} ({info['backend']})")
    if info["type"] == "cuda":
        print(f"  GPU: {info['name']}")
        print(f"  Memory: {info['total_memory_gb']} GB")
        print(f"  GPU count: {info['gpu_count']}")


def to_device(tensor_or_model, device: torch.device):
    """
    Move a tensor or model to the specified device.

    Args:
        tensor_or_model: A torch.Tensor or nn.Module
        device: Target device

    Returns:
        The tensor/model on the target device
    """
    return tensor_or_model.to(device)
