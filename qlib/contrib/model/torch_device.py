# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Optional, Union

import torch

from qlib.config import C
from qlib.log import get_module_logger

_NPU_AVAILABLE: Optional[bool] = None


def is_npu_available() -> bool:
    global _NPU_AVAILABLE
    if _NPU_AVAILABLE is not None:
        return _NPU_AVAILABLE
    try:
        import torch_npu  # noqa: F401

        _NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
    except (ImportError, AttributeError):
        _NPU_AVAILABLE = False
    return _NPU_AVAILABLE


def resolve_torch_device(
    gpu: Union[int, str] = 0,
    device_type: Optional[str] = None,
) -> torch.device:
    """
    Resolve a torch device for Qlib PyTorch models.

    Parameters
    ----------
    gpu : int or str
        Device index for cuda/npu when ``gpu >= 0``, or a full device string (e.g. ``cuda:0``, ``cpu``).
        ``gpu < 0`` forces CPU.
    device_type : str, optional
        ``auto`` | ``cuda`` | ``npu`` | ``cpu``. Defaults to ``C["accelerator"]["device_type"]`` or ``auto``.
    """
    logger = get_module_logger("torch_device")

    if isinstance(gpu, str):
        return torch.device(gpu)

    acc = C.get("accelerator") or {}
    if device_type is None:
        device_type = acc.get("device_type", "auto")
    device_type = str(device_type).lower()

    if gpu < 0 or device_type == "cpu":
        return torch.device("cpu")

    device_id = int(gpu)
    cuda_ok = torch.cuda.is_available()
    npu_ok = is_npu_available()

    if device_type == "cuda":
        if cuda_ok:
            return torch.device(f"cuda:{device_id}")
        logger.warning("CUDA is not available; falling back to CPU for training.")
        return torch.device("cpu")

    if device_type == "npu":
        if npu_ok:
            return torch.device(f"npu:{device_id}")
        logger.warning("NPU is not available; falling back to CPU for training.")
        return torch.device("cpu")

    # auto: prefer CUDA, then NPU, then CPU
    if cuda_ok:
        return torch.device(f"cuda:{device_id}")
    if npu_ok:
        return torch.device(f"npu:{device_id}")
    return torch.device("cpu")


def use_torch_device(device: Union[torch.device, str]) -> bool:
    if isinstance(device, str):
        device = torch.device(device)
    return device.type != "cpu"


def empty_torch_cache(device: Union[torch.device, str]) -> None:
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "npu" and is_npu_available():
        torch.npu.empty_cache()
