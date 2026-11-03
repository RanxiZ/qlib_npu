# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helpers for accelerator (NPU) inference.

These mirror the pattern introduced for the MLP model (``pytorch_nn.py``) and
are reused by the RNN models so the inference data pipeline behaves the same
way everywhere:

- upload once to the device (``to_infer_device``), reusing a cached device
  tensor when ``predict`` is called repeatedly on the same feature object;
- avoid the slow strided host-to-device copy that ``DataFrame.values``
  (Fortran-ordered) triggers by uploading the flat contiguous buffer and
  transposing on the accelerator;
- optional fp16 host->device transfer (halved bytes) with a cast back to fp32
  on the accelerator before the forward pass.
"""
from __future__ import annotations

from typing import Callable

import weakref

import numpy as np
import pandas as pd
import torch


class InferTensorCache:
    """Weakref-based cache for the last uploaded inference features."""

    def __init__(self):
        self._src = None
        self._tensor = None

    def get(self, data):
        src = self._src
        if src is not None and self._tensor is not None and src() is data:
            return self._tensor
        return None

    def set(self, data, tensor):
        self._src = weakref.ref(data)
        self._tensor = tensor


def to_infer_device(data, device, use_fp16=False, cache=None):
    """Upload feature data to ``device`` for inference (see module docstring)."""
    if isinstance(data, torch.Tensor):
        return data.to(device)

    if cache is not None:
        hit = cache.get(data)
        if hit is not None:
            return hit

    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    if arr.ndim == 2 and arr.flags["F_CONTIGUOUS"] and not arr.flags["C_CONTIGUOUS"]:
        # zero-copy contiguous view of a Fortran-ordered array
        cpu = torch.from_numpy(arr.ravel(order="F"))
        need_transpose = True
    else:
        cpu = torch.from_numpy(np.ascontiguousarray(arr))
        need_transpose = False

    if use_fp16:
        cpu = cpu.half()
    cpu = cpu.to(device)
    if need_transpose:
        cpu = cpu.view(arr.shape[1], arr.shape[0]).t().contiguous()
    if use_fp16:
        cpu = cpu.float()

    if cache is not None:
        cache.set(data, cpu)
    return cpu


def predict_in_batches(
    data: torch.Tensor,
    batch_size: int,
    forward: Callable[[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    """Run ``forward`` over a resident device tensor in batches.

    Returns the concatenated predictions as a contiguous CPU numpy array. All
    models share the same batched inference loop; this keeps the per-model
    ``predict`` implementation to a one-liner around the common device-upload
    helper (:func:`to_infer_device`) plus this loop.
    """
    preds = []
    with torch.no_grad():
        for begin in range(0, len(data), batch_size):
            x = data[begin : begin + batch_size]
            preds.append(forward(x).detach())
    return torch.cat(preds).cpu().numpy()
