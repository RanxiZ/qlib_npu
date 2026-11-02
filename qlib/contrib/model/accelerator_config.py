# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Dict, Optional

from qlib.config import C


def merge_accelerator_config(accelerator: Optional[Dict[str, Any]]) -> None:
    """Merge ``qlib_init.accelerator`` into global config ``C``."""
    if not accelerator:
        return
    base = dict(C.get("accelerator") or {})
    base.update(accelerator)
    C["accelerator"] = base


def apply_accelerator_to_model_config(model_config: Optional[Dict[str, Any]]) -> None:
    """
    Fill missing model kwargs from ``C["accelerator"]`` (e.g. ``GPU`` from ``device_id``).
    Does not override explicit user settings in the workflow YAML.
    """
    if not model_config:
        return
    acc = C.get("accelerator") or {}
    if not acc:
        return
    kwargs = model_config.setdefault("kwargs", {})
    if "GPU" not in kwargs and "device_id" in acc:
        kwargs["GPU"] = acc["device_id"]
