"""Runtime configuration helpers that must run before importing PyTorch."""

from __future__ import annotations

import os
import platform
from typing import Optional


def configure_mps_environment(
    *,
    low_watermark_ratio: str = "0.4",
    high_watermark_ratio: str = "0.6",
) -> Optional[dict[str, str]]:
    """Apply conservative macOS MPS defaults without overriding user settings."""
    if platform.system() != "Darwin":
        return None

    defaults = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": low_watermark_ratio,
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": high_watermark_ratio,
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in defaults}
