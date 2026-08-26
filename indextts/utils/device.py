"""Shared helpers for accelerator memory management."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    """Process and accelerator memory captured without synchronizing the device."""

    device_type: str
    process_rss: int | None
    tensor_allocated: int | None = None
    driver_allocated: int | None = None
    reserved: int | None = None
    recommended_max: int | None = None

    @property
    def non_tensor_driver(self) -> int | None:
        if self.driver_allocated is None or self.tensor_allocated is None:
            return None
        return max(self.driver_allocated - self.tensor_allocated, 0)


def device_type(device: Any) -> str:
    """Return a normalized device type for strings and ``torch.device`` objects."""
    if isinstance(device, torch.device):
        return device.type
    return str(device or "cpu").split(":", 1)[0].lower()


def _process_rss_bytes() -> int | None:
    """Return the process's current resident set size when psutil is available."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except (psutil.Error, OSError):
        return None


def get_device_memory_snapshot(device: Any) -> DeviceMemorySnapshot:
    """Capture current process and accelerator allocator memory.

    This intentionally does not synchronize the accelerator, so adding
    diagnostics does not change normal inference timing.
    """
    kind = device_type(device)
    process_rss = _process_rss_bytes()
    backend = getattr(torch, kind, None)
    if backend is None or not backend.is_available():
        return DeviceMemorySnapshot(device_type=kind, process_rss=process_rss)

    if kind == "mps":
        recommended_max = None
        recommended_max_memory = getattr(backend, "recommended_max_memory", None)
        if recommended_max_memory is not None:
            recommended_max = recommended_max_memory()
        return DeviceMemorySnapshot(
            device_type=kind,
            process_rss=process_rss,
            tensor_allocated=backend.current_allocated_memory(),
            driver_allocated=backend.driver_allocated_memory(),
            recommended_max=recommended_max,
        )

    if kind in {"cuda", "xpu"}:
        memory_allocated = getattr(backend, "memory_allocated", None)
        memory_reserved = getattr(backend, "memory_reserved", None)
        return DeviceMemorySnapshot(
            device_type=kind,
            process_rss=process_rss,
            tensor_allocated=memory_allocated(device) if memory_allocated else None,
            reserved=memory_reserved(device) if memory_reserved else None,
        )

    return DeviceMemorySnapshot(device_type=kind, process_rss=process_rss)


def _format_gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / (1024 ** 3):.2f} GiB"


def format_device_memory(snapshot: DeviceMemorySnapshot, stage: str) -> str:
    """Format a compact, stable memory line shared by all inference entrypoints."""
    fields = [f"process RSS={_format_gib(snapshot.process_rss)}"]
    if snapshot.device_type == "mps" and snapshot.tensor_allocated is not None:
        fields.extend(
            [
                f"MPS tensors={_format_gib(snapshot.tensor_allocated)}",
                f"MPS driver={_format_gib(snapshot.driver_allocated)}",
                f"MPS non-tensor driver={_format_gib(snapshot.non_tensor_driver)}",
            ]
        )
        if snapshot.recommended_max is not None:
            fields.append(f"MPS recommended={_format_gib(snapshot.recommended_max)}")
            if snapshot.driver_allocated is not None and snapshot.recommended_max:
                ratio = 100 * snapshot.driver_allocated / snapshot.recommended_max
                fields.append(f"MPS driver/recommended={ratio:.1f}%")
    elif snapshot.device_type in {"cuda", "xpu"} and snapshot.tensor_allocated is not None:
        label = snapshot.device_type.upper()
        fields.extend(
            [
                f"{label} tensors={_format_gib(snapshot.tensor_allocated)}",
                f"{label} reserved={_format_gib(snapshot.reserved)}",
            ]
        )
    return f">> [Memory] {stage}: " + ", ".join(fields)


def log_device_memory(device: Any, stage: str) -> DeviceMemorySnapshot:
    """Print and return a non-synchronizing process/device memory snapshot."""
    snapshot = get_device_memory_snapshot(device)
    print(format_device_memory(snapshot, stage))
    return snapshot


def clear_device_cache(
    device: Any,
    *,
    collect_garbage: bool = False,
    synchronize: bool = False,
) -> bool:
    """Release unused allocator memory for CUDA, MPS, or XPU.

    CPU and unavailable accelerators are intentional no-ops. The return value
    indicates whether an accelerator cache was cleared.
    """
    if collect_garbage:
        gc.collect()

    kind = device_type(device)
    if kind not in {"cuda", "mps", "xpu"}:
        return False

    backend = getattr(torch, kind, None)
    if backend is None or not backend.is_available():
        return False

    if synchronize:
        if kind == "mps":
            backend.synchronize()
        else:
            backend.synchronize(device)
    backend.empty_cache()
    return True
