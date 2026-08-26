"""Shared helpers for accelerator memory management."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    """Process and accelerator memory captured at one diagnostic checkpoint."""

    device_type: str
    process_rss: int | None
    tensor_allocated: int | None = None
    driver_allocated: int | None = None
    reserved: int | None = None
    recommended_max: int | None = None

    @property
    def driver_overhead_cache(self) -> int | None:
        """Driver memory not represented by live tensors.

        PyTorch documents driver memory as including allocator pools and
        MPS/MPSGraph allocations, so this delta is deliberately not labelled
        as strictly "non-tensor" memory.
        """
        if self.driver_allocated is None or self.tensor_allocated is None:
            return None
        return max(self.driver_allocated - self.tensor_allocated, 0)

    @property
    def non_tensor_driver(self) -> int | None:
        """Backward-compatible alias for the former, imprecise name."""
        return self.driver_overhead_cache


@dataclass
class DeviceMemoryTracker:
    """Log ordered memory checkpoints with optional synchronization and deltas."""

    device: Any
    synchronize: bool = False
    include_deltas: bool = False
    _previous: DeviceMemorySnapshot | None = field(default=None, init=False, repr=False)

    def log(self, stage: str, *, synchronize: bool | None = None) -> DeviceMemorySnapshot:
        should_synchronize = self.synchronize if synchronize is None else synchronize
        snapshot = log_device_memory(
            self.device,
            stage,
            synchronize=should_synchronize,
            previous=self._previous if self.include_deltas else None,
        )
        self._previous = snapshot
        return snapshot

    def reset(self) -> None:
        """Forget the previous checkpoint while retaining tracker settings."""
        self._previous = None


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


def synchronize_device(device: Any) -> bool:
    """Synchronize a supported accelerator, returning whether it was available."""
    kind = device_type(device)
    if kind not in {"cuda", "mps", "xpu"}:
        return False

    backend = getattr(torch, kind, None)
    if backend is None or not backend.is_available():
        return False

    if kind == "mps":
        backend.synchronize()
    else:
        backend.synchronize(device)
    return True


def get_device_memory_snapshot(
    device: Any,
    *,
    synchronize: bool = False,
) -> DeviceMemorySnapshot:
    """Capture current process and accelerator allocator memory.

    Synchronization is opt-in because it changes normal asynchronous execution
    timing. Detailed diagnostics enable it at stage boundaries so measurements
    from different backends are comparable.
    """
    if synchronize:
        synchronize_device(device)

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


def _format_delta(value: int | None) -> str:
    if value is None:
        return "n/a"
    gib = value / (1024 ** 3)
    if abs(gib) < 0.01:
        return f"{value / (1024 ** 2):+.1f} MiB"
    return f"{gib:+.2f} GiB"


def _snapshot_delta(
    current: DeviceMemorySnapshot,
    previous: DeviceMemorySnapshot | None,
) -> list[str]:
    if previous is None or previous.device_type != current.device_type:
        return []

    fields = []
    for label, current_value, previous_value in (
        ("RSS", current.process_rss, previous.process_rss),
        ("tensors", current.tensor_allocated, previous.tensor_allocated),
        ("driver", current.driver_allocated, previous.driver_allocated),
        ("overhead/cache", current.driver_overhead_cache, previous.driver_overhead_cache),
        ("reserved", current.reserved, previous.reserved),
    ):
        if current_value is not None and previous_value is not None:
            fields.append(f"{label}={_format_delta(current_value - previous_value)}")
    return fields


def format_device_memory(
    snapshot: DeviceMemorySnapshot,
    stage: str,
    *,
    previous: DeviceMemorySnapshot | None = None,
) -> str:
    """Format a compact, stable memory line shared by all inference entrypoints."""
    fields = [f"process RSS={_format_gib(snapshot.process_rss)}"]
    if snapshot.device_type == "mps" and snapshot.tensor_allocated is not None:
        fields.extend(
            [
                f"MPS tensors={_format_gib(snapshot.tensor_allocated)}",
                f"MPS driver={_format_gib(snapshot.driver_allocated)}",
                f"MPS driver overhead/cache={_format_gib(snapshot.driver_overhead_cache)}",
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
    message = f">> [Memory] {stage}: " + ", ".join(fields)
    delta_fields = _snapshot_delta(snapshot, previous)
    if delta_fields:
        message += "; delta since previous: " + ", ".join(delta_fields)
    return message


def log_device_memory(
    device: Any,
    stage: str,
    *,
    synchronize: bool = False,
    previous: DeviceMemorySnapshot | None = None,
) -> DeviceMemorySnapshot:
    """Print and return a process/device memory snapshot."""
    snapshot = get_device_memory_snapshot(device, synchronize=synchronize)
    print(format_device_memory(snapshot, stage, previous=previous))
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
        synchronize_device(device)
    backend.empty_cache()
    return True
