from __future__ import annotations

"""Device abstraction layer for Mini-SGLang.

This module provides a centralized interface for device-specific operations,
enabling seamless switching between CUDA and CPU execution modes.
"""

import contextlib
from typing import Generator, Tuple

import torch

_DEVICE = None


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE


def set_device(device: str) -> None:
    global _DEVICE
    if device == "auto":
        _DEVICE = None  # Will be detected in get_device()
    else:
        _DEVICE = torch.device(device)


def is_cuda() -> bool:
    return get_device().type == "cuda"


def is_cpu() -> bool:
    return get_device().type == "cpu"


def synchronize() -> None:
    if is_cuda():
        torch.cuda.synchronize()


def empty_cache() -> None:
    if is_cuda():
        torch.cuda.empty_cache()


def mem_get_info() -> Tuple[int, int]:
    if is_cuda():
        return torch.cuda.mem_get_info()
    else:
        # psutil dependency is required for CPU memory info
        import psutil
        mem = psutil.virtual_memory()
        return mem.available, mem.total


@contextlib.contextmanager
def nvtx_range(msg: str) -> Generator[None, None, None]:
    if is_cuda():
        import torch.cuda.nvtx as nvtx
        with nvtx.range(msg):
            yield
    else:
        yield


@contextlib.contextmanager
def noop_context() -> Generator[None, None, None]:
    yield


__all__ = [
    "get_device",
    "set_device",
    "is_cuda",
    "is_cpu",
    "synchronize",
    "empty_cache",
    "mem_get_info",
    "nvtx_range",
    "noop_context",
]
