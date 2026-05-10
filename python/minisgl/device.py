"""Explicit device helpers for Mini-SGLang runtime code."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch


def resolve_device(device: str | torch.device, *, rank: int = 0) -> torch.device:
    if str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{rank}")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device(f"cuda:{rank}")
    return resolved


def is_cuda(device: str | torch.device) -> bool:
    return torch.device(device).type == "cuda"


def is_cpu(device: str | torch.device) -> bool:
    return torch.device(device).type == "cpu"


def supports_pinned_memory(device: str | torch.device) -> bool:
    return is_cuda(device)


def mem_get_info(device: str | torch.device) -> tuple[int, int]:
    resolved = torch.device(device)
    if is_cuda(resolved):
        return torch.cuda.mem_get_info(resolved)

    import psutil

    mem = psutil.virtual_memory()
    return mem.available, mem.total


@contextmanager
def nvtx_range(device: str | torch.device, msg: str) -> Generator[None, None, None]:
    if is_cuda(device):
        import torch.cuda.nvtx as nvtx

        with nvtx.range(msg):
            yield
    else:
        _ = msg
        yield


__all__ = [
    "resolve_device",
    "is_cuda",
    "is_cpu",
    "supports_pinned_memory",
    "mem_get_info",
    "nvtx_range",
]
