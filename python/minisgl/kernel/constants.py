"""Shared constants for kernel modules."""

from __future__ import annotations

from .utils import KernelConfig

DEFAULT_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
