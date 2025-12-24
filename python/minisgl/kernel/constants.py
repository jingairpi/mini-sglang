from __future__ import annotations

"""Shared constants for kernel modules."""

from .utils import KernelConfig

# Default kernel configuration used by index and store kernels
DEFAULT_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
