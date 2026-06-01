from __future__ import annotations

from typing import Tuple

import torch
from minisgl import device as device_mod

from .base import BaseOP


def _cpu_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None
) -> torch.Tensor:
    """CPU implementation of RMSNorm. Writes to `out` if provided."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    out_tensor = x * weight
    if out is not None:
        out.copy_(out_tensor)
        return out
    return out_tensor


def _cpu_fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """Fused add + rmsnorm: residual += x, then x = rmsnorm(residual). Both in-place."""
    residual.add_(x)
    normed = _cpu_rmsnorm(residual, weight, eps)
    x.copy_(normed)


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.ones(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        _rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.ones(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _rmsnorm(x, self.weight, self.eps), x
        _fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


def _rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None
) -> torch.Tensor:
    if device_mod.is_cpu(x.device):
        return _cpu_rmsnorm(x, weight, eps, out=out)

    from flashinfer import rmsnorm

    return rmsnorm(x, weight, eps, out=out)


def _fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    if device_mod.is_cpu(x.device):
        return _cpu_fused_add_rmsnorm(x, residual, weight, eps)

    from flashinfer import fused_add_rmsnorm

    return fused_add_rmsnorm(x, residual, weight, eps)
