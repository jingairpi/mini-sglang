from __future__ import annotations

from typing import Tuple

import torch

from .base import BaseOP
from minisgl import device as device_mod


def _cpu_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
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
        if device_mod.is_cpu():
            self.rmsnorm = _cpu_rmsnorm
        else:
            from flashinfer import rmsnorm
            self.rmsnorm = rmsnorm

        self.eps = eps
        self.weight = torch.ones(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        if device_mod.is_cpu():
            self.rmsnorm = _cpu_rmsnorm
            self.fused_add_rmsnorm = _cpu_fused_add_rmsnorm
        else:
            from flashinfer import fused_add_rmsnorm, rmsnorm
            self.rmsnorm = rmsnorm
            self.fused_add_rmsnorm = fused_add_rmsnorm

        self.eps = eps
        self.weight = torch.ones(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
