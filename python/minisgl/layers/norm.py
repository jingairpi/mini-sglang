from typing import Tuple

import torch

from .base import BaseOP
from minisgl import device as device_mod


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        if device_mod.is_cpu():
            def manual_rmsnorm(x, weight, eps, out=None):
                """CPU implementation of RMSNorm."""
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
            self.rmsnorm = manual_rmsnorm
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
            self.rmsnorm = RMSNorm(size, eps).rmsnorm

            def manual_fused_cpu(x, residual, weight, eps):
                # Fused add + rmsnorm: residual += x, then x = rmsnorm(residual)
                residual.add_(x)
                normed = self.rmsnorm(residual, weight, eps)
                x.copy_(normed)

            self.fused_add_rmsnorm = manual_fused_cpu
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
