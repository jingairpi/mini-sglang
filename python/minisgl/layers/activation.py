from __future__ import annotations

import torch
import torch.nn.functional as F
from minisgl import device as device_mod


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute SiLU activation on gated portion and multiply.
    
    x is expected to have shape [..., 2 * hidden_dim], where the first half
    is the gate and the second half is the value.
    """
    if device_mod.is_cpu():
        # CPU fallback using PyTorch
        gate, up = x.chunk(2, dim=-1)
        if out is not None:
            torch.mul(F.silu(gate), up, out=out)
            return out
        return F.silu(gate) * up
    else:
        from flashinfer import silu_and_mul

        return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
