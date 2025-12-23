from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP
from minisgl import device as device_mod


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        if device_mod.is_cpu():
            def cpu_rope(positions, query, key, head_size, cos_sin_cache, is_neox=True):
                # Manual RoPE implementation
                # query: [num_tokens, num_q_heads, head_size]
                # key:   [num_tokens, num_k_heads, head_size]
                # positions: [num_tokens]
                # cos_sin_cache: [max_pos, rot_dim * 2] (cos, sin)
                
                # Extract cos, sin for positions
                # cos_sin shape: [num_tokens, rot_dim * 2] (where rot_dim is head_size)
                # cache stores [cos, sin], each of size head_size/2
                # Wait, init says:
                # inv_freq size is rotary_dim / 2
                # cos is size rotary_dim / 2
                # cache is cat(cos, sin) -> size rotary_dim
                
                # So cos_sin[positions] has size [num_tokens, head_size]
                # We need to split it into cos and sin, each size head_size/2
                cos_sin = cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                
                # Repeat to matching head_size [num_tokens, head_size]
                cos = torch.cat([cos, cos], dim=-1)
                sin = torch.cat([sin, sin], dim=-1)
                
                # Reshape for broadcasting [num_tokens, 1, head_size]
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
                
                def rotate_half(x):
                    # split at half of the last dim
                    mid = x.shape[-1] // 2
                    x1 = x[..., :mid]
                    x2 = x[..., mid:]
                    return torch.cat((-x2, x1), dim=-1)

                # Split q, k into rotary and non-rotary parts if layout allows?
                # FlashInfer assumes head_size == rot_dim usually, or handles it.
                # Here we assert rot_dim == head_size, so full rotation.
                
                q_embed = (query * cos) + (rotate_half(query) * sin)
                k_embed = (key * cos) + (rotate_half(key) * sin)
                
                query.copy_(q_embed)
                key.copy_(k_embed)

            self.apply_rope_with_cos_sin_cache_inplace = cpu_rope
        else:
            from flashinfer import apply_rope_with_cos_sin_cache_inplace
            self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace


    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
