from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata
from minisgl.core import Batch

if TYPE_CHECKING:
    from minisgl.kvcache import BaseKVCachePool
    from minisgl.models import ModelConfig


@dataclass
class CPUAttnMetadata(BaseAttnMetadata):
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_extend_lens: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_extend_lens[1 : 1 + bs] - 1


class CPUAttentionBackend(BaseAttnBackend):
    """CPU attention backend using PyTorch scaled dot product attention."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        kvcache: BaseKVCachePool,
        page_table: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.config = config
        self.kvcache = kvcache
        self.page_table = page_table
        self.device = device
        self.dim = config.head_dim

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        seqlens = [req.device_len for req in reqs]
        cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32, device=self.device).cumsum(0)
        extend_lens = [req.extend_len for req in reqs]
        cu_extend_lens = torch.tensor(
            [0] + extend_lens, dtype=torch.int32, device=self.device
        ).cumsum(0)

        indices = torch.cat([self.page_table[req.table_idx, : req.device_len] for req in reqs]).to(
            self.device
        )

        batch.attn_metadata = CPUAttnMetadata(
            indices=indices,
            cu_seqlens=cu_seqlens,
            cu_extend_lens=cu_extend_lens,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        meta = batch.attn_metadata
        if not isinstance(meta, CPUAttnMetadata):
            raise TypeError(f"Expected CPUAttnMetadata, got {type(meta).__name__}")

        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        local_kv_heads = k_cache.shape[-2]
        k_cache = k_cache.view(-1, local_kv_heads, self.dim)
        v_cache = v_cache.view(-1, local_kv_heads, self.dim)
        all_k = k_cache[meta.indices]
        all_v = v_cache[meta.indices]

        num_q_heads = q.shape[1]
        num_kv_heads = all_k.shape[1]
        if num_q_heads % num_kv_heads != 0:
            raise RuntimeError(
                f"Query heads ({num_q_heads}) must be divisible by KV heads ({num_kv_heads})."
            )

        output: list[torch.Tensor] = []
        q_start = 0
        for i, req in enumerate(batch.reqs):
            q_len = req.extend_len
            if batch.is_decode:
                qi = q[i : i + 1]
            else:
                qi = q[q_start : q_start + q_len]
                q_start += q_len

            k_start = meta.cu_seqlens[i].item()
            k_end = meta.cu_seqlens[i + 1].item()
            ki = all_k[k_start:k_end]
            vi = all_v[k_start:k_end]

            if num_q_heads > num_kv_heads:
                repeat_factor = num_q_heads // num_kv_heads
                ki = _expand_kv_heads(ki, repeat_factor, num_q_heads, self.dim)
                vi = _expand_kv_heads(vi, repeat_factor, num_q_heads, self.dim)

            qi = qi.transpose(0, 1)
            ki = ki.transpose(0, 1)
            vi = vi.transpose(0, 1)
            attn_mask = _causal_prefix_mask(req.cached_len, q_len, ki.shape[1], q.device)
            out_i = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=attn_mask)
            output.append(out_i.transpose(0, 1))

        out = torch.cat(output, dim=0)
        return out.reshape(out.shape[0], -1)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        pass

    def prepare_for_capture(self, batch: Batch) -> None:
        pass

    def prepare_for_replay(self, batch: Batch) -> None:
        pass


def _expand_kv_heads(
    tensor: torch.Tensor, repeat_factor: int, num_q_heads: int, head_dim: int
) -> torch.Tensor:
    return (
        tensor.unsqueeze(2)
        .expand(-1, -1, repeat_factor, -1)
        .reshape(tensor.shape[0], num_q_heads, head_dim)
    )


def _causal_prefix_mask(
    cached_len: int, query_len: int, key_len: int, device: torch.device
) -> torch.Tensor:
    q_pos = cached_len + torch.arange(query_len, device=device).unsqueeze(1)
    k_pos = torch.arange(key_len, device=device).unsqueeze(0)
    return k_pos <= q_pos
