from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from minisgl import device as device_mod
from minisgl.core import Batch
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata
from minisgl.attention.utils import make_positions


@dataclass
class CPUAttnMetadata(BaseAttnMetadata):
    """Metadata for CPU attention: indices, cu_seqlens, cu_extend_lens."""
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_extend_lens: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # Use extend_len because model output only contains new tokens
        return self.cu_extend_lens[1 : 1 + bs] - 1


class CPUAttentionBackend(BaseAttnBackend):
    """CPU attention backend using PyTorch's scaled_dot_product_attention.
    
    Handles GQA via head expansion. CUDA graph methods are no-ops on CPU.
    """
    
    def __init__(self, config, kvcache, page_table):
        self.config = config
        self.kvcache = kvcache
        self.page_table = page_table
        self.dim = config.head_dim

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        device = device_mod.get_device()

        # Calculate seqlens (total length for KV cache)
        seqlens = [req.device_len for req in reqs]
        cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32, device=device).cumsum(0)

        # Calculate extend_lens (new tokens for Q and Output)
        extend_lens = [req.extend_len for req in reqs]
        cu_extend_lens = torch.tensor([0] + extend_lens, dtype=torch.int32, device=device).cumsum(0)

        # Flatten page indices
        indices_list = []
        for req in reqs:
            indices_list.append(self.page_table[req.table_idx, : req.device_len])
        indices = torch.cat(indices_list).to(device)

        batch.attn_metadata = CPUAttnMetadata(
            positions=make_positions(device, reqs),
            indices=indices,
            cu_seqlens=cu_seqlens,
            cu_extend_lens=cu_extend_lens,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # q: [num_tokens, num_q_heads, head_dim] - already 3D
        # k: [num_tokens, num_kv_heads * head_dim] - 2D, for storage
        # v: [num_tokens, num_kv_heads * head_dim] - 2D, for storage
        
        # 1. Store KV (k, v are 2D)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        # 2. Compute Attention
        meta = batch.attn_metadata
        assert isinstance(meta, CPUAttnMetadata)
        
        # Get cached KV - shape is [num_pages, 1, num_kv_heads, head_dim]
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        
        # Remove the extra dimension: [num_pages, num_kv_heads, head_dim]
        k_cache = k_cache.squeeze(1)
        v_cache = v_cache.squeeze(1)

        # Gather all K/V for the batch using indices
        # meta.indices is flat list of page indices for all tokens in the batch
        all_k = k_cache[meta.indices]  # [total_tokens, num_kv_heads, head_dim]
        all_v = v_cache[meta.indices]

        # Q is already 3D: [total_q_tokens, num_q_heads, head_dim]
        num_q_heads = q.shape[1]
        num_kv_heads = all_k.shape[1]

        output = []
        start = 0
        for i, req in enumerate(batch.reqs):  # Use actual reqs, not padded
            q_len = req.extend_len
            
            if batch.is_decode:
                qi = q[i : i + 1]  # [1, num_q_heads, head_dim]
            else:
                qi = q[start : start + q_len]  # [q_len, num_q_heads, head_dim]
                start += q_len

            k_start = meta.cu_seqlens[i].item()
            k_end = meta.cu_seqlens[i + 1].item()
            ki = all_k[k_start:k_end]  # [k_len, num_kv_heads, head_dim]
            vi = all_v[k_start:k_end]

            # Handle GQA: repeat KV heads to match Q heads using expand instead of repeat_interleave
            if num_q_heads > num_kv_heads:
                # [k_len, num_kv_heads, head_dim] -> [k_len, num_kv_heads, group, head_dim]
                repeat_factor = num_q_heads // num_kv_heads
                ki = (
                    ki.unsqueeze(2)
                    .expand(-1, -1, repeat_factor, -1)
                    .reshape(ki.shape[0], num_q_heads, self.dim)
                )
                vi = (
                    vi.unsqueeze(2)
                    .expand(-1, -1, repeat_factor, -1)
                    .reshape(vi.shape[0], num_q_heads, self.dim)
                )

            # Transpose to [num_heads, seq_len, head_dim] for SDPA
            qi = qi.transpose(0, 1)  # [num_q_heads, q_len, head_dim]
            ki = ki.transpose(0, 1)  # [num_q_heads, k_len, head_dim]
            vi = vi.transpose(0, 1)

            # Use causal mask only for prefill (>1 tokens)
            is_causal = q_len > 1
            out_i = F.scaled_dot_product_attention(qi, ki, vi, is_causal=is_causal)

            # Transpose back: [num_q_heads, q_len, head_dim] -> [q_len, num_q_heads, head_dim]
            output.append(out_i.transpose(0, 1))

        # Concatenate - output will be [total_q_tokens, num_q_heads, head_dim]
        out = torch.cat(output, dim=0)
        # Flatten to [total_q_tokens, num_q_heads * head_dim]
        return out.reshape(out.shape[0], -1)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        pass

    def prepare_for_capture(self, batch: Batch) -> None:
        pass

    def prepare_for_replay(self, batch: Batch) -> None:
        pass
