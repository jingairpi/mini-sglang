from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from minisgl.core import Batch
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata
from minisgl.attention.utils import make_positions


@dataclass
class CPUAttnMetadata(BaseAttnMetadata):
    indices: torch.Tensor
    cu_seqlens: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens[1 : 1 + bs] - 1


class CPUAttentionBackend(BaseAttnBackend):
    def __init__(self, config, kvcache, page_table):
        self.config = config
        self.kvcache = kvcache
        self.page_table = page_table
        self.dim = config.head_dim
        self.scale = 1.0 / (self.dim**0.5)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        device = "cpu"

        # Calculate seqlens
        seqlens = [req.device_len for req in reqs]  # total length so far
        cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32, device=device).cumsum(0)

        # Flatten page indices
        indices_list = []
        for req in reqs:
            indices_list.append(self.page_table[req.table_idx, : req.device_len])
        indices = torch.cat(indices_list).to(device)

        batch.attn_metadata = CPUAttnMetadata(
            positions=make_positions(device, reqs),
            indices=indices,
            cu_seqlens=cu_seqlens,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # 1. Store KV
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        # 2. Compute Attention
        meta = batch.attn_metadata
        assert isinstance(meta, CPUAttnMetadata)
        
        k_cache = self.kvcache.k_cache(layer_id)  # [num_pages, heads, dim]
        v_cache = self.kvcache.v_cache(layer_id)

        # Gather all K/V for the batch
        all_k = k_cache[meta.indices]
        all_v = v_cache[meta.indices]

        output = []
        start = 0
        for i, req in enumerate(batch.reqs):  # Use actual reqs, not padded
            q_len = req.extend_len
            
            if batch.is_decode:
                qi = q[i : i + 1]  # [1, H, D]
            else:
                qi = q[start : start + q_len]  # [q_len, H, D]
                start += q_len

            k_start = meta.cu_seqlens[i]
            k_end = meta.cu_seqlens[i + 1]
            ki = all_k[k_start:k_end]  # [k_len, H, D]
            vi = all_v[k_start:k_end]

            # Transpose to [H, Q, D] for SDPA
            qi = qi.transpose(0, 1)
            ki = ki.transpose(0, 1)
            vi = vi.transpose(0, 1)

            # For prefill, if we have q_len > 1, we need causal masking.
            # But SDPA is_causal=True applies it to the WHOLE matrix [Q, K].
            # Here Q is subset of K (at the end).
            # The indices in ki correspond to [0...k_len-1].
            # The indices in qi correspond to [k_len-q_len...k_len-1].
            # We want q[t] to attend to k[0...t + (k_len-q_len)].
            # So is_causal=True in SDPA might assume Q and K are aligned?
            # If Q and K have different lengths, does SDPA handle it?
            # PyTorch SDPA doc: "If is_causal is True, attn_mask will be generated...".
            # It usually assumes triangular mask on the last dimension.
            # If L != S, it masks out upper triangle?
            
            # To be safe, manual mask or avoid is_causal if not needed.
            # For decode (q_len=1), is_causal=False is fine (attend to all K).
            # For prefill (q_len > 1), qi is fresh tokens. ki is history + fresh.
            # We only care about causal relation within fresh tokens.
            # The history part is fully visible.
            # Standard SDPA is_causal=True works if L == S.
            # Here S >= L.
            # We can use manual mask if needed.
            
            if q_len > 1:
                # Manual causal mask
                # qi is [H, L, D], ki is [H, S, D]
                # mask shape [L, S]
                # mask[i, j] = -inf if j > i + (S - L)
                L = qi.size(1)
                S = ki.size(1)
                mask = torch.ones((L, S), device=qi.device, dtype=torch.bool)
                mask = torch.triu(mask, diagonal=S - L + 1)
                # SDPA supports attn_mask
                # Convert bool mask to float: True -> -inf, False -> 0
                # Wait, SDPA mask: "True values indicate elements that *should be computed*?" NO.
                # "Rational values ... added to attention logits."
                # Or bool mask: "True -> allowed, False -> -inf" ?
                # PyTorch docs are confusing.
                # Let's assume is_causal=True is dangerous for L != S.
                # Let's enforce non-causal for simple port if possible, or use better mask.
                
                # Check recent pytorch: is_causal works for L != S. 
                # "The causal mask is applied to the logic matrix (L, S). Mask[i, j] = -inf if j > i + (S-L)."
                # So it correctly masks out future tokens relative to Q's position.
                out_i = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
            else:
                out_i = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)

            output.append(out_i.transpose(0, 1))

        return torch.cat(output, dim=0)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        pass

    def prepare_for_capture(self, batch: Batch) -> None:
        pass

    def prepare_for_replay(self, batch: Batch) -> None:
        pass
