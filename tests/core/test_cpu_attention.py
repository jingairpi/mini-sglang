from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from minisgl.attention.cpu import CPUAttentionBackend


def test_cpu_attention_uses_rank_local_kv_heads_from_cache_shape() -> None:
    class KVCache:
        def __init__(self) -> None:
            self.k = torch.zeros((2, 2, 2, 2))
            self.v = torch.zeros((2, 2, 2, 2))

        def store_kv(
            self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
        ) -> None:
            _ = layer_id
            self.k.view(-1, 2, 2)[out_loc] = k
            self.v.view(-1, 2, 2)[out_loc] = v

        def k_cache(self, layer_id: int) -> torch.Tensor:
            _ = layer_id
            return self.k

        def v_cache(self, layer_id: int) -> torch.Tensor:
            _ = layer_id
            return self.v

    req = SimpleNamespace(device_len=4, extend_len=4, cached_len=0, table_idx=0)
    batch = SimpleNamespace(
        reqs=[req],
        padded_reqs=[req],
        out_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        is_decode=False,
    )
    backend = CPUAttentionBackend(
        SimpleNamespace(head_dim=2, num_kv_heads=4),
        kvcache=KVCache(),
        page_table=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    backend.prepare_metadata(batch)

    q = torch.arange(32, dtype=torch.float32).reshape(4, 4, 2) / 10
    k = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2) / 10
    v = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)

    out = backend.forward(q, k, v, layer_id=0, batch=batch)

    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.repeat_interleave(2, dim=1).transpose(0, 1),
        v.repeat_interleave(2, dim=1).transpose(0, 1),
        attn_mask=torch.tril(torch.ones((4, 4), dtype=torch.bool)),
    )
    assert torch.allclose(out, expected.transpose(0, 1).flatten(1))
