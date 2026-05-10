from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from minisgl.attention.cpu import CPUAttentionBackend


class _Req:
    def __init__(self, device_len: int, extend_len: int, table_idx: int = 0) -> None:
        self.device_len = device_len
        self.extend_len = extend_len
        self.table_idx = table_idx

    @property
    def cached_len(self) -> int:
        return self.device_len - self.extend_len


def test_cpu_attention_forward_handles_paged_cache_and_gqa() -> None:
    """CPU attention should flatten paged KV cache storage and expand KV heads for GQA."""

    class KVCache:
        def __init__(self) -> None:
            self.k = torch.zeros((2, 2, 1, 2))
            self.v = torch.zeros((2, 2, 1, 2))

        def store_kv(
            self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
        ) -> None:
            _ = layer_id
            self.k.view(-1, 1, 2)[out_loc] = k.view(-1, 1, 2)
            self.v.view(-1, 1, 2)[out_loc] = v.view(-1, 1, 2)

        def k_cache(self, layer_id: int) -> torch.Tensor:
            _ = layer_id
            return self.k

        def v_cache(self, layer_id: int) -> torch.Tensor:
            _ = layer_id
            return self.v

    page_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    kvcache = KVCache()
    req = _Req(device_len=3, extend_len=3, table_idx=0)
    batch = SimpleNamespace(
        reqs=[req],
        padded_reqs=[req],
        out_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_decode=False,
    )
    config = SimpleNamespace(head_dim=2, num_kv_heads=1)
    backend = CPUAttentionBackend(
        config,
        kvcache=kvcache,
        page_table=page_table,
        device=torch.device("cpu"),
    )
    backend.prepare_metadata(batch)

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [0.25, 0.75]],
            [[1.0, 1.0], [0.0, 1.0]],
        ]
    )
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    out = backend.forward(q, k, v, layer_id=0, batch=batch)

    expanded_k = k.view(3, 1, 2).expand(-1, 2, -1).transpose(0, 1)
    expanded_v = v.view(3, 1, 2).expand(-1, 2, -1).transpose(0, 1)
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        expanded_k,
        expanded_v,
        attn_mask=torch.tril(torch.ones((3, 3), dtype=torch.bool)),
    )
    assert torch.allclose(out, expected.transpose(0, 1).flatten(1))


def test_cpu_attention_uses_rank_local_kv_heads_from_cache_shape() -> None:
    """CPU attention must read TP-sharded KV cache with the local KV head count."""

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

    req = _Req(device_len=4, extend_len=4, table_idx=0)
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

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
            [[0.5, 1.0], [1.0, 0.5], [0.0, 1.0], [1.0, 0.0]],
            [[1.0, -0.5], [-0.5, 1.0], [0.25, 0.75], [0.75, 0.25]],
            [[0.0, 0.5], [0.5, 0.0], [1.0, 0.25], [0.25, 1.0]],
        ]
    )
    k = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 0.5]],
            [[1.0, 1.0], [0.25, 0.75]],
        ]
    )
    v = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
            [[7.0, 70.0], [8.0, 80.0]],
        ]
    )

    out = backend.forward(q, k, v, layer_id=0, batch=batch)

    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.repeat_interleave(2, dim=1).transpose(0, 1),
        v.repeat_interleave(2, dim=1).transpose(0, 1),
        attn_mask=torch.tril(torch.ones((4, 4), dtype=torch.bool)),
    )
    assert torch.allclose(out, expected.transpose(0, 1).flatten(1))
