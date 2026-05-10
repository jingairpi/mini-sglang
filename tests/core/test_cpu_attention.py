from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from minisgl.attention.cpu import CPUAttentionBackend, CPUAttnMetadata


@dataclass
class MockReq:
    device_len: int
    extend_len: int
    table_idx: int = 0

    @property
    def cached_len(self) -> int:
        return self.device_len - self.extend_len


@dataclass
class MockBatch:
    padded_reqs: list[Any]
    attn_metadata: Any = None


def test_cpu_attention_metadata_indices() -> None:
    """Test that get_last_indices uses extend_len correctly."""

    # Scene: 2 requests
    # Req 1: total 10 tokens, but this step extending by 3 (prefix match 7)
    # Req 2: total 5 tokens, this step extending by 5 (fresh)

    req1 = MockReq(device_len=10, extend_len=3, table_idx=0)
    req2 = MockReq(device_len=5, extend_len=5, table_idx=1)

    batch = MockBatch(padded_reqs=[req1, req2])

    # Mock components for backend
    @dataclass
    class MockConfig:
        head_dim: int = 64

    page_table = torch.zeros((2, 20), dtype=torch.int32)
    backend = CPUAttentionBackend(
        MockConfig(),
        kvcache=object(),
        page_table=page_table,
        device=torch.device("cpu"),
    )

    backend.prepare_metadata(batch)

    meta = batch.attn_metadata
    assert isinstance(meta, CPUAttnMetadata)

    # Verify cu_extend_lens
    # [0, 3, 8]
    assert torch.equal(meta.cu_extend_lens, torch.tensor([0, 3, 8], dtype=torch.int32))

    # Verify cu_seqlens (total length)
    # [0, 10, 15]
    assert torch.equal(meta.cu_seqlens, torch.tensor([0, 10, 15], dtype=torch.int32))

    # Verify get_last_indices uses extend_lens (new tokens only)
    # Req 1: extends by 3, occupies 0,1,2. Last is 2.
    # Req 2: extends by 5, occupies 3,4,5,6,7. Last is 7.
    last_indices = meta.get_last_indices(bs=2)
    assert torch.equal(last_indices, torch.tensor([2, 7], dtype=torch.int32))


def test_cpu_attention_forward_handles_paged_cache_and_gqa() -> None:
    """CPU attention should flatten paged KV cache storage and expand KV heads for GQA."""

    class MockKVCache:
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
    kvcache = MockKVCache()
    req = MockReq(device_len=3, extend_len=3, table_idx=0)
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

    class MockKVCache:
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

    req = MockReq(device_len=4, extend_len=4, table_idx=0)
    batch = SimpleNamespace(
        reqs=[req],
        padded_reqs=[req],
        out_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        is_decode=False,
    )
    backend = CPUAttentionBackend(
        SimpleNamespace(head_dim=2, num_kv_heads=4),
        kvcache=MockKVCache(),
        page_table=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    backend.prepare_metadata(batch)

    q = torch.randn(4, 4, 2)
    k = torch.randn(4, 2, 2)
    v = torch.randn(4, 2, 2)

    out = backend.forward(q, k, v, layer_id=0, batch=batch)

    assert out.shape == (4, 8)
