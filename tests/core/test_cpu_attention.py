from __future__ import annotations

import torch
import pytest
from dataclasses import dataclass
from typing import Any
from minisgl import device as device_mod
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
    kvcache: Any = None  # Not used, prepare_metadata only accesses page_table
    
    backend = CPUAttentionBackend(MockConfig(), kvcache, page_table)
    
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
