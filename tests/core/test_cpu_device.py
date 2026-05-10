from __future__ import annotations

import pytest
import torch
from minisgl import device as device_mod
from minisgl.kernel.index import indexing
from minisgl.kernel.store import store_cache
from minisgl.layers.activation import silu_and_mul
from minisgl.layers.norm import RMSNorm, RMSNormFused
from minisgl.layers.rotary import RotaryEmbedding


@pytest.mark.skipif(torch.cuda.is_available(), reason="Running on CUDA device")
def test_auto_device_on_cpu_only():
    """Test device abstraction layer reports correctly on CPU only env."""
    device = device_mod.resolve_device("auto")
    assert device_mod.is_cpu(device)
    assert not device_mod.is_cuda(device)
    assert device.type == "cpu"


def test_cpu_rope_via_embedding():
    """Test CPU RoPE via RotaryEmbedding class."""
    head_size = 64
    rotary_dim = 64
    max_pos = 100
    batch_size = 2
    seq_len = 10
    num_heads = 4

    rope = RotaryEmbedding(head_size, rotary_dim, max_pos, base=10000.0)

    q = torch.randn(batch_size * seq_len, num_heads * head_size)
    k = torch.randn(batch_size * seq_len, num_heads * head_size)
    positions = torch.randint(0, max_pos, (batch_size * seq_len,))

    q_out, k_out = rope.forward(positions, q.clone(), k.clone())

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert not torch.allclose(q, q_out)


def test_cpu_rmsnorm_class():
    """Test CPU RMSNorm class."""
    size = 128
    eps = 1e-5
    norm = RMSNorm(size, eps)

    x = torch.randn(10, size)
    out = norm.forward(x)

    variance = x.pow(2).mean(-1, keepdim=True)
    expected = x * torch.rsqrt(variance + eps) * norm.weight

    assert torch.allclose(out, expected, atol=1e-5)


def test_cpu_rmsnorm_fused_class():
    """Test CPU RMSNormFused class."""
    size = 128
    eps = 1e-5
    norm = RMSNormFused(size, eps)

    x = torch.randn(10, size)
    residual = torch.randn(10, size)

    out, new_resid = norm.forward(x.clone(), residual.clone())

    expected_resid = residual + x
    assert torch.allclose(new_resid, expected_resid)

    variance = expected_resid.pow(2).mean(-1, keepdim=True)
    expected_out = expected_resid * torch.rsqrt(variance + eps) * norm.weight
    assert torch.allclose(out, expected_out, atol=1e-5)


def test_cpu_silu_and_mul():
    """Test CPU SiLU and mul activation function."""
    import torch.nn.functional as F

    hidden_dim = 64
    x = torch.randn(10, 2 * hidden_dim)

    out = silu_and_mul(x)

    # Manual verification
    gate, up = x.chunk(2, dim=-1)
    expected = F.silu(gate) * up

    assert out.shape == (10, hidden_dim)
    assert torch.allclose(out, expected)


def test_cpu_indexing():
    """Test CPU indexing."""
    vocab_size = 100
    embedding_dim = 64

    weights = torch.randn(vocab_size, embedding_dim)
    indices = torch.randint(0, vocab_size, (20,))

    out = indexing(weights, indices)

    expected = torch.nn.functional.embedding(indices, weights)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected)


def test_cpu_indexing_with_vocab_range():
    """Test CPU indexing with vocab-parallel mode."""
    vocab_size = 100
    embedding_dim = 64
    shard_start = 25
    shard_length = 50

    weights = torch.randn(shard_length, embedding_dim)  # Only local shard
    indices = torch.randint(0, vocab_size, (20,))

    out = indexing(weights, indices, vocab_range=(shard_start, shard_length))

    # Manual verification: only indices in [25, 75) should be filled
    mask = (indices >= shard_start) & (indices < shard_start + shard_length)
    local_indices = indices[mask] - shard_start

    # Check that masked positions have correct values
    assert torch.allclose(out[mask], weights[local_indices])
    # Check that non-masked positions are zero
    if (~mask).any():
        assert torch.allclose(out[~mask], torch.zeros_like(out[~mask]))


def test_cpu_store_cache():
    """Test CPU store_cache operation."""
    num_pages = 100
    num_kv_heads = 4
    head_dim = 64
    num_tokens = 10

    k_cache = torch.zeros(num_pages, num_kv_heads * head_dim)
    v_cache = torch.zeros(num_pages, num_kv_heads * head_dim)

    # Use unique indices to avoid overwrites making verification fail
    indices = torch.arange(0, num_tokens, dtype=torch.int32)
    k = torch.randn(num_tokens, num_kv_heads * head_dim)
    v = torch.randn(num_tokens, num_kv_heads * head_dim)

    store_cache(k_cache, v_cache, indices, k, v)

    # Verify that the values were stored correctly
    for i, idx in enumerate(indices):
        assert torch.allclose(k_cache[idx], k[i])
        assert torch.allclose(v_cache[idx], v[i])


def test_cpu_mem_get_info():
    """Test CPU memory info retrieval."""
    available, total = device_mod.mem_get_info(torch.device("cpu"))

    assert isinstance(available, int)
    assert isinstance(total, int)
    assert available > 0
    assert total > 0
    assert available <= total
