from __future__ import annotations

import torch
import pytest
from minisgl import device as device_mod
from minisgl.layers.rotary import RotaryEmbedding
from minisgl.layers.norm import RMSNorm, RMSNormFused
from minisgl.attention.cpu import CPUAttentionBackend

@pytest.mark.skipif(torch.cuda.is_available(), reason="Running on CUDA device, skipping CPU specific tests")
def test_device_abstraction():
    """Test device abstraction layer reports correctly on CPU only env."""
    assert device_mod.is_cpu()
    assert not device_mod.is_cuda()
    assert device_mod.get_device().type == "cpu"
    
    # Test context managers don't crash
    with device_mod.nvtx_range("test"):
        pass
    
    with device_mod.noop_context():
        pass

def test_cpu_rope():
    """Test CPU RoPE implementation runs without error."""
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
    
    # CPU RoPE takes flattened view
    q_out, k_out = rope.forward(positions, q.clone(), k.clone())
    
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert not torch.allclose(q, q_out) # Rotation should change values

def test_cpu_rmsnorm():
    """Test CPU RMSNorm implementation runs and is correct."""
    size = 128
    eps = 1e-5
    norm = RMSNorm(size, eps)
    
    x = torch.randn(10, size)
    out = norm.forward(x)
    
    # Manual verification
    variance = x.pow(2).mean(-1, keepdim=True)
    expected = x * torch.rsqrt(variance + eps) * norm.weight
    
    assert torch.allclose(out, expected, atol=1e-5)

def test_cpu_rmsnorm_fused():
    """Test CPU RMSNormFused implementation runs."""
    size = 128
    eps = 1e-5
    norm = RMSNormFused(size, eps)
    
    x = torch.randn(10, size)
    residual = torch.randn(10, size)
    
    # Forward with residual
    out, new_resid = norm.forward(x.clone(), residual.clone())
    
    # Manual check: resid += x
    expected_resid = residual + x
    assert torch.allclose(new_resid, expected_resid)
    
    # norm(resid)
    variance = expected_resid.pow(2).mean(-1, keepdim=True)
    expected_out = expected_resid * torch.rsqrt(variance + eps) * norm.weight
    assert torch.allclose(out, expected_out, atol=1e-5)
