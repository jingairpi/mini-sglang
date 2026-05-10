from __future__ import annotations

import torch
from minisgl.engine.sample import _apply_top_p


def test_cpu_top_p_keeps_boundary_token() -> None:
    probs = torch.tensor([[0.40, 0.35, 0.25]], dtype=torch.float32)

    filtered = _apply_top_p(probs, 0.70)

    expected = torch.tensor([[0.40 / 0.75, 0.35 / 0.75, 0.0]], dtype=torch.float32)
    assert torch.allclose(filtered, expected)
