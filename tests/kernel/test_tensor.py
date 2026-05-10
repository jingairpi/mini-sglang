from __future__ import annotations

import pytest
import torch
from minisgl.kernel import test_tensor
from minisgl.utils import call_if_main


@call_if_main()
def main():
    if torch.cuda.device_count() < 2:
        pytest.skip(
            "tensor kernel smoke test requires at least two CUDA devices",
            allow_module_level=True,
        )

    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    test_tensor(x, y)
