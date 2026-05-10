from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _adjust_config


def _config(**kwargs) -> EngineConfig:
    config = EngineConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        **kwargs,
    )
    object.__setattr__(config, "model_config", SimpleNamespace(is_moe=False))
    return config


def test_cpu_auto_attention_backend_selects_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(device="cpu")
    monkeypatch.setattr("minisgl.engine.engine.logger.info_rank0", lambda _: None)

    _adjust_config(config)

    assert config.attention_backend == "cpu"


@pytest.mark.parametrize(
    ("device", "backend", "message"),
    [
        ("cpu", "fi", "CPU execution requires attention backend 'cpu'"),
        ("cuda", "cpu", "CUDA execution requires CUDA attention backends"),
    ],
)
def test_rejects_mixed_attention_device_modes(
    device: str, backend: str, message: str
) -> None:
    config = _config(device=device, attention_backend=backend)

    with pytest.raises(ValueError, match=message):
        _adjust_config(config)
