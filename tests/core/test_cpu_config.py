from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import Engine, _adjust_config


def _config(**kwargs) -> EngineConfig:
    tp_info = kwargs.pop("tp_info", DistributedInfo(0, 1))
    config = EngineConfig(
        model_path="unused",
        tp_info=tp_info,
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


def test_cpu_rejects_moe_model() -> None:
    config = _config(device="cpu", attention_backend="cpu")
    object.__setattr__(config, "model_config", SimpleNamespace(is_moe=True))

    with pytest.raises(ValueError, match="CPU execution does not support MoE models"):
        _adjust_config(config)


def test_cpu_tensor_parallel_uses_gloo_when_pynccl_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(
        device="cpu",
        tp_info=DistributedInfo(0, 2),
        use_pynccl=False,
    )
    engine = object.__new__(Engine)
    engine.device = torch.device("cpu")
    engine.dtype = torch.float32
    world = object()
    calls = []

    def init_process_group(**kwargs):
        calls.append(kwargs)

    def new_group(**kwargs):
        raise AssertionError(f"CPU tensor parallelism should not create an NCCL group: {kwargs}")

    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)
    monkeypatch.setattr(torch.distributed, "group", SimpleNamespace(WORLD=world))
    monkeypatch.setattr(torch.distributed, "new_group", new_group)

    group = engine._init_communication(config)

    assert group is world
    assert calls[0]["backend"] == "gloo"
