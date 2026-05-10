from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    from minisgl import device as device_mod

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            device = _infer_device(self, args, kwargs)
            with device_mod.nvtx_range(device, display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


def _infer_device(self, args, kwargs):
    import torch

    if (device := getattr(self, "device", None)) is not None:
        return torch.device(device)

    for value in list(args) + list(kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
