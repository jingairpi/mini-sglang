from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args
from .constants import DEFAULT_KERNEL_CONFIG

if TYPE_CHECKING:
    from tvm_ffi import Module

import torch
from minisgl import device as device_mod


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    if device_mod.is_cpu():
        # CPU fallback for embedding lookup
        if vocab_range is not None:
            # Vocab-parallel mode: zero-init for all-reduce, gather only local shard
            start, length = vocab_range
            mask = (indices >= start) & (indices < (start + length))
            local_indices = indices[mask] - start
            output.zero_()
            output[mask] = weights[local_indices]
        else:
            # Standard embedding lookup
            output.copy_(torch.nn.functional.embedding(indices, weights))
        return output

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
