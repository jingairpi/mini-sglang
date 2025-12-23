from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module
    from tvm_ffi import Module
    
import torch
from minisgl import device as device_mod

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
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
         # CPU fallback using advanced indexing
         # weights: [num_tokens, embedding_dim] or [num_embeddings_tp, embedding_dim] ?
         # In embedding.py: weights=self.weight (tp), indices=x
         # self.weight is [num_embeddings_tp, embedding_dim]
         # x is [num_tokens] (token ids)
         
         # The CUDA kernel logic:
         # It gathers or scatters?
         # VocabParallelEmbedding calls indexing(weights=self.weight, indices=x, vocab_range=...)
         # If vocab_range is set, it means we are doing vocab parallel lookup?
         # wait, VocabParallelEmbedding forward:
         # y = indexing(weights, indices, vocab_range)
         # It seems it's gathering embeddings.
         
         # Let's check indexing kernel usage.
         # self.weight is the embedding table.
         # indices are input tokens.
         # output is [batch_size, embedding_dim]
         
         # BUT, VocabParallelEmbedding passes `weights=self.weight`.
         # If tp_size > 1, vocab_range is used to mask out indices not on this rank?
         
         if vocab_range is not None:
             start, length = vocab_range
             # Create mask for indices in this range
             mask = (indices >= start) & (indices < (start + length))
             # Shift indices to local range
             local_indices = indices[mask] - start
             # Gather
             output[mask] = weights[local_indices]
             # For indices not in mask, output should be 0? CUDA kernel probably handles it.
             # In TP embedding, we allow gathering 0s if index is out of range, then allreduce sums them up.
             # So we should initialize output to 0.
             # In `indexing` function, `output = weights.new_empty(...)` is used.
             # We should probably use new_zeros if specialized logic is needed, 
             # but check if new_empty is safe validation.
             # For TP, we need zeroes for reduction.
             output.zero_()
             output[mask] = weights[local_indices]
         else:
             # Standard embedding lookup
             # weights is [vocab, dim]
             # indices is [batch]
             # output is [batch, dim]
             # PyTorch embedding is F.embedding(indices, weights)
             # But here we are passed manually.
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
