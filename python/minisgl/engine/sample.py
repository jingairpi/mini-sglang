from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl import device as device_mod
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=device.type == "cuda").to(
        device, non_blocking=True
    )


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    if device_mod.is_cpu(logits.device):
        scores = logits / temperatures.unsqueeze(-1)
        if top_k is not None:
            scores = _apply_top_k(scores, top_k)
        probs = torch.softmax(scores, dim=-1)
        if top_p is not None:
            probs = _apply_top_p(probs, top_p)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def _apply_top_k(scores: torch.Tensor, top_k: torch.Tensor | int) -> torch.Tensor:
    if isinstance(top_k, int):
        top_k = torch.full((scores.shape[0],), top_k, dtype=torch.int64, device=scores.device)
    else:
        top_k = top_k.to(device=scores.device, dtype=torch.int64)

    top_k = top_k.clamp(min=1, max=scores.shape[-1])
    max_k = int(top_k.max().item())
    topk_indices = torch.topk(scores, max_k, dim=-1).indices
    keep = torch.zeros_like(scores, dtype=torch.bool)
    ranks = torch.arange(max_k, device=scores.device).unsqueeze(0)
    keep.scatter_(1, topk_indices, ranks < top_k.unsqueeze(1))
    return scores.masked_fill(~keep, float("-inf"))


def _apply_top_p(probs: torch.Tensor, top_p: torch.Tensor | float) -> torch.Tensor:
    if isinstance(top_p, float):
        top_p = torch.full((probs.shape[0],), top_p, dtype=probs.dtype, device=probs.device)
    else:
        top_p = top_p.to(device=probs.device, dtype=probs.dtype)

    top_p = top_p.clamp(min=0.0, max=1.0)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    remove = sorted_probs.cumsum(dim=-1) > top_p.unsqueeze(1)
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with device_mod.nvtx_range(self.device, "Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
