"""
Test that CacheManager._allocate correctly handles eviction with page_size > 1.
"""

from __future__ import annotations

import minisgl.core as core
import pytest
import torch
from minisgl.scheduler.cache import CacheManager


@pytest.fixture(autouse=True)
def reset_global_ctx():
    """Reset global context before and after each test."""
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _make_cache_manager(num_pages: int, page_size: int) -> CacheManager:
    """Helper to create a CacheManager with radix cache on CPU."""
    page_table = torch.empty((1,))
    ctx = core.Context(page_size=page_size)
    core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, type="radix")


def _insert_evictable(cm: CacheManager, input_ids: torch.Tensor, indices: torch.Tensor):
    """Insert a prefix into the radix cache so it becomes evictable."""
    cm.prefix_cache.insert_prefix(input_ids, indices)


def test_allocate_evicts_page_aligned_blocks() -> None:
    page_size = 4
    cm = _make_cache_manager(num_pages=8, page_size=page_size)

    cached_pages = cm._allocate(2)
    token_indices = cm._page_to_token(cached_pages)
    _insert_evictable(cm, torch.arange(len(token_indices), dtype=torch.int32), token_indices)
    cm.check_integrity()

    cm._allocate(6)
    assert len(cm.free_slots) == 0

    evicted_page = cm._allocate(1)
    assert len(evicted_page) == 1
    assert evicted_page.item() % page_size == 0
    assert torch.all(cm.free_slots % page_size == 0)
