from __future__ import annotations

from argparse import ArgumentTypeError

import pytest
from minisgl.attention import validate_attn_backend


@pytest.mark.parametrize("backend", ["cpu,fi", "fi,cpu"])
def test_cpu_attention_backend_is_not_hybrid(backend: str) -> None:
    with pytest.raises(
        ArgumentTypeError,
        match="CPU attention backend must be specified as 'cpu'",
    ):
        validate_attn_backend(backend)
