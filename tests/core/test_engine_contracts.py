from __future__ import annotations


def test_forward_output_uses_device_explicit_names() -> None:
    from minisgl.engine import ForwardOutput

    assert ForwardOutput._fields == ("next_tokens_device", "next_tokens_cpu", "copy_done")
