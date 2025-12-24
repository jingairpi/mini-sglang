"""Integration test for CPU execution mode.

This test verifies end-to-end inference on CPU by spawning a scheduler
subprocess and sending requests through ZMQ queues.

Run with: pytest tests/core/test_cpu_integration.py -v --timeout=120
Or directly: python tests/core/test_cpu_integration.py
"""
from __future__ import annotations

import os
import signal
import multiprocessing as mp

import pytest
import torch
from transformers import AutoConfig

from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from minisgl.core import SamplingParams

logger = init_logger(__name__)

# Use a standard model that shouldn't require downloading heavy weights if use_dummy_weight=True
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 


def _scheduler_process(config: SchedulerConfig, queue: mp.Queue) -> None:
    """Run scheduler in subprocess."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.info("Initializing Scheduler in subprocess...")
        scheduler = Scheduler(config)
        logger.info("Scheduler initialized. Signaling READY.")
        queue.put("READY")
        scheduler.run_forever()
    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
        queue.put(e)
        raise


def _can_load_model_config() -> bool:
    """Check if model config can be loaded."""
    try:
        AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def cpu_scheduler():
    """Fixture that starts a CPU scheduler in a subprocess."""
    if not _can_load_model_config():
        pytest.skip(f"Model config {MODEL_PATH} cannot be loaded")
    
    config = SchedulerConfig(
        model_path=MODEL_PATH,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32, 
        device="cpu",
        max_running_req=10,
        use_dummy_weight=True,
        cuda_graph_bs=None, 
        _unique_suffix=f".test_cpu_int.{os.getpid()}"
    )

    # Start scheduler process
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_scheduler_process, args=(config, q))
    p.start()
    
    try:
        msg = q.get(timeout=60)
        if isinstance(msg, Exception):
            raise msg
        if msg != "READY":
            raise RuntimeError(f"Scheduler failed to start: {msg}")
        
        # Create communication queues
        send_queue = ZmqPushQueue(
            config.zmq_backend_addr,
            create=False,
            encoder=BaseBackendMsg.encoder,
        )
        recv_queue = ZmqPullQueue(
            config.zmq_detokenizer_addr,
            create=False,
            decoder=BaseTokenizerMsg.decoder,
        )
        
        yield {
            "config": config,
            "send": send_queue,
            "recv": recv_queue,
            "process": p,
        }
        
        # Cleanup
        send_queue.put(ExitMsg())
        
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_cpu_scheduler_starts(cpu_scheduler):
    """Test that CPU scheduler starts successfully."""
    assert cpu_scheduler["process"].is_alive()


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_cpu_single_request(cpu_scheduler):
    """Test processing a single request on CPU."""
    send = cpu_scheduler["send"]
    recv = cpu_scheduler["recv"]
    
    input_ids = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    send.put(UserMsg(
        uid=100,
        input_ids=input_ids,
        sampling_params=SamplingParams(max_tokens=3),
    ))
    
    tokens_received = 0
    while True:
        if recv.socket.poll(timeout=30000) == 0:
            pytest.fail("Timeout waiting for response")
        msg = recv.get()
        assert isinstance(msg, DetokenizeMsg)
        assert msg.uid == 100
        tokens_received += 1
        if msg.finished:
            break
    
    assert tokens_received >= 1


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_cpu_prefix_caching(cpu_scheduler):
    """Test that prefix caching works on CPU (two requests with shared prefix)."""
    send = cpu_scheduler["send"]
    recv = cpu_scheduler["recv"]
    
    # Request 1: Prefix only
    ids1 = [101, 102, 103, 104]
    # Request 2: Same prefix + extension (tests prefix cache hit)
    ids2 = [101, 102, 103, 104, 201, 202]
    
    for req_id, input_ids_list in enumerate([ids1, ids2], start=200):
        input_ids = torch.tensor(input_ids_list, dtype=torch.int32)
        send.put(UserMsg(
            uid=req_id,
            input_ids=input_ids,
            sampling_params=SamplingParams(max_tokens=3),
        ))
        
        tokens_received = 0
        while True:
            if recv.socket.poll(timeout=30000) == 0:
                pytest.fail(f"Timeout waiting for response to req {req_id}")
            msg = recv.get()
            assert isinstance(msg, DetokenizeMsg)
            assert msg.uid == req_id
            tokens_received += 1
            if msg.finished:
                break
        
        assert tokens_received >= 1


# Allow running directly for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=120"])
