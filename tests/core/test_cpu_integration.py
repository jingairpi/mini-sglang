from __future__ import annotations

import torch
import multiprocessing as mp
import time
import os
import signal
from transformers import AutoConfig

from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from minisgl.core import SamplingParams

logger = init_logger(__name__)

# Use a standard model that shouldn't require downloading heavy weights if use_dummy_weight=True
# We use a path that 'transformers' knows about.
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

def scheduler_process(config: SchedulerConfig, queue: mp.Queue) -> None:
    try:
        # We need to ensure we don't capture signals meant for the parent
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        logger.info("Initializing Scheduler in subprocess...")
        scheduler = Scheduler(config)
        logger.info("Scheduler initialized. Signaling READY.")
        queue.put("READY")
        scheduler.run_forever()
    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
        queue.put(e)
        raise e

def run_test():
    # 1. Setup Config
    # Check if we can load config first (to fail fast if model not found)
    try:
        AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Skipping test because model config {MODEL_PATH} cannot be loaded: {e}")
        return

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

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    p = mp.Process(target=scheduler_process, args=(config, q))
    p.start()
    
    try:
        # Wait for ready
        msg = q.get(timeout=60)
        if isinstance(msg, Exception):
            raise msg
        if msg != "READY":
            raise RuntimeError(f"Scheduler failed to start: {msg}")
        
        print("Scheduler started.")

        # 2. Setup Queues
        send_backend = ZmqPushQueue(
            config.zmq_backend_addr,
            create=False,
            encoder=BaseBackendMsg.encoder,
        )

        recv_backend = ZmqPullQueue(
            config.zmq_detokenizer_addr,
            create=False,
            decoder=BaseTokenizerMsg.decoder,
        )
        
        # 3. Request Sequence
        # Req 1: Prefix A
        # Req 2: Prefix A + Suffix B (Matches Prefix A)
        
        # Mock token IDs
        ids1 = [101, 102, 103, 104]
        ids2 = [101, 102, 103, 104, 201, 202]
        
        reqs = [
            (ids1, 5), 
            (ids2, 5),
        ]
        
        for i, (input_ids_list, max_tokens) in enumerate(reqs):
            print(f"Sending request {i}...")
            input_ids = torch.tensor(input_ids_list, dtype=torch.int32)
            
            send_backend.put(
                UserMsg(
                    uid=i,
                    input_ids=input_ids,
                    sampling_params=SamplingParams(max_tokens=max_tokens),
                )
            )
            
            # Collect response
            tokens_received = 0
            while True:
                if recv_backend.socket.poll(timeout=30000) == 0:
                     raise TimeoutError(f"Timeout waiting for response to req {i}")
                msg = recv_backend.get() 
                assert isinstance(msg, DetokenizeMsg)
                tokens_received += 1
                if msg.finished:
                    print(f"Request {i} finished. Generated {tokens_received} tokens.")
                    break
                    
        print("All requests finished successfully.")
        send_backend.put(ExitMsg())
    
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

if __name__ == "__main__":
    run_test()
