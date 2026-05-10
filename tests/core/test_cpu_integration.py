"""CPU scheduler integration coverage with a generated local model."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
from pathlib import Path

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

logger = init_logger(__name__)


def _scheduler_process(config: SchedulerConfig, queue: mp.Queue) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.info("Initializing scheduler subprocess...")
        scheduler = Scheduler(config)
        logger.info("Scheduler subprocess ready.")
        queue.put("READY")
        scheduler.run_forever()
    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
        queue.put(e)
        raise


@pytest.fixture(scope="module")
def local_tiny_llama_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    model_dir = tmp_path_factory.mktemp("tiny-llama")
    _write_tiny_llama_config(model_dir)
    _write_tiny_tokenizer(model_dir)
    return str(model_dir)


@pytest.fixture(scope="module")
def cpu_scheduler(local_tiny_llama_path: str):
    config = SchedulerConfig(
        model_path=local_tiny_llama_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        device="cpu",
        max_running_req=10,
        use_dummy_weight=True,
        cuda_graph_bs=None,
        _unique_suffix=f".test_cpu_int.{os.getpid()}",
    )

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

        send_queue.put(ExitMsg())

    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()


def _write_tiny_llama_config(model_dir: Path) -> None:
    (model_dir / "config.json").write_text(
        """{
  "architectures": ["LlamaForCausalLM"],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 128,
  "initializer_range": 0.02,
  "intermediate_size": 256,
  "max_position_embeddings": 32,
  "model_type": "llama",
  "num_attention_heads": 2,
  "num_hidden_layers": 1,
  "num_key_value_heads": 2,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "vocab_size": 320
}
""",
        encoding="utf-8",
    )


def _write_tiny_tokenizer(model_dir: Path) -> None:
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({str(i): i for i in range(4, 320)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(model_dir)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_cpu_prefix_caching(cpu_scheduler):
    send = cpu_scheduler["send"]
    recv = cpu_scheduler["recv"]

    ids1 = [101, 102, 103, 104]
    ids2 = [101, 102, 103, 104, 201, 202]

    for req_id, input_ids_list in enumerate([ids1, ids2], start=200):
        input_ids = torch.tensor(input_ids_list, dtype=torch.int32)
        send.put(
            UserMsg(
                uid=req_id,
                input_ids=input_ids,
                sampling_params=SamplingParams(max_tokens=3),
            )
        )

        while True:
            if recv.socket.poll(timeout=30000) == 0:
                pytest.fail(f"Timeout waiting for response to req {req_id}")
            msg = recv.get()
            assert isinstance(msg, DetokenizeMsg)
            assert msg.uid == req_id
            if msg.finished:
                break
