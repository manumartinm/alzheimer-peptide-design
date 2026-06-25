from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_ddp(backend: str = "nccl") -> tuple[int, int, int]:
    if not is_distributed():
        return 0, 0, 1
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank0_only(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    v = value.clone()
    dist.all_reduce(v, op=dist.ReduceOp.SUM)
    v /= dist.get_world_size()
    return v
