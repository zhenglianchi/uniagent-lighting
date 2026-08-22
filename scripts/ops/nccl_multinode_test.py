#!/usr/bin/env python3
"""多机 NCCL 连通性/带宽测试（UCloud 双机验证用）。

用法（同一脚本放到两台机器，分别执行）：
  nodeA: NCCL_SOCKET_IFNAME=eth0 MASTER_ADDR=<nodeA内网IP> MASTER_PORT=29500 \
         RANK=0 WORLD_SIZE=2 python3 nccl_multinode_test.py
  nodeB: NCCL_SOCKET_IFNAME=eth0 MASTER_ADDR=<nodeA内网IP> MASTER_PORT=29500 \
         RANK=1 WORLD_SIZE=2 python3 nccl_multinode_test.py

前置：两台机器已装 torch（CUDA 版），GPU 驱动正常（nvidia-smi 有卡）。
通过标准：两 rank 都打印 "NCCL 多机测试通过"，且带宽>0。
"""

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://")
    local = rank % torch.cuda.device_count()
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    print(
        f"[rank {rank}] init ok: {torch.cuda.get_device_name(dev)} | "
        f"NCCL {torch.cuda.nccl.version()}",
        flush=True,
    )

    nbytes = 1024**3  # 1 GiB
    tensor = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
    for _ in range(3):  # warmup
        dist.all_reduce(tensor)
    dist.barrier()

    iters = 5
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(tensor)
    dist.barrier()
    dt = time.time() - t0
    # all_reduce 每次每 rank 收发各 1 GiB（两机 ring），估算带宽
    bw = nbytes * 2 * iters / dt / 1e9
    print(
        f"[rank {rank}] all_reduce {iters}x 耗时 {dt:.2f}s，估算带宽 {bw:.2f} GB/s",
        flush=True,
    )
    dist.destroy_process_group()
    print(f"[rank {rank}] NCCL 多机测试通过", flush=True)


if __name__ == "__main__":
    main()
