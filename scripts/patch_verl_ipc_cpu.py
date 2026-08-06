#!/usr/bin/env python3
"""修复 verl 权重同步对 CPU 大张量的 IPC 崩溃（2026-08-05 实测）。

现象：LoRA 首次基座权重同步（base_sync_done=False）把权重搬到 CPU 收集，
embedding ~1.09GB 超过 bucket 时走 _direct_send_large_weight，对 CPU 张量
reduce_tensor 生成的句柄参数不足 7 个，接收端 rebuild_ipc 的
list_args[6] = device_id 抛 IndexError（multiproc_executor 里 worker 崩溃）。

修复：_direct_send_large_weight 里发送前把 CPU 张量挪到 CUDA，使句柄为标准
CUDA IPC 格式。幂等：已含标记则跳过。

用法（两台都执行）：
  /home/ubuntu/miniforge3/envs/swe-rl/bin/python patch_verl_ipc_cpu.py
"""

import pathlib

TARGET = pathlib.Path(
    "/home/ubuntu/uni-agent/verl/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"
)
MARKER = "# [swe-rl patch] move CPU weights to CUDA before reduce_tensor"


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("patch already applied, skip")
        return

    anchor = "        handle = reduce_tensor(weight)\n"
    assert anchor in src, f"anchor not found in {TARGET}"
    insertion = (
        "        # [swe-rl patch] move CPU weights to CUDA before reduce_tensor\n"
        "        # LoRA first base-weight sync collects params on CPU; a CPU tensor\n"
        "        # yields a short handle that rebuild_ipc (list_args[6]) can't parse.\n"
        "        if weight.device.type != \"cuda\":\n"
        "            weight = weight.to(f\"cuda:{get_device_id()}\")\n"
    )
    src = src.replace(anchor, insertion + anchor, 1)
    TARGET.write_text(src, encoding="utf-8")
    print(f"patched: {TARGET}")


if __name__ == "__main__":
    main()
