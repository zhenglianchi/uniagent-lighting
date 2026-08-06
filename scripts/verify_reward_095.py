"""验证回滚后 reward 评估链路（astroid-1268 干净沙箱，预期 20/21 = 0.9523）。

用法（node2）：
  cd /home/ubuntu/swe-rl && set -a && source tencent_sandbox.env && set +a && \
  export E2B_DOMAIN=ap-guangzhou.tencentags.com E2B_API_KEY=$TENCENT_SANDBOX_E2B_TOKEN && \
  /home/ubuntu/miniforge3/envs/swe-rl/bin/python \
    /home/ubuntu/uniagent-lighting/scripts/verify_reward_095.py
"""

from __future__ import annotations

import asyncio
import json
import os

from uni_agent_ext.agents.mini_swe_agent_runner import evaluate_reward, extract_task
from uni_agent_ext.sandbox.tencent_agent_runtime import TencentAgentRuntimeSandbox


DATA = "/home/ubuntu/swe-rl/data/agentic_train.jsonl"
INSTANCE_ID = "pylint-dev__astroid-1268"


async def main() -> None:
    # 读数据行，构造 task（复用 runner 的 extract_task）
    rec = None
    for line in open(DATA):
        d = json.loads(line)
        md = d.get("extra_info", {}).get("tools_kwargs", {}).get("reward", {}).get("metadata", {})
        if md.get("instance_id") == INSTANCE_ID:
            rec = d
            break
    if rec is None:
        raise SystemExit(f"{INSTANCE_ID} not in {DATA}")
    tk = rec["extra_info"]["tools_kwargs"]
    task = extract_task(rec["prompt"], tk)

    image = tk["env"]["image"]
    sb = TencentAgentRuntimeSandbox(image=image)
    await sb.start()
    try:
        score, details = await evaluate_reward(sb, task, include_p2p=True)
        print(f"=== reward for {INSTANCE_ID} ===")
        print("score:", round(score, 6))
        print("passed/total:", details["passed"], "/", details["total"])
        print("p2p_sampled:", details.get("p2p_sampled"))
        print("resolved:", details["resolved"])
        passed = [t for t in details["per_test"] if t.endswith("PASS")]
        failed = [t for t in details["per_test"] if t.endswith("FAIL")]
        print(f"PASS {len(passed)} / FAIL {len(failed)}")
        print("failed tests:", failed[:5])
        assert abs(score - 20 / 21) < 1e-6, f"unexpected score {score}"
        print("VERIFY OK: reward == 0.9523 (F2P fail + P2P 20/20 pass)")
    finally:
        await sb.stop()


if __name__ == "__main__":
    asyncio.run(main())
