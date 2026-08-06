"""验证 P2P 抽样测试在干净沙箱中的基线通过率（诊断 reward 全 0 用）。

用法（node2）：
  cd /home/ubuntu/swe-rl && set -a && source tencent_sandbox.env && set +a && \
  /home/ubuntu/miniforge3/envs/swe-rl/bin/python \
    /home/ubuntu/uniagent-lighting/scripts/verify_p2p_baseline.py [instance_id] [sample_n]
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys

from uni_agent_ext.sandbox.tencent_agent_runtime import TencentAgentRuntimeSandbox


def _load_metadata(data_path: str, instance_id: str) -> dict:
    for line in open(data_path):
        rec = json.loads(line)
        md = rec.get("extra_info", {}).get("tools_kwargs", {}).get("reward", {}).get("metadata", {})
        if md.get("instance_id") == instance_id:
            return md
    raise SystemExit(f"instance {instance_id} not found in {data_path}")


def _parse_list(v) -> list[str]:
    if isinstance(v, list):
        if all(len(str(x)) == 1 for x in v):
            v = "".join(str(x) for x in v)
        else:
            return [str(x) for x in v]
    if isinstance(v, str):
        v = v.strip()
        try:
            parsed = json.loads(v)
            return [str(x) for x in parsed]
        except json.JSONDecodeError:
            return [x.strip() for x in v.splitlines() if x.strip()]
    return []


async def main() -> None:
    instance_id = sys.argv[1] if len(sys.argv) > 1 else "pylint-dev__astroid-1268"
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    data_path = "/home/ubuntu/swe-rl/data/agentic_train.jsonl"
    md = _load_metadata(data_path, instance_id)
    p2p = _parse_list(md.get("PASS_TO_PASS", []))
    f2p = _parse_list(md.get("FAIL_TO_PASS", []))
    rng = random.Random(instance_id)
    sample = rng.sample(p2p, min(sample_n, len(p2p)))
    print(f"instance={instance_id} F2P={len(f2p)} P2P_total={len(p2p)} sample={len(sample)}")

    image = md.get("image") or f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"
    sb = TencentAgentRuntimeSandbox(image=image)
    await sb.start()
    try:
        await sb.write_file("/tmp/_p2p.txt", "\n".join(sample))
        res = await sb.exec_shell(
            "cd /testbed && python -m pytest -q --no-header -p no:cacheprovider --tb=short "
            "$(cat /tmp/_p2p.txt | tr '\\n' ' ')",
            timeout=600,
        )
        out = (res.stdout or "")[-4000:]
        err = res.stderr or ""
        print("=== pytest tail ===")
        print(out)
        if err:
            print("=== stderr tail ===")
            print(err[-1500:])
    finally:
        await sb.stop()


if __name__ == "__main__":
    asyncio.run(main())
