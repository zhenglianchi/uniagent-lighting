#!/usr/bin/env python3
"""列出并停止/尝试销毁腾讯云 Agent Runtime 沙箱实例。

背景：训练结束后部分沙箱未自动销毁，占满 50 核配额（LimitExceeded.CPU），
导致评测无法创建新沙箱。本脚本遍历 DescribeSandboxInstanceList 并停止全部实例。

销毁说明（2026-08-10 实测）：腾讯云 ags 控制面 API 只有 Start/Stop/Pause/Resume/
Update，**没有删除实例的接口**；E2B 兼容端点（ap-guangzhou.tencentags.com）的
DELETE /sandboxes/{id}（即 e2b Sandbox.kill）也被映射为"停止"，对已 STOPPED 实例
返回 UnsupportedOperation（400）。因此 --destroy 只是对每个实例发出 kill 请求并
如实报告结果；STOPPED 实例无法真正销毁，靠腾讯侧"超时删除/空闲回收"自动清理，
且 STOPPED 不占用 CPU 配额、不产生运行计费。

用法（服务器）：
  python tencent_stop_all_instances.py            # 仅列出并停止运行中实例
  python tencent_stop_all_instances.py --destroy  # 额外对每个实例发 kill 请求
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ags.v20250920 import ags_client, models


def load_creds() -> None:
    candidates = [
        Path("/home/ubuntu/swe-rl/tencent_sandbox.env"),
        Path(__file__).resolve().parent.parent / "work" / "tencent_sandbox.env",
    ]
    for c in candidates:
        if c.exists():
            load_dotenv(c)
            print("creds from", c)
            return
    raise SystemExit("no tencent_sandbox.env found")


def try_destroy(instance_id: str, tool_name: str) -> str:
    """通过 E2B 兼容端点对实例发 kill（DELETE）请求。

    腾讯侧将 kill 映射为 StopSandboxInstance：RUNNING 等状态可停，STOPPED 返回
    UnsupportedOperation。返回结果描述字符串。
    """
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return "SKIP (e2b_code_interpreter 未安装)"
    os.environ.setdefault("E2B_API_KEY", os.environ.get("TENCENT_SANDBOX_E2B_TOKEN", ""))
    os.environ.setdefault("E2B_DOMAIN", "ap-guangzhou.tencentags.com")
    try:
        ok = Sandbox.kill(sandbox_id=instance_id)
        return "KILL_OK(True)" if ok else "KILL_NOT_FOUND(404)"
    except Exception as exc:  # noqa: BLE001 - 上报腾讯返回的具体错误
        msg = str(exc).split("(", 1)[0].strip()
        return f"KILL_FAIL({msg})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--destroy",
        action="store_true",
        help="对每个实例发 kill 请求（腾讯映射为停止，STOPPED 实例会报错）",
    )
    args = p.parse_args()

    load_creds()
    cred = credential.Credential(
        os.environ["TENCENT_SECRET_ID"], os.environ["TENCENT_SECRET_KEY"]
    )
    profile = ClientProfile(HttpProfile(endpoint="ags.tencentcloudapi.com"))
    profile.signMethod = "HmacSHA256"
    client = ags_client.AgsClient(cred, "ap-guangzhou", profile)

    instances = []
    offset = 0
    while True:
        req = models.DescribeSandboxInstanceListRequest()
        req.Limit = 100
        req.Offset = offset
        resp = client.DescribeSandboxInstanceList(req)
        batch = resp.InstanceSet or []
        instances.extend(batch)
        if len(batch) < 100:
            break
        offset += 100

    print(f"total instances: {len(instances)}")
    stopped = 0
    destroyed = 0
    for inst in instances:
        status = inst.Status
        print(
            f"  {inst.InstanceId} status={status} tool={inst.ToolName} "
            f"create={inst.CreateTime} expires={inst.ExpiresAt}"
        )
        if status.upper() in ("RUNNING", "PENDING", "STARTING", "CREATED"):
            req = models.StopSandboxInstanceRequest()
            req.InstanceId = inst.InstanceId
            client.StopSandboxInstance(req)
            print(f"  -> stopped {inst.InstanceId}")
            stopped += 1
        if args.destroy:
            result = try_destroy(inst.InstanceId, inst.ToolName)
            print(f"  [destroy] {inst.InstanceId} -> {result}")
            if result.startswith("KILL_OK"):
                destroyed += 1
    print(f"stopped {stopped} instances")
    if args.destroy:
        print(f"destroy attempts done: {destroyed} confirmed (see per-instance results)")


if __name__ == "__main__":
    main()
