#!/usr/bin/env python3
"""列出并停止腾讯云 Agent Runtime 沙箱实例（释放 CPU 配额）。

背景：训练结束后部分沙箱未自动销毁，占满 50 核配额（LimitExceeded.CPU），
导致评测无法创建新沙箱。本脚本遍历 DescribeSandboxInstanceList 并停止全部实例。

用法（服务器）：
  python tencent_stop_all_instances.py
"""

from __future__ import annotations

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


def main() -> None:
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
    print(f"stopped {stopped} instances")


if __name__ == "__main__":
    main()
