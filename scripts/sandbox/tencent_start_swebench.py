#!/usr/bin/env python
"""启动腾讯云 SWE-bench 沙箱实例（系统镜像仓库，无需推 TCR）。

用法（swe-rl 环境）：
    python scripts/sandbox/tencent_start_swebench.py swebench/sweb.eval.x86_64.django_1776_django-13447:latest

流程：
1. StartSandboxInstance + CustomConfiguration.Image=<镜像>（ImageRegistryType=system，
   系统仓库已内置 SWE-bench 镜像，如 swebench/sweb.eval.x86_64.<org>_<repo>-<pr>:latest）
2. 返回 InstanceId；E2B 连接：Sandbox.connect(sandbox_id=<InstanceId>)（配合 e2b_ Key）
3. 用 --kill 可销毁实例

参考：swebench 沙箱工具（ToolType=swebench）内置 swerex runtime（/nix），
8000 端口跑 swerex server；/testbed 为题目仓库、conda env testbed 为 Python 环境。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env_file = ROOT / "work" / "tencent_sandbox.env"
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key, value)


def _client(region: str):
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.ags.v20250920 import ags_client

    cred = credential.Credential(
        os.environ["TENCENT_SECRET_ID"], os.environ["TENCENT_SECRET_KEY"]
    )
    http_profile = HttpProfile(endpoint="ags.tencentcloudapi.com")
    return ags_client.AgsClient(cred, region, ClientProfile(httpProfile=http_profile))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="SWE-bench 系统镜像，如 swebench/sweb.eval.x86_64.django_1776_django-13447:latest")
    parser.add_argument("--tool", default="swebench-v1")
    parser.add_argument("--timeout", default="10m")
    parser.add_argument("--region", default="ap-guangzhou")
    parser.add_argument("--kill", default="", help="销毁指定 InstanceId 后退出")
    args = parser.parse_args()
    load_env()

    from tencentcloud.ags.v20250920 import models

    client = _client(args.region)
    if args.kill:
        req = models.StopSandboxInstanceRequest()
        req.InstanceId = args.kill
        client.StopSandboxInstance(req)
        print(f"stopped {args.kill}")
        return 0

    req = models.StartSandboxInstanceRequest()
    req.ToolName = args.tool
    req.Timeout = args.timeout
    req.CustomConfiguration = models.CustomConfiguration()
    req.CustomConfiguration.Image = args.image
    req.CustomConfiguration.ImageRegistryType = "system"
    resp = client.StartSandboxInstance(req)
    inst = resp.Instance
    print(f"InstanceId: {inst.InstanceId}")
    print(f"Status: {inst.Status}")
    print(f"Image: {inst.CustomConfiguration.Image}")
    print(f"ExpiresAt: {inst.ExpiresAt}")
    print(f"E2B 连接: Sandbox.connect(sandbox_id={inst.InstanceId!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
