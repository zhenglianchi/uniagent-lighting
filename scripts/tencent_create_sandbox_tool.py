#!/usr/bin/env python
"""通过腾讯云 Cloud API 创建 Agent Runtime 沙箱工具（替代控制台操作）。

用法（swe-rl 环境）：
    python scripts/tencent_create_sandbox_tool.py \
        [--name code-interpreter-v1] [--type code-interpreter] \
        [--network PUBLIC|SANDBOX|VPC] [--timeout 1h]

- 凭据从 work/tencent_sandbox.env 加载（TENCENT_SECRET_ID/SECRET_KEY）
- 请求域名 ags.tencentcloudapi.com，Region 默认 ap-guangzhou（自定义沙箱仅广州）
- 创建后可用 scripts/tencent_list_sandbox_tools.py 查询状态
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="code-interpreter-v1")
    parser.add_argument("--type", default="code-interpreter")
    parser.add_argument("--network", default="PUBLIC", choices=["PUBLIC", "SANDBOX", "VPC"])
    parser.add_argument("--timeout", default="1h")
    parser.add_argument("--region", default="ap-guangzhou")
    args = parser.parse_args()

    load_env()
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.ags.v20250920 import ags_client, models

    cred = credential.Credential(
        os.environ["TENCENT_SECRET_ID"], os.environ["TENCENT_SECRET_KEY"]
    )
    http_profile = HttpProfile(endpoint="ags.tencentcloudapi.com")
    client = ags_client.AgsClient(cred, args.region, ClientProfile(httpProfile=http_profile))

    req = models.CreateSandboxToolRequest()
    req.ToolName = args.name
    req.ToolType = args.type
    req.DefaultTimeout = args.timeout
    req.NetworkConfiguration = models.NetworkConfiguration()
    req.NetworkConfiguration.NetworkMode = args.network
    req.ClientToken = f"codex-{args.name}-{os.getpid()}"

    resp = client.CreateSandboxTool(req)
    print(f"ToolId: {resp.ToolId}")
    print(f"RequestId: {resp.RequestId}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
