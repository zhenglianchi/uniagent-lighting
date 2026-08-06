#!/usr/bin/env python
"""查询腾讯云 Agent Runtime 沙箱工具列表（含状态/失败原因）。

用法（swe-rl 环境）：
    python scripts/tencent_list_sandbox_tools.py [--region ap-guangzhou]
"""

from __future__ import annotations

import argparse
import json
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

    req = models.DescribeSandboxToolListRequest()
    resp = client.DescribeSandboxToolList(req)
    print(json.dumps(resp.to_json_string(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
