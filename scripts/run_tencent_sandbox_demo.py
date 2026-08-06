#!/usr/bin/env python
"""用 uni-agent 官方 sandbox demo 验证腾讯云沙箱后端。

用法（swe-rl 环境）：
    python scripts/run_tencent_sandbox_demo.py [--template code-interpreter-v1]

作用：
1. 从 work/tencent_sandbox.env 加载 E2B_DOMAIN / E2B_API_KEY
2. 把 uni_agent / uni_agent_ext 加入 sys.path（本地源码，不 pip 安装）
3. 以 SANDBOX_PROVIDER=tencent_agent_runtime 跑
   work/uni-agent/examples/quickstart/sandbox/demo.py（安装包→写文件→执行→状态保持）
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env_file = ROOT / "work" / "tencent_sandbox.env"
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key, value)
    os.environ.setdefault("E2B_DOMAIN", "ap-guangzhou.tencentags.com")
    os.environ.setdefault("E2B_API_KEY", os.environ["TENCENT_SANDBOX_E2B_TOKEN"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="code-interpreter-v1")
    parser.add_argument("--image", default="python:3.12")
    args = parser.parse_args()

    load_env()
    os.environ["TENCENT_SANDBOX_TEMPLATE"] = args.template
    os.environ["SANDBOX_PROVIDER"] = "tencent_agent_runtime"
    os.environ["IMAGE"] = args.image
    os.environ["DEBUG_MODE"] = "1"  # demo 的 INFO 走查日志

    sys.path.insert(0, str(ROOT / "work" / "uni-agent"))
    sys.path.insert(0, str(ROOT / "platform"))
    import uni_agent_ext  # noqa: F401  注册 tencent_agent_runtime provider

    demo = ROOT / "work" / "uni-agent" / "examples" / "quickstart" / "sandbox" / "demo.py"
    runpy.run_path(str(demo), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
