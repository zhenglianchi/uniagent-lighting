#!/usr/bin/env python
"""腾讯云 Agent Runtime（Agent Sandbox）最小连通验证脚本。

用法（在 swe-rl conda 环境中）：
    python scripts/tencent_sandbox_demo.py [template_name]

- 默认 template = code-interpreter-v1（须先在腾讯云控制台创建同名"沙箱工具"）
- 凭据从 work/tencent_sandbox.env 加载（E2B 兼容路径：e2b_* Key）
- 动作：创建沙箱 → run_code 打印 hello → kill，全流程带耗时输出
"""

from __future__ import annotations

import os
import sys
import time
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
    # E2B SDK 要求 e2b_* 前缀，走官方"兼容 E2B"路径
    os.environ.setdefault("E2B_API_KEY", os.environ["TENCENT_SANDBOX_E2B_TOKEN"])


def main() -> int:
    template = sys.argv[1] if len(sys.argv) > 1 else "code-interpreter-v1"
    load_env()
    print(f"[demo] E2B_DOMAIN={os.environ['E2B_DOMAIN']}")
    print(f"[demo] template={template}")

    from e2b_code_interpreter import Sandbox

    sandbox = None
    t0 = time.time()
    try:
        sandbox = Sandbox.create(template=template, timeout=3600)
        print(f"[demo] sandbox created in {time.time() - t0:.1f}s id={sandbox.sandbox_id}")
        result = sandbox.run_code(
            "import platform; print('hello from tencent sandbox'); print(platform.platform())",
            timeout=120,
        )
        stdout = "".join(result.logs.stdout)
        stderr = "".join(result.logs.stderr)
        print("[demo] stdout:", stdout.strip())
        if stderr.strip():
            print("[demo] stderr:", stderr.strip())
        if result.error:
            print("[demo] error:", result.error)
            return 1
    finally:
        if sandbox is not None:
            sandbox.kill()
            print("[demo] sandbox killed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
