"""黑盒平台化本地 agent（WSL 端，2026-08-12）：Claude Code 在本地编排，
工具经 MCP 转发到云端腾讯沙箱，模型调用经隧道指向云端 Gateway。

配合训练侧 ``external_agent_runner``（建沙箱 + task.json + done + reward）：
1. paramiko 连训练机读 ``<PLATFORM_TEST_DIR>/<session>.task.json``
2. 起 direct-tcpip 隧道（本地 → 训练机内网 Gateway 8001）
3. 写 MCP config（sandbox server：E2B_SANDBOX_ID=训练侧建的沙箱）
4. 本地跑 ``claude --bare``（--disallowedTools 内置工具 + MCP Bash/Read/Edit/Write/
   Glob），ANTHROPIC_BASE_URL 指向隧道后的 Gateway session（去 /v1）
5. 完成后经 SSH 创建远程 done 标记

用法（WSL；训练先启动，等 task.json）：
  PYTHONPATH=$PWD/vendor/uni-agent:$PWD \
  python scripts/platform/platform_local_claude.py --wait --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import paramiko

from platform_local_agent import (
    TunnelForwarder,
    load_sandbox_env,
    load_ucloud_env,
    wait_for_task,
)

ROOT = Path(__file__).resolve().parents[1]  # 仓库根（uniagent-lighting）
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", str(Path.home() / ".npm-global/bin/claude"))
MCP_SERVER = str(ROOT / "scripts/platform/sandbox_mcp_server.py")
PYTHON_BIN = os.environ.get("PLATFORM_PYTHON", "/home/zhenglianchi/miniconda3/envs/swe-rl/bin/python")


def build_task_text(payload: dict) -> str:
    """复用 claude_code_runner.build_claude_task 生成任务文本。"""
    from uni_agent_ext.agents.claude_code_runner import build_claude_task

    raw_prompt = payload.get("raw_prompt")
    tools_kwargs = payload.get("tools_kwargs") or {}
    return build_claude_task(raw_prompt, tools_kwargs)


def build_mcp_config(instance_id: str) -> dict:
    return {
        "mcpServers": {
            "sandbox": {
                "command": PYTHON_BIN,
                "args": [MCP_SERVER],
                "env": {
                    "E2B_SANDBOX_ID": instance_id,
                    "E2B_API_KEY": os.environ["E2B_API_KEY"],
                    "E2B_DOMAIN": os.environ.get("E2B_DOMAIN", "ap-guangzhou.tencentags.com"),
                },
            }
        }
    }


def build_claude_command(task_text: str, tunnel_url: str, mcp_config_path: str) -> list[str]:
    from urllib.parse import urlparse

    parsed = urlparse(tunnel_url)
    anthropic_base_url = tunnel_url  # 已去 /v1
    model = os.environ.get("PLATFORM_MODEL", "Qwen3-8B")
    max_turns = int(os.environ.get("PLATFORM_MAX_TURNS", "60"))
    env = {
        "ANTHROPIC_BASE_URL": anthropic_base_url,
        "ANTHROPIC_API_KEY": "not-needed",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_FORK_SUBAGENT": "0",
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "8192"),
        "DISABLE_AUTOUPDATER": "1",
        "IS_SANDBOX": "1",
    }
    for key, value in env.items():
        os.environ[key] = value
    return [
        CLAUDE_BIN,
        "--bare",
        "-p",
        task_text,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--permission-mode",
        "bypassPermissions",
        "--disallowedTools",
        "Bash",
        "Edit",
        "Read",
        "Write",
        "Glob",
        "Grep",
        "Agent",
        "Task",
        "WebFetch",
        "WebSearch",
        "--mcp-config",
        mcp_config_path,
        "--output-format",
        "text",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--remote-dir", default="/home/ubuntu/swe-rl/platform_test")
    args = parser.parse_args()

    load_sandbox_env()
    env = load_ucloud_env()
    host = env.get("UCLOUD1_HOST", "")
    user = env.get("UCLOUD1_USER", "ubuntu")
    password = env.get("UCLOUD1_PASS", "")
    port = int(env.get("UCLOUD1_PORT", "22"))
    if not host or not password:
        print("missing ucloud credentials", flush=True)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=30)
    sftp = client.open_sftp()
    transport = client.get_transport()
    forwarder = None
    try:
        name, payload = wait_for_task(sftp, args.remote_dir, args.timeout)
        print(f"task: {name} instance={payload['instance_id']}", flush=True)

        from urllib.parse import urlparse

        parsed = urlparse(payload["base_url"])
        forwarder = TunnelForwarder(transport, parsed.hostname, parsed.port or 8001)
        forwarder.start()
        # Anthropic 客户端拼 /v1/messages：base_url 去 /v1
        tunnel_url = f"http://127.0.0.1:{forwarder.port}{parsed.path.removesuffix('/v1')}"
        print(f"tunnel up: 127.0.0.1:{forwarder.port} -> {parsed.hostname}:{parsed.port}", flush=True)

        task_text = build_task_text(payload)
        mcp_cfg = build_mcp_config(payload["instance_id"])
        mcp_path = f"/tmp/platform_mcp_{payload['session_id'][-8:]}.json"
        Path(mcp_path).write_text(json.dumps(mcp_cfg, indent=1), encoding="utf-8")
        print(f"mcp config: {mcp_path}", flush=True)

        cmd = build_claude_command(task_text, tunnel_url, mcp_path)
        print("claude cmd: " + " ".join(cmd[:12]) + " ...", flush=True)
        started = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(args.timeout),
            env={**os.environ, "NO_PROXY": "127.0.0.1,localhost"},
        )
        elapsed = time.time() - started
        print(f"claude rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
        print("--- stdout tail ---", flush=True)
        print((proc.stdout or "")[-3000:], flush=True)
        print("--- stderr tail ---", flush=True)
        print((proc.stderr or "")[-1500:], flush=True)

        done_marker = payload["done_marker"]
        stdin, stdout, stderr = client.exec_command(f"mkdir -p {args.remote_dir} && touch {done_marker}")
        stdout.channel.recv_exit_status()
        print(f"done marker created: {done_marker}", flush=True)
        return 0 if proc.returncode == 0 else 1
    finally:
        if forwarder is not None:
            forwarder.close()
        sftp.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
