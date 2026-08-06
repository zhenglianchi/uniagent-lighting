#!/usr/bin/env python3
"""轻量轮询 node1（短超时、短间隔），用于训练运行中机器无响应时的状态监控。

用法：
  conda run -n swe-rl python scripts/ssh_poll_node1.py [tries]
  tries 默认 40（12s 间隔 ≈ 8 分钟）；连上后打印内存/进程/日志尾/dmesg OOM 并退出。
"""

import os
import sys
import time

import paramiko


def load_env(path: str) -> dict:
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


STATUS_CMD = (
    "echo ===MEM===; free -g | head -2; "
    "echo ===PROC===; pgrep -f verl.trainer.main_ppo | head -3; "
    "echo ===LOG===; tail -n 16 /home/ubuntu/swe-rl/grpo_multinode.log 2>/dev/null | tr -d '\\033' | tr -d '\\r'; "
    "echo ===OOM===; sudo dmesg 2>/dev/null | grep -i 'killed process' | tail -4"
)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = load_env(os.path.join(root, "work", "ucloud.env"))
    host = env["UCLOUD1_HOST"]
    user = env["UCLOUD1_USER"]
    password = env["UCLOUD1_PASS"]
    port = int(env.get("UCLOUD1_PORT", "22"))
    tries = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    for i in range(1, tries + 1):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(host, port=port, username=user, password=password, timeout=8)
        except Exception as e:
            print(f"try {i}/{tries}: {type(e).__name__}: {e}", flush=True)
            client.close()
            time.sleep(12)
            continue
        try:
            _, stdout, stderr = client.exec_command(STATUS_CMD, timeout=40)
            print("CONNECTED:", flush=True)
            print(stdout.read().decode(errors="replace"), flush=True)
            err = stderr.read().decode(errors="replace")
            if err.strip():
                print("STDERR:", err, flush=True)
        finally:
            client.close()
        return
    print(f"still unreachable after {tries} tries", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
