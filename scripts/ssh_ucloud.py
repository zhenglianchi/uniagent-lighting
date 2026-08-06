#!/usr/bin/env python3
"""通过 SSH 在 UCloud GPU 云主机上执行命令。

凭据从 work/ucloud.env 读取，支持多台（--node 1/2/...）：
  UCLOUD1_HOST / UCLOUD1_USER / UCLOUD1_PASS / UCLOUD1_PORT
  UCLOUD2_HOST ...

用法：
  conda run -n swe-rl python scripts/ssh_ucloud.py 'nvidia-smi'
  conda run -n swe-rl python scripts/ssh_ucloud.py --node 2 'hostname'
  conda run -n swe-rl python scripts/ssh_ucloud.py --sftp-local f /root/
"""

import argparse
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


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = load_env(os.path.join(root, "work", "ucloud.env"))
    p = argparse.ArgumentParser(description="UCloud SSH 执行/传输")
    p.add_argument("--node", type=int, default=1, help="实例编号（默认 1）")
    p.add_argument("command", nargs="?", help="远程要执行的命令")
    p.add_argument("--sftp-local", nargs=2, metavar=("LOCAL", "REMOTE"), help="上传本地文件到服务器")
    p.add_argument("--sftp-remote", nargs=2, metavar=("REMOTE", "LOCAL"), help="下载服务器文件到本地")
    args = p.parse_args()

    prefix = f"UCLOUD{args.node}"
    host = env.get(f"{prefix}_HOST")
    user = env.get(f"{prefix}_USER")
    password = env.get(f"{prefix}_PASS")
    port = int(env.get(f"{prefix}_PORT", "22"))
    jump = env.get(f"{prefix}_JUMP")  # 跳板机（无公网实例经此中转，如 node2 -> node1）
    if not (host and user and password):
        sys.exit(f"缺少凭据：请检查 work/ucloud.env 的 {prefix}_HOST/USER/PASS")

    print(f"[node{args.node}] {host} jump={jump or '-'}", file=sys.stderr)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sock = None
    jump_client = None
    if jump:
        jump_client = paramiko.SSHClient()
        jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        jump_ok = False
        for attempt in range(1, 4):
            try:
                jump_client.connect(
                    jump,
                    port=int(env.get("UCLOUD1_PORT", "22")),
                    username=env.get("UCLOUD1_USER", user),
                    password=env.get("UCLOUD1_PASS", password),
                    timeout=60,
                )
                jump_ok = True
                break
            except Exception as e:
                print(f"跳板机连接尝试 {attempt}/3 失败：{e}", file=sys.stderr)
                time.sleep(5)
        if not jump_ok:
            sys.exit("跳板机连接失败（3 次重试后仍失败）")
        sock = jump_client.get_transport().open_channel(
            "direct-tcpip", (host, port), ("127.0.0.1", 0)
        )
    connected = False
    for attempt in range(1, 4):
        try:
            client.connect(
                host, port=port, username=user, password=password, timeout=60, sock=sock
            )
            connected = True
            break
        except Exception as e:
            print(f"SSH 连接尝试 {attempt}/3 失败：{e}", file=sys.stderr)
            time.sleep(5)
    if not connected:
        if jump_client:
            jump_client.close()
        sys.exit("SSH 连接失败（3 次重试后仍失败）")

    if args.sftp_local:
        local, remote = args.sftp_local
        sftp = client.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        print(f"已上传：{local} -> {remote}")
        client.close()
        if jump_client:
            jump_client.close()
        return
    if args.sftp_remote:
        remote, local = args.sftp_remote
        sftp = client.open_sftp()
        sftp.get(remote, local)
        sftp.close()
        print(f"已下载：{remote} -> {local}")
        client.close()
        if jump_client:
            jump_client.close()
        return
    if not args.command:
        client.close()
        sys.exit("请提供要执行的远程命令")

    stdin, stdout, stderr = client.exec_command(args.command, timeout=7200)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    client.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
