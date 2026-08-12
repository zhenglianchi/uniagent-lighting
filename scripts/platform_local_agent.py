"""平台化测试本地 agent（WSL 端，2026-08-12 §D P0）。

配合训练侧 ``uni_agent_ext.agents.external_agent_runner``：
1. paramiko 连训练机（读 ``work/ucloud.env`` 凭据），等待/读取远程
   ``<PLATFORM_TEST_DIR>/<session_id>.task.json``
2. 起 paramiko direct-tcpip 隧道（本地随机端口 → 训练机 10.60.56.10:8001）
3. 用 task.json 生成 mini-swe config（api_base=隧道后的 Gateway session URL、
   attach 已建沙箱实例），跑 mini-swe-agent（Python API 模式，humaneval_fix 任务）
4. 完成后经 SSH 创建远程 ``<session_id>.done`` 标记，训练侧 runner 收到后评估 reward

用法（WSL，swe-rl 环境；训练先启动，等 runner 写好 task.json）：
  PYTHONPATH=/home/zhenglianchi/swe-rl-local/work/uni-agent:/home/zhenglianchi/swe-rl-local/uniagent-lighting \
  python uniagent-lighting/scripts/platform_local_agent.py --wait --timeout 1800
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from pathlib import Path

import paramiko
import yaml

ROOT = Path(__file__).resolve().parents[2]  # swe-rl-local


def load_ucloud_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / "work/ucloud.env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def load_sandbox_env() -> None:
    """加载本地腾讯沙箱凭据（work/tencent_sandbox.env）并映射 E2B_* 环境变量。"""
    env_path = ROOT / "work/tencent_sandbox.env"
    if not env_path.exists():
        raise SystemExit(f"missing {env_path} (腾讯云沙箱凭据)")
    env: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    os.environ.setdefault("E2B_DOMAIN", env.get("E2B_DOMAIN", "ap-guangzhou.tencentags.com"))
    os.environ.setdefault("E2B_API_KEY", env.get("E2B_API_KEY") or env.get("TENCENT_SANDBOX_E2B_TOKEN", ""))
    if not os.environ.get("E2B_API_KEY"):
        raise SystemExit("E2B_API_KEY 未设置（tencent_sandbox.env 缺 TENCENT_SANDBOX_E2B_TOKEN）")
    print(f"E2B ready: domain={os.environ['E2B_DOMAIN']}", flush=True)


class TunnelForwarder:
    """paramiko direct-tcpip 端口转发：本地 127.0.0.1:<port> → 远端 host:port。"""

    def __init__(self, transport: paramiko.Transport, remote_host: str, remote_port: int):
        self._transport = transport
        self._remote = (remote_host, remote_port)
        self._local = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._local.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._local.bind(("127.0.0.1", 0))
        self._local.listen(16)
        self.port = self._local.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._local.accept()
            except OSError:
                break
            threading.Thread(target=self._forward, args=(conn,), daemon=True).start()

    def _forward(self, conn: socket.socket) -> None:
        try:
            channel = self._transport.open_channel(
                "direct-tcpip", self._remote, conn.getpeername()
            )
        except Exception:
            conn.close()
            return

        def _pump(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                try:
                    dst.close()
                except Exception:
                    pass

        threading.Thread(target=_pump, args=(conn, channel), daemon=True).start()
        threading.Thread(target=_pump, args=(channel, conn), daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._local.close()
        except OSError:
            pass


def wait_for_task(sftp: paramiko.SFTPClient, remote_dir: str, timeout: float) -> tuple[str, dict]:
    deadline = time.time() + timeout
    last_note = ""
    while time.time() < deadline:
        try:
            files = sorted(sftp.listdir(remote_dir), reverse=True)
        except FileNotFoundError:
            files = []
        tasks = [f for f in files if f.endswith(".task.json")]
        if tasks:
            name = tasks[0]
            with sftp.open(f"{remote_dir}/{name}") as fh:
                payload = json.loads(fh.read().decode())
            return name, payload
        note = f"[{int(time.time())}] waiting for task.json ({len(files)} files)"
        if note != last_note:
            print(note, flush=True)
            last_note = note
        time.sleep(5)
    raise TimeoutError(f"no platform task.json within {timeout}s")


def build_config(payload: dict, local_port: int, output_path: str) -> str:
    """生成 mini-swe config：attach 已建沙箱 + api_base 指向隧道后的 Gateway。"""
    from uni_agent_ext.agents.mini_swe_agent_runner import build_mini_swe_config

    base_url = payload["base_url"]
    # 隧道后：http://127.0.0.1:<local_port><session path>/v1
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    tunnel_url = f"http://127.0.0.1:{local_port}{parsed.path}"
    task = payload.get("task", {})
    model_cfg = ((payload.get("tools_kwargs") or {}).get("task") or {}).get("model", {})
    return build_mini_swe_config(
        base_url=tunnel_url,
        model=os.environ.get("PLATFORM_MODEL", "Qwen3-8B"),
        max_turns=int(os.environ.get("PLATFORM_MAX_TURNS", "60")),
        instance_id=payload["instance_id"],
        image=payload["image"],
        temperature=model_cfg.get("temperature"),
        output_path=output_path,
    )


def run_local_agent(config_path: str, task: dict, timeout: int) -> int:
    """照 mini_swe_agent_runner.run_mini_swe_agent_api：Python API 模式跑 humaneval_fix。"""
    from minisweagent.agents import get_agent
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment
    import minisweagent.environments.extra.tencent_e2b  # noqa: F401

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    instance = {
        "instance_id": task["instance_id"],
        "problem_statement": task["issue"],
        "image_name": config.get("environment", {}).get("image", "python:3.12"),
    }

    def _run() -> int:
        env = get_sb_environment(config, instance)
        agent = get_agent(
            get_model(config=config["model"]), env, config["agent"], default_type="default"
        )
        agent.run(instance["problem_statement"])
        return 0

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("local agent timed out", flush=True)
            return -1
        except Exception as exc:  # noqa: BLE001
            print(f"local agent failed: {type(exc).__name__}: {exc}", flush=True)
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="轮询等待远程 task.json")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--remote-dir", default="/home/ubuntu/swe-rl/platform_test")
    parser.add_argument("--gateway-port", type=int, default=8001)
    args = parser.parse_args()

    load_sandbox_env()
    env = load_ucloud_env()
    host = env.get("UCLOUD1_HOST", "")
    user = env.get("UCLOUD1_USER", "ubuntu")
    password = env.get("UCLOUD1_PASS", "")
    port = int(env.get("UCLOUD1_PORT", "22"))
    if not host or not password:
        print("missing ucloud credentials in work/ucloud.env", flush=True)
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

        # 隧道：本地随机端口 → 训练机内网 Gateway
        parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(payload["base_url"])
        forwarder = TunnelForwarder(transport, parsed.hostname, parsed.port or args.gateway_port)
        forwarder.start()
        print(f"tunnel up: 127.0.0.1:{forwarder.port} -> {parsed.hostname}:{parsed.port}", flush=True)

        config_path = f"/tmp/platform_mini_swe_{payload['session_id'][-8:]}.yaml"
        traj_path = f"/tmp/platform_mini_swe_{payload['session_id'][-8:]}.traj.json"
        cfg_text = build_config(payload, forwarder.port, traj_path)
        Path(config_path).write_text(cfg_text, encoding="utf-8")
        print(f"config written: {config_path}", flush=True)

        rc = run_local_agent(config_path, payload["task"], timeout=int(args.timeout))
        print(f"local agent rc={rc}", flush=True)

        # 创建远程 done 标记
        done_marker = payload["done_marker"]
        stdin, stdout, stderr = client.exec_command(f"mkdir -p {args.remote_dir} && touch {done_marker}")
        stdout.channel.recv_exit_status()
        print(f"done marker created: {done_marker}", flush=True)
        return 0 if rc == 0 else 1
    finally:
        if forwarder is not None:
            forwarder.close()
        sftp.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
