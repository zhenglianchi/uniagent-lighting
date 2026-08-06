"""mini-swe-agent 训练 runner（uni-agent ``AgentRunner`` 协议）。

在腾讯云 Agent Runtime 沙箱（provider=``tencent_agent_runtime``）内**黑盒**运行
mini-swe-agent，模型端点指向训练 Gateway（``session.base_url``），完成后在同一沙箱内
按 SWE-bench ``FAIL_TO_PASS`` 评估 reward 并上报 ``reward_info``。

与 ``examples/blackbox_recipes/claude_code/claude_code_runner.py`` 同构，职责划分：

1. :func:`extract_task`          —— 从 raw_prompt / tools_kwargs 解析任务（issue、测试、实例 id）
2. :func:`create_task_sandbox`   —— 沙箱工厂（扩展点：换沙箱后端/镜像在这里改）
3. :func:`build_agent_command`   —— agent 调用命令构建（扩展点：换 agent/入参方式）
4. :func:`evaluate_reward`       —— reward 评估（扩展点：换打分逻辑）
5. :func:`mini_swe_agent_runner` —— 主流程：建沙箱 → 跑 agent → 打分 → 上报 → 清理

用法（训练配置里注册，runner_fqn 指向本模块）::

    agent_runners:
      mini_swe_agent:
        runner_fqn: uni_agent_ext.agents.mini_swe_agent_runner.mini_swe_agent_runner
        dispatch_mode: ray_task
        max_concurrent_sessions: 4
        runner_kwargs:
          max_turns: 60

部署注意：本包（``uni_agent_ext``）需与 uni-agent 一起放在训练机 Python 可导入路径下；
沙箱需能访问 ``session.base_url``（Gateway 公网可达或隧道），否则需在
:func:`create_task_sandbox` 里做 URL 改写/内网隧道。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from uni_agent.gateway.session import SessionHandle
from uni_agent.sandbox import Sandbox, SandboxConfig, build_sandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量（均可用环境变量覆盖，便于不改代码调参）
# ---------------------------------------------------------------------------
DEFAULT_SANDBOX_PROVIDER = "tencent_agent_runtime"
DEFAULT_RUNTIME_TIMEOUT = float(os.getenv("MSA_SANDBOX_RUNTIME_TIMEOUT", "3600"))
DEFAULT_AGENT_RUN_TIMEOUT = int(os.getenv("MSA_AGENT_RUN_TIMEOUT", "3600"))
DEFAULT_MAX_TURNS = int(os.getenv("MSA_AGENT_MAX_TURNS", "60"))
# 沙箱内是否现场 pip install mini-swe-agent（预装镜像可关掉，省启动时间）
DEFAULT_INSTALL_AGENT = os.getenv("MSA_INSTALL_AGENT", "0") == "1"
MINI_SWE_AGENT_PACKAGE = "mini-swe-agent"

# Gateway 隧道（沙箱内 ssh -L 走训练机 22 端口，转发 Gateway 端口到沙箱本地）
GATEWAY_TUNNEL_ENABLED = os.getenv("MSA_GATEWAY_TUNNEL", "1") == "1"
GATEWAY_SSH_KEY_PATH = os.getenv("MSA_GATEWAY_SSH_KEY_PATH", "/home/ubuntu/.ssh/gateway_tunnel_key")
GATEWAY_SSH_USER = os.getenv("MSA_GATEWAY_SSH_USER", "ubuntu")
GATEWAY_SSH_HOST = os.getenv("MSA_GATEWAY_SSH_HOST", "")
GATEWAY_LOCAL_PORT = int(os.getenv("MSA_GATEWAY_LOCAL_PORT", "8000"))
GATEWAY_TUNNEL_WAIT = int(os.getenv("MSA_GATEWAY_TUNNEL_WAIT", "60"))

# Reward 评估（真实 SWE-bench 口径）
REWARD_TEST_TIMEOUT = int(os.getenv("MSA_REWARD_TEST_TIMEOUT", "300"))
REWARD_INCLUDE_P2P = os.getenv("MSA_REWARD_INCLUDE_P2P", "0") == "1"
REWARD_PYTHON = os.getenv("MSA_REWARD_PYTHON", "python")


# ---------------------------------------------------------------------------
# 任务解析
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> list[str]:
    """把 list/tuple / JSON 字符串 / 换行或逗号分隔字符串统一成 list[str]。

    SWE-bench 数据集字段经 verl tensordict 序列化后可能变成字符串，
    这里做多种格式的容错解析（单个测试名也可能含逗号，优先按换行拆）。
    """
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        try:
            parsed = json.loads(v)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
        if "\n" in v:
            parts = [p.strip() for p in v.split("\n") if p.strip()]
            if len(parts) > 1:
                return parts
        if "," in v:
            parts = [p.strip() for p in v.split(",") if p.strip()]
            if len(parts) > 1:
                return parts
        return [v]
    return [str(value)]


def _extract_issue_text(task: str) -> str:
    """从 SWE-bench prompt 里抠出 issue 正文（兼容 <issue_description> 与裸文本）。"""
    start = task.find("<issue_description>")
    end = task.find("</issue_description>")
    if start >= 0 and end > start:
        return task[start + len("<issue_description>") : end].strip()
    marker = "\nFollow these steps to resolve the issue:"
    if marker in task:
        return task.split(marker, 1)[0].strip()
    return task.strip()


def extract_task(raw_prompt: Any, tools_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """解析任务元数据 -> dict(instance_id, issue, fail_to_pass[], test_patch)。

    ``tools_kwargs["reward"]["metadata"]`` 为 SWE-bench 样本的标准字段：
    problem_statement / FAIL_TO_PASS / PASS_TO_PASS / test_patch / instance_id。
    """
    tools_kwargs = tools_kwargs or {}
    metadata = (tools_kwargs.get("reward") or {}).get("metadata") or {}

    if isinstance(raw_prompt, str):
        task_text = raw_prompt
    else:
        task_text = next(
            (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
            str(raw_prompt),
        )

    instance_id = metadata.get("instance_id") or metadata.get("task_id") or ""
    issue = metadata.get("problem_statement") or _extract_issue_text(task_text)
    fail_to_pass = _as_list(metadata.get("FAIL_TO_PASS"))
    pass_to_pass = _as_list(metadata.get("PASS_TO_PASS"))
    raw_ftp = metadata.get("FAIL_TO_PASS")
    logger.info(
        "extract_task[%s]: FAIL_TO_PASS raw type=%s repr=%r (parsed %d items)",
        instance_id, type(raw_ftp).__name__, str(raw_ftp)[:300], len(fail_to_pass),
    )
    test_patch = metadata.get("test_patch") or ""
    return {
        "instance_id": str(instance_id),
        "issue": issue,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "test_patch": test_patch,
        "task_text": task_text,
    }


# ---------------------------------------------------------------------------
# 沙箱工厂（扩展点）
# ---------------------------------------------------------------------------
def create_task_sandbox(
    *,
    image: str,
    gateway_url: str | None,
    provider: str = DEFAULT_SANDBOX_PROVIDER,
    runtime_timeout: float = DEFAULT_RUNTIME_TIMEOUT,
    **sandbox_kwargs: Any,
) -> Sandbox:
    """创建任务沙箱（默认腾讯云 Agent Runtime）。

    ``image`` 为 SWE-bench 实例镜像（``sweb.eval.x86_64.<org>_1776_<repo>-<pr>:latest``），
    由 tencent 后端映射为沙箱工具；``sandbox_kwargs`` 可传 ``template/startup_timeout`` 等。
    Gateway 需从沙箱可达——公网不通时在此改写 URL 或建立内网隧道。
    """
    config = SandboxConfig(
        provider=provider,
        image=image,
        runtime_timeout=runtime_timeout,
        sandbox_kwargs=sandbox_kwargs,
    )
    sandbox = build_sandbox(config)
    return sandbox


async def ensure_gateway_tunnel(
    sandbox: Sandbox,
    gateway_url: str,
    *,
    ssh_key_path: str = GATEWAY_SSH_KEY_PATH,
    ssh_host: str = GATEWAY_SSH_HOST,
    ssh_user: str = GATEWAY_SSH_USER,
    local_port: int = GATEWAY_LOCAL_PORT,
    wait_seconds: int = GATEWAY_TUNNEL_WAIT,
) -> str:
    """在沙箱内建立到训练机的 SSH 隧道，返回沙箱本地可用的 Gateway URL。

    背景：训练机公网只开放 22 端口，腾讯沙箱无法直连 Gateway 自定义端口；
    方案：沙箱内 `ssh -N -L 127.0.0.1:<local>:127.0.0.1:<gateway_port>` 走 22，
    把 Gateway 端口转发到沙箱本地。返回 ``http://127.0.0.1:<local><path>``。
    """
    from urllib.parse import urlparse

    parsed = urlparse(gateway_url)
    gateway_port = parsed.port or 80
    gateway_path = parsed.path or "/v1"
    if not ssh_host:
        raise ValueError("MSA_GATEWAY_SSH_HOST 未设置（训练机公网 IP），无法建立隧道")

    key_b64 = Path(ssh_key_path).read_bytes()
    key_dst = "/root/.ssh/id_gateway"
    await sandbox.exec_shell("mkdir -p /root/.ssh && chmod 700 /root/.ssh", timeout=30)
    await sandbox.write_file(key_dst, key_b64)
    await sandbox.exec_shell(f"chmod 600 {key_dst}", timeout=30)

    # openssh-client 缺失时补装（幂等）
    await sandbox.exec_shell(
        "which ssh >/dev/null 2>&1 || (DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-client)",
        timeout=300,
    )

    cmd = (
        f"nohup ssh -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes "
        f"-o ServerAliveInterval=15 -N -L 127.0.0.1:{local_port}:127.0.0.1:{gateway_port} "
        f"-i {key_dst} {ssh_user}@{ssh_host} > /tmp/gw_tunnel.log 2>&1 &"
    )
    await sandbox.exec_shell(cmd, timeout=60)

    ok = False
    for _ in range(wait_seconds):
        probe = await sandbox.exec_shell(f"bash -c '</dev/tcp/127.0.0.1/{local_port}' && echo UP || echo DOWN", timeout=30)
        if probe.stdout.strip().endswith("UP"):
            ok = True
            break
        await asyncio.sleep(1)
    if not ok:
        log = await sandbox.exec_shell("cat /tmp/gw_tunnel.log 2>/dev/null || true", timeout=30)
        raise RuntimeError(f"gateway tunnel failed: {log.stdout[-2000:]}{log.stderr[-2000:]}")

    logger.info("gateway tunnel up: 127.0.0.1:%s -> %s:%s", local_port, ssh_host, gateway_port)
    return f"http://127.0.0.1:{local_port}{gateway_path}"


# ---------------------------------------------------------------------------
# Agent 调用命令（扩展点）
# ---------------------------------------------------------------------------
def build_mini_swe_config(
    *,
    base_url: str,
    model: str,
    max_turns: int,
    instance_id: str,
    image: str,
    output_path: str = "/tmp/mini_swe_traj.json",
) -> str:
    """生成 mini-swe-agent 配置：harness 在训练机，环境类 attach 已建沙箱实例。

    基于随包模板（含必填的 system/instance_template），只覆写运行期参数。
    """
    import yaml

    template_path = Path(__file__).with_name("mini_swe_config.template.yaml")
    cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    cfg["agent"]["step_limit"] = int(max_turns)
    cfg["agent"]["output_path"] = output_path
    cfg["environment"]["attach_instance_id"] = instance_id
    cfg["environment"]["image"] = image
    cfg["model"]["model_name"] = model
    cfg["model"]["model_kwargs"]["api_base"] = base_url
    cfg["model"]["model_kwargs"]["api_key"] = "EMPTY"  # LiteLLM/OpenAI provider 必须有 key；Gateway 接受任意非空值
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


async def run_mini_swe_agent(
    *,
    task: dict[str, Any],
    config_path: str = "/tmp/mini_swe_config.yaml",
    traj_path: str = "/tmp/traj.json",
    run_timeout: int = DEFAULT_AGENT_RUN_TIMEOUT,
) -> tuple[int, str]:
    """在训练机本地驱动 mini-swe-agent（harness 在外、沙箱当环境）。

    返回 ``(exit_code, log_tail)``；模型调用走配置里的 ``api_base``（隧道后的 Gateway），
    轨迹由 Gateway session 记录，同时落盘 traj.json 备用。
    """
    import sys

    args = [
        sys.executable,
        "-m",
        "minisweagent.run.utilities.mini_extra",
        "swebench-single",
        "-c",
        config_path,
        "-o",
        traj_path,
        "-i",
        task["instance_id"],
        "--exit-immediately",
        "-y",
    ]
    logger.info("[sample %s] run mini-swe-agent locally: %s", task["instance_id"], shlex.join(args))
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")  # LiteLLM 兜底，防 Missing credentials
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=run_timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, f"mini-swe-agent timeout after {run_timeout}s"
    log = out.decode(errors="replace")
    return proc.returncode or 0, log[-4000:]


# ---------------------------------------------------------------------------
# Reward 评估（扩展点）
# ---------------------------------------------------------------------------
class SandboxEnvForReward:
    """把 :class:`Sandbox` 适配成 reward 评估用的异步 env 接口。"""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def communicate(self, input: str, timeout: int = 600) -> str:
        result = await self._sandbox.exec_shell(input, timeout=timeout)
        return result.stdout

    async def write_file(self, path: str | Path, content: str) -> None:
        await self._sandbox.write_file(str(path), content.encode())

    async def read_file(self, path: str | Path, **_: Any) -> str:
        data = await self._sandbox.read_file(str(path))
        return data.decode(errors="replace")

    async def exec_shell(self, command: str, *, workdir: str | None = None, timeout: int = 600):
        return await self._sandbox.exec_shell(command, workdir=workdir, timeout=timeout)


async def evaluate_reward(
    sandbox: Sandbox,
    task: dict[str, Any],
    *,
    timeout: int = REWARD_TEST_TIMEOUT,
    include_p2p: bool = REWARD_INCLUDE_P2P,
    test_python: str = REWARD_PYTHON,
) -> tuple[float, dict[str, Any]]:
    """真实 SWE-bench reward：写入 test_patch，跑 FAIL_TO_PASS（可选 PASS_TO_PASS）的 pytest。

    返回 ``(score, details)``，score = 通过测试数 / 总测试数（0.0~1.0，分级）；
    ``resolved`` 表示全部通过。测试列表解析兼容 list/JSON 字符串/换行分隔。
    注意：``test_patch`` 只用于评估，**不注入给 agent**（无测试泄露约定）。
    """
    env = SandboxEnvForReward(sandbox)
    details: dict[str, Any] = {
        "resolved": False, "passed": 0, "total": 0,
        "per_test": [], "log": "", "apply_status": "",
    }
    fail_to_pass = task["fail_to_pass"]
    # 防御：若被序列化成字符级列表，先合并重解析
    if fail_to_pass and all(len(str(x)) == 1 for x in fail_to_pass):
        fail_to_pass = _as_list("".join(str(x) for x in fail_to_pass))
        logger.info("evaluate_reward: FAIL_TO_PASS was char-split, merged -> %d tests", len(fail_to_pass))
    test_names = fail_to_pass + (task["pass_to_pass"] if include_p2p else [])
    if not test_names:
        logger.warning("evaluate_reward: no FAIL_TO_PASS for %s", task["instance_id"])
        return 0.0, details

    if task["test_patch"]:
        await env.write_file("/testbed/test_patch.diff", task["test_patch"])
        apply = await env.exec_shell("cd /testbed && git apply --3way test_patch.diff", timeout=120)
        if apply.exit_code != 0:
            apply = await env.exec_shell("cd /testbed && patch -p1 < test_patch.diff", timeout=120)
        details["apply_status"] = "ok" if apply.exit_code == 0 else (
            "failed: " + (apply.stderr or apply.stdout or "")[-300:]
        )
        if apply.exit_code != 0:
            logger.warning("evaluate_reward: test_patch apply failed (%s)", details["apply_status"])

    passed = 0
    logs: list[str] = []
    for test in test_names:
        result = await env.exec_shell(
            f"cd /testbed && {test_python} -m pytest {shlex.quote(test)} -q --no-header -p no:cacheprovider",
            timeout=timeout,
        )
        ok = result.exit_code == 0
        passed += int(ok)
        logs.append(f"{test}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            details["log"] = (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]

    score = passed / len(test_names)
    details.update(
        {"resolved": score == 1.0, "passed": passed, "total": len(test_names), "per_test": logs}
    )
    logger.info("evaluate_reward: %s -> %s (%s)", task["instance_id"], score, "; ".join(logs))
    return score, details


# ---------------------------------------------------------------------------
# 主 runner（AgentRunner 协议：async callable）
# ---------------------------------------------------------------------------
async def mini_swe_agent_runner(
    *,
    raw_prompt: Any,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict[str, Any] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    run_timeout: int = DEFAULT_AGENT_RUN_TIMEOUT,
    model_name: str = "default",
    **_: Any,
) -> None:
    """跑一条 mini-swe-agent 轨迹并上报 reward（harness 在训练机，沙箱为执行环境）。

    ``model_name`` 由 agent_framework 的 runner_kwargs.model_name 注入（Gateway served model）；
    harness 在训练机本地，直接调 ``session.base_url``（本机 Gateway），不需要沙箱内隧道。
    """
    tools_kwargs = tools_kwargs or {}
    task = extract_task(raw_prompt, tools_kwargs)
    env_config = tools_kwargs.get("env") or {}
    image = env_config.get("image")
    if not image:
        raise ValueError(f"tools_kwargs.env.image missing for sample {sample_index} ({task['instance_id']})")

    gateway_url = session.base_url
    if not gateway_url:
        raise ValueError(f"session.base_url empty for sample {sample_index}")

    sandbox = await asyncio.to_thread(create_task_sandbox, image=image, gateway_url=gateway_url)
    try:
        await sandbox.start()
        instance_id = sandbox.instance_id
        if not instance_id:
            raise RuntimeError(f"sandbox instance_id empty for sample {sample_index}")
        config_path = "/tmp/mini_swe_config.yaml"
        traj_path = "/tmp/mini_swe_traj.json"
        Path(config_path).write_text(
            build_mini_swe_config(
                base_url=gateway_url,
                model=model_name,
                max_turns=max_turns,
                instance_id=instance_id,
                image=image,
                output_path=traj_path,
            ),
            encoding="utf-8",
        )
        started = time.perf_counter()
        rc, log_tail = await run_mini_swe_agent(
            task=task, config_path=config_path, traj_path=traj_path, run_timeout=run_timeout
        )
        logger.info(
            "[sample %d] mini-swe-agent rc=%s elapsed=%.1fs traj=%s",
            sample_index, rc, time.perf_counter() - started, task["instance_id"],
        )
        if rc != 0:
            logger.warning("[sample %d] mini-swe-agent failed rc=%s tail=%s", sample_index, rc, log_tail[-1500:])

        score, details = await evaluate_reward(sandbox, task)
        reward_info = {"reward_score": score, "agent_exit_code": rc, **details}
        if not session.reward_info_url:
            raise ValueError(f"reward_info_url empty for session {session.session_id}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(session.reward_info_url, json={"reward_info": reward_info})
            response.raise_for_status()
    finally:
        await sandbox.stop()
