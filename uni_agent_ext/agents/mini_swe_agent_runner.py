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


# ---------------------------------------------------------------------------
# 任务解析
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> list[str]:
    """把字符串 / list / JSON 字符串统一成 list[str]。"""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
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
    test_patch = metadata.get("test_patch") or ""
    return {
        "instance_id": str(instance_id),
        "issue": issue,
        "fail_to_pass": fail_to_pass,
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


# ---------------------------------------------------------------------------
# Agent 调用命令（扩展点）
# ---------------------------------------------------------------------------
def build_mini_swe_config(
    *,
    base_url: str,
    model: str,
    max_turns: int,
    environment_class: str = "local",
    cwd: str = "/testbed",
) -> str:
    """生成 mini-swe-agent 的 YAML 配置（对齐其 agent/environment/model schema）。

    ``environment_class=local`` 表示 agent 直接跑在沙箱内（/testbed 即题目仓库）；
    ``base_url`` 指向训练 Gateway（``session.base_url``），model 为 Gateway 的 served model。
    """
    return (
        "agent:\n"
        f"  step_limit: {int(max_turns)}\n"
        "  cost_limit: 0.\n"
        "  mode: yolo\n"
        "  wall_time_limit_seconds: 0\n"
        "  output_path: /tmp/mini_swe_traj.json\n"
        "environment:\n"
        f"  cwd: {cwd}\n"
        "  timeout: 60\n"
        f"  environment_class: {environment_class}\n"
        "model:\n"
        f"  model_name: {model}\n"
        "  cost_tracking: ignore_errors\n"
        "  model_kwargs:\n"
        "    custom_llm_provider: openai\n"
        f"    api_base: {base_url}\n"
    )


def build_agent_command(
    *,
    task: dict[str, Any],
    base_url: str,
    model: str,
    max_turns: int,
    config_path: str = "/tmp/mini_swe_config.yaml",
    traj_path: str = "/tmp/traj.json",
    conda_env: str | None = "testbed",
    install_agent: bool = DEFAULT_INSTALL_AGENT,
) -> str:
    """构造沙箱内执行 mini-swe-agent 的 shell 命令。"""
    parts: list[str] = ["set -euo pipefail", "cd /testbed"]
    if install_agent:
        parts.append(
            f"{'source activate ' + shlex.quote(conda_env) + ' && ' if conda_env else ''}"
            f"pip install {MINI_SWE_AGENT_PACKAGE} -q"
        )
    args = [
        "mini-extra",
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
    parts.append(shlex.join(args))
    return " && ".join(parts)


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
    timeout: int = 1800,
) -> tuple[float, dict[str, Any]]:
    """SWE-bench 式 reward：写入 test_patch，跑 FAIL_TO_PASS 的 pytest。

    返回 ``(score, details)``，score 为通过比例（0.0~1.0）；后续可换分级 reward。
    注意：``test_patch`` 只用于评估，**不注入给 agent**（无测试泄露约定）。
    """
    env = SandboxEnvForReward(sandbox)
    details: dict[str, Any] = {"resolved": False, "passed": 0, "total": 0, "log": ""}
    fail_to_pass = task["fail_to_pass"]
    if not fail_to_pass:
        logger.warning("evaluate_reward: no FAIL_TO_PASS for %s", task["instance_id"])
        return 0.0, details

    if task["test_patch"]:
        await env.write_file("/testbed/test_patch.diff", task["test_patch"])
        await env.communicate("cd /testbed && git apply test_patch.diff", timeout=120)

    passed = 0
    logs: list[str] = []
    for test in fail_to_pass:
        result = await env.exec_shell(f"cd /testbed && python -m pytest {shlex.quote(test)} -x -q", timeout=timeout)
        ok = result.exit_code == 0
        passed += int(ok)
        logs.append(f"{test}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            details["log"] = (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]

    score = passed / len(fail_to_pass)
    details.update({"resolved": score == 1.0, "passed": passed, "total": len(fail_to_pass)})
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
    install_agent: bool = DEFAULT_INSTALL_AGENT,
    **_: Any,
) -> None:
    """跑一条 mini-swe-agent 轨迹并上报 reward（黑盒、单样本）。"""
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
        config_path = "/tmp/mini_swe_config.yaml"
        await sandbox.write_file(
            config_path,
            build_mini_swe_config(base_url=gateway_url, model="default", max_turns=max_turns).encode(),
        )
        command = build_agent_command(
            task=task,
            base_url=gateway_url,
            model="default",
            max_turns=max_turns,
            install_agent=install_agent,
        )
        started = time.perf_counter()
        result = await sandbox.exec_shell(command, timeout=run_timeout)
        logger.info(
            "[sample %d] mini-swe-agent rc=%s elapsed=%.1fs traj=%s",
            sample_index, result.exit_code, time.perf_counter() - started, task["instance_id"],
        )

        score, details = await evaluate_reward(sandbox, task)
        reward_info = {"reward_score": score, "agent_exit_code": result.exit_code, **details}
        if not session.reward_info_url:
            raise ValueError(f"reward_info_url empty for session {session.session_id}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(session.reward_info_url, json={"reward_info": reward_info})
            response.raise_for_status()
    finally:
        await sandbox.stop()
