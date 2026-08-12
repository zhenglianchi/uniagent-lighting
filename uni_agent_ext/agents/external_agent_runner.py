"""平台化外部 agent runner（2026-08-12，§D P0 测试）。

agent 跑在**用户侧/本地**，云端只提供：Gateway（模型端点 + token-truth 轨迹）、
腾讯沙箱（执行环境）、reward 评估。本 runner 在训练侧（framework 驱动）负责：

1. 建腾讯沙箱 + 注入任务文件（humaneval_fix 的 solution.py，无测试泄露）
2. 写任务文件 ``<PLATFORM_TEST_DIR>/<session_id>.task.json``
   （session base_url / 沙箱 instance_id / tools_kwargs / task 信息）
3. 轮询 ``<session_id>.done`` 标记（本地 agent 完成后由本地脚本经 SSH 创建）
4. 检测到 done → 云侧 reward（沙箱 pytest）→ POST reward_info
5. 清理沙箱与标记文件

本地侧配套：``scripts/platform_local_agent.py``（WSL：paramiko 隧道 + 读 task.json +
跑 mini-swe-agent + 创建 done 标记）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

try:
    from uni_agent.gateway.session import SessionHandle
except Exception:  # noqa: BLE001 - 本地开发（无 ray）也能 import 纯函数
    SessionHandle = Any  # type: ignore[misc,assignment]
from uni_agent.sandbox import Sandbox

from uni_agent_ext.agents.mini_swe_agent_runner import (
    create_task_sandbox,
    evaluate_reward as evaluate_reward_msa,
    extract_task as extract_task_meta,
)

logger = logging.getLogger(__name__)

PLATFORM_TEST_DIR = Path(os.environ.get("PLATFORM_TEST_DIR", "/home/ubuntu/swe-rl/platform_test"))
POLL_INTERVAL = float(os.environ.get("PLATFORM_POLL_INTERVAL", "5"))


def _write_task_file(
    *,
    session: SessionHandle,
    sandbox: Sandbox,
    raw_prompt: Any,
    tools_kwargs: dict[str, Any],
    task: dict[str, Any],
) -> Path:
    """把外部 agent 需要的任务信息落盘（本地脚本经 SSH 读取）。"""
    PLATFORM_TEST_DIR.mkdir(parents=True, exist_ok=True)
    task_file = PLATFORM_TEST_DIR / f"{session.session_id}.task.json"
    payload = {
        "session_id": session.session_id,
        "base_url": session.base_url,
        "instance_id": getattr(sandbox, "instance_id", ""),
        "image": (tools_kwargs.get("env") or {}).get("image", ""),
        "raw_prompt": raw_prompt,
        "tools_kwargs": tools_kwargs,
        "task": task,
        "done_marker": str(PLATFORM_TEST_DIR / f"{session.session_id}.done"),
        "created_at": time.time(),
    }
    task_file.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("platform task written: %s", task_file)
    return task_file


async def _wait_for_done(session_id: str, run_timeout: float) -> bool:
    """轮询 done 标记（本地 agent 完成后经 SSH 创建）。"""
    done_file = PLATFORM_TEST_DIR / f"{session_id}.done"
    deadline = time.time() + run_timeout
    while time.time() < deadline:
        if done_file.exists():
            return True
        await asyncio.sleep(POLL_INTERVAL)
    logger.warning("platform session %s timed out waiting for external agent", session_id)
    return False


async def external_agent_runner(
    *,
    raw_prompt: Any,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict[str, Any] | None = None,
    run_timeout: int = 7200,
    **_: Any,
) -> None:
    """外部 agent 平台化 runner：建沙箱 → 暴露任务 → 等本地 agent → reward。"""
    tools_kwargs = tools_kwargs or {}
    task = extract_task_meta(raw_prompt, tools_kwargs)
    env_config = tools_kwargs.get("env") or {}
    image = env_config.get("image")
    if not image:
        raise ValueError(f"tools_kwargs.env.image missing for sample {sample_index}")
    if not session.base_url:
        raise ValueError(f"session.base_url empty for sample {sample_index}")

    sandbox = create_task_sandbox(image=image, gateway_url=session.base_url)
    task_file: Path | None = None
    try:
        await sandbox.start()

        # 任务文件注入（humaneval_fix）：/testbed git 仓库 + solution.py（无测试泄露）
        env_files = env_config.get("files") or {}
        if env_files:
            await sandbox.exec_shell(
                "mkdir -p /testbed && cd /testbed && "
                "(git --version >/dev/null 2>&1 || "
                "(DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git)) && "
                "git init -q && git config user.email t@example.com && git config user.name t",
                timeout=300,
            )
            for rel_path, content in env_files.items():
                await sandbox.write_file(f"/testbed/{rel_path}", content)
            await sandbox.exec_shell("cd /testbed && git add -A", timeout=60)

        task_file = _write_task_file(
            session=session,
            sandbox=sandbox,
            raw_prompt=raw_prompt,
            tools_kwargs=tools_kwargs,
            task=task,
        )
        logger.info(
            "[sample %d] platform task ready: session=%s instance=%s base_url=%s",
            sample_index,
            session.session_id,
            getattr(sandbox, "instance_id", ""),
            session.base_url,
        )

        done = await _wait_for_done(session.session_id, run_timeout)
        logger.info("[sample %d] external agent done=%s, evaluating reward", sample_index, done)

        eval_timeout = int(os.environ.get("SWE_AGENT_EVAL_TIMEOUT", "600"))
        score, eval_result = await evaluate_reward_msa(sandbox, task, timeout=eval_timeout)
        logger.info("[sample %d] reward done score=%s resolved=%s", sample_index, score, eval_result.get("resolved"))

        reward_info = {
            "reward": score,  # framework._score_from_reward_info 消费此键
            "reward_score": score,
            "external_agent": True,
            **eval_result,
        }
        if not session.reward_info_url:
            raise ValueError(f"reward_info_url empty for session {session.session_id}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(session.reward_info_url, json={"reward_info": reward_info})
            response.raise_for_status()
    finally:
        if task_file is not None:
            try:
                task_file.unlink()
                (PLATFORM_TEST_DIR / f"{session.session_id}.done").unlink(missing_ok=True)
            except OSError:
                pass
        await sandbox.stop()
