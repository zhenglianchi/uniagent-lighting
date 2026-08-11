"""Claude Code 黑盒 runner（腾讯云 Agent Runtime direct-URL 版，2026-08-11）。

对标 uni-agent 官方 ``examples/blackbox_recipes/claude_code/claude_code_runner.py``，
针对腾讯 E2B 后端的两处改造（TODO §G.5 调研定稿）：

1. **direct-URL 模式**：官方 runner 把 ``ANTHROPIC_BASE_URL`` 重写成沙箱内隧道
   ``127.0.0.1:38197``（openyuanrong 后端的 mounts/upstream 机制），腾讯 E2B 后端
   不支持 → 这里直接指向公网可达的 Gateway URL（``session.base_url`` 去掉 ``/v1``），
   腾讯沙箱在云端，公网可达性天然满足；
2. **沙箱内安装 claude-code**：官方用 sidecar 镜像挂载 ``/opt/claude-code``（腾讯后端
   不支持 mounts）→ 这里在沙箱内 ``npm install -g @anthropic-ai/claude-code@2.1.153``
   （pin < 2.1.154，避开 vLLM 0.11.1 严格 role 校验的 system/ctx/msg 问题；用 npmmirror
   加速国内安装）。

reward 评估复用 uni-agent 的 SWE-bench reward（``uni_agent.tasks.swe_bench.reward``），
与 mini-swe-agent runner 同口径。

配置（训练 yaml 的 agent_runners）：
  runner_fqn: uni_agent_ext.agents.claude_code_runner.claude_code_runner
  runner_kwargs:
    model_name: <gateway served model>
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from uni_agent.gateway.session import SessionHandle
from uni_agent.sandbox import Sandbox

from uni_agent_ext.agents.mini_swe_agent_runner import create_task_sandbox

logger = logging.getLogger(__name__)

# vLLM 0.11.1 严格校验 Claude Code 2.1.154+ 的非标准 role（system/ctx/msg），
# 修复 2026-06 才合入 vLLM → 必须 pin < 2.1.154
CLAUDE_CODE_VERSION = "2.1.153"
NPM_REGISTRY = "https://registry.npmmirror.com"
DEFAULT_MAX_TURNS = 100
DEFAULT_AGENT_RUN_TIMEOUT = 7200


def extract_image(env_config: dict) -> str:
    """从 tools_kwargs.env 提取 SWE-bench 实例镜像（flat/nested 均支持）。"""
    image = env_config.get("image")
    if image:
        return image
    deployment = env_config.get("deployment")
    if isinstance(deployment, dict):
        image = deployment.get("image")
        if image:
            return image
    return ""


def extract_task(raw_prompt) -> str:
    if isinstance(raw_prompt, str):
        return raw_prompt
    return next(
        (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
        str(raw_prompt),
    )


def _extract_issue_text(task: str) -> str:
    start = task.find("<issue_description>")
    end = task.find("</issue_description>")
    if start >= 0 and end > start:
        return task[start + len("<issue_description>") : end].strip()
    marker = "\nFollow these steps to resolve the issue:"
    if marker in task:
        return task.split(marker, 1)[0].strip()
    return task.strip()


def _decode_metadata_list(value) -> list[str]:
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


def build_claude_task(raw_prompt, tools_kwargs: dict | None = None) -> str:
    """构建 claude -p 任务文本（issue + FAIL_TO_PASS 测试，无 test_patch 泄露）。"""
    tools_kwargs = tools_kwargs or {}
    task = extract_task(raw_prompt)
    metadata = (tools_kwargs.get("reward") or {}).get("metadata") or {}
    issue = metadata.get("problem_statement") or _extract_issue_text(task)
    tests = _decode_metadata_list(metadata.get("FAIL_TO_PASS"))
    lines = [
        "You are a software engineer working in /testbed.",
        "Fix the issue described below. Do not change unrelated behavior.",
        "",
        "<issue_description>",
        issue,
        "</issue_description>",
    ]
    if tests:
        lines += [
            "",
            "The following tests must pass after your fix:",
            *[f"- {t}" for t in tests],
        ]
    lines += [
        "",
        "- Do not run `pytest --collect-only`, `git log`, or any other command that does not",
        "  directly validate the fix.",
    ]
    return "\n".join(lines)


def build_claude_command(
    *,
    task: str,
    base_url: str,
    max_turns: int,
    model: str = "default",
    permission_mode: str = "bypassPermissions",
    conda_env: str | None = "testbed",
    disable_web_tools: bool = True,
    disable_slash_commands: bool = True,
) -> str:
    """构建沙箱内 claude -p 命令（direct URL，claude 走 PATH）。"""
    env = {
        "ANTHROPIC_BASE_URL": base_url,
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
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",  # vLLM ≤0.17.1：避免请求 hash 破坏 prefix cache
        "DISABLE_AUTOUPDATER": "1",
        "IS_SANDBOX": "1",
    }
    env_assignments = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    if conda_env:
        conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
        env_assignments.extend(
            [
                f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)}",
                f"CONDA_PREFIX={shlex.quote(conda_prefix)}",
                f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/miniconda3/bin:$PATH",
            ]
        )
    env_prefix = " ".join(env_assignments)
    argv = [
        "claude",
        "-p",
        task,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--permission-mode",
        permission_mode,
    ]
    if disable_slash_commands:
        argv.append("--disable-slash-commands")
    if disable_web_tools:
        argv.extend(["--disallowedTools", "Agent", "Task", "WebFetch", "WebSearch"])
    return (
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
        "cd /testbed; "
        f"{env_prefix} " + shlex.join(argv)
    )


async def install_claude_in_sandbox(sandbox: Sandbox) -> None:
    """在腾讯沙箱内安装 pin 版 claude-code（幂等：已装则跳过）。"""
    check = await sandbox.exec_shell("claude --version", timeout=60)
    if check.exit_code == 0 and CLAUDE_CODE_VERSION in (check.stdout or ""):
        logger.info("claude-code already installed: %s", (check.stdout or "").strip())
        return
    install_cmd = (
        f"npm install -g @anthropic-ai/claude-code@{CLAUDE_CODE_VERSION} "
        f"--registry={NPM_REGISTRY}"
    )
    result = await sandbox.exec_shell(install_cmd, timeout=600)
    if result.exit_code != 0:
        raise RuntimeError(
            f"claude-code install failed rc={result.exit_code}: "
            f"{(result.stdout or '')[-500:]} {(result.stderr or '')[-500:]}"
        )
    logger.info("claude-code %s installed in sandbox", CLAUDE_CODE_VERSION)


class SandboxEnvForReward:
    """把 :class:`Sandbox` 适配成 reward 评估用的 async env 接口。"""

    def __init__(self, sandbox: Sandbox):
        self._sandbox = sandbox

    async def communicate(self, input: str, timeout=600, check="ignore", error_msg="Command failed") -> str:
        result = await self._sandbox.exec_shell(input, timeout=int(timeout))
        if check == "raise" and result.exit_code != 0:
            raise RuntimeError(
                f"{error_msg} (exit_code={result.exit_code}) "
                f"stdout={result.stdout[:200]} stderr={result.stderr[:200]}"
            )
        return result.stdout

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        await self.communicate(f"echo {encoded} | base64 -d > {path}", check="raise", error_msg=f"write {path}")

    async def read_file(self, path: str | Path, **_) -> str:
        return await self.communicate(f"cat {path}")

    async def exec_shell(self, command: str, *, workdir=None, timeout=600):
        return await self._sandbox.exec_shell(command, workdir=workdir, timeout=int(timeout))


async def evaluate_in_env(env, metadata: dict, eval_timeout: int = 600) -> tuple[float, dict]:
    """在沙箱内跑 SWE-bench reward（与 mini-swe-agent runner 同口径）。"""
    data_source = metadata.get("data_source", "unknown")
    reward_model = metadata.get("reward_model", {})
    if data_source != "swe_bench":
        raise ValueError(f"Unsupported reward data source: {data_source}")
    from uni_agent.tasks.swe_bench.reward import compute_reward

    spec_metadata = reward_model.get("ground_truth", reward_model)
    result = await compute_reward(spec_metadata, env, eval_timeout=eval_timeout)
    score = 1.0 if result.get("resolved", False) else 0.0
    return score, result


async def claude_code_runner(
    *,
    raw_prompt: Any,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    run_timeout: int = DEFAULT_AGENT_RUN_TIMEOUT,
    model_name: str = "default",
    **_: Any,
) -> None:
    """跑一条 Claude Code 黑盒轨迹并上报 reward（腾讯沙箱 direct-URL 版）。"""
    tools_kwargs = tools_kwargs or {}
    task = build_claude_task(raw_prompt, tools_kwargs)
    env_config = tools_kwargs.get("env", {})
    image = extract_image(env_config)
    if not image:
        raise ValueError(f"No Docker image found in tools_kwargs.env for sample {sample_index}")
    gateway_url = session.base_url
    if not gateway_url:
        raise ValueError(f"gateway_url is empty for sample {sample_index}")

    sandbox = create_task_sandbox(image=image, gateway_url=gateway_url)
    try:
        post_setup_cmd = env_config.get("post_setup_cmd", "")
        if post_setup_cmd:
            setup_result = await sandbox.exec_shell(post_setup_cmd, timeout=120)
            if setup_result.exit_code != 0:
                logger.warning(
                    "post_setup_cmd failed rc=%s: %.300s",
                    setup_result.exit_code,
                    setup_result.stdout + setup_result.stderr,
                )

        await install_claude_in_sandbox(sandbox)

        # direct URL：session.base_url 是 /v1 API root，去掉 /v1 供 Anthropic 客户端拼 /v1/messages
        claude_base_url = gateway_url.removesuffix("/v1")
        agent_cmd = build_claude_command(
            task=task,
            base_url=claude_base_url,
            max_turns=max_turns,
            model=model_name,
            conda_env=None,  # 腾讯 code-interpreter 沙箱无 /opt/miniconda3，用默认 PATH
        )

        started_at = time.perf_counter()
        result = await sandbox.exec_shell(agent_cmd, timeout=int(run_timeout))
        elapsed = time.perf_counter() - started_at
        logger.info(
            "[sample %d] claude-code finished rc=%s elapsed=%.1fs", sample_index, result.exit_code, elapsed
        )
        if result.exit_code != 0:
            logger.warning(
                "[sample %d] claude-code failed stdout_tail=%r stderr_tail=%r",
                sample_index,
                (result.stdout or "")[-4000:],
                (result.stderr or "")[-4000:],
            )

        metadata = {
            "data_source": (tools_kwargs.get("reward") or {}).get("name", "unknown"),
            "reward_model": (tools_kwargs.get("reward") or {}).get("metadata", {}),
        }
        eval_timeout = int(os.environ.get("SWE_AGENT_EVAL_TIMEOUT", "600"))
        score, eval_result = await evaluate_in_env(SandboxEnvForReward(sandbox), metadata, eval_timeout)
        logger.info("[sample %d] reward done score=%s resolved=%s", sample_index, score, eval_result.get("resolved"))

        reward_info = {
            "reward_score": score,
            "claude_code_exit_code": result.exit_code,
            **eval_result,
        }
        if not session.reward_info_url:
            raise ValueError(f"reward_info_url is empty for session {session.session_id}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(session.reward_info_url, json={"reward_info": reward_info})
            response.raise_for_status()
    finally:
        await sandbox.stop()
