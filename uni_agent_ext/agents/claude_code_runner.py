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

import json
import logging
import os
import shlex
import time
from typing import Any
from urllib.parse import urlparse

import httpx

try:
    from uni_agent.gateway.session import SessionHandle
except Exception:  # noqa: BLE001 - 本地开发（无 ray）也能 import runner 的纯函数部分
    SessionHandle = Any  # type: ignore[misc,assignment] - 仅用于类型注解
from uni_agent.sandbox import Sandbox

from uni_agent_ext.agents.mini_swe_agent_runner import (
    create_task_sandbox,
    ensure_gateway_tunnel,
    evaluate_reward as evaluate_reward_msa,
    extract_task as extract_task_meta,
)

logger = logging.getLogger(__name__)

# vLLM 0.11.1 严格校验 Claude Code 2.1.154+ 的非标准 role（system/ctx/msg），
# 修复 2026-06 才合入 vLLM → 必须 pin < 2.1.154
CLAUDE_CODE_VERSION = "2.1.153"
NPM_REGISTRY = "https://registry.npmmirror.com"
# 与白盒 mini-swe-agent 同口径（2026-08-12 用户定）：MSA_AGENT_MAX_TURNS 默认 60
DEFAULT_MAX_TURNS = int(os.getenv("CLAUDE_AGENT_MAX_TURNS", "60"))
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
        # 与 Gateway 的 GATEWAY_MAX_GENERATION_TOKENS 截断一致（v0.38.2）：
        # claude-code 用它作为请求 max_tokens，太大（32000/128000）会让 vLLM
        # 400（max_model_len - max_tokens 为负）
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "8192"),
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
        "--bare",
        "-p",
        task,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--permission-mode",
        permission_mode,
        "--debug-file",
        "/tmp/claude-debug.log",
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
    result = await sandbox.exec_shell(install_cmd, timeout=900)
    if result.exit_code != 0:
        raise RuntimeError(
            f"claude-code install failed rc={result.exit_code}: "
            f"{(result.stdout or '')[-500:]} {(result.stderr or '')[-500:]}"
        )
    logger.info("claude-code %s installed in sandbox", CLAUDE_CODE_VERSION)


async def evaluate_in_env(sandbox: Sandbox, raw_prompt, tools_kwargs: dict, eval_timeout: int = 600) -> tuple[float, dict]:
    """在沙箱内跑 reward，与白盒 mini-swe-agent runner 完全同口径。

    复用 :func:`mini_swe_agent_runner.evaluate_reward`（swe_bench / humaneval_fix 双口径：
    test_patch git apply + FAIL_TO_PASS/P2P 分级打分；humaneval_fix 写 hidden_files 后
    跑 test_solution.py::test_all）。这样黑白盒 reward 口径一致，便于同条件对比。
    """
    task = extract_task_meta(raw_prompt, tools_kwargs)
    score, details = await evaluate_reward_msa(sandbox, task, timeout=eval_timeout)
    return score, details


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
        await sandbox.start()
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

        # 任务文件注入（humaneval_fix）：沙箱预置 /testbed git 仓库 + solution.py。
        # 只注入任务文件，隐藏测试由 evaluate_in_env 在完成后写入（无测试泄露）。
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
            # 让 solution.py 被 git 跟踪，否则 agent 提交时 `git diff` 拿不到 patch
            await sandbox.exec_shell("cd /testbed && git add -A", timeout=60)

        # 沙箱内 claude-code 访问 Gateway：
        # - direct-URL 模式（默认，定稿方案 v0.35.1）：ANTHROPIC_BASE_URL 直连公网
        #   Gateway（GATEWAY_PORT 固定 + 安全组放行 + CLAUDE_GATEWAY_PUBLIC_HOST=公网 IP）
        # - 隧道模式（CLAUDE_GATEWAY_TUNNEL=1，备选）：沙箱内 ssh -N -L 走训练机公网
        #   22 端口转发 Gateway（仅需放行 22）
        if os.environ.get("CLAUDE_GATEWAY_TUNNEL", "0") == "1":
            claude_base_url = await ensure_gateway_tunnel(sandbox, gateway_url)
        else:
            public_host = os.environ.get("CLAUDE_GATEWAY_PUBLIC_HOST", "")
            if public_host:
                parsed = urlparse(gateway_url)
                gateway_url = (
                    f"{parsed.scheme}://{public_host}:{parsed.port}{parsed.path}"
                )
            claude_base_url = gateway_url
        # session.base_url 是 /v1 API root，去掉 /v1 供 Anthropic 客户端拼 /v1/messages
        claude_base_url = claude_base_url.removesuffix("/v1")
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
        # 完整输出落盘（调试黑盒请求/ECONNRESET 用）
        claude_stdout = result.stdout or ""
        claude_stderr = result.stderr or ""
        logger.info("[sample %d] claude-code stdout_tail=%.3000s", sample_index, claude_stdout[-3000:])
        if claude_stderr:
            logger.info("[sample %d] claude-code stderr_tail=%.2000s", sample_index, claude_stderr[-2000:])
        if result.exit_code != 0:
            logger.warning(
                "[sample %d] claude-code failed stdout_tail=%r stderr_tail=%r",
                sample_index,
                claude_stdout[-4000:],
                claude_stderr[-4000:],
            )
            # 取回 claude-debug 日志（API 请求/响应，排障用）
            debug_log = await sandbox.exec_shell(
                "tail -80 /tmp/claude-debug.log 2>/dev/null || true", timeout=30
            )
            if debug_log.stdout:
                logger.info(
                    "[sample %d] claude-debug tail:\n%s",
                    sample_index,
                    (debug_log.stdout or "")[-6000:],
                )

        eval_timeout = int(os.environ.get("SWE_AGENT_EVAL_TIMEOUT", "600"))
        score, eval_result = await evaluate_in_env(sandbox, raw_prompt, tools_kwargs, eval_timeout)
        logger.info("[sample %d] reward done score=%s resolved=%s", sample_index, score, eval_result.get("resolved"))

        reward_info = {
            # framework._score_from_reward_info 消费 "reward" 键（reward_score 只是兼容字段）
            "reward": score,
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
