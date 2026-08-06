#!/usr/bin/env python3
"""本地开发脚本：腾讯 E2B 沙箱 + 本地 OpenAI 兼容 API（阿里云百炼）跑 humaneval_fix 冒烟轨迹。

等价于 ``mini_swe_agent_runner`` 的 humaneval_fix 分支，但模型端点指向本地配置的
百炼 API（不走 Gateway / 训练机），用于在服务器恢复前先采几条真实轨迹：
看 8B 级模型修 HumanEvalFix 大概需要几轮、reward 是否可算、轨迹是否完整。

流程（复用同一个沙箱实例，逐样本重置 /testbed，省沙箱按小时计费的钱）：
    1. 建 code-interpreter-v1 沙箱（python:3.12），装 pytest
    2. 每个样本：重置 /testbed + git init + 注入 solution.py（可见任务文件）
    3. mini-swe-agent 走百炼 API 跑轨迹（step_limit=60），轨迹落 work/swebench/
    4. reward：注入隐藏测试 test_solution.py（无测试泄露）→ pytest 打分
    5. 汇总 JSON：交互轮数 / 耗时 / reward / 是否 resolved

用法：
    python scripts/run_humanevalfix_local.py --data work/data/humanevalfix_train.jsonl \
        --indices 0 1 2 --model qwen3.7-plus

环境变量（脚本自动加载，无需手动 export）：
    - work/tencent_sandbox.env  ：TENCENT_SANDBOX_E2B_TOKEN（e2b_*）、TENCENT_SECRET_ID/KEY
    - ~/.config/mini-swe-agent/.env ：OPENAI_API_KEY（百炼 sk-ws-*）
    - config/tencent_swebench.yaml  ：model.model_kwargs.api_base（百炼 OpenAI 兼容端点）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "work/uni-agent"))  # uni_agent 源码副本（未 pip 安装）

from dotenv import load_dotenv

from uni_agent_ext.agents.mini_swe_agent_runner import (
    build_mini_swe_config,
    create_task_sandbox,
    evaluate_reward,
    extract_task,
    run_mini_swe_agent_api,
)

REPO_WORK = ROOT.parent / "work"  # 采样产物统一放 work/（勿用 /tmp）


def load_envs() -> None:
    """加载沙箱凭据 + 百炼 API key（不覆盖已有环境变量）。"""
    load_dotenv(REPO_WORK / "tencent_sandbox.env")
    load_dotenv(Path.home() / ".config/mini-swe-agent/.env")
    os.environ.setdefault("E2B_DOMAIN", "ap-guangzhou.tencentags.com")
    os.environ["E2B_API_KEY"] = os.environ["TENCENT_SANDBOX_E2B_TOKEN"]


def load_records(data_path: Path, indices: list[int]) -> list[dict]:
    rows = [json.loads(line) for line in data_path.open(encoding="utf-8")]
    return [rows[i] for i in indices]


def read_model_config() -> tuple[str, str]:
    """从 config/tencent_swebench.yaml 读 model_name / api_base。"""
    import yaml

    candidates = [ROOT / "config/tencent_swebench.yaml", ROOT.parent / "config/tencent_swebench.yaml"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("tencent_swebench.yaml 未找到，请用 --model/--base-url 指定")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = cfg["model"]
    return model["model_name"], model["model_kwargs"]["api_base"]


async def reset_testbed(sandbox) -> None:
    """重置 /testbed：清空 + git init（agent 提交时依赖 git diff）。"""
    result = await sandbox.exec_shell(
        "rm -rf /testbed && mkdir -p /testbed && cd /testbed && "
        "(git --version >/dev/null 2>&1 || "
        "(DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git)) && "
        "git init -q && git config user.email t@example.com && git config user.name t",
        timeout=300,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"reset_testbed failed: {result.stderr[-500:]}")


async def ensure_pytest(sandbox) -> None:
    result = await sandbox.exec_shell(
        "python -c 'import pytest' 2>/dev/null || python -m pip install -q pytest",
        timeout=300,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"pytest install failed: {result.stderr[-500:]}")


async def run_one(
    sandbox,
    task: dict,
    *,
    model_name: str,
    api_base: str,
    api_key: str,
    max_turns: int,
    out_dir: Path,
) -> dict:
    instance_id = task["instance_id"]
    env_files = task["env_files"]
    await reset_testbed(sandbox)
    for rel_path, content in env_files.items():
        await sandbox.write_file(f"/testbed/{rel_path}", content)
    await sandbox.exec_shell("cd /testbed && git add -A", timeout=60)

    run_id = uuid.uuid4().hex
    config_path = Path(f"/tmp/mini_swe_config_{run_id}.yaml")
    traj_path = out_dir / f"humanevalfix_{instance_id}.traj.json"
    cfg_text = build_mini_swe_config(
        base_url=api_base,
        model=model_name,
        max_turns=max_turns,
        instance_id=sandbox.instance_id,
        image="python:3.12",
        temperature=0.8,
        output_path=str(traj_path),
    )
    import yaml

    cfg = yaml.safe_load(cfg_text)
    cfg["model"]["model_kwargs"]["api_key"] = api_key  # 百炼要真实 key，不用 "EMPTY"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    started = time.perf_counter()
    rc, log_tail = await run_mini_swe_agent_api(task=task, config_path=str(config_path), run_timeout=3600)
    elapsed = time.perf_counter() - started
    if rc != 0:
        print(f"[sample {instance_id}] agent rc={rc}, full log:\n{log_tail}")

    traj = {}
    if traj_path.exists():
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
    rounds = traj.get("info", {}).get("model_stats", {}).get("api_calls", -1)
    exit_status = traj.get("info", {}).get("exit_status", "")

    score, details = await evaluate_reward(sandbox, task)
    summary = {
        "instance_id": instance_id,
        "model": model_name,
        "rounds": rounds,
        "elapsed_s": round(elapsed, 1),
        "agent_rc": rc,
        "agent_exit_status": exit_status,
        "reward": score,
        "resolved": details.get("resolved"),
        "passed": details.get("passed"),
        "total": details.get("total"),
        "per_test": details.get("per_test"),
        "traj": str(traj_path),
        "log_tail": log_tail[-800:],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "work/data/humanevalfix_train.jsonl"))
    p.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--model", default="", help="默认读 config/tencent_swebench.yaml")
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--out-dir", default=str(REPO_WORK / "swebench"))
    args = p.parse_args()

    load_envs()
    default_model, api_base = read_model_config()
    model_name = args.model or default_model
    api_key = os.environ["OPENAI_API_KEY"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.data), args.indices)
    tasks = []
    for rec in records:
        tk = rec["extra_info"]["tools_kwargs"]
        task = extract_task(rec["prompt"], tk)
        task["env_files"] = tk["env"]["files"]
        tasks.append(task)

    print(f"[run] {len(tasks)} samples model={model_name} base={api_base} max_turns={args.max_turns}")
    sandbox = create_task_sandbox(image="python:3.12", gateway_url=None)
    try:
        await sandbox.start()
        print(f"[run] sandbox up: {sandbox.instance_id}")
        await ensure_pytest(sandbox)
        results = []
        for i, task in enumerate(tasks):
            print(f"\n=== sample {i + 1}/{len(tasks)}: {task['instance_id']} ===")
            results.append(
                await run_one(
                    sandbox,
                    task,
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    max_turns=args.max_turns,
                    out_dir=out_dir,
                )
            )
        summary_path = out_dir / "humanevalfix_local_summary.json"
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[run] summary -> {summary_path}")
    finally:
        await sandbox.stop()


if __name__ == "__main__":
    asyncio.run(main())
