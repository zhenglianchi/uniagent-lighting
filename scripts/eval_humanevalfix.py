#!/usr/bin/env python3
"""HumanEvalFix 通过率评测：n=1，每个样本跑一次，并发沙箱，输出 pass rate。

用于"训练前（基座）vs 训练后（final LoRA 合并模型）"对比。模型端点指向
本机 vLLM（OpenAI 兼容），走与训练一致的 mini-swe-agent harness + 腾讯沙箱 +
隐藏测试 reward（无测试泄露）。

用法（先起 vLLM，两个模型分别跑一遍）：
  vllm serve /home/ubuntu/models/Qwen3-8B \\
      --served-model-name qwen3-8b-base --port 8001 \\
      --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.75
  python scripts/eval_humanevalfix.py \\
      --base-url http://127.0.0.1:8001/v1 --model qwen3-8b-base \\
      --data work/data/humanevalfix_train161.jsonl --concurrency 16 \\
      --out work/eval/base.json
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
sys.path.insert(0, str(ROOT.parent / "work/uni-agent"))

from dotenv import load_dotenv

from uni_agent_ext.agents.mini_swe_agent_runner import (
    build_mini_swe_config,
    create_task_sandbox,
    evaluate_reward,
    extract_task,
    run_mini_swe_agent_api,
)

REPO_WORK = ROOT.parent / "work"


def load_envs() -> None:
    candidates = [
        REPO_WORK / "tencent_sandbox.env",                    # 本地布局
        ROOT.parent / "swe-rl" / "tencent_sandbox.env",       # 服务器布局
    ]
    for c in candidates:
        if c.exists():
            load_dotenv(c)
            break
    os.environ.setdefault("E2B_DOMAIN", "ap-guangzhou.tencentags.com")
    os.environ["E2B_API_KEY"] = os.environ["TENCENT_SANDBOX_E2B_TOKEN"]
    # 跳过沙箱内 tmux 安装（E2B 请求可能挂起，白耗时间）
    os.environ["TENCENT_SANDBOX_SKIP_TMUX"] = "1"


def load_tasks(data_path: Path) -> list[dict]:
    tasks = []
    for line in data_path.open(encoding="utf-8"):
        rec = json.loads(line)
        tk = rec["extra_info"]["tools_kwargs"]
        task = extract_task(rec["prompt"], tk)
        task["env_files"] = tk["env"]["files"]
        tasks.append(task)
    return tasks


async def reset_testbed(sandbox) -> None:
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
    temperature: float,
    out_dir: Path,
) -> dict:
    instance_id = task["instance_id"]
    await reset_testbed(sandbox)
    for rel_path, content in task["env_files"].items():
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
        temperature=temperature,
        output_path=str(traj_path),
    )
    import yaml

    cfg = yaml.safe_load(cfg_text)
    cfg["model"]["model_kwargs"]["api_key"] = api_key
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    started = time.perf_counter()
    rc, log_tail = await run_mini_swe_agent_api(task=task, config_path=str(config_path), run_timeout=3600)
    elapsed = time.perf_counter() - started

    traj = {}
    if traj_path.exists():
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
    rounds = traj.get("info", {}).get("model_stats", {}).get("api_calls", -1)

    score, details = await evaluate_reward(sandbox, task)
    return {
        "instance_id": instance_id,
        "model": model_name,
        "rounds": rounds,
        "elapsed_s": round(elapsed, 1),
        "agent_rc": rc,
        "reward": score,
        "resolved": details.get("resolved"),
        "passed": details.get("passed"),
        "total": details.get("total"),
        "per_test": details.get("per_test"),
        "traj": str(traj_path),
        "log_tail": (log_tail or "")[-500:],
    }


async def worker(
    chunk: list[dict],
    *,
    model_name: str,
    api_base: str,
    api_key: str,
    max_turns: int,
    temperature: float,
    out_dir: Path,
    worker_id: int,
) -> list[dict]:
    results = []
    sandbox = create_task_sandbox(image="python:3.12", gateway_url=None)
    try:
        await sandbox.start()
        await ensure_pytest(sandbox)
        print(f"[worker {worker_id}] sandbox up: {sandbox.instance_id}", flush=True)
        for i, task in enumerate(chunk):
            print(f"[worker {worker_id}] {i + 1}/{len(chunk)} {task['instance_id']}", flush=True)
            try:
                res = await run_one(
                    sandbox,
                    task,
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    max_turns=max_turns,
                    temperature=temperature,
                    out_dir=out_dir,
                )
                results.append(res)
                print(
                    f"[worker {worker_id}] done {task['instance_id']} "
                    f"reward={res.get('reward')} resolved={res.get('resolved')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - 单个样本失败不拖垮整个 worker
                results.append(
                    {
                        "instance_id": task["instance_id"],
                        "model": model_name,
                        "reward": 0.0,
                        "resolved": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        await sandbox.stop()
    return results


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "work/data/humanevalfix_train161.jsonl"))
    p.add_argument("--base-url", required=True, help="vLLM OpenAI 兼容端点（如 http://127.0.0.1:8001/v1）")
    p.add_argument("--model", required=True, help="vLLM served model 名")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.2, help="评测采样温度（默认 0.2，更稳定）")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", required=True, help="汇总 JSON 输出路径")
    p.add_argument("--out-dir", default=str(REPO_WORK / "eval"))
    args = p.parse_args()

    load_envs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(Path(args.data))
    print(f"[eval] {len(tasks)} samples model={args.model} base={args.base_url} concurrency={args.concurrency}")

    chunks = [tasks[i:: args.concurrency] for i in range(args.concurrency)]
    chunks = [c for c in chunks if c]
    # 真正的并发：asyncio.gather 同时跑所有 worker（每个 worker 一个沙箱）
    results_lists = await asyncio.gather(
        *[
            worker(
                chunk,
                model_name=args.model,
                api_base=args.base_url,
                api_key=args.api_key,
                max_turns=args.max_turns,
                temperature=args.temperature,
                out_dir=out_dir,
                worker_id=wid,
            )
            for wid, chunk in enumerate(chunks)
        ]
    )
    results = [r for rl in results_lists for r in rl]

    total = len(results)
    passed = sum(1 for r in results if r.get("resolved"))
    rewards = [r.get("reward", 0.0) for r in results]
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "per_sample": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] done: {passed}/{total} passed, pass_rate={summary['pass_rate']}, mean_reward={summary['reward_mean']}")
    print(f"[eval] summary -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
