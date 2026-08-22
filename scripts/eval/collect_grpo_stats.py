#!/usr/bin/env python3
"""GRPO 训练逐步统计收集器：把每个 batch（step）的 rollout / 训练 / 奖励明细落盘 JSONL。

数据源（都不需要额外依赖，stdlib only）：
1. verl 训练主日志（--log）里的 step 指标行
   （``step:N - ... critic/rewards/mean:... - timing_s/gen:... - timing_s/update_actor:...``）
2. AgentFrameworkWorker 的 ``generate_sequences summary:`` 行（每 batch 一条：
   num_input_prompts / num_success_sessions / num_success_outputs ...）
3. 会话日志（--sessions 根目录下 ``step_N/session-*/task.log``）里的
   ``evaluate_reward: <instance> -> <score> (...)`` 逐条 reward

输出：一行一个 step 的 JSONL，字段见 :func:`row_for_step`。

用法：
  # 一次性解析
  python scripts/eval/collect_grpo_stats.py \\
      --log /home/ubuntu/swe-rl/grpo_humanevalfix_train8.log \\
      --sessions /home/ubuntu/swe-rl/logs/humanevalfix \\
      --out /home/ubuntu/swe-rl/logs/grpo_stats_train8.jsonl
  # 常驻监听（训练过程中每 --interval 秒增量落盘一次，可 nohup 后台跑）
  python scripts/eval/collect_grpo_stats.py ... --watch --interval 30
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# verl step 指标行：step:3 - key:val - key:val ...
_STEP_RE = re.compile(r"step:(\d+)\s+-")
_KV_RE = re.compile(r"([a-z_][\w/]*):(-?(?:\d+\.?\d*|np\.(?:float64|int64)\([\d.eE+-]+\)))")

# AgentFrameworkWorker 每 batch 汇总
_SUMMARY_RE = re.compile(
    r"generate_sequences summary:\s*"
    r"num_input_prompts=(\d+)\s+"
    r"num_success_sessions=(\d+)\s+"
    r"num_failed_sessions=(\d+)\s+"
    r"num_success_outputs=(\d+)\s+"
    r"num_unfinished_episodes=(\d+)\s+"
    r"num_failed_uids=(\d+)"
)

# 会话日志：evaluate_reward: <instance> -> <score> (...)
_REWARD_RE = re.compile(r"evaluate_reward:\s*(\S+)\s*->\s*([\d.]+)\s*\(")

_FLOAT_RE = re.compile(r"np\.(?:float64|int64)\(([\d.eE+-]+)\)")


def _num(value: str) -> float:
    m = _FLOAT_RE.search(value)
    if m:
        return float(m.group(1))
    return float(value)


def parse_step_metrics(log: str) -> dict[int, dict[str, float]]:
    """从训练主日志提取每个 global_step 的指标（保留最后一次出现的同 step 行）。"""
    steps: dict[int, dict[str, float]] = {}
    for line in log.splitlines():
        m = _STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        kvs = {}
        for key, val in _KV_RE.findall(line):
            try:
                kvs[key] = _num(val)
            except ValueError:
                continue
        if "training/global_step" in kvs:
            steps[step] = kvs
    return steps


def parse_summaries(log: str) -> list[dict[str, int]]:
    rows = []
    for line in log.splitlines():
        m = _SUMMARY_RE.search(line)
        if m:
            rows.append(
                {
                    "num_input_prompts": int(m.group(1)),
                    "num_success_sessions": int(m.group(2)),
                    "num_failed_sessions": int(m.group(3)),
                    "num_success_outputs": int(m.group(4)),
                    "num_unfinished_episodes": int(m.group(5)),
                    "num_failed_uids": int(m.group(6)),
                }
            )
    return rows


def parse_session_rewards(sessions_root: Path) -> dict[int, list[dict]]:
    """按 step 目录收集逐条 reward：{step: [{"instance": ..., "reward": ...}, ...]}。"""
    out: dict[int, list[dict]] = {}
    if not sessions_root.exists():
        return out
    for step_dir in sorted(sessions_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
        step = int(step_dir.name.split("_")[1])
        for task_log in step_dir.glob("session-*/task.log"):
            try:
                text = task_log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in _REWARD_RE.finditer(text):
                out.setdefault(step, []).append(
                    {"instance": m.group(1), "reward": float(m.group(2))}
                )
    return out


def row_for_step(
    step: int,
    metrics: dict[str, float],
    summary: dict[str, int] | None,
    rewards: list[dict],
    samples_hint: list[str],
    run_name: str,
    ts: str,
) -> dict:
    per_session: dict[str, list[float]] = {}
    for r in rewards:
        per_session.setdefault(r["instance"], []).append(r["reward"])
    reward_scores = [r["reward"] for r in rewards] or [metrics.get("critic/rewards/mean")]
    row = {
        "run": run_name,
        "ts": ts,
        "step": step,
        "samples": samples_hint,
        "num_sessions": len(rewards) or (summary or {}).get("num_success_sessions"),
        "rewards": {
            "mean": metrics.get("critic/rewards/mean"),
            "min": metrics.get("critic/rewards/min"),
            "max": metrics.get("critic/rewards/max"),
            "per_session": per_session,
            "scores": reward_scores,
        },
        "advantages": {
            "mean": metrics.get("critic/advantages/mean"),
            "min": metrics.get("critic/advantages/min"),
            "max": metrics.get("critic/advantages/max"),
        },
        "rollout_gen_s": metrics.get("timing_s/gen"),
        "rollout_old_logprob_s": metrics.get("timing_s/old_log_prob"),
        "train_update_actor_s": metrics.get("timing_s/update_actor"),
        "save_checkpoint_s": metrics.get("timing_s/save_checkpoint"),
        "step_total_s": metrics.get("timing_s/step"),
        "num_tokens": metrics.get("perf/total_num_tokens"),
        "throughput_tok_s": metrics.get("perf/throughput"),
        "num_turns": {
            "mean": metrics.get("training/num_turns/mean"),
            "min": metrics.get("training/num_turns/min"),
            "max": metrics.get("training/num_turns/max"),
        },
        "actor_grad_norm": metrics.get("actor/grad_norm"),
        "actor_pg_loss": metrics.get("actor/pg_loss"),
        "response_len": {
            "mean": metrics.get("response_length/mean"),
            "min": metrics.get("response_length/min"),
            "max": metrics.get("response_length/max"),
        },
    }
    if summary:
        row.update(
            {
                "num_input_prompts": summary["num_input_prompts"],
                "num_success_sessions": summary["num_success_sessions"],
                "num_failed_sessions": summary["num_failed_sessions"],
                "num_success_outputs": summary["num_success_outputs"],
                "num_unfinished_episodes": summary["num_unfinished_episodes"],
                "num_failed_uids": summary["num_failed_uids"],
            }
        )
    return row


def collect(
    log_path: Path,
    sessions_root: Path,
    run_name: str,
    samples_by_step: dict[int, list[str]] | None = None,
) -> list[dict]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_step_metrics(log_text)
    summaries = parse_summaries(log_text)
    session_rewards = parse_session_rewards(sessions_root)
    samples_by_step = samples_by_step or {}
    rows = []
    for step in sorted(metrics):
        summary = summaries[step - 1] if step - 1 < len(summaries) else None
        rows.append(
            row_for_step(
                step,
                metrics[step],
                summary,
                session_rewards.get(step, []),
                samples_by_step.get(step, []),
                run_name,
                datetime.now(timezone.utc).isoformat(),
            )
        )
    return rows


def load_samples_by_step(sessions_root: Path) -> dict[int, list[str]]:
    """从会话日志里的 extract_task[<instance>] 反推每个 step 处理的样本。"""
    out: dict[int, list[str]] = {}
    if not sessions_root.exists():
        return out
    for step_dir in sorted(sessions_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
        step = int(step_dir.name.split("_")[1])
        seen: list[str] = []
        for task_log in step_dir.glob("session-*/task.log"):
            try:
                text = task_log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in re.finditer(r"extract_task\[([\w-]+)\]", text):
                if m.group(1) not in seen:
                    seen.append(m.group(1))
        out[step] = seen
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="verl 训练主日志路径")
    p.add_argument("--sessions", required=True, help="agent framework 会话日志根目录（含 step_N/）")
    p.add_argument("--out", required=True, help="输出 JSONL 路径")
    p.add_argument("--run", default="grpo", help="run 名称（写入每行 run 字段）")
    p.add_argument("--watch", action="store_true", help="常驻监听，增量落盘")
    p.add_argument("--interval", type=int, default=30, help="watch 轮询间隔（秒）")
    args = p.parse_args()

    log_path = Path(args.log)
    sessions_root = Path(args.sessions)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_steps: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                seen_steps.add(int(json.loads(line)["step"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    def flush_once() -> None:
        nonlocal seen_steps
        if not log_path.exists():
            return
        samples_by_step = load_samples_by_step(sessions_root)
        rows = collect(log_path, sessions_root, args.run, samples_by_step)
        new_rows = [r for r in rows if r["step"] not in seen_steps]
        if not new_rows:
            return
        with out_path.open("a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        seen_steps.update(r["step"] for r in new_rows)
        print(f"[stats] appended {len(new_rows)} rows (steps {[r['step'] for r in new_rows]}), total {len(seen_steps)}", flush=True)

    if not args.watch:
        flush_once()
        return
    print(f"[stats] watching {log_path} every {args.interval}s -> {out_path}", flush=True)
    while True:
        try:
            flush_once()
        except Exception as exc:  # noqa: BLE001 - 监听进程不能因为单次解析失败退出
            print(f"[stats] parse error: {exc!r}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
