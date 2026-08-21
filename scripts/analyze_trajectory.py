#!/usr/bin/env python3
"""分析一条训练轨迹（framework.log / trajectory.json / trajectory.npz）。

用法：
  # 轨迹已压缩归档，先解压再分析
  tar -xzf work/logs/humanevalfix_full_20260809/humanevalfix_trajectories.tgz -C /tmp
  python analyze_trajectory.py /tmp/humanevalfix/step_1/session-sample-0-rollout-0-xxx

输出：会话摘要 + token 级字段形状 + logprob/reward 统计。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def analyze_session(session_dir: Path) -> None:
    print(f"== 会话目录: {session_dir.name} ==")

    # 1. framework.log：会话摘要
    fw = session_dir / "framework.log"
    if fw.exists():
        for line in fw.read_text(errors="ignore").splitlines():
            if "trajectory(ies)" in line or "turns=" in line:
                print(f"[framework] {line.strip()}")

    # 2. trajectory.json：可读摘要
    tj = session_dir / "trajectory.json"
    if tj.exists():
        data = json.loads(tj.read_text())
        print(f"[trajectory.json] {data.get('num_trajectories')} 条轨迹")
        for i, t in enumerate(data.get("trajectories", [])):
            print(
                f"  traj[{i}]: turns={t.get('num_turns')} "
                f"prompt={t.get('prompt_len')} response={t.get('response_len')} "
                f"reward={t.get('reward_score')} "
                f"reason={t.get('materialization_reason')}"
            )

    # 3. trajectory.npz：token 级数据
    npz = session_dir / "trajectory.npz"
    if npz.exists():
        d = np.load(npz)
        print("[trajectory.npz] token 级字段:")
        for key in sorted(d.keys()):
            arr = d[key]
            extra = ""
            if key.endswith("response_logprobs") and arr.size:
                extra = f" logprob mean={arr.mean():.4f}"
            if key.endswith("response_mask") and arr.size:
                extra = f" mask=1 占比 {arr.mean():.1%}"
            print(f"  {key}: {arr.shape} {arr.dtype}{extra}")

    # 4. task.log 尾部：agent 行为
    tl = session_dir / "task.log"
    if tl.exists():
        lines = tl.read_text(errors="ignore").splitlines()
        tail = [l for l in lines if "mini-swe-agent" in l or "evaluate_reward" in l][-3:]
        if tail:
            print("[task.log] 关键行为:")
            for l in tail:
                print(f"  {l.strip()}")


def main() -> None:
    p = argparse.ArgumentParser(description="分析一条训练轨迹")
    p.add_argument("session_dir", help="会话目录路径（含 trajectory.json/npz）")
    args = p.parse_args()
    d = Path(args.session_dir)
    if not d.is_dir():
        sys.exit(f"目录不存在: {d}")
    analyze_session(d)


if __name__ == "__main__":
    main()
