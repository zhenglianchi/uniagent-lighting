#!/usr/bin/env bash
# 挂后台 watcher：增量收集投机 run 的逐步统计（每 30s 一轮）
set -u
cd /home/ubuntu/swe-rl
nohup /home/ubuntu/miniforge3/envs/swe-rl/bin/python collect_grpo_stats.py \
  --log /home/ubuntu/swe-rl/grpo_humanevalfix_spec.log \
  --sessions /home/ubuntu/swe-rl/logs/humanevalfix_spec \
  --out /home/ubuntu/swe-rl/logs/grpo_stats_spec.jsonl \
  --watch --interval 30 > logs/stats_watch_spec.log 2>&1 &
echo "WATCHER_PID=$!"
