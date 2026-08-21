#!/usr/bin/env bash
# 停掉当前训练 run（投机 logprobs bug 排查用）
set -u
pkill -9 -f 'verl.trainer.main_ppo' 2>/dev/null
sleep 2
/home/ubuntu/miniforge3/envs/swe-rl/bin/ray stop --force >/dev/null 2>&1
sleep 2
pkill -9 -f 'vllm' 2>/dev/null
pkill -9 -f 'GatewayActor' 2>/dev/null
sleep 2
echo "REMAIN=$(ps aux | grep -cE '[m]ain_ppo|[v]llm|[G]atewayActor')"
echo "GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
