#!/usr/bin/env bash
# 投机解码 A/B 启动脚本（服务器上执行）：spec off → on，后台跑，日志 spec_ab.log
set -u
cd /home/ubuntu/swe-rl
export VLLM_USE_V1=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
nohup /home/ubuntu/miniforge3/envs/swe-rl/bin/python spec_bench_ab.py \
  --prompts /home/ubuntu/swe-rl/data/humanevalfix_train164.jsonl \
  --num-prompts 32 --n 4 --max-tokens 512 --temperature 0.8 \
  --model /home/ubuntu/models/Qwen3-8B \
  --draft /home/ubuntu/models/Qwen3-8B-speculator.eagle3 \
  --gpu-memory-utilization 0.7 --max-num-seqs 16 --max-model-len 8192 \
  > spec_ab.log 2>&1 &
echo "PID=$!"
sleep 3
tail -5 spec_ab.log
