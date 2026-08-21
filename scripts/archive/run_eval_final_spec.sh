#!/usr/bin/env bash
# 投机 run 最终权重评测（n=1 / temp 0.8 / 161 条 / 并发 24，与基线评测同口径）
set -u
cd /home/ubuntu/swe-rl

# 起 vLLM server（Qwen3-8B-final-spec，参数与训练一致）
pkill -f "Qwen3-8B-final-spec" 2>/dev/null
sleep 2
nohup /home/ubuntu/miniforge3/envs/swe-rl/bin/vllm serve /home/ubuntu/models/Qwen3-8B-final-spec \
  --port 8001 \
  --served-model-name Qwen3-8B-final-spec \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 128 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > vllm_final_spec.log 2>&1 &
echo "VLLM_PID=$!"

# 等 server 就绪（最多 5 分钟）
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "VLLM_READY after ${i}x5s"
    break
  fi
  sleep 5
done

# 评测（n=1、temp 0.8、161 条、并发 24）
set -a
source /home/ubuntu/swe-rl/tencent_sandbox.env
set +a
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}"
/home/ubuntu/miniforge3/envs/swe-rl/bin/python eval_humanevalfix.py \
  --data /home/ubuntu/swe-rl/data/humanevalfix_train161.jsonl \
  --base-url http://127.0.0.1:8001/v1 \
  --model Qwen3-8B-final-spec \
  --max-turns 40 \
  --temperature 0.8 \
  --concurrency 16 \
  --out /home/ubuntu/swe-rl/eval_final_spec.json \
  --out-dir /home/ubuntu/swe-rl/eval_final_spec_dir \
  > eval_final_spec.log 2>&1
echo "EVAL_EXIT=$?"
tail -5 eval_final_spec.log
