#!/usr/bin/env bash
# 仅评测（vLLM server 已就绪，不重启）：n=1 / temp 0.8 / 161 条 / 并发 24
set -u
cd /home/ubuntu/swe-rl
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
  --concurrency 24 \
  --out /home/ubuntu/swe-rl/eval_final_spec.json \
  --out-dir /home/ubuntu/swe-rl/eval_final_spec_dir \
  > eval_final_spec.log 2>&1
echo "EVAL_EXIT=$?"
tail -5 eval_final_spec.log
