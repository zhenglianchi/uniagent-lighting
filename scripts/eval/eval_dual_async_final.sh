#!/usr/bin/env bash
# 双机分离式全异步 + Mooncake 正式训练（25 步）最终权重评测
# 与 baseline/spec 评估口径一致：n=1 / temperature 0.8 / 161 条
set -u
cd /home/ubuntu/swe-rl
ENV_PY=/home/ubuntu/miniforge3/envs/swe-rl/bin/python
export VLLM_USE_V1=1
# 注入腾讯沙箱凭据（eval_humanevalfix.py 在服务器布局下路径推断失效）
set -a
source /home/ubuntu/swe-rl/tencent_sandbox.env
set +a
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}"
export TENCENT_SANDBOX_SKIP_TMUX=1

MODEL_PATH=/home/ubuntu/models/Qwen3-8B-final-dual-async
SERVED_NAME=qwen3-8b-final-dual-async

echo "== vllm serve $SERVED_NAME =="
nohup "$ENV_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --port 8001 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 8192 --enforce-eager \
  --served-model-name "$SERVED_NAME" \
  --gpu-memory-utilization 0.8 --enable-prefix-caching \
  --max-num-seqs 128 --enable-chunked-prefill \
  > logs/vllm_dual_async_final.log 2>&1 &
VLLM_PID=$!
echo "vllm pid=$VLLM_PID, waiting for ready..."
for i in $(seq 1 60); do
  if curl -s -o /dev/null http://127.0.0.1:8001/v1/models; then echo READY; break; fi
  sleep 5
done

echo "== eval $SERVED_NAME =="
"$ENV_PY" eval_humanevalfix.py \
  --data /home/ubuntu/swe-rl/data/humanevalfix_train161.jsonl \
  --base-url http://127.0.0.1:8001/v1 \
  --model "$SERVED_NAME" \
  --temperature 0.8 --concurrency 16 \
  --out logs/eval_dual_async_final.json --out-dir logs/eval_dual_async_final_dir

kill "$VLLM_PID" 2>/dev/null
echo "== DONE =="
"$ENV_PY" -c "import json; d=json.load(open('logs/eval_dual_async_final.json')); print('final-dual-async:', d['passed'], '/', d['total'], '=', d['pass_rate'])"
