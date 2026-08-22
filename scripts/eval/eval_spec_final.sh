#!/usr/bin/env bash
# 投机 run（修复后 5 epoch）最终权重评测：base vs final-spec 通过率对比
# 与 2026-08-09 基线评测口径一致：n=1 / temperature 0.8 / 161 条 / 并发 24
set -u
cd /home/ubuntu/swe-rl
ENV_PY=/home/ubuntu/miniforge3/envs/swe-rl/bin/python
export VLLM_USE_V1=1
# eval_humanevalfix.py 的 load_envs 路径推断在服务器布局下失效（ROOT=parents[1]
# 算成 /home/ubuntu），这里直接注入腾讯沙箱凭据，不依赖脚本内部加载
set -a
source /home/ubuntu/swe-rl/tencent_sandbox.env
set +a
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}"
export TENCENT_SANDBOX_SKIP_TMUX=1

# 1. 合并 step 25 LoRA 到 HF 权重
echo "== convert step25 -> Qwen3-8B-final-spec =="
"$ENV_PY" convert_verl_lora_to_hf.py \
  --ckpt /home/ubuntu/swe-rl/checkpoints/humanevalfix_spec/global_step_25/actor/model_world_size_1_rank_0.pt \
  --base /home/ubuntu/models/Qwen3-8B \
  --lora-meta /home/ubuntu/swe-rl/checkpoints/humanevalfix_spec/global_step_25/actor/lora_train_meta.json \
  --out /home/ubuntu/models/Qwen3-8B-final-spec || exit 1

run_eval() {
  local MODEL_PATH="$1" SERVED_NAME="$2" OUT="$3" OUTDIR="$4" VLLM_LOG="$5"
  echo "== vllm serve $SERVED_NAME =="
  nohup "$ENV_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --port 8001 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --enforce-eager \
    --served-model-name "$SERVED_NAME" \
    --gpu-memory-utilization 0.8 --enable-prefix-caching \
    --max-num-seqs 128 --enable-chunked-prefill \
    > "$VLLM_LOG" 2>&1 &
  local VLLM_PID=$!
  echo "vllm pid=$VLLM_PID, waiting for ready..."
  for i in $(seq 1 60); do
    if curl -s -o /dev/null http://127.0.0.1:8001/v1/models; then break; fi
    sleep 5
  done
  echo "== eval $SERVED_NAME =="
  "$ENV_PY" eval_humanevalfix.py \
    --data /home/ubuntu/swe-rl/data/humanevalfix_train161.jsonl \
    --base-url http://127.0.0.1:8001/v1 \
    --model "$SERVED_NAME" \
    --temperature 0.8 --concurrency 8 \
    --out "$OUT" --out-dir "$OUTDIR"
  kill "$VLLM_PID" 2>/dev/null
  sleep 5
}

run_eval /home/ubuntu/models/Qwen3-8B qwen3-8b-base \
  logs/eval_spec_base.json logs/eval_spec_base_dir logs/vllm_spec_base.log
run_eval /home/ubuntu/models/Qwen3-8B-final-spec qwen3-8b-final-spec \
  logs/eval_spec_final.json logs/eval_spec_final_dir logs/vllm_spec_final.log

echo "== DONE =="
echo "base:  $(python3 -c "import json; d=json.load(open('logs/eval_spec_base.json')); print(d['passed'], '/', d['total'], '=', d['pass_rate'])")"
echo "final: $(python3 -c "import json; d=json.load(open('logs/eval_spec_final.json')); print(d['passed'], '/', d['total'], '=', d['pass_rate'])")"
