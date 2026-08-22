#!/usr/bin/env bash
# 正式双机训练链（node1 后台执行）：
#   separate_async + Mooncake + EAGLE-3 训练（失败自动 resume，最多 3 次）
#   → 收集 node2 本地 actor checkpoint → LoRA 合并 → vLLM 评估
#
# 背景：separate_async 下 FSDP worker 在 node2，actor 权重存 node2 本地
# /home/ubuntu/swe-rl/checkpoints/<CKPT>/global_step_N/actor/；
# trainer 侧 data.pt + latest_checkpointed_iteration.txt 在 node1。
# 训练完成后必须从 node2 拉回 actor 目录，node1 才能合并/评估。
#
# 用法（node1）：
#   setsid nohup bash /home/ubuntu/swe-rl/run_dual_formal_chain.sh \
#     > /home/ubuntu/swe-rl/grpo_dual_chain.log 2>&1 < /dev/null &
set -u
cd /home/ubuntu/swe-rl
ENV_PY=/home/ubuntu/miniforge3/envs/swe-rl/bin/python
NODE2=ubuntu@10.60.253.166
LOG=grpo_humanevalfix_dual_async_mooncake.log
CKPT_DIR=/home/ubuntu/swe-rl/checkpoints/humanevalfix_dual_async
OUT=/home/ubuntu/models/Qwen3-8B-final-dual-async
EVAL_OUT=/home/ubuntu/swe-rl/eval_dual_async.json
EVAL_DIR=/home/ubuntu/swe-rl/eval_dual_async_dir

echo "==== dual formal chain start $(date) ====" >> "$LOG"

# 1. 正式训练（失败自动 resume 重试，最多 3 次；保留最近 3 个 checkpoint 便于回退）
TRAIN_EXIT=1
for attempt in 1 2 3; do
  echo "==== train attempt $attempt $(date) ====" >> "$LOG"
  # 先清理残留沙箱（崩溃/被杀会泄漏 E2B 实例，64 并发 × 2 次尝试即可打爆 CPU 配额）
  set -a
  source /home/ubuntu/swe-rl/tencent_sandbox.env
  set +a
  "$ENV_PY" /home/ubuntu/swe-rl/tencent_stop_all_instances.py >> "$LOG" 2>&1
  sleep 10
  MAX_CKPT_KEEP=3 MOONCAKE=1 bash run_grpo_dual_async_mooncake_ucloud.sh >> "$LOG" 2>&1
  TRAIN_EXIT=$?
  echo "TRAIN_EXIT=$TRAIN_EXIT attempt=$attempt $(date)" >> "$LOG"
  if [ "$TRAIN_EXIT" -eq 0 ]; then
    break
  fi
  sleep 30
done
[ "$TRAIN_EXIT" -ne 0 ] && { echo "training failed after retries, skip eval" >> "$LOG"; exit 1; }

# 2. 定位最终 step（tracker 在 node1）
LATEST=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | tr -d " \n")
[ -z "$LATEST" ] && LATEST=$(ls "$CKPT_DIR" 2>/dev/null | grep global_step | sort -V | tail -1 | sed "s/global_step_//")
echo "latest_step=$LATEST $(date)" >> "$LOG"

# 3. 收集 actor checkpoint（FSDP worker 落点不固定：本地 node1 优先，node2 兜底）
CKPT_SUB="global_step_${LATEST}"
mkdir -p "$CKPT_DIR/$CKPT_SUB"
echo "==== collect actor from node2 $(date) ====" >> "$LOG"
if [ -d "$CKPT_DIR/$CKPT_SUB/actor" ]; then
  echo "actor already on node1" >> "$LOG"
else
  scp -r -o StrictHostKeyChecking=no "$NODE2:$CKPT_DIR/$CKPT_SUB/actor" "$CKPT_DIR/$CKPT_SUB/" >> "$LOG" 2>&1 \
    && echo "actor collected from node2" >> "$LOG" \
    || echo "actor missing on node1 and node2" >> "$LOG"
fi
CKPT="$CKPT_DIR/$CKPT_SUB/actor/model_world_size_1_rank_0.pt"
LORA_META="$CKPT_DIR/$CKPT_SUB/actor/lora_train_meta.json"
if [ ! -f "$CKPT" ] || [ ! -f "$LORA_META" ]; then
  echo "collect failed: missing $CKPT or $LORA_META" >> "$LOG"
  exit 1
fi

# 4. LoRA 合并
echo "==== convert lora $(date) ====" >> "$LOG"
"$ENV_PY" convert_verl_lora_to_hf.py --ckpt "$CKPT" --base /home/ubuntu/models/Qwen3-8B \
  --lora-meta "$LORA_META" --out "$OUT" >> "$LOG" 2>&1
CONV_EXIT=$?
echo "CONVERT_EXIT=$CONV_EXIT $(date)" >> "$LOG"
[ "$CONV_EXIT" -ne 0 ] && exit 1

# 5. vLLM server + 评估（等训练释放 GPU）
echo "==== vllm serve + eval $(date) ====" >> "$LOG"
sleep 60
pkill -f "Qwen3-8B-final-dual-async" 2>/dev/null || true
sleep 2
nohup "$ENV_PY" -m vllm.entrypoints.openai.api_server \
  --model "$OUT" --port 8001 --served-model-name Qwen3-8B-final-dual-async \
  --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.8 \
  --max-num-seqs 128 --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > vllm_dual_final.log 2>&1 &
VLLM_PID=$!
echo "VLLM_PID=$VLLM_PID"
for i in $(seq 1 60); do
  if curl -s -o /dev/null http://127.0.0.1:8001/v1/models; then
    echo "VLLM_READY after ${i}x5s $(date)" >> "$LOG"
    break
  fi
  sleep 5
done

set -a
source /home/ubuntu/swe-rl/tencent_sandbox.env
set +a
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}"
export TENCENT_SANDBOX_SKIP_TMUX=1
"$ENV_PY" eval_humanevalfix.py \
  --data /home/ubuntu/swe-rl/data/humanevalfix_train161.jsonl \
  --base-url http://127.0.0.1:8001/v1 \
  --model Qwen3-8B-final-dual-async \
  --max-turns 40 --temperature 0.8 --concurrency 24 \
  --out "$EVAL_OUT" --out-dir "$EVAL_DIR" > eval_dual_async_mooncake.log 2>&1
echo "EVAL_EXIT=$? $(date)" >> "$LOG"
tail -5 eval_dual_async_mooncake.log >> "$LOG"
