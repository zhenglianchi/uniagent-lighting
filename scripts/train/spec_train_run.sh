#!/usr/bin/env bash
# 投机解码 + gateway 修复全样本训练（2026-08-09，117.50.189.37）
# 与 2026-08-08~09 全样本 run 配置一致（train161 / batch32 / mini16 / micro4 /
# 并发64 / vllm 128 / util 0.8 / 5 epoch / ckpt keep 1），仅新增：
#   LORA_MERGE=1（LoRA×SD 互斥，merge 全量权重同步）+ SPEC_ON=1（EAGLE-3）
# 日志/checkpoint 独立目录，不碰 final 与旧 run。
set -u
cd /home/ubuntu/swe-rl

# 备份 gateway 验证的旧会话目录（防统计污染，可随时恢复对比）
if [ -d logs/humanevalfix ] && [ ! -d logs/humanevalfix_pre_spec ]; then
  mv logs/humanevalfix logs/humanevalfix_pre_spec
fi

export TRAIN_FILE=/home/ubuntu/swe-rl/data/humanevalfix_train161.jsonl
export VAL_FILE=/home/ubuntu/swe-rl/data/humanevalfix_val.jsonl
export TOTAL_EPOCHS=5
export TRAIN_BATCH_SIZE=32
export PPO_MINI_BATCH=16
export PPO_MICRO_BATCH=4
export MAX_CKPT_KEEP=1
export CONCURRENCY=64
export VLLM_MAX_NUM_SEQS=128
export VLLM_GPU_MEM_UTIL=0.8
export CKPT_DIR=/home/ubuntu/swe-rl/checkpoints/humanevalfix_spec
export LOG_DIR=/home/ubuntu/swe-rl/logs/humanevalfix_spec
export LORA_MERGE=1
export SPEC_ON=1

nohup bash run_grpo_humanevalfix_ucloud.sh > grpo_humanevalfix_spec.log 2>&1 &
echo "PID=$!"
sleep 8
tail -8 grpo_humanevalfix_spec.log
