#!/usr/bin/env bash
# verl GRPO 单机双卡冒烟（UCloud node1：2×RTX 4090 / 64GB，2026-08-05 准备）
# 版本链：torch 2.9.0+cu128 / vllm 0.11.1 / verl 0.9.0.dev（node2 克隆镜像前先在 node1 上跑通这版）
# 数据：work/data/smoke_train.jsonl(2条) + smoke_val.jsonl(1条)；奖励：非空+1（假奖励，验证链路用）
#
# 与单卡冒烟（run_grpo_smoke_ucloud.sh）的差异：
#   - n_gpus_per_node=2 + rollout tp=2（单机双卡一个 vLLM 引擎，7B bf16 14.3GB 占 2×24GB 无压力）
#   - FSDP2 不开 CPU offload（offload=True 时 CPU↔GPU 搬运 + 每步 14GB 权重同步会把 CPU 打满，
#     导致 sshd 无响应、机器"假死"；offload=False 参数常驻 GPU，但 vLLM 显存必须压到 0.4：
#     FSDP 参数 7G/卡 + vLLM(0.4×24=9.6G) = 16.6G/卡，避开 0.5 时的 CUDA OOM），model_dtype=bf16
#   - GRPO n=2、train_batch_size=2（冒烟 2 条 prompt；batch>2 会算出 0 步 → ZeroDivisionError）
# 若想 A/B dp=2/tp=1（每卡一个引擎）：把 rollout.tensor_model_parallel_size 改 1、
#   +actor_rollout_ref.rollout.data_parallel_size=2
#
# 用法（node1 上执行）：
#   bash /home/ubuntu/swe-rl/run_grpo_dualgpu_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_dualgpu.log
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

ENV=/home/ubuntu/miniforge3/envs/swe-rl
MODEL=/home/ubuntu/models/Qwen3-8B
TRAIN_FILE=/home/ubuntu/swe-rl/data/smoke_train.jsonl
VAL_FILE=/home/ubuntu/swe-rl/data/smoke_val.jsonl
REWARD_PATH=/home/ubuntu/swe-rl/reward_smoke.py
cd /home/ubuntu/uni-agent/verl

ls -la "$TRAIN_FILE" "$VAL_FILE" "$REWARD_PATH" "$MODEL" >/dev/null

"$ENV/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=2 \
  data.val_batch_size=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.offload_policy=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.optimizer=SGD \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  reward.custom_reward_function.path=$REWARD_PATH \
  reward.custom_reward_function.name=smoke_reward \
  reward.num_workers=1 \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
  transfer_queue.backend.SimpleStorage.total_storage_size=1000 \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=swe-rl-smoke \
  trainer.experiment_name=qwen25-7b-grpo-dualgpu-ucloud \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  "$@"
