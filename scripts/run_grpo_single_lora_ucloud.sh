#!/usr/bin/env bash
# 单机单卡 LoRA GRPO 冒烟（48G 4090 / 94GB RAM，2026-08-05）
# 配置 = 多机定稿的 LoRA 版（rank=32 / AdamW fp32 / offload 关 / 梯度检查点 / fused kernels），
# 仅改 nnodes=1 / n_gpus_per_node=1 / tp=1，且不预起 Ray（verl 内部自建本地 Ray）。
# ⚠️ use_fused_kernels 必须关（2026-08-05 实测）：fused monkey patch 与 LoRA(PEFT) 冲突，
#    training 步 aten.mm 报 mixed torch.Tensor and DTensor；全参 A/B 时可再开。
# 显存账：FSDP 基座 14G（单卡不分片）+ vLLM(0.5×48G=24G) 共存 ≈ 38G < 48G；
# 内存账：首次基座同步峰值 ~60G < 94G，无需 swap。
# 用法（node2 上执行）：
#   bash /home/ubuntu/swe-rl/run_grpo_single_lora_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_single_lora.log
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

ENV=/home/ubuntu/miniforge3/envs/swe-rl
MODEL=/home/ubuntu/models/Qwen2.5-Coder-7B-Instruct
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
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.offload_policy=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
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
  trainer.experiment_name=qwen25-7b-grpo-single-lora-ucloud \
  trainer.save_freq=1 \
  trainer.resume_mode=auto \
  trainer.default_local_dir=/home/ubuntu/swe-rl/checkpoints/single_lora_smoke \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  "$@"
