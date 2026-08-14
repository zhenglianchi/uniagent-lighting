#!/usr/bin/env bash
# 多机 GRPO 冒烟训练（UCloud 双机 4090，2026-08-05 定稿：LoRA + AdamW(fp32) + offload 关闭）
# 硬件：node1(10.60.173.163) + node2(10.60.46.121)，各 1×48GB 4090 / 94GB RAM
# 前置：Ray 集群已起（node1 head + node2 join，ray status 显示 2 节点 4 GPU）
# 用法（node1 上执行）：
#   bash /home/ubuntu/swe-rl/run_grpo_multinode_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_multinode.log
# 配置要点：
#   - LoRA 微调（lora_rank=32，PEFT，非全参；verl 用 model.lora_rank>0 启用）
#   - 优化器默认 AdamW（fp32，torch.optim）——LoRA 可训练参数仅 ~0.05B，fp32 状态 ~560MB
#     分片后每卡 ~140MB，完全无压力；不需要 8bit（那是全参内存紧张时的方案）
#   - FSDP2 不开 CPU offload（LoRA + 4 卡分片：基座权重 3.5G/卡 + vLLM 12G ≈ 15.5G/卡，24G 放得下；
#     2026-08-05 实测 offload 全开会把 14G 权重来回搬 CPU + 首次同步物化，60G 内存峰值 + 全核占满 → 机器"假死"）
#   - 梯度检查点 + fused kernels（model.use_fused_kernels=True，verl monkey patch，默认 torch backend）
#   - 注意：首次基座权重同步仍有 ~60G 内存峰值 + CPU 搬运（一次性），
#     64GB 内存机器建议保留 swap（20G）兜底，必要时用控制台 Web shell 操作
#   - 换全参微调（后续 A/B）：把 model.lora_rank 改成 0（verl 0=关闭 LoRA），lr 1e-5→1e-6，
#     **全参时 fp32 AdamW 状态 84GB 放不下，需改回 AdamW8bit**（bitsandbytes，两台都要装）
#     + offload 全开，4 卡分片后每卡峰值 ~13G 物理可行
set -xeuo pipefail

export RAY_ADDRESS=10.60.188.85:6379
export MODEL_PATH=/home/ubuntu/models/Qwen3-8B
export TRAIN_FILE=/home/ubuntu/swe-rl/data/smoke_train.jsonl
export VAL_FILE=/home/ubuntu/swe-rl/data/smoke_val.jsonl
export REWARD_PATH=/home/ubuntu/swe-rl/reward_smoke.py
export PYTHON=/home/ubuntu/miniforge3/envs/swe-rl/bin/python
# 投机解码（2026-08-14：对照实验统一开启）
LORA_MERGE=${LORA_MERGE:-1}
SPEC_ON=${SPEC_ON:-1}
SPEC_DRAFT=${SPEC_DRAFT:-/home/ubuntu/models/Qwen3-8B-speculator.eagle3}
SPEC_TOKENS=${SPEC_TOKENS:-3}

# 多机通信：指定内网网卡
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_memory_monitor_refresh_ms=0   # 关 Ray OOM 杀手（防御性保留）

ls -la "$TRAIN_FILE" "$VAL_FILE" "$REWARD_PATH" "$MODEL_PATH" >/dev/null

EXTRA_OPTS=(
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.offload_policy=False
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16   # 4090 支持 bf16
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  # LoRA 微调（lora_rank>0 启用；alpha 默认 32；target_modules 默认 all-linear）
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.use_fused_kernels=False
  # vllm 0.11.1 多节点：dp=2/tp=1 = 每节点一个单卡引擎（新硬件每节点 1×48G）
  actor_rollout_ref.rollout.data_parallel_size=2
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  +ray_kwargs.ray_init.address="$RAY_ADDRESS"
)

EXTRA_ARGS=()
if [ "$LORA_MERGE" = "1" ]; then
  EXTRA_ARGS+=(actor_rollout_ref.model.lora.merge=True)
fi
if [ "$SPEC_ON" = "1" ]; then
  EXTRA_ARGS+=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config='{\"method\": \"eagle3\", \"model\": \"$SPEC_DRAFT\", \"num_speculative_tokens\": $SPEC_TOKENS, \"draft_tensor_parallel_size\": 1}'"
  )
fi

"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['$TRAIN_FILE']" \
  data.val_files="['$VAL_FILE']" \
  data.train_batch_size=2 \
  data.val_batch_size=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=smoke_reward \
  reward.num_workers=1 \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
  transfer_queue.backend.SimpleStorage.total_storage_size=1000 \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.nnodes=2 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name=swe-rl-smoke \
  trainer.experiment_name=qwen25-7b-grpo-multinode-lora-ucloud \
  "${EXTRA_OPTS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
