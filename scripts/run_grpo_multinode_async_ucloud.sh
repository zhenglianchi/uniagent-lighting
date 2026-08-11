#!/usr/bin/env bash
# 双机全异步 GRPO 冒烟（UCloud node1 + node2，2026-08-11 调研定稿）
#
# 调研结论（见 TODO §6.4）：
#   - verl v1 数据流已全程走 TQ（agent framework 写轨迹 + trainer kv_put/get）；
#     全异步 = trainer.v1.trainer_mode=colocate_async（Trainer 与 rollout 同机重叠，
#     partial rollout 已启用），uni-agent 官方 recipe 即此模式
#   - 换 mooncake = transfer_queue.backend.storage_backend=MooncakeStore
#     （TransferQueue 0.1.8 已内置；无 RDMA 时 protocol=tcp，双机小数据量收益有限，
#     作为对照实验验证）
#
# 用法（node1 上执行，Ray 集群已起）：
#   # 双机 colocate_async + TQ SimpleStorage（默认）
#   bash /home/ubuntu/swe-rl/run_grpo_multinode_async_ucloud.sh 2>&1 | tee grpo_multinode_async.log
#   # 换 mooncake 后端（对照实验）
#   MOONCAKE=1 bash /home/ubuntu/swe-rl/run_grpo_multinode_async_ucloud.sh 2>&1 | tee grpo_multinode_async_mooncake.log
#
# 关键环境变量：
#   RAY_ADDRESS             Ray head 地址（默认 10.60.173.163:6379）
#   TRAINER_MODE            colocate_async（默认）/ separate_async（实验性）
#   NUM_WARMUP_BATCHES      预热批次（默认 1，colocate_async 预填充 rollout 流水线）
#   MOONCAKE                0（默认 SimpleStorage）/ 1（MooncakeStore）
#   MOONCAKE_PROTOCOL       tcp（默认）/ rdma（需网卡支持）
#   MOONCAKE_MASTER         mooncake master 地址（默认 node1:50124）
#   MOONCAKE_METADATA       metadata 地址（默认 node1:50123）
#   VLLM_GPU_MEM_UTIL       vLLM 显存利用率（colocate 重叠时建议 0.4~0.5）
#
# 注意：
#   - separate_async 要求 checkpoint_engine.backend != naive（nccl/nixl/mooncake），
#     且 train_batch_size == parameter_sync_step * ppo_mini_batch_size；未实测，实验性
#   - colocate_async 下训练与生成同时进行，显存账：FSDP 基座 14G + vLLM(0.5×48G=24G)
#     ≈ 38G < 48G；若 OOM 调低 VLLM_GPU_MEM_UTIL
#   - mooncake 双机需各节点能互连 master 端口；local_hostname 自动取 Ray node IP
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0   # 关 Ray OOM 杀手（防御性保留）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

ENV=/home/ubuntu/miniforge3/envs/swe-rl
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/models/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-/home/ubuntu/swe-rl/data/smoke_train.jsonl}
VAL_FILE=${VAL_FILE:-/home/ubuntu/swe-rl/data/smoke_val.jsonl}
REWARD_PATH=${REWARD_PATH:-/home/ubuntu/swe-rl/reward_smoke.py}
PYTHON=${PYTHON:-$ENV/bin/python}

RAY_ADDRESS=${RAY_ADDRESS:-10.60.173.163:6379}
TRAINER_MODE=${TRAINER_MODE:-colocate_async}
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}
MOONCAKE=${MOONCAKE:-0}
MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
MOONCAKE_MASTER=${MOONCAKE_MASTER:-10.60.173.163:50124}
MOONCAKE_METADATA=${MOONCAKE_METADATA:-10.60.173.163:50123}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.5}

# 多机通信：指定内网网卡
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}

ls -la "$TRAIN_FILE" "$VAL_FILE" "$REWARD_PATH" "$MODEL_PATH" >/dev/null

EXTRA_OPTS=(
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.offload_policy=False
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.use_fused_kernels=False
  actor_rollout_ref.rollout.data_parallel_size=2
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  +ray_kwargs.ray_init.address="$RAY_ADDRESS"
  # ---- 全异步（v1 colocate_async：Trainer 与 rollout 同机重叠 + partial rollout）----
  trainer.use_v1=True
  trainer.v1.trainer_mode="$TRAINER_MODE"
  trainer.v1.colocate_async.num_warmup_batches="$NUM_WARMUP_BATCHES"
  transfer_queue.enable=True
)

if [ "$TRAINER_MODE" = "separate_async" ]; then
  # separate_async：Trainer/Rollout 分机，需非 naive checkpoint engine 做权重同步
  EXTRA_OPTS+=(
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    trainer.v1.separate_async.num_warmup_batches="$NUM_WARMUP_BATCHES"
    trainer.v1.separate_async.parameter_sync_step=1
  )
fi

if [ "$MOONCAKE" = "1" ]; then
  EXTRA_OPTS+=(
    transfer_queue.backend.storage_backend=MooncakeStore
    transfer_queue.backend.MooncakeStore.auto_init=True
    transfer_queue.backend.MooncakeStore.metadata_server="$MOONCAKE_METADATA"
    transfer_queue.backend.MooncakeStore.master_server_address="$MOONCAKE_MASTER"
    transfer_queue.backend.MooncakeStore.protocol="$MOONCAKE_PROTOCOL"
    transfer_queue.backend.MooncakeStore.global_segment_size=8589934592
    transfer_queue.backend.MooncakeStore.local_buffer_size=2147483648
  )
  echo "== MooncakeStore enabled: master=$MOONCAKE_MASTER protocol=$MOONCAKE_PROTOCOL =="
else
  EXTRA_OPTS+=(
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2
    transfer_queue.backend.SimpleStorage.total_storage_size=1000
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
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=smoke_reward \
  reward.num_workers=1 \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.nnodes=2 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name=swe-rl-smoke \
  trainer.experiment_name=qwen3-8b-grpo-multinode-async-ucloud \
  "${EXTRA_OPTS[@]}" \
  "$@"
