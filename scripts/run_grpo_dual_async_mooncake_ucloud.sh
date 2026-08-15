#!/usr/bin/env bash
# 双机分离式全异步 + Mooncake + EAGLE-3 白盒（mini-swe-agent）GRPO 正式训练
# 2026-08-14：基于 run_grpo_humanevalfix_blackbox_ucloud.sh（单机正式，
# 并发 64 / max_num_seqs 128 / util 0.8 口径不变），架构切换为：
#   - trainer.v1.trainer_mode=separate_async：trainer 占 node1 1 卡、
#     独立 rollout 引擎占 node2 1 卡（dp=1/tp=1，生成与训练重叠）
#   - transfer_queue backend=MooncakeStore（master 预启动 + auto_init=False；
#     无 RDMA 走 tcp，双机小数据量收益有限，作为架构验证）
#   - 投机解码 EAGLE-3 开启（SPEC_ON=1）：独立引擎 dp=1 单节点，
#     避开 vLLM 0.11.1 EAGLE+dp>1 死锁，恢复单机 spec 的吞吐收益（+41.7%）
#   - 权重同步 checkpoint_engine.backend=nccl，
#     parameter_sync_step=2（train_batch=32 == 2*mini=16）
#
# 小样本验证（train3 / batch3 / mini3 / micro1 / pss1 / n=2 / conc6）：
#   TRAIN_FILE=/home/ubuntu/swe-rl/data/humanevalfix_train3.jsonl \
#   TRAIN_BATCH_SIZE=3 PPO_MINI_BATCH=3 PPO_MICRO_BATCH=1 PARAM_SYNC_STEP=1 \
#   TOTAL_EPOCHS=1 ROLLOUT_N=2 CONCURRENCY=6 VLLM_MAX_NUM_SEQS=16 VLLM_GPU_MEM_UTIL=0.5 \
#   CKPT_DIR=/home/ubuntu/swe-rl/checkpoints/humanevalfix_dual_smoke \
#   LOG_DIR=/home/ubuntu/swe-rl/logs/humanevalfix_dual_smoke \
#   bash run_grpo_dual_async_mooncake_ucloud.sh
#
# 正式训练（node1 上执行，后台跑；训练完成后自动接评估）：
#   setsid nohup bash /home/ubuntu/swe-rl/run_grpo_dual_async_mooncake_ucloud.sh \
#     > /home/ubuntu/swe-rl/grpo_humanevalfix_dual_async_mooncake.log 2>&1 < /dev/null &
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MC_STORE_MEMCPY=0   # mooncake 禁用 GPU memcpy，避免与 vLLM 同卡冲突
# 腾讯云沙箱凭据（E2B_API_KEY / E2B_DOMAIN / TENCENT_*）
set -a
source /home/ubuntu/swe-rl/tencent_sandbox.env
set +a
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}"
# 白盒（mini-swe-agent）：harness 在训练机 node1，沙箱为腾讯云执行环境
export GATEWAY_PORT=${GATEWAY_PORT:-8001}
export MSA_GATEWAY_TUNNEL=${MSA_GATEWAY_TUNNEL:-0}
export MSA_INSTALL_AGENT=${MSA_INSTALL_AGENT:-1}
export MSA_REWARD_INCLUDE_P2P=${MSA_REWARD_INCLUDE_P2P:-1}
export MSA_REWARD_P2P_SAMPLE=${MSA_REWARD_P2P_SAMPLE:-20}
export TENCENT_SANDBOX_SKIP_TMUX=1

ENV=/home/ubuntu/miniforge3/envs/swe-rl
# MooncakeStore auto_init 需要 mooncake_master 二进制，确保 Ray worker 的 PATH 带 conda bin
export PATH="$ENV/bin:$PATH"
MODEL=${MODEL:-/home/ubuntu/models/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_train161.jsonl}
VAL_FILE=${VAL_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_val.jsonl}
TOOL_PARSER=${TOOL_PARSER:-hermes}
GATEWAY_COUNT=${GATEWAY_COUNT:-1}
CONCURRENCY=${CONCURRENCY:-64}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "$MODEL")"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-16}
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-4}
PARAM_SYNC_STEP=${PARAM_SYNC_STEP:-2}      # separate_async: batch == pss*mini
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-1}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.8}
CKPT_DIR=${CKPT_DIR:-/home/ubuntu/swe-rl/checkpoints/humanevalfix_dual_async}
LOG_DIR=${LOG_DIR:-/home/ubuntu/swe-rl/logs/humanevalfix_dual_async}
RAY_ADDRESS=${RAY_ADDRESS:-10.60.216.3:6379}
MOONCAKE_MASTER=${MOONCAKE_MASTER:-10.60.216.3:50124}
MOONCAKE_METADATA=${MOONCAKE_METADATA:-10.60.216.3:50123}
MOONCAKE=${MOONCAKE:-0}   # 0=SimpleStorage（正式训练默认，稳定）；1=MooncakeStore（实验）
SPEC_ON=${SPEC_ON:-1}
SPEC_DRAFT=${SPEC_DRAFT:-/home/ubuntu/models/Qwen3-8B-speculator.eagle3}
SPEC_TOKENS=${SPEC_TOKENS:-3}

cd /home/ubuntu/uni-agent/verl

ls -la "$TRAIN_FILE" "$VAL_FILE" "$MODEL" >/dev/null

EXTRA_OPTS=(
  # ---- 双机分离式全异步（v1 separate_async：trainer 1 卡 + 独立 rollout 1 卡）----
  trainer.use_v1=True
  trainer.v1.trainer_mode=separate_async
  trainer.v1.separate_async.num_warmup_batches=1
  trainer.v1.separate_async.parameter_sync_step="$PARAM_SYNC_STEP"
  transfer_queue.enable=True
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
  # trainer 占 1 卡（node1）；rollout 独立 1 卡（node2）
  trainer.nnodes=1
  trainer.n_gpus_per_node=1
  actor_rollout_ref.rollout.nnodes=1
  actor_rollout_ref.rollout.n_gpus_per_node=1
  actor_rollout_ref.rollout.data_parallel_size=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  +ray_kwargs.ray_init.address="$RAY_ADDRESS"
)
if [ "$MOONCAKE" = "1" ]; then
  EXTRA_OPTS+=(
    transfer_queue.backend.storage_backend=MooncakeStore
    transfer_queue.backend.MooncakeStore.auto_init=False
    transfer_queue.backend.MooncakeStore.metadata_server="$MOONCAKE_METADATA"
    transfer_queue.backend.MooncakeStore.master_server_address="$MOONCAKE_MASTER"
    transfer_queue.backend.MooncakeStore.protocol=tcp
    transfer_queue.backend.MooncakeStore.local_hostname=""
    transfer_queue.backend.MooncakeStore.global_segment_size=8589934592
    transfer_queue.backend.MooncakeStore.local_buffer_size=2147483648
  )
else
  EXTRA_OPTS+=(
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2
    transfer_queue.backend.SimpleStorage.total_storage_size=1000
  )
fi

EXTRA_ARGS=()
EXTRA_ARGS+=(
  # 只保留最近 1 个完整 checkpoint（旧的大文件滚动删除，防磁盘累积爆盘）
  trainer.max_actor_ckpt_to_keep="$MAX_CKPT_KEEP"
)
if [ "$SPEC_ON" = "1" ]; then
  EXTRA_ARGS+=(
    actor_rollout_ref.model.lora.merge=True
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config='{\"method\": \"eagle3\", \"model\": \"$SPEC_DRAFT\", \"num_speculative_tokens\": $SPEC_TOKENS, \"draft_tensor_parallel_size\": 1}'"
  )
fi

"$ENV/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size=1 \
  ++data.apply_chat_template_kwargs.enable_thinking=false \
  data.max_prompt_length=8192 \
  data.max_response_length=8192 \
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
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.max_model_len=16384 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_num_seqs="$VLLM_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="$LOG_DIR" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.runner_fqn=uni_agent_ext.agents.mini_swe_agent_runner.mini_swe_agent_runner \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.max_concurrent_sessions=${CONCURRENCY} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.runner_kwargs.max_turns=60 \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.runner_kwargs.run_timeout=7200 \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
  reward.reward_manager.name=naive \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=swe-rl-blackbox-dual \
  trainer.experiment_name=qwen3-8b-grpo-humanevalfix-dual-async \
  trainer.save_freq=1 \
  trainer.resume_mode=auto \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  "${EXTRA_OPTS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
