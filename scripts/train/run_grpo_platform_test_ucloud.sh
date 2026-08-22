#!/usr/bin/env bash
# 平台化（§D P0）单步验证：外部 agent（本地 WSL）→ 云端 Gateway → 轨迹 → GRPO 1 step
# 2026-08-12：加载已有训练权重（baseline final / spec final），**save_freq=-1 不保存
# 新权重**（绝不覆盖 models/Qwen3-8B-final* 与 checkpoints/humanevalfix*，红线）。
#
# 形态：runner = external_agent_runner（建沙箱 + 暴露任务 + 等本地 agent done 标记 +
# 云侧 reward）；本地侧 scripts/platform/platform_local_agent.py（paramiko 隧道 + 跑
# mini-swe-agent + touch done）。
#
# 用法（训练机 node1，后台）：
#   # baseline final 权重
#   MODEL=/home/ubuntu/models/Qwen3-8B-final \
#     setsid nohup bash /home/ubuntu/swe-rl/run_grpo_platform_test_ucloud.sh \
#     > /home/ubuntu/swe-rl/grpo_platform_test_baseline.log 2>&1 < /dev/null &
#   # spec final 权重（MODEL 换 final-spec + 独立目录）
#
# 本地（WSL，训练起后等 task.json 出现）：
#   python scripts/platform/platform_local_agent.py --wait --timeout 1800
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
set -a
source /home/ubuntu/swe-rl/tencent_sandbox.env
set +a
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}"
export GATEWAY_PORT=${GATEWAY_PORT:-8001}
export TENCENT_SANDBOX_SKIP_TMUX=1
export PLATFORM_TEST_DIR=${PLATFORM_TEST_DIR:-/home/ubuntu/swe-rl/platform_test}

ENV=/home/ubuntu/miniforge3/envs/swe-rl
MODEL=${MODEL:-/home/ubuntu/models/Qwen3-8B-final}   # 测试加载的已有权重（final / final-spec）
TRAIN_FILE=${TRAIN_FILE:-/home/ubuntu/swe-rl/data/platform_test_train.jsonl}
VAL_FILE=${VAL_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_val.jsonl}
TOOL_PARSER=${TOOL_PARSER:-hermes}
GATEWAY_COUNT=${GATEWAY_COUNT:-1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "$MODEL")"}
RUN_NAME=${RUN_NAME:-"$(basename "$MODEL")"}          # 独立日志/checkpoint 目录后缀
CKPT_DIR=${CKPT_DIR:-/home/ubuntu/swe-rl/checkpoints/platform_test_$RUN_NAME}
LOG_DIR=${LOG_DIR:-/home/ubuntu/swe-rl/logs/platform_test_$RUN_NAME}
cd /home/ubuntu/uni-agent/verl

ls -la "$TRAIN_FILE" "$VAL_FILE" "$MODEL" >/dev/null
echo "== platform test: MODEL=$MODEL RUN_NAME=$RUN_NAME (save_freq=-1, no new ckpt) =="

"$ENV/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=1 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=16384 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="$LOG_DIR" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.external_agent.runner_fqn=uni_agent_ext.agents.external_agent_runner.external_agent_runner \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.external_agent.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.external_agent.max_concurrent_sessions=1 \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.external_agent.runner_kwargs.run_timeout=3600 \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
  reward.reward_manager.name=naive \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=swe-rl-platform \
  trainer.experiment_name=qwen3-8b-platform-test \
  trainer.save_freq=-1 \
  trainer.resume_mode=auto \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
