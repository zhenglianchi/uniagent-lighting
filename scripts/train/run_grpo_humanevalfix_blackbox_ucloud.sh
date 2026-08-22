#!/usr/bin/env bash
# 单机黑盒（Claude Code）GRPO 正式训练：**HumanEvalFix 全样本 161 条**
# 2026-08-12 v0.39.0：基于 run_grpo_humanevalfix_ucloud.sh（白盒）+ 黑盒 runner
# （claude_code_runner，v0.35.1+）。并发/吞吐配置与 baseline / 投机 run 一致
# （并发 64 / max_num_seqs 128 / util 0.8），保证速度对比同条件可比（用户定）。
#
# 黑盒差异：
#   - runner = uni_agent_ext.agents.claude_code_runner.claude_code_runner
#   - 沙箱内 npm 装 claude-code 2.1.153（pin < 2.1.154）+ --bare 精简系统提示
#   - GATEWAY_PORT=8001 固定 Gateway 端口（补丁 verl_gateway_fixed_port +
#     gateway_fixed_port）；默认隧道模式（CLAUDE_GATEWAY_TUNNEL=1，走公网 22）
#     ——direct-URL（CLAUDE_GATEWAY_TUNNEL=0 + CLAUDE_GATEWAY_PUBLIC_HOST）需
#     安全组放行 8001，放行后可直接切换
#   - max_turns=60（与白盒 MSA_AGENT_MAX_TURNS=60 一致，用户定）
#   - max_prompt_length/max_response_length=8192、max_model_len=16384
#     （Claude Code 系统提示较大，小样本实测 prompt 1625 / 长轨迹 11K）
#
# 用法（训练机 node1 上执行，后台跑）：
#   setsid nohup bash /home/ubuntu/swe-rl/run_grpo_humanevalfix_blackbox_ucloud.sh \
#     > /home/ubuntu/swe-rl/grpo_humanevalfix_blackbox.log 2>&1 < /dev/null &
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
# 腾讯云沙箱凭据（E2B_API_KEY / E2B_DOMAIN / TENCENT_*）
set -a
source /home/ubuntu/swe-rl/tencent_sandbox.env
set +a
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}"
# 黑盒：GATEWAY 固定端口 + 隧道（走公网 22，无需放行 8001）
export GATEWAY_PORT=${GATEWAY_PORT:-8001}
export CLAUDE_GATEWAY_TUNNEL=${CLAUDE_GATEWAY_TUNNEL:-1}
export MSA_GATEWAY_SSH_HOST=${MSA_GATEWAY_SSH_HOST:-117.50.199.93}
# 跳过沙箱内 tmux 安装（黑盒 claude-code 不需要；每会话白耗 180s 超时）
export TENCENT_SANDBOX_SKIP_TMUX=1

ENV=/home/ubuntu/miniforge3/envs/swe-rl
MODEL=${MODEL:-/home/ubuntu/models/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_train161.jsonl}
VAL_FILE=${VAL_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_val.jsonl}
TOOL_PARSER=${TOOL_PARSER:-hermes}         # gateway tool-call parser；Qwen3-8B 用 hermes
GATEWAY_COUNT=${GATEWAY_COUNT:-1}
CONCURRENCY=${CONCURRENCY:-64}             # 与 baseline/投机 run 一致（用户定，配额只作余量）
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "$MODEL")"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-16}
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-4}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-1}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}  # 与 CONCURRENCY 匹配（baseline 口径）
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.8}
CKPT_DIR=${CKPT_DIR:-/home/ubuntu/swe-rl/checkpoints/humanevalfix_blackbox}
LOG_DIR=${LOG_DIR:-/home/ubuntu/swe-rl/logs/humanevalfix_blackbox}
cd /home/ubuntu/uni-agent/verl

ls -la "$TRAIN_FILE" "$VAL_FILE" "$MODEL" >/dev/null

EXTRA_ARGS=()
if [ -n "$MAX_CKPT_KEEP" ]; then
  EXTRA_ARGS+=(trainer.max_actor_ckpt_to_keep="$MAX_CKPT_KEEP")
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
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.max_model_len=16384 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=4 \
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
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.runner_fqn=uni_agent_ext.agents.claude_code_runner.claude_code_runner \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.max_concurrent_sessions=${CONCURRENCY} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.runner_kwargs.max_turns=60 \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.runner_kwargs.run_timeout=7200 \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
  reward.reward_manager.name=naive \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=swe-rl-blackbox \
  trainer.experiment_name=qwen3-8b-grpo-humanevalfix-blackbox \
  trainer.save_freq=1 \
  trainer.resume_mode=auto \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  "${EXTRA_ARGS[@]}" \
