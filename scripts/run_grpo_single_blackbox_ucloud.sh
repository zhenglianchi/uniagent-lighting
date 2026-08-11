#!/usr/bin/env bash
# 单机黑盒（Claude Code）GRPO 小样本测试（uni-agent agent framework + claude_code_runner）
# 2026-08-11 v0.37.0：基于 run_grpo_single_agentic_ucloud.sh 改造——
#   runner 换成 uni_agent_ext.agents.claude_code_runner.claude_code_runner（腾讯 E2B
#   direct-URL 版）：沙箱内 npm 装 claude-code 2.1.153 + ANTHROPIC_BASE_URL 直连公网
#   Gateway；reward 复用白盒 mini_swe_agent_runner.evaluate_reward（swe_bench /
#   humaneval_fix 双口径，与白盒完全同分）。
#
# 前置：
#   1) 数据：work/data/humanevalfix_train3.jsonl（tools_kwargs.env.files 注入
#      solution.py + reward.metadata 含 problem_statement/FAIL_TO_PASS/hidden_files）
#   2) uni_agent_ext 已放训练机 PYTHONPATH；腾讯沙箱凭据 tencent_sandbox.env
#   3) 训练机 Gateway 公网端口已放行（黑盒沙箱内 Claude Code 直连，docs/vllm_access.md）
#   4) 沙箱内 npm 可访问 npmmirror（安装 claude-code@2.1.153）
#
# 用法（训练机 node1 上执行）：
#   bash /home/ubuntu/swe-rl/run_grpo_single_blackbox_ucloud.sh 2>&1 | tee grpo_blackbox.log
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

ENV=/home/ubuntu/miniforge3/envs/swe-rl
MODEL=/home/ubuntu/models/Qwen3-8B
TRAIN_FILE=${TRAIN_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_train3.jsonl}
VAL_FILE=${VAL_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_val.jsonl}
TOOL_PARSER=${TOOL_PARSER:-hermes}         # gateway tool-call parser；Qwen3-8B 用 hermes
GATEWAY_COUNT=${GATEWAY_COUNT:-1}          # 单机冒烟 1 个 gateway actor
CONCURRENCY=${CONCURRENCY:-4}              # 并发 rollout sessions（= 同时跑的沙箱数，控制成本）
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "$MODEL")"}
cd /home/ubuntu/uni-agent/verl

ls -la "$TRAIN_FILE" "$VAL_FILE" "$MODEL" >/dev/null

"$ENV/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  ++data.apply_chat_template_kwargs.enable_thinking=false \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=/home/ubuntu/swe-rl/logs/blackbox \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.runner_fqn=uni_agent_ext.agents.claude_code_runner.claude_code_runner \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.max_concurrent_sessions=${CONCURRENCY} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.runner_kwargs.max_turns=100 \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.claude_code.runner_kwargs.run_timeout=7200 \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
  reward.reward_manager.name=naive \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=swe-rl-blackbox \
  trainer.experiment_name=qwen3-8b-grpo-blackbox-claude \
  trainer.save_freq=1 \
  trainer.resume_mode=auto \
  trainer.default_local_dir=/home/ubuntu/swe-rl/checkpoints/blackbox \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  "$@"
