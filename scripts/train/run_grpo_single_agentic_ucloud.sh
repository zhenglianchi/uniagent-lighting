#!/usr/bin/env bash
# 单机 agentic GRPO 训练（uni-agent agent framework + 自定义 mini-swe-agent runner）
# 2026-08-06 v0.20.0：基座切换 Qwen3-8B（支持 function calling）
#   - 关 thinking：data.apply_chat_template_kwargs.enable_thinking=false
#     （uni-agent gateway codec 渲染 chat template 时生效，entry.py 读取）
#   - TOOL_PARSER=hermes：Qwen3 官方推荐工具格式（vllm ToolParserManager）
# 2026-08-05 v0.3.0：对齐官方 examples/quickstart/training/train_qwen3p5_dense.sh 的接线
#   （multi_turn + AgentFrameworkRolloutAdapter + agent_framework.agent_runners），
#   runner 换成我们自己的 uni_agent_ext.agents.mini_swe_agent_runner。
#
# 前置：
#   1) 数据：scripts/data/make_agentic_data.py 生成的 agentic_train/val.jsonl（含 tools_kwargs）
#   2) uni_agent_ext 包已放到训练机 PYTHONPATH（部署步骤）
#   3) 沙箱/本地 agent 访问 Gateway：SSH 隧道或公网放行（docs/vllm_access.md）
#   4) 腾讯沙箱凭据环境变量（work/tencent_sandbox.env）
#
# 用法（训练机 node2 上执行）：
#   bash /home/ubuntu/swe-rl/run_grpo_single_agentic_ucloud.sh 2>&1 | tee grpo_agentic.log
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
# E2B 兼容端点映射（凭据文件里是 TENCENT_SANDBOX_E2B_TOKEN）
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}"
# harness 在训练机本地，直接调本机 Gateway（session.base_url），不需要沙箱内隧道
export MSA_GATEWAY_TUNNEL=0
export MSA_INSTALL_AGENT=1   # 沙箱内现场 pip install mini-swe-agent（预装镜像后可关）
# reward：F2P 全量 + P2P 固定抽样 20 条（连续梯度，防回归破坏）
export MSA_REWARD_INCLUDE_P2P=1
export MSA_REWARD_P2P_SAMPLE=20

ENV=/home/ubuntu/miniforge3/envs/swe-rl
MODEL=/home/ubuntu/models/Qwen3-8B
TRAIN_FILE=/home/ubuntu/swe-rl/data/agentic_train.jsonl
VAL_FILE=/home/ubuntu/swe-rl/data/agentic_val.jsonl
TOOL_PARSER=${TOOL_PARSER:-hermes}         # gateway tool-call parser；Qwen3-8B 用 hermes（官方推荐格式）
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
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=/home/ubuntu/swe-rl/logs/agentic \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.runner_fqn=uni_agent_ext.agents.mini_swe_agent_runner.mini_swe_agent_runner \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.max_concurrent_sessions=${CONCURRENCY} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
  reward.reward_manager.name=naive \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=swe-rl-agentic \
  trainer.experiment_name=qwen25-7b-grpo-agentic-lora \
  trainer.save_freq=1 \
  trainer.resume_mode=auto \
  trainer.default_local_dir=/home/ubuntu/swe-rl/checkpoints/agentic \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  "$@"
