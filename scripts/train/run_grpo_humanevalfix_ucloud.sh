#!/usr/bin/env bash
# 单机 agentic GRPO 训练：**HumanEvalFix 数据集**（2026-08-06 新增，原始 SWE-bench 方案
# 保留在 run_grpo_single_agentic_ucloud.sh，本脚本只改数据/实验名/checkpoint 目录）。
#
# 前置：
#   1) 数据：scripts/data/make_humanevalfix_data.py 生成的 humanevalfix_train/val.jsonl
#      （沙箱预置 /testbed/solution.py，隐藏测试只在 reward 阶段注入，无测试泄露）
#   2) uni_agent_ext 包已放到训练机 PYTHONPATH（runner 已支持 humaneval_fix 任务类型）
#   3) 腾讯沙箱凭据环境变量（work/tencent_sandbox.env）
#
# 用法（训练机 node2 上执行）：
#   bash /home/ubuntu/swe-rl/run_grpo_humanevalfix_ucloud.sh 2>&1 | tee grpo_humanevalfix.log
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
# 跳过沙箱内 tmux 安装（mini-swe-agent harness 自带 pexpect shell，不需要 tmux；
# 该 apt-get 经常等到 E2B 180s 超时，白耗每会话 3 分钟）
export TENCENT_SANDBOX_SKIP_TMUX=1
# HumanEvalFix 无 PASS_TO_PASS，P2P 抽样不适用
export MSA_REWARD_INCLUDE_P2P=0

ENV=/home/ubuntu/miniforge3/envs/swe-rl
MODEL=${MODEL:-/home/ubuntu/models/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_train161.jsonl}
VAL_FILE=${VAL_FILE:-/home/ubuntu/swe-rl/data/humanevalfix_val.jsonl}
TOOL_PARSER=${TOOL_PARSER:-hermes}         # gateway tool-call parser；Qwen3-8B 用 hermes（官方推荐格式）
GATEWAY_COUNT=${GATEWAY_COUNT:-1}          # 单机冒烟 1 个 gateway actor
CONCURRENCY=${CONCURRENCY:-4}              # 并发 rollout sessions（= 同时跑的沙箱数，控制成本）
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "$MODEL")"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}          # 全样本长训可覆盖（如 20），resume_mode=auto 已默认开启
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}  # 每 step 样本数（全样本 164 条建议 16）
PPO_MINI_BATCH=${PPO_MINI_BATCH:-2}      # 训练更新 mini-batch（可随 train_batch 调高，如 8）
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-1}    # 每 GPU micro-batch（48G + offload 可调高到 2）
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-}         # 只保留最近 N 个 checkpoint（如 1 = 一直覆盖最新）
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-4}  # vLLM 并发序列数，与 CONCURRENCY 同步调高（如 8）
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.5}  # vLLM 显存利用率（16 并发建议 0.7）
CKPT_DIR=${CKPT_DIR:-/home/ubuntu/swe-rl/checkpoints/humanevalfix}  # 独立验证 run 可覆盖，避免续训污染
LOG_DIR=${LOG_DIR:-/home/ubuntu/swe-rl/logs/humanevalfix}          # 会话/轨迹日志目录（独立 run 用新目录防混杂）
LORA_MERGE=${LORA_MERGE:-0}              # 1 = lora.merge=True（每步合并全量权重同步；投机解码 SD 的前置，vLLM LoRA×SD 互斥）
SPEC_ON=${SPEC_ON:-0}                    # 1 = 开启投机解码（EAGLE-3 drafter，需 LORA_MERGE=1）
SPEC_DRAFT=${SPEC_DRAFT:-/home/ubuntu/models/Qwen3-8B-speculator.eagle3}
SPEC_TOKENS=${SPEC_TOKENS:-3}            # num_speculative_tokens（官方编码基准 k=3）
cd /home/ubuntu/uni-agent/verl

ls -la "$TRAIN_FILE" "$VAL_FILE" "$MODEL" >/dev/null

EXTRA_ARGS=()
if [ -n "$MAX_CKPT_KEEP" ]; then
  EXTRA_ARGS+=(trainer.max_actor_ckpt_to_keep="$MAX_CKPT_KEEP")
fi
if [ "$LORA_MERGE" = "1" ]; then
  EXTRA_ARGS+=(actor_rollout_ref.model.lora.merge=True)
fi
if [ "$SPEC_ON" = "1" ]; then
  # 独立 drafter（非 MTP）不走权重同步（verl _iter_all_models 只同步 actor+MTP）；
  # drafter 保持静态，训练中 LoRA 漂移会导致接受率下降（step1 vs step5 各记录一次）
  EXTRA_ARGS+=(
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
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.max_model_len=8192 \
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
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.runner_fqn=uni_agent_ext.agents.mini_swe_agent_runner.mini_swe_agent_runner \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.max_concurrent_sessions=${CONCURRENCY} \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mini_swe_agent.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
  reward.reward_manager.name=naive \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=swe-rl-humanevalfix \
  trainer.experiment_name=qwen3-8b-grpo-humanevalfix-lora \
  trainer.save_freq=1 \
  trainer.resume_mode=auto \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  "${EXTRA_ARGS[@]}" \
