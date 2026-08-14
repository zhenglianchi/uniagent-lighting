#!/usr/bin/env bash
# 在 ray start 之前 source 并 export 所有 Ray worker 需要的环境变量。
#
# 背景（2026-08-15 实测踩坑）：Ray worker 的环境变量在 `ray start` 时固定，
# 不继承训练脚本内的 export。若在 ray start 后才 export E2B_API_KEY /
# E2B_DOMAIN / GATEWAY_PORT，agent 起沙箱会报
# `AuthenticationException: API key is required`，全部会话失败。
#
# 用法（node1 / node2 相同，必须在 ray start 之前）：
#   source scripts/bootstrap_ray_env.sh
#   ray stop --force
#   ray start --head --port=6379 --num-gpus=1        # node1
#   ray start --address=node1内网IP:6379 --num-gpus=1  # node2 加入

set -a
source /home/ubuntu/swe-rl/tencent_sandbox.env
set +a

export E2B_DOMAIN=${E2B_DOMAIN:-ap-guangzhou.tencentags.com}
export E2B_API_KEY=${E2B_API_KEY:-${TENCENT_SANDBOX_E2B_TOKEN}}

# Gateway 固定端口（白盒 harness 在训练机本地直连，无需沙箱内隧道）
export GATEWAY_PORT=${GATEWAY_PORT:-8001}
export MSA_GATEWAY_TUNNEL=${MSA_GATEWAY_TUNNEL:-0}
export MSA_INSTALL_AGENT=${MSA_INSTALL_AGENT:-1}
export MSA_REWARD_INCLUDE_P2P=${MSA_REWARD_INCLUDE_P2P:-1}
export MSA_REWARD_P2P_SAMPLE=${MSA_REWARD_P2P_SAMPLE:-20}
export TENCENT_SANDBOX_SKIP_TMUX=1

if [ -z "${E2B_API_KEY}" ]; then
  echo "[bootstrap_ray_env] ERROR: E2B_API_KEY empty; check tencent_sandbox.env" >&2
  return 1 2>/dev/null || exit 1
fi
echo "[bootstrap_ray_env] E2B_API_KEY=${E2B_API_KEY:0:5}... E2B_DOMAIN=${E2B_DOMAIN} GATEWAY_PORT=${GATEWAY_PORT}"
