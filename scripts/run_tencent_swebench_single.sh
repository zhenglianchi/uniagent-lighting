#!/usr/bin/env bash
# 在腾讯云沙箱上跑一条真实 SWE-bench 轨迹（mini-swe-agent + tencent_e2b 环境）
#
# 用法：bash scripts/run_tencent_swebench_single.sh [instance_id]
#   默认 instance_id=django__django-13447（SWE-Bench full/test）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. 加载腾讯云凭据（TENCENT_SECRET_ID/KEY、TENCENT_SANDBOX_E2B_TOKEN）
set -a
source "$ROOT/work/tencent_sandbox.env"
set +a

# 2. E2B 兼容端点 + HF 镜像（数据集加载）
export E2B_DOMAIN="${E2B_DOMAIN:-ap-guangzhou.tencentags.com}"
export E2B_API_KEY="${TENCENT_SANDBOX_E2B_TOKEN}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

INSTANCE_ID="${1:-marshmallow-code__marshmallow-1359}"
echo "[run] instance=$INSTANCE_ID template=swebench-v1 model=qwen3.7-plus subset=lite split=dev"

exec mini-extra swebench-single \
  -c config/tencent_swebench.yaml \
  -o "work/swebench/tencent_${INSTANCE_ID}.traj.json" \
  --subset lite \
  --split dev \
  -i "$INSTANCE_ID" \
  --exit-immediately \
  -y
