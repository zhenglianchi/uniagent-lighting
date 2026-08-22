#!/usr/bin/env bash
# 一键启动采样流水线：预拉镜像 -> mini-swe-agent 采样 -> 打包上传(UCloud SFTP 直传)
#
# 用法：
#   # 测试期：只跑 train.jsonl 里前 1 个实例，step_limit=30，打包但不实际上传
#   conda run -n swe-rl bash scripts/sampling/start_sampling.sh --limit 1 --dry-run
#
#   # 正式批量：跑前 40 个实例，step_limit=40，跑完删镜像，真实上传
#   conda run -n swe-rl bash scripts/sampling/start_sampling.sh --limit 40 \
#       --step-limit 40 --rm-image
#
#   # 单实例指定 ID，跳过预拉（镜像已在本地）
#   conda run -n swe-rl bash scripts/sampling/start_sampling.sh \
#       --instance sympy__sympy-13043 --no-pull --dry-run
#
# 关键参数：
#   --list <jsonl>     实例清单（默认 work/data/train.jsonl，取 instance_id 列）
#   --limit N          只跑前 N 个实例（默认 1，省 token）
#   --step-limit N     agent 最大步数（默认 30；完整提交建议 40）
#   --instance <id>    只跑指定实例（覆盖 --list/--limit）
#   --no-pull          不预拉镜像（本地已有时省时间）
#   --rm-image         每个实例跑完后 docker rmi 删镜像（正式批量用）
#   --no-upload        采样完不调上传器
#   --dry-run          传给上传器：只打包不上传
#   --plan-only        只打印执行计划，不采样不拉镜像（预览用）
#   --config <yaml>    模型配置（默认 config/mini_aliyun.yaml；
#                       正式阶段可切换为指向云端 vLLM 的 OpenAI 兼容配置）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEBENCH_YAML="$ROOT/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml"
MODEL_CONFIG="${MODEL_CONFIG:-$ROOT/config/mini_aliyun.yaml}"
LIST_FILE="$ROOT/work/data/train.jsonl"
OUT_DIR="$ROOT/work/swebench"
INSTANCE_DIR="$ROOT/work/swebench"
LIMIT=1
STEP_LIMIT=30
INSTANCE=""
PULL=1
RM_IMAGE=0
UPLOAD=1
DRY_RUN=0
PLAN_ONLY=0

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST_FILE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --step-limit) STEP_LIMIT="$2"; shift 2 ;;
    --instance) INSTANCE="$2"; shift 2 ;;
    --config) MODEL_CONFIG="$2"; shift 2 ;;
    --no-pull) PULL=0; shift ;;
    --rm-image) RM_IMAGE=1; shift ;;
    --no-upload) UPLOAD=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done

if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "错误：模型配置不存在 $MODEL_CONFIG" >&2; exit 1
fi

# 构造实例 ID 列表
if [[ -n "$INSTANCE" ]]; then
  INSTANCE_IDS=("$INSTANCE")
elif [[ -f "$LIST_FILE" ]]; then
  mapfile -t INSTANCE_IDS < <(conda run -n swe-rl python -c "
import json, sys
ids = []
with open('$LIST_FILE', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        ids.append(json.loads(line)['instance_id'])
print('\n'.join(ids[:$LIMIT]))
")
else
  echo "错误：找不到实例清单 $LIST_FILE（可用 --instance 指定）" >&2; exit 1
fi

if [[ ${#INSTANCE_IDS[@]} -eq 0 ]]; then
  echo "实例列表为空，退出。" >&2; exit 1
fi

echo "==== 采样流水线启动 ===="
echo "实例数: ${#INSTANCE_IDS[@]} | step_limit: $STEP_LIMIT | 模型配置: $MODEL_CONFIG"
echo "预拉镜像: $([ $PULL -eq 1 ] && echo 是 || echo 否) | 跑完删镜像: $([ $RM_IMAGE -eq 1 ] && echo 是 || echo 否)"
echo "目标目录: $OUT_DIR | 上传: $([ $UPLOAD -eq 1 ] && echo 是 || echo 否)"

if [[ $PLAN_ONLY -eq 1 ]]; then
  echo ""
  echo "==== 执行计划（plan-only，不执行）===="
  for iid in "${INSTANCE_IDS[@]}"; do
    image="docker.io/swebench/sweb.eval.x86_64.$(echo "$iid" | sed 's/__/_1776_/g'):latest"
    echo "  - $iid"
    echo "      镜像: $image"
    echo "      命令: mini-extra swebench-single --subset lite --split test --instance $iid -c agent.step_limit=$STEP_LIMIT"
  done
  echo ""
  if [[ $UPLOAD -eq 1 ]]; then
    echo "  上传: trajectory_uploader.py --input-dir $INSTANCE_DIR$([ $DRY_RUN -eq 1 ] && echo ' --dry-run')"
  fi
  exit 0
fi

run_one() {
  local iid="$1"
  local image
  image="docker.io/swebench/sweb.eval.x86_64.$(echo "$iid" | sed 's/__/_1776_/g'):latest"

  echo ""
  echo "==== 开始实例: $iid ===="
  if [[ $PULL -eq 1 ]]; then
    echo "[1/3] 预拉镜像: $image"
    docker pull "$image"
  fi

  echo "[2/3] 采样（step_limit=$STEP_LIMIT）"
  conda run -n swe-rl mini-extra swebench-single \
    --yolo --exit-immediately \
    --subset lite --split test --instance "$iid" \
    -c "$SWEBENCH_YAML" -c "$MODEL_CONFIG" \
    -c agent.step_limit="$STEP_LIMIT" -c agent.cost_limit=0 \
    --environment-class docker \
    -o "$OUT_DIR/$iid.traj.json"

  if [[ $RM_IMAGE -eq 1 ]]; then
    echo "[3/3] 删除镜像: $image"
    docker rmi "$image"
  else
    echo "[3/3] 保留镜像（测试期调试用）"
  fi
  echo "==== 完成实例: $iid ===="
}

for iid in "${INSTANCE_IDS[@]}"; do
  run_one "$iid"
done

echo ""
echo "==== 全部采样完成 ===="

if [[ $UPLOAD -eq 1 ]]; then
  echo "调用上传器（--input-dir $INSTANCE_DIR）..."
  if [[ $DRY_RUN -eq 1 ]]; then
    conda run -n swe-rl python "$ROOT/scripts/sampling/trajectory_uploader.py" \
      --input-dir "$INSTANCE_DIR" --dry-run
  else
    conda run -n swe-rl python "$ROOT/scripts/sampling/trajectory_uploader.py" \
      --input-dir "$INSTANCE_DIR"
  fi
else
  echo "跳过上传（--no-upload）。轨迹在 $INSTANCE_DIR/"
fi
