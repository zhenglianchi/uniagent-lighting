#!/usr/bin/env bash
# UCloud node1 环境升级脚本（2026-08-04 node1 实测通过）
# 升级到：torch 2.9.0+cu128 / vllm 0.11.1
# 保持：transformers 4.57.6 / verl 0.9.0.dev（uni-agent 捆绑）/ ray 2.56.1 / TransferQueue==0.1.9
#
# 为什么必须升（verl 0.9 多机 GRPO 在 vllm 0.10.1 下跑不起来）：
#   - tp=4            → AssertionError: For multi-node MP Executor, either set
#                       data_parallel_size>1 or upgrade vLLM to >= 0.11.1
#   - dp=2/tp=2       → vllm 0.10.1 不认 verl 传的 --master-addr/--node-rank/--nnodes
#                       （unrecognized arguments → 进程 exit 2）
#   结论：多机必须 vllm>=0.11.1（官方报错 + verl 源码双重确认）；verl 版本不降级（用户约束）。
#
# 用法：bash upgrade_vllm_0111.sh
#   RUN_VLLM_SMOKE=1 bash upgrade_vllm_0111.sh   # 升级后再跑 7B tp=2 引擎冒烟
# 说明：node2 不需要跑本脚本——UCloud 控制台直接用 node1 镜像克隆即可。
set -euo pipefail

ENV="$HOME/miniforge3/envs/swe-rl"
PY="$ENV/bin/python"
PIP="$ENV/bin/pip"

echo "== 1/4 卸载旧 torch 三件套（torchvision/torchaudio 无 GPU 需求方，一并移除）=="
"$PIP" uninstall -y torch torchvision torchaudio || true

echo "== 2/4 安装 torch 2.9.0+cu128（PyTorch 官方 cu128 index；清华源大文件限速 ~600KB/s，别用清华拉 torch wheel）=="
"$PIP" install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

echo "== 3/4 升级 vllm 0.11.1（PyPI/清华源；pip 会顺带校验 torch 依赖）=="
"$PIP" install vllm==0.11.1
"$PIP" install "transformers>=4.55.0,<4.60"   # 保持 4.57.6（verl 0.9 实测安全区间）

echo "== 4/4 验证 =="
if ! grep -q "strenum" "$HOME/uni-agent/verl/verl/utils/tokenizer/continuous_token_wiring.py"; then
  echo "WARN: verl StrEnum 补丁缺失（升级不碰 verl 源码，正常不会丢；若丢请重跑 setup_ucloud_uniagent.sh）"
fi
"$PY" - <<'PYEOF'
import torch, vllm, transformers
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| avail:", torch.cuda.is_available(), "| cc:", torch.cuda.get_device_capability())
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
import verl, uni_agent
print("verl / uni_agent import OK")
PYEOF

if [ "${RUN_VLLM_SMOKE:-0}" = "1" ]; then
  echo "== 可选：vLLM 引擎冒烟（Qwen3-8B / tp=2 / FLASH_ATTN）=="
  "$PY" - <<'PYEOF'
from vllm import LLM, SamplingParams
llm = LLM(model="/home/ubuntu/models/Qwen3-8B",
          tensor_parallel_size=2, enforce_eager=True,
          dtype="bfloat16", gpu_memory_utilization=0.5)
out = llm.generate(["def add(a, b):"], SamplingParams(max_tokens=32))
print("SMOKE OUTPUT:", out[0].outputs[0].text.strip())
PYEOF
fi

echo "== 清理 pip 缓存（省磁盘/镜像体积）=="
"$PIP" cache purge || true
echo "升级完成 ✅"
