#!/usr/bin/env bash
# UCloud 裸机一键安装（2026-08-05，最终版本链直装，跳过先装旧版再升级的重复下载）
# 目标链：torch 2.9.0+cu128 / vllm 0.11.1 / transformers 4.57.6 / verl 0.9.0.dev / ray 2.56.1
# 前置：Ubuntu 24.04 + NVIDIA 驱动（570+，CUDA 12.8）
# 包含：miniforge + swe-rl env、清华 pip/HF 镜像、uni-agent、verl 三处补丁
#       （StrEnum + fsdp2 单卡 + IPC CPU 大权重）、uni-agent codec 补丁（vllm 0.11.1
#       hermes 工具解析）、mini-swe-agent 2.4.6 + tencent_e2b 补丁、
#       模型下载（Qwen3-8B，hf-mirror 15G）
# 用法：
#   bash install_ucloud_from_scratch.sh
#   CREATE_SWAP_SIZE_GB=20 bash install_ucloud_from_scratch.sh  # 可选加 swap
# 安装后需从本地仓库上传冒烟文件（见尾部提示）
set -euo pipefail

MINIFORGE_DIR="$HOME/miniforge3"
ENV="$MINIFORGE_DIR/envs/swe-rl"
PY="$ENV/bin/python"
PIP="$ENV/bin/pip"

echo "== 1/8 Miniforge（GitHub 失败退清华镜像）=="
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  cd /tmp
  (curl -fsSL --max-time 300 -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    || curl -fsSL --max-time 300 -o Miniforge3.sh https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh)
  bash Miniforge3.sh -b -p "$MINIFORGE_DIR"
  "$MINIFORGE_DIR/bin/conda" init bash >/dev/null
fi

echo "== 2/8 conda env（python 3.10）+ 清华 conda 源 =="
"$MINIFORGE_DIR/bin/conda" config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ >/dev/null 2>&1 || true
"$MINIFORGE_DIR/bin/conda" config --set channel_priority flexible >/dev/null 2>&1 || true
"$MINIFORGE_DIR/bin/conda" create -n swe-rl python=3.10 -y || true

echo "== 3/8 pip 清华源 + HF 镜像（永久写入）=="
"$PIP" config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$PIP" config set global.trusted-host pypi.tuna.tsinghua.edu.cn
grep -q HF_ENDPOINT "$HOME/.bashrc" || {
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> "$HOME/.bashrc"
  echo 'export HF_HUB_DISABLE_XET=1' >> "$HOME/.bashrc"
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> "$HOME/.profile"
  echo 'export HF_HUB_DISABLE_XET=1' >> "$HOME/.profile"
}

echo "== 4/8 torch 2.9.0+cu128（PyTorch 官方 cu128 index，清华大文件限速别用）=="
"$PIP" install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

echo "== 5/8 vllm 0.11.1 + transformers（多机必须 vllm>=0.11.1；4.57.6 实测安全）=="
"$PIP" install vllm==0.11.1
"$PIP" install "transformers>=4.55.0,<4.60"

echo "== 6/8 uni-agent + bundled verl（浅克隆，GitHub 断流重试）=="
cd "$HOME"
if [ ! -d uni-agent/.git ]; then
  git config --global http.postBuffer 524288000
  git config --global http.lowSpeedLimit 1000
  git config --global http.lowSpeedTime 60
  for i in 1 2 3; do
    timeout 300 git clone --depth 1 https://github.com/verl-project/uni-agent.git && break || { echo "clone retry $i"; sleep 5; }
  done
  cd uni-agent
  for i in 1 2 3; do
    timeout 600 git submodule update --init --recursive --depth 1 && break || { echo "submodule retry $i"; sleep 5; }
  done
fi
cd "$HOME/uni-agent"
"$PIP" install --no-deps -e ./verl
"$PIP" install -e .
"$PIP" install "ray[default]" loguru pydantic pydantic_settings aiohttp \
  tensordict torchdata wandb omegaconf pylatexenc tensorboard pybind11 peft hydra-core codetiming
"$PIP" install "TransferQueue==0.1.9"
"$PIP" install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
"$PIP" install -q StrEnum || true

echo "== 7/8 verl 补丁（StrEnum + fsdp2 单卡 + IPC CPU 大权重）=="
"$PY" - <<'PYEOF'
import io
import os
import pathlib

home = os.path.expanduser("~")
verl_root = os.path.join(home, "uni-agent", "verl", "verl")

# 7.1 StrEnum
old = "from enum import StrEnum"
new = "try:\n    from enum import StrEnum\nexcept ImportError:\n    from strenum import StrEnum"
for p in pathlib.Path(verl_root).rglob("*.py"):
    s = p.read_text(encoding="utf-8")
    if old in s and new not in s:
        p.write_text(s.replace(old, new), encoding="utf-8")
        print("StrEnum patched:", p)

# 7.2 fsdp2 单卡跳过 state_dict 拷贝
path = os.path.join(verl_root, "workers", "engine", "fsdp", "transformer_impl.py")
src = io.open(path, encoding="utf-8").read()
old2 = """            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)"""
new2 = """            if torch.distributed.get_world_size() > 1:
                # multi-rank: full_state is needed to broadcast weights from rank 0
                full_state = module.state_dict()
                apply_fsdp2(module, fsdp_kwargs, self.engine_config)
                fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
            else:
                # single-rank: apply_fsdp2 keeps already-loaded weights in place;
                # skip state_dict copy + reload to avoid ~2x memory peak on 32GB hosts
                apply_fsdp2(module, fsdp_kwargs, self.engine_config)"""
if old2 in src:
    src = src.replace(old2, new2, 1)
    io.open(path, "w", encoding="utf-8").write(src)
    print("[patch] fsdp2 single-rank applied")

# 7.3 IPC CPU 大权重：_direct_send_large_weight 发送前 CPU->CUDA
path3 = os.path.join(
    verl_root, "workers", "rollout", "vllm_rollout", "bucketed_weight_transfer.py"
)
src3 = io.open(path3, encoding="utf-8").read()
marker = "# [swe-rl patch] move CPU weights to CUDA before reduce_tensor"
if marker not in src3:
    anchor = "        handle = reduce_tensor(weight)\n"
    assert anchor in src3, "IPC anchor not found"
    insertion = (
        "        # [swe-rl patch] move CPU weights to CUDA before reduce_tensor\n"
        "        # LoRA first base-weight sync collects params on CPU; a CPU tensor\n"
        "        # yields a short handle that rebuild_ipc (list_args[6]) can't parse.\n"
        "        if weight.device.type != \"cuda\":\n"
        "            weight = weight.to(f\"cuda:{get_device_id()}\")\n"
    )
    src3 = src3.replace(anchor, insertion + anchor, 1)
    io.open(path3, "w", encoding="utf-8").write(src3)
    print("[patch] IPC CPU-weight fix applied")
PYEOF

echo "== 8/9 模型（hf-mirror，15GB；已存在则跳过）=="
mkdir -p "$HOME/models"
if [ ! -f "$HOME/models/Qwen3-8B/config.json" ]; then
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 "$PY" -c \
    "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='$HOME/models/Qwen3-8B')"
fi

echo "== 9/9 mini-swe-agent + 补丁（v0.26.0：tencent_e2b / codec 路径）=="
"$PIP" install "mini-swe-agent==2.4.6"
SP="$ENV/lib/python3.10/site-packages"
REPO_RAW=https://raw.githubusercontent.com/zhenglianchi/uniagent-lighting/main
for i in 1 2 3; do
  curl -fsSL --max-time 60 -o /tmp/tencent_e2b.py "$REPO_RAW/patches/tencent_e2b.py" && break || { echo "retry $i"; sleep 3; }
done
cp /tmp/tencent_e2b.py "$SP/minisweagent/environments/extra/tencent_e2b.py"
# uni-agent gateway codec：vllm 0.11.1 的 ChatCompletionToolsParam 路径修复（hermes 工具解析）
for i in 1 2 3; do
  curl -fsSL --max-time 60 -o /tmp/uni_agent_vllm0111.patch "$REPO_RAW/patches/uni_agent_vllm0111_toolparsers.patch" && break || { echo "retry $i"; sleep 3; }
done
cd "$HOME/uni-agent" && patch -p1 -N < /tmp/uni_agent_vllm0111.patch || true

mkdir -p "$HOME/swe-rl/data"

# 可选 swap（94GB 内存不需要；64GB 建议 20G）
if [ -n "${CREATE_SWAP_SIZE_GB:-}" ]; then
  sudo fallocate -l "${CREATE_SWAP_SIZE_GB}G" /swapfile \
    && sudo chmod 600 /swapfile \
    && sudo mkswap /swapfile \
    && sudo swapon /swapfile
  grep -q /swapfile /etc/fstab || echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
fi

echo "== 验证 =="
"$PY" -c "import torch,vllm,transformers,verl,uni_agent,ray,peft;print('OK',torch.__version__,vllm.__version__,transformers.__version__,verl.__version__)"
"$PY" -c "import verl.trainer.main_ppo; print('main_ppo import OK')"
echo "SETUP_COMPLETE"
