# uniagent-lighting

把 **uni-agent（verl 之上的 Agent RL 编排层）改造成 agentlighting 式异步架构**：本地采样/rollout 与云端训练通过轨迹存储解耦。

## 定位

- **上游**：[verl-project/uni-agent](https://github.com/verl-project/uni-agent)（构建于 verl 之上）+ 本项目扩展包 `uni_agent_ext`
- **目标形态**（对标 agentlighting 的 Algorithm ↔ Store ↔ Runner 解耦）：
  - 本地 runner（mini-swe-agent + 腾讯云 Agent Runtime 沙箱）采集轨迹，异步进入云端轨迹存储
  - 云端 verl trainer 消费轨迹做 GRPO 训练，权重回传供采样侧使用
- **训练基线（2026-08-05 定稿）**：LoRA rank=32 + AdamW(fp32) + CPU offload 关 + 梯度检查点 + fused kernels 关（与 LoRA 冲突）；单机 48G 4090 / 94GB RAM 已验证完整跑通 1 步

## 仓库结构

```
uni_agent_ext/            # uni-agent 扩展包（不改上游源码的增量）
  sandbox/                # 腾讯云 Agent Runtime 沙箱后端（E2B 兼容）
  agents/                 # mini-swe-agent 训练 runner（AgentRunner 协议）
scripts/                  # 数据构建 / 训练 / 上传 / 采样脚本
docs/architecture.md      # agentlighting 式改造方案（三档路线）
CHANGELOG.md              # 版本记录（每完成一项 commit 一次）
```

## 当前进度

- ✅ 单机 LoRA GRPO 训练引擎跑通（纯 verl，无 agent；step ~50s，训练显存峰值 18.8G）
- ✅ verl 续训机制确认并启用（`resume_mode=auto` + `save_freq` + `default_local_dir`）
- ✅ mini-swe-agent 训练 runner（`uni_agent_ext/agents/mini_swe_agent_runner.py`）
- ✅ 7.2 任务数据（`scripts/make_agentic_data.py`，agentic schema 已上机对齐）
- ✅ vLLM 访问方案（`docs/vllm_access.md`：SSH 隧道走 22 端口 + `scripts/vllm_tunnel.sh`）
- ✅ 7.3 agentic 训练全链路跑通（`scripts/run_grpo_single_agentic_ucloud.sh`，v0.15.1 实测）
- ⏳ 轨迹异步上传 + 云端重放训练（方案 2）
- ⏳ TQ 解耦改造（方案 1，正式改造目标）
- ✅ 训练机部署 git 化（仓库 clone + 软链，v0.18.1）
- ✅ 腾讯沙箱 SWE-bench 实例接入（StartSandboxInstance + E2B connect，v0.9.0）

## 从零部署（新机器 → 跑通训练）

面向 UCloud GPU 云主机单机部署（node2 口径：单张 RTX 4090 48G / 94GB 内存；
64GB 内存机器也够，见排坑）。多机见文末「多机（可选）」。

### 0. 前置

- UCloud 实例：Ubuntu 24.04 + NVIDIA 驱动 570+（CUDA 12.8），≥64GB 内存，公网 IP + 22 端口
- 腾讯云沙箱已开通（凭据在本地 `work/tencent_sandbox.env`，不入库）
- 本地已 clone 本仓库，conda 环境 `swe-rl` 已装好（数据生成用）

### 1. 一键安装基础环境

在服务器上（预计 1~2 小时，主要是 torch/vllm/模型下载）：

```bash
cd ~ && curl -fsSL -o install_ucloud_from_scratch.sh \
  https://raw.githubusercontent.com/zhenglianchi/uniagent-lighting/main/scripts/install_ucloud_from_scratch.sh
bash install_ucloud_from_scratch.sh
# 64GB 内存机器建议：CREATE_SWAP_SIZE_GB=20 bash install_ucloud_from_scratch.sh
```

脚本完成内容（版本链 = **torch 2.9.0+cu128 / vllm 0.11.1 / transformers 4.57.x /
verl 0.9.0.dev / ray 2.56.1**）：

- Miniforge + `swe-rl` env（Python 3.10）、清华 pip 源、HF 镜像（永久写入 `~/.bashrc`）
- clone `verl-project/uni-agent`（含 verl 子模块）+ `pip install -e`
- verl 三处补丁：StrEnum（py3.10）、fsdp2 单卡跳过 state_dict 拷贝、IPC CPU 大权重
- 模型 `Qwen3-8B` 下载到 `~/models/`（hf-mirror，约 15.3GB；BF16，支持 function calling）
- 结束时打印 `SETUP_COMPLETE` 与版本验证

### 2. 部署改造仓（git 管理）

```bash
cd /home/ubuntu
git clone https://github.com/zhenglianchi/uniagent-lighting.git
# uni_agent_ext 进 PYTHONPATH（.pth 内容必须是包的父目录 /home/ubuntu）
ln -sfn /home/ubuntu/uniagent-lighting/uni_agent_ext /home/ubuntu/uni_agent_ext
echo "/home/ubuntu" > /home/ubuntu/miniforge3/envs/swe-rl/lib/python3.10/site-packages/uni_agent_ext.pth
# 运行目录（数据/凭据/checkpoint/日志，非仓库内容）
mkdir -p /home/ubuntu/swe-rl/{data,logs,checkpoints,outputs}
# 训练脚本软链到仓库（后续 git pull 即更新）
cd /home/ubuntu/swe-rl
for f in run_grpo_single_agentic_ucloud.sh run_grpo_single_lora_ucloud.sh \
         run_grpo_smoke_ucloud.sh run_grpo_multinode_ucloud.sh \
         fix_multinode_hosts.sh nccl_multinode_test.py patch_verl_ipc_cpu.py \
         ray_import_test.py reward_smoke.py; do
  ln -sf /home/ubuntu/uniagent-lighting/scripts/$f "$f"
done
```

### 3. 数据与凭据

数据在**本地**生成（依赖本地 SWE-bench Lite 缓存），再上传：

```bash
# 本地
conda run -n swe-rl python scripts/make_agentic_data.py --train-num 2 --val-num 1
scp work/data/agentic_train.jsonl work/data/agentic_val.jsonl \
  ubuntu@<服务器IP>:/home/ubuntu/swe-rl/data/
scp work/tencent_sandbox.env ubuntu@<服务器IP>:/home/ubuntu/swe-rl/tencent_sandbox.env
```

服务器上把凭据权限收紧（`chmod 600 /home/ubuntu/swe-rl/tencent_sandbox.env`）。

### 4. mini-swe-agent 补丁（训练机必做）

runner 在训练机本地驱动 mini-swe-agent（harness 在外、腾讯沙箱当执行环境），
需要给训练机安装的包打 `tencent_e2b` 环境补丁（attach 复用实例 + cleanup 只断开不停实例）：

```bash
/home/ubuntu/miniforge3/envs/swe-rl/bin/pip install "mini-swe-agent==2.4.6"
SP=/home/ubuntu/miniforge3/envs/swe-rl/lib/python3.10/site-packages
cp /home/ubuntu/uniagent-lighting/patches/tencent_e2b.py \
   "$SP/minisweagent/environments/extra/tencent_e2b.py"
```

### 5. 验证

```bash
E=/home/ubuntu/miniforge3/envs/swe-rl
$E/bin/python -c "import torch,vllm,transformers,verl,uni_agent,ray,peft; print(torch.__version__, vllm.__version__, transformers.__version__)"
$E/bin/python -c "import uni_agent_ext; from uni_agent_ext.agents.mini_swe_agent_runner import mini_swe_agent_runner; print('uni_agent_ext OK:', mini_swe_agent_runner.__name__)"
$E/bin/python -c "import verl.trainer.main_ppo; print('main_ppo OK')"
```

### 6. 跑 agentic GRPO 训练

```bash
bash /home/ubuntu/swe-rl/run_grpo_single_agentic_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_agentic.log
```

脚本做的事：起 verl（内部自动起 Ray + vLLM Gateway）→ runner 在腾讯沙箱里跑
mini-swe-agent 多轮轨迹 → 真实 SWE-bench reward → GRPO 更新 LoRA（rank=32）→
adapter 同步回 vLLM 引擎。默认 2 条 train + 1 条 val、`n=2`、`step_limit=60`、
续训已启用（`resume_mode=auto`，checkpoint 在 `/home/ubuntu/swe-rl/checkpoints/agentic`）。

> ⚠️ 训练中 CPU 峰值内存约 65GB，SSH 可能长时间无响应——用 UCloud 控制台 Web shell 操作，
> 不要误杀进程。腾讯沙箱按小时计费：只跑单实例、任务间隙停实例（`StopSandboxInstance`）。

### 常见排坑

- HF 下载必须 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`，否则 Xet 协议 401
- torch 大文件走 PyTorch 官方 cu128 index（清华源大文件限速 ~600KB/s）
- `docker pull` swebench 镜像超 120s 会判超时：先显式 `docker pull` 再跑实例（本地采样链路）
- 训练数据不要注入 `test_patch`（防泄露），`test_patch` 只用于 reward 评估
- 所有模型/数据/checkpoint 走服务器本地，**不用腾讯云 COS**

### 多机（可选）

单机跑通后，多机 = 两台同 VPC/子网的机器（vllm 0.11.1 已满足多节点要求）：
`fix_multinode_hosts.sh` 写内网 hosts → Ray head/join → `run_grpo_multinode_ucloud.sh`
（LoRA rank=32 / dp=2 / tp=2 / batch=2）。详见 `docs/architecture.md`。

## 使用

各脚本头部有详细注释；架构与改造路线见 `docs/architecture.md`；训练机部署细节见
`docs/deployment.md`。
