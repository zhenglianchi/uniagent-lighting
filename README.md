# uniagent-lighting

把 **uni-agent（verl 之上的 Agent RL 编排层）改造成 agentlighting 式异步架构**：本地采样/rollout
与云端训练通过轨迹存储解耦。当前以 HumanEvalFix 为基准完成单机全样本训练 + 投机解码加速，
下一步推进双机全异步 + TransferQueue 的 mooncake 存储后端。

## 定位

- **上游**：[verl-project/uni-agent](https://github.com/verl-project/uni-agent)（构建于 verl 之上）
  + 本项目扩展包 `uni_agent_ext`
- **目标形态**（对标 agentlighting 的 Algorithm ↔ Store ↔ Runner 解耦）：
  - 本地 runner（mini-swe-agent + 腾讯云 Agent Runtime 沙箱）采集轨迹，异步进入云端轨迹存储
  - 云端 verl trainer 消费轨迹做 GRPO 训练，权重回传供采样侧使用
- **训练基线（2026-08-05 定稿，实测执行）**：`Qwen3-8B` + LoRA rank=32 + AdamW(fp32)
  + 梯度检查点；单机 4090 24GB 即可跑冒烟（全样本 32GB 机器亦可，见部署）

## 我们做了什么（截至 2026-08-10，v0.33.4）

- ✅ **单机 agentic GRPO 全链路**：verl 内部 Ray + vLLM Gateway + mini-swe-agent runner
  （`uni_agent_ext/agents/mini_swe_agent_runner.py`）+ 腾讯云沙箱（E2B 兼容端点），
  LoRA rank=32、续训 `resume_mode=auto`、checkpoint keep=1
- ✅ **全样本 HumanEvalFix 训练（baseline）**：train161 / batch32 / mini16 / micro4 / 并发64 /
  5 epoch（26 步），评测 **基座 76.4% → final 83.2%**（n=1 / temp 0.8 / 161 条）
- ✅ **gateway hermes 工具解析容错**（JSON repair + 合成重试，v0.31.8）：消除解析错误杀会话
- ✅ **投机解码（EAGLE-3 + LoRA merge）**：25/25 步训练完成（v0.32.x），rollout 吞吐
  **+40%**（199 → 282 tok/s），最终模型评测 **82.6%（133/161）**，与 baseline final 几乎持平
- ✅ **训练日志双存档**：本地 `swe-rl-local/work/server_logs/`（完整会话目录）+ 本仓
  `work/logs/spec_run_20260810/`（压缩日志 + 逐步统计 + 评测结果，含 sha256）
- ⏳ **双机全异步 + TQ mooncake 存储后端**（进行中）：保存 node1 镜像 → 恢复双机 →
  GRPO 冒烟 → 验证全异步链路 → 对比 mooncake 存储后端吞吐

## 仓库结构

```
uni_agent_ext/            # uni-agent 扩展包（不改上游源码的增量）
  sandbox/                # 腾讯云 Agent Runtime 沙箱后端（E2B 兼容）
  agents/                 # mini-swe-agent 训练 runner（AgentRunner 协议）
scripts/                  # 数据构建 / 训练（各模式）/ 评测 / 运维脚本
patches/                  # verl / uni-agent / mini-swe-agent 补丁（部署时应用）
docs/                     # architecture.md（改造方案）/ deployment.md / vllm_access.md 等
work/                     # 数据（data/）、训练日志存档（logs/）、环境配置（config/）
CHANGELOG.md              # 版本记录（每完成一项 commit 一次，语义化递增）
```

## 训练模式一览

| 模式 | 脚本 | 用途 / 数据 | 关键参数 |
| --- | --- | --- | --- |
| 冒烟（纯 verl） | `run_grpo_smoke_ucloud.sh` | 验证链路，smoke 2 条假奖励 | batch2 / n=2 |
| 单机 LoRA | `run_grpo_single_lora_ucloud.sh` | 验证 LoRA + vLLM 共存 | rank=32 / offload 关 |
| agentic 单机 | `run_grpo_single_agentic_ucloud.sh` | 完整 agent 链路，agentic 数据 | step_limit=60 |
| 全样本 HumanEvalFix | `run_grpo_humanevalfix_ucloud.sh` | train161 / 5 epoch | batch32 / mini16 / micro4 / 并发64 |
| **投机推理全样本** | `spec_train_run.sh` | 全样本 + EAGLE-3 投机解码 | `LORA_MERGE=1` `SPEC_ON=1` |
| 多机 | `run_grpo_multinode_ucloud.sh` | 双机 GRPO 冒烟 | dp=2 / tp=2 / batch=2 |
| **多机全异步** | `run_grpo_multinode_async_ucloud.sh` | 双机 colocate_async（Trainer 与 rollout
  重叠）+ TQ，可切 mooncake 后端 | `MOONCAKE=0/1` / 预热 1 / dp=2 / tp=1 |

### 启动示例

```bash
# 冒烟（最快验证）
bash /home/ubuntu/swe-rl/run_grpo_smoke_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_smoke.log

# agentic 单机（2 条 train + 1 条 val，真实 SWE-bench reward）
bash /home/ubuntu/swe-rl/run_grpo_single_agentic_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_agentic.log

# 全样本 HumanEvalFix（baseline 口径，环境变量在脚本内可覆盖）
bash /home/ubuntu/swe-rl/run_grpo_humanevalfix_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_humanevalfix.log

# 投机推理全样本（EAGLE-3 + LoRA merge；独立 checkpoint/日志目录，不污染 baseline）
bash /home/ubuntu/swe-rl/spec_train_run.sh
tail -f /home/ubuntu/swe-rl/grpo_humanevalfix_spec.log

# 多机（node1 上，前置 Ray 集群已起）
bash /home/ubuntu/swe-rl/run_grpo_multinode_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_multinode.log

# 多机全异步（TQ SimpleStorage；MOONCAKE=1 切 MooncakeStore 做对照）
MOONCAKE=0 bash /home/ubuntu/swe-rl/run_grpo_multinode_async_ucloud.sh 2>&1 | tee grpo_multinode_async.log
MOONCAKE=1 bash /home/ubuntu/swe-rl/run_grpo_multinode_async_ucloud.sh 2>&1 | tee grpo_multinode_async_mooncake.log
```

> ⚠️ 训练中 CPU 峰值内存较高（agentic ~65GB），SSH 可能长时间无响应——用 UCloud 控制台
> Web shell 操作，不要误杀进程。腾讯沙箱按量计费：只跑单实例、任务间隙停实例
> （`tencent_stop_all_instances.py`，仅支持停止；STOPPED 实例不计费、过期自动回收）。

## 从零部署（新机器 → 跑通训练）

面向 UCloud GPU 云主机单机部署（RTX 4090 24GB/48G 均可；48G 跑全样本 batch32 更宽裕）。
多机见文末「多机（可选）」。

### 0. 前置

- UCloud 实例：Ubuntu 24.04 + NVIDIA 驱动 570+（CUDA 12.8），≥32GB 内存（全样本建议 64G+），
  公网 IP + 22 端口
- 腾讯云沙箱已开通（凭据在本地 `work/tencent_sandbox.env`，不入库，服务器 chmod 600）
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

- Miniforge + `swe-rl` env（Python 3.10）、清华 pip 源、HF 镜像（永久写入 `~/.bashrc`，
  且必须 `HF_HUB_DISABLE_XET=1`，否则 hf-mirror 报 401）
- clone `verl-project/uni-agent`（含 verl 子模块）+ `pip install -e`
- verl 补丁：StrEnum（py3.10）、fsdp2 单卡跳过 state_dict 拷贝、IPC CPU 大权重、
  logprobs 投机修复（`patches/verl_vllm_logprobs_spec_fix.patch`）、LoRA merge 物化修复
  （`patches/verl_merged_lora_materialize_fix.patch`）、debug 指标缺键保护
- 模型 `Qwen3-8B` 下载到 `~/models/`（hf-mirror，约 15.3GB；BF16，支持 function calling）
- 投机解码 drafter（可选，跑投机推理才需要）：`Qwen3-8B-speculator.eagle3`
  （RedHatAI EAGLE-3，约 2GB，hf-mirror）
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
# 训练/运维脚本软链到仓库（后续 git pull 即更新）
cd /home/ubuntu/swe-rl
for f in run_grpo_smoke_ucloud.sh run_grpo_single_lora_ucloud.sh \
         run_grpo_single_agentic_ucloud.sh run_grpo_humanevalfix_ucloud.sh \
         run_grpo_multinode_ucloud.sh spec_train_run.sh kill_train.sh \
         start_stats_watch.sh collect_grpo_stats.py eval_humanevalfix.py \
         convert_verl_lora_to_hf.py fix_multinode_hosts.sh nccl_multinode_test.py \
         patch_verl_ipc_cpu.py ray_import_test.py reward_smoke.py \
         tencent_stop_all_instances.py kill_eval.sh eval_spec_final.sh \
         run_eval_only.sh run_eval_final_spec.sh; do
  ln -sf /home/ubuntu/uniagent-lighting/scripts/$f "$f"
done
```

### 3. 数据与凭据

数据在**本地**生成（依赖本地 SWE-bench Lite 缓存），再上传：

```bash
# 本地：生成 agentic / HumanEvalFix 数据
conda run -n swe-rl python scripts/make_agentic_data.py --train-num 2 --val-num 1
conda run -n swe-rl python scripts/make_humanevalfix_data.py
scp work/data/agentic_train.jsonl work/data/agentic_val.jsonl \
  work/data/humanevalfix_train161.jsonl work/data/humanevalfix_val.jsonl \
  ubuntu@<服务器IP>:/home/ubuntu/swe-rl/data/
scp work/tencent_sandbox.env ubuntu@<服务器IP>:/home/ubuntu/swe-rl/tencent_sandbox.env
```

服务器上把凭据权限收紧（`chmod 600 /home/ubuntu/swe-rl/tencent_sandbox.env`）。
数据文件 sha256 与 `work/data/` 一致（本仓已内置一份，含 val 正确版与 smoke 数据）。

### 4. mini-swe-agent 补丁（agentic/评测必做）

runner 在训练机本地驱动 mini-swe-agent（harness 在外、腾讯沙箱当执行环境），
需要给训练机安装的包打 `tencent_e2b` 环境补丁：

```bash
/home/ubuntu/miniforge3/envs/swe-rl/bin/pip install "mini-swe-agent==2.4.6"
SP=/home/ubuntu/miniforge3/envs/swe-rl/lib/python3.10/site-packages
cp /home/ubuntu/uniagent-lighting/patches/tencent_e2b.py \
   "$SP/minisweagent/environments/extra/tencent_e2b.py"
# pip 官方版 swebench.py 的镜像注入列表不含 tencent_e2b（否则沙箱实例无 /testbed）：
cp /home/ubuntu/uniagent-lighting/patches/miniswe_swebench.py \
   "$SP/minisweagent/run/benchmarks/swebench.py"
```

### 5. 验证

```bash
E=/home/ubuntu/miniforge3/envs/swe-rl
$E/bin/python -c "import torch,vllm,transformers,verl,uni_agent,ray,peft; print(torch.__version__, vllm.__version__, transformers.__version__)"
$E/bin/python -c "import uni_agent_ext; from uni_agent_ext.agents.mini_swe_agent_runner import mini_swe_agent_runner; print('uni_agent_ext OK:', mini_swe_agent_runner.__name__)"
$E/bin/python -c "import verl.trainer.main_ppo; print('main_ppo OK')"
```

### 6. 跑训练

按「训练模式一览」选择脚本，冒烟建议先跑（2~5 分钟验证链路）：

```bash
bash /home/ubuntu/swe-rl/run_grpo_smoke_ucloud.sh 2>&1 | tee /home/ubuntu/swe-rl/grpo_smoke.log
```

agentic 训练脚本做的事：起 verl（内部自动起 Ray + vLLM Gateway）→ runner 在腾讯沙箱里跑
mini-swe-agent 多轮轨迹 → 真实 reward → GRPO 更新 LoRA（rank=32）→ adapter 同步回
vLLM 引擎。默认 `resume_mode=auto` 续训，checkpoint 在 `/home/ubuntu/swe-rl/checkpoints/`。

### 常见排坑

- HF 下载必须 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`，否则 Xet 协议 401
- torch 大文件走 PyTorch 官方 cu128 index（清华源大文件限速 ~600KB/s）
- `use_fused_kernels` 与 LoRA(PEFT) 冲突（aten.mm mixed DTensor 报错）：LoRA 训练必须关
- 投机解码：LoRA 必须 `lora.merge=True`（LoRA×SD 互斥）；EAGLE-3 下 vLLM `logprobs=0`
  会全丢（vllm#30059），需应用 `patches/verl_vllm_logprobs_spec_fix.patch`（0→1）
- 腾讯沙箱：一次性并发创建过多实例会 `AUTHENTICATION_FAILED` / `LimitExceeded.CPU`，
  训练渐进派发无碍；评测若批量失败，先停残留实例（`tencent_stop_all_instances.py`）再重跑
- 训练数据不要注入 `test_patch`（防泄露），`test_patch` 只用于 reward 评估
- 所有模型/数据/checkpoint 走服务器本地，**不用腾讯云 COS**

### 多机（可选）

单机跑通后，多机 = 两台同 VPC/子网的机器（vllm 0.11.1 已满足多节点要求）：
`fix_multinode_hosts.sh` 写内网 hosts → Ray head/join → `run_grpo_multinode_ucloud.sh`
（LoRA rank=32 / dp=2 / tp=2 / batch=2，冒烟口径 2 条 prompt）。详见
`docs/architecture.md` 与 `docs/deployment.md`。下一步计划：双机全异步 rollout +
TransferQueue 的 mooncake 存储后端对比（见 `docs/ROADMAP.md`）。

## 使用

各脚本头部有详细注释；架构与改造路线见 `docs/architecture.md`；训练机部署细节见
`docs/deployment.md`；版本记录见 `CHANGELOG.md`。
