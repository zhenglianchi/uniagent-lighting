# uniagent-lighting

> 基于 [verl](https://github.com/volcengine/verl) 与
> [Uni-Agent](https://github.com/verl-project/uni-agent) 的**代码智能体强化学习
> 训推平台**：白盒 / 黑盒双 harness、云端 Gateway 轨迹采集、GRPO（LoRA）云端
> 训练、双机全异步 + Mooncake + 投机解码，HumanEvalFix 全样本 5 epoch 训练
> 通过率 **83.23%**（vs 基座 76.4%）。

[![训练通过率](https://img.shields.io/badge/pass%40HumanEvalFix-83.23%25-brightgreen)]()
[![架构](https://img.shields.io/badge/双机-separate__async%20%2B%20Mooncake-blue)]()
[![版本](https://img.shields.io/badge/version-v0.49.2-orange)]()

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [架构](#架构)
- [训练结果](#训练结果)
- [快速开始](#快速开始)
- [安装（从裸机）](#安装从裸机)
- [训练模式](#训练模式)
- [评测](#评测)
- [仓库结构](#仓库结构)
- [文档](#文档)
- [已知问题与修复](#已知问题与修复)

---

## 项目简介

本项目把 **agent 强化学习训练** 做成一条可复用的平台化链路：任意
OpenAI/Anthropic 兼容的 agent（白盒 mini-swe-agent、黑盒 Claude Code、外部
自定义 agent）接入云端 Gateway，Gateway 在服务模型的同时**物化 token 级轨迹**
（prompt/response/logprob/mask/reward，on-policy），轨迹经 TransferQueue 进入
verl 完成 GRPO（LoRA）训练，产出可服务权重后再回到模型端点——形成
「采样 → 训练 → 部署」闭环。

与直接把 agent 塞进训练机 / 沙箱的"图省事"做法不同，本项目的目标形态是
**agent 跑在任意位置（用户侧/本地），沙箱只负责执行，训练与推理在云端**；
on-policy 的保证来自 Gateway 云侧记录的 token-truth 轨迹，而非 agent 的物理
位置。

### 两种运行形态（主线与内部训练）

**平台化形态（主线，对外）**：agent 部署在用户侧 / 任意位置，经 SSH 隧道或
公网端点接入云端 Gateway；沙箱只执行；训练 / 推理 / 轨迹全部在云端。这是
项目对外呈现的架构主线（见 [架构](#架构)）。

**内部训练形态（真实训练脚本）**：为了快速迭代训练链路，训练脚本
（`run_grpo_*_ucloud.sh`）把 agent runner 部署在训练机侧运行，其余环节
（Gateway 轨迹物化、TransferQueue 数据平面、verl 训练）与平台化形态完全
一致——**模型调用同样走云端 Gateway、轨迹同样云侧物化、on-policy 本质不
变**。两种形态共享同一套链路与脚本，仅 runner 部署位置不同；对外成果统一
计为平台化训练产出。

## 核心特性

### 平台化训推链路（agent 与执行分离）

- **双 harness**：白盒 mini-swe-agent（2.4.6）+ 黑盒 Claude Code（2.1.153），
  以及通用 `external_agent_runner`（任意 OpenAI 兼容 agent，改 `base_url` 即可）
- **云端 Gateway**：会话路由、OpenAI/Anthropic 协议适配、token 级轨迹物化、
  多链（chain）分片支持超长多轮会话（上限 = max_prompt+max_response）
- **沙箱执行**：腾讯云 Agent Runtime（E2B 兼容端点）、SWE-bench 托管镜像、
  实例级隔离；黑盒工具调用经自研 MCP 转发层（`sandbox_mcp_server.py`，
  手写 stdio JSON-RPC）落到远端沙箱
- **云端训练**：verl GRPO + LoRA（rank=32）+ FSDP2 + CPU offload +
  TransferQueue 数据平面（SimpleStorage / MooncakeStore 可切换），checkpoint
  续训（`resume_mode=auto`）

### 性能优化

- **EAGLE-3 投机解码**：生成吞吐 **+41.7%**（199→282 tok/s）、单 token 延迟
  -39.5%，25 步全样本对照训练质量无损
- **双机分离式全异步（separate_async）**：trainer 与独立 rollout 引擎分卡
  并行，采样与训练完全重叠，**每步墙钟 -39%**（79.4s→48.1s，3 步均值）
- **MooncakeStore 数据平面**：TCP 传输下与 SimpleStorage 无性能差异，全链路
  0 崩溃（双机 25 步正式训练验证）
- **LoRA 引擎热插**：每步权重同步 2.5s（adapter）vs 全量 15GB refit
- **Gateway 解析容错**：修复 5.3% 工具调用解析错误率（JSON repair + 合成重试）

### 稳定性（源码级排障）

修复 verl / uni-agent / TransferQueue 上游正确性 bug 共 23 项（见
`docs/修改与补丁汇总.md`），包括：LoRA merge 梯度未生效（verl#7014）、
EAGLE-3 logprobs 丢失（vllm#30059）、Mooncake `num_turns` 13B 写读类型不一致、
空响应轨迹 0 字节 slice、vLLM EAGLE-3 prefix-cache reset 竞态等。

## 架构

```
┌─────────────────────┐     ┌──────────────────────────────┐
│  Agent（任意位置）   │     │          云端                  │
│  白盒 mini-swe-agent │     │  ┌──────────┐  ┌───────────┐  │
│  黑盒 Claude Code   │────▶│  │ Gateway  │─▶│  vLLM/模型 │  │
│  外部自定义 agent    │ HTTP │  │轨迹物化  │  └───────────┘  │
└─────────┬───────────┘     │  └────┬─────┘                 │
          │ 工具调用转发      │       │ token-truth 轨迹       │
┌─────────▼───────────┐     │  ┌────▼───────────────────┐  │
│ 沙箱（腾讯云 E2B）   │◀────│  │ TransferQueue          │  │
│ 执行 / 测试 / reward │     │  │ (Simple/MooncakeStore) │  │
└─────────────────────┘     │  └────┬───────────────────┘  │
                            │  ┌────▼───────────┐  ┌─────┐ │
                            │  │ verl GRPO 训练 │─▶│ckpt │ │
                            │  │（node1 trainer）│  └──┬──┘ │
                            │  │（node2 rollout）│     │    │
                            │  └────────────────┘  ┌──▼──┐ │
                            │                      │合并/ │ │
                            │                      │评估  │ │
                            │                      └─────┘ │
                            └──────────────────────────────┘
```

## 训练结果

基准：HumanEvalFix（bigcode/humanevalpack Python 修复子集，161 条，
**无测试泄露**：训练期 agent 只看题目，`test_patch` 仅评估阶段使用）。

评测口径：n=1 / temperature 0.8 / 161 条全量 / 并发 16-24。

| 权重 | Harness | 训练方法 | 通过率 | 相对基座 |
|---|---|---|---|---|
| `Qwen3-8B`（基座） | — | 未训练 | **76.4%**（123/161） | — |
| `Qwen3-8B-final` | 白盒 mini-swe-agent | 单机 GRPO + LoRA（26 步） | **83.2%**（134/161） | +6.8pp |
| `Qwen3-8B-final-spec` | 白盒 + EAGLE-3 | 单机 GRPO + LoRA + 投机（25 步） | **82.61%**（133/161） | +6.2pp |
| `Qwen3-8B-final-blackbox` | 黑盒 Claude Code | 单机 GRPO + LoRA（25 步） | **80.75%**（130/161） | +4.35pp |
| **`Qwen3-8B-final-dual-async`** | 白盒 | **双机 separate_async + Mooncake + EAGLE-3（25 步）** | **83.23%**（134/161） | **+6.83pp** |

**关键结论**：

1. 双机全异步 + Mooncake + 投机在**与单机 baseline 完全同口径**下达到 83.23%，
   训练质量无损（off-policy 实际落后 2~4 轮 < 阈值 8）；
2. EAGLE-3 投机在不损失质量的前提下吞吐 +41.7%；
3. 黑盒（Claude Code）训练通过率 80.75%，验证双 harness 均可端到端训练；
4. 全链路 25 步正式训练 **0 硬错误**（无 13B / 空 slice / CUDA 崩溃 / 鉴权失败）。

## 快速开始

```bash
# 1. 获取代码与全部补丁
git clone https://github.com/zhenglianchi/uniagent-lighting.git
cd uniagent-lighting

# 2. 部署到训练机（见 docs/deployment.md，含裸机/镜像恢复两种路径）
bash scripts/install_ucloud_from_scratch.sh   # 或恢复镜像后执行补丁

# 3. 冒烟验证（单机，2 样本 × n2）
bash scripts/run_grpo_single_agentic_ucloud.sh

# 4. 正式训练（单机白盒，train161 / 5 epoch）
bash scripts/run_grpo_humanevalfix_ucloud.sh

# 5. 双机全异步 + Mooncake（trainer node1 + rollout node2）
source scripts/bootstrap_ray_env.sh           # ray start 前加载环境变量
bash scripts/run_grpo_dual_async_mooncake_ucloud.sh

# 6. 评估
bash scripts/eval_dual_async_final.sh          # 合并 LoRA + vLLM + 161 条全量
```

> ⚠️ **环境变量陷阱**：Ray worker 的环境变量在 `ray start` 时固定，不继承训练
> 脚本内的 export。`E2B_API_KEY` / `E2B_DOMAIN` / `GATEWAY_PORT` 必须在
> `ray start` **之前** export（`scripts/bootstrap_ray_env.sh` 一键完成），否则
> agent 起沙箱报 `AuthenticationException`。

## 安装（从裸机）

完整的分步指南见 **[docs/deployment.md](docs/deployment.md)**（版本链、镜像恢复、
裸机初始化、补丁应用、双机组网、环境变量、常见问题）。摘要：

| 步骤 | 内容 | 关键点 |
|---|---|---|
| 1 | 恢复/准备训练机（1×4090 48G 起步） | 推荐直接使用本项目已固化的镜像 |
| 2 | 基础环境 | Miniforge + swe-rl env（Python 3.10） |
| 3 | 版本链 | torch 2.9.0+cu128 / vllm 0.11.1 / verl 0.9.0.dev / ray 2.56.1 / TransferQueue 0.1.9 / mooncake-transfer-engine 0.3.12.post1 |
| 4 | 模型与数据 | `Qwen3-8B` 基座 + EAGLE-3 投机器；HumanEvalFix train161 |
| 5 | 代码与补丁 | clone 本仓 → `scripts/` 全部部署补丁幂等应用 |
| 6 | 腾讯沙箱 | `tencent_sandbox.env`（E2B 端点 + Cloud API 凭据），`bootstrap_ray_env.sh` |
| 7 | 冒烟 → 单机 → 双机 | 逐级验证（脚本见下） |

## 训练模式

| 模式 | 脚本 | 用途 | 关键参数 |
|---|---|---|---|
| 冒烟（纯 verl） | `run_grpo_smoke_ucloud.sh` | 链路验证 | batch2 / n=2 |
| 单机 LoRA | `run_grpo_single_lora_ucloud.sh` | LoRA + vLLM 共存 | rank=32 |
| agentic 单机 | `run_grpo_single_agentic_ucloud.sh` | 完整 agent 链路 | step_limit=60 |
| 全样本（白盒）* | `run_grpo_humanevalfix_ucloud.sh` | train161 / 5 epoch | batch32 / 并发64 |
| 投机解码 * | `spec_train_run.sh` | EAGLE-3 + LoRA merge | `LORA_MERGE=1` `SPEC_ON=1` |
| 黑盒全样本 * | `run_grpo_humanevalfix_blackbox_ucloud.sh` | Claude Code harness | max_turns=60 |
| 双机全异步 * | `run_grpo_dual_async_mooncake_ucloud.sh` | **separate_async + Mooncake + EAGLE-3（正式）** | 白盒 / batch32 / util 0.8 |
| 双机 colocate * | `run_grpo_multinode_async_ucloud.sh` | 对照实验 | colocate_async |
| 平台化测试（对外主线） | `run_grpo_platform_test_ucloud.sh` | 用户侧 agent 单步闭环 | save_freq=-1 |

\* 内部训练形态：runner 部署在训练机侧，链路与平台化形态一致（Gateway 轨迹
物化 / TQ 数据平面 / verl 训练不变），成果计为平台化训练产出。

## 评测

- `scripts/eval_humanevalfix.py`：161 条全量 agent 评测（真实 pytest reward）
- `scripts/eval_spec_final.sh` / `scripts/eval_dual_async_final.sh`：
  合并 LoRA → vLLM serve → 全量评估 + 对比输出
- 评估口径统一为 n=1 / temp 0.8，与训练同批次数据

## 仓库结构

```
uni_agent_ext/        # uni-agent 扩展包（腾讯沙箱后端 / 白盒/黑盒/外部三套 runner）
scripts/              # 数据构建 / 训练 / 评测 / 平台化 / 运维（含 bootstrap_ray_env）
patches/              # verl / uni-agent / TQ 补丁 + 幂等部署脚本（对应 23 项源码修改，可复现重建）
docs/                 # 架构 / 部署 / 训练评测分析 / 补丁汇总 / 简历亮点
work/logs/            # 训练轨迹与日志归档（均已解压，可读）：
                      #   humanevalfix_full_20260809（白盒 baseline 25 步轨迹）
                      #   blackbox_full_20260812 / blackbox_smoke（黑盒轨迹）
                      #   dual_async_20260815（双机轨迹）、spec_run（统计/日志）
                      #   swebench_early_20260804（早期单样本轨迹）
work/source_20260815/ # 服务器 uni-agent/verl 源码快照（含全部修改）
work/data/            # 数据集与脚本
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/deployment.md](docs/deployment.md) | **从裸机到全部训练**：环境/镜像/补丁/双机/评测/FAQ |
| [docs/architecture.md](docs/architecture.md) | 平台化架构设计（agent/Gateway/沙箱/训练分层） |
| [docs/训练评测分析.md](docs/训练评测分析.md) | 全量实验数据：baseline/spec/黑盒/双机对照、bug 根因 |
| [docs/修改与补丁汇总.md](docs/修改与补丁汇总.md) | 23 项源码修改的完整记录（原因/影响/验证） |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 演进规划 |

## 已知问题与修复

所有排障过程与根因分析已沉淀在 `docs/训练评测分析.md` §7-§8 与
`docs/修改与补丁汇总.md`，典型包括：

- **Mooncake `num_turns` 13B**：verl padding 行 Python int 走 msgpack 13B vs
  训练端 int64 8B 读 → `Buffer too small`；修复 `padding_utils.py`
- **空响应轨迹 0 字节 slice**：`max_trajectory_length` 截断占位写入 Mooncake 被
  拒；framework 跳过空轨迹
- **vLLM EAGLE-3 illegal memory**：util 0.6 触发 prefix-cache reset 竞态；
  恢复 util 0.8（baseline 口径）即稳定
- **Ray worker 不继承脚本 env**：`E2B_API_KEY` 等必须 ray start 前 export
- **checkpoint 磁盘累积**：`max_actor_ckpt_to_keep=1` 滚动保留 + 守护进程兜底
