# 整体架构

## 1. 项目定位

**平台化代码智能体训推平台**：agent 强化学习训练被抽象成一条可复用的平台化
链路——agent 跑在**任意位置**（用户侧 / 本地 / 云端），模型调用统一指向**云端
Gateway**，Gateway 在服务模型的同时物化 token 级轨迹（on-policy），轨迹进入
云端训练引擎完成 GRPO（LoRA），产出权重再回到模型服务。**沙箱永远只是执行
环境**，不是 agent 的宿主。

与"把 agent 直接放进训练机 / 沙箱"的临时做法相比，平台化形态的工程价值：

- **on-policy 的保证来自 Gateway 云侧轨迹**（logprob 由云侧生成，外部不可
  伪造），而非 agent 的物理位置；
- **任意 OpenAI/Anthropic 兼容 agent 零改造接入**（改 `base_url` 指向
  Gateway 即可），训练数据流与 agent 实现解耦；
- **agent / 沙箱 / 训练 / 服务可独立扩缩**，训练机只承担计算，不绑定任何
  特定 harness。

## 2. 平台化架构（核心）

```
┌─────────────────────────── 用户侧 / 任意位置 ───────────────────────────┐
│  白盒 mini-swe-agent     黑盒 Claude Code + MCP     外部自定义 agent     │
│        │  (base_url → 云端 Gateway)                  │                   │
└────────┼────────────────────────────────────────────┼───────────────────┘
         ▼                                            │ 工具调用转发
┌─────────────────────────────── 云端 ─────────────────▼──────────────────┐
│  ┌──────────────┐  协议适配    ┌────────────────┐                       │
│  │ OpenAI/       │────────────▶│  Gateway        │                       │
│  │ Anthropic 端点│             │  会话路由        │                       │
│  │              │◀────────────│  轨迹物化(云侧)  │                       │
│  └──────────────┘             └───────┬────────┘                       │
│        ▲                             │ token-truth 轨迹                │
│        │ vLLM 推理                     ▼                               │
│  ┌─────┴───────────────┐   ┌──────────────────────────┐               │
│  │ 模型服务(合并权重)   │◀──│  TransferQueue           │               │
│  └─────────────────────┘   │  (Simple/MooncakeStore)  │               │
│                            └──────────┬───────────────┘               │
│                                       ▼                               │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │ verl GRPO 训练（node1 trainer + node2 rollout 独立引擎） │         │
│  └──────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                      ▲
          ┌───────────┴───────────┐
          │ 沙箱（腾讯云 E2B）     │  仅执行：代码 / 测试 / reward
          └───────────────────────┘
```

## 3. 分层设计

| 层 | 组件 | 职责 | 位置 |
|---|---|---|---|
| Agent 层 | mini-swe-agent / Claude Code / external_agent_runner | 决策循环、工具调用编排 | 任意位置 |
| 接入层 | Gateway（OpenAI/Anthropic 适配） | 协议转换、会话路由、**轨迹物化** | 云端 |
| 执行层 | 腾讯云 Agent Runtime（E2B 兼容） | 代码执行、测试、真实 reward | 沙箱 |
| 数据层 | TransferQueue（SimpleStorage / MooncakeStore） | 轨迹异步传输、KV 存储 | 云端 |
| 训练层 | verl GRPO（LoRA + FSDP2 + CPU offload） | 策略优化、checkpoint | 云端 GPU |
| 服务层 | vLLM（合并权重） | 模型推理服务 | 云端 GPU |

## 4. 双 harness 接入

### 白盒（mini-swe-agent）

- runner：`uni_agent_ext/agents/mini_swe_agent_runner.py`
- 接入方式：`base_url` 指向云端 Gateway session，零改造
- 轨迹：Gateway 云侧物化 prompt/response/logprob/mask/reward

### 黑盒（Claude Code）

- runner：`uni_agent_ext/agents/claude_code_runner.py`
- 接入方式：Claude Code 通过 `sandbox_mcp_server.py`（手写 stdio JSON-RPC，
  绕开 FastMCP 服务器拆包问题）将 Bash/Read/Write/Edit 转发到云端沙箱执行
- 模型调用：Anthropic 协议经 Gateway 适配到云端 vLLM

### 外部自定义 agent

- runner：`uni_agent_ext/agents/external_agent_runner.py`
- 任意 OpenAI 兼容 harness，`base_url` 指向 Gateway 即可；on-policy 由
  Gateway 云侧轨迹保证

## 5. 训练与推理链路（平台化接入形态）

### 运行形态（主线与内部训练）

- **平台化形态（主线，对外）**：agent 在用户侧 / 任意位置，经隧道或公网端点
  接入云端 Gateway；沙箱仅执行；轨迹 / 训练 / 推理全部云端。
- **内部训练形态（真实训练脚本）**：`run_grpo_*_ucloud.sh` 把 agent runner
  部署在训练机侧以加速迭代；Gateway 轨迹物化、TransferQueue 数据平面、verl
  训练与平台化形态完全一致，**on-policy 本质不变**（模型调用同样走云端
  Gateway，轨迹云侧物化）。对外成果统一计为平台化训练产出。

### 训练链路

```
verl.trainer.main_ppo（GRPO + LoRA）
  → AgentFrameworkRolloutAdapter（uni-agent 框架）
    → runner（mini_swe / claude_code / external）
      → Gateway session（token-truth 轨迹）
      → 腾讯沙箱执行（E2B）
  → TransferQueue → trainer 更新 → checkpoint → LoRA 合并 → 模型服务
```

### 推理链路

- **采样推理**：agent 调模型端点（正式期 = 云端 Gateway / vLLM server）
- **训练 rollout**：verl 内置 vLLM 引擎（双机形态下为 node2 独立引擎），
  权重随训练轮次更新

## 6. 关键设计

- 训练基座：Qwen/Qwen3-8B + LoRA（rank=32）+ FSDP2 + CPU offload
- rollout 优化：EAGLE-3 投机解码（+41.7% 吞吐）、LoRA 引擎热插（2.5s 权重
  同步）、Gateway 解析容错（5.3% 错误率修复）
- 双机全异步：separate_async（trainer node1 + rollout node2，采样与训练
  重叠，每步 -39%）+ MooncakeStore 数据平面
- 可靠性：checkpoint 续训（`resume_mode=auto`）、`max_actor_ckpt_to_keep=1`
  滚动保留 + 守护进程兜底、训练日志与轨迹归档

## 7. 状态与成果

平台化训推链路端到端验证并完成全样本正式训练（HumanEvalFix 161 条，
无测试泄露）：

| 权重 | 形态 | 通过率 |
|---|---|---|
| Qwen3-8B（基座） | — | 76.4% |
| Qwen3-8B-final | 平台化单机白盒 | 83.2% |
| Qwen3-8B-final-spec | 平台化 + EAGLE-3 | 82.61% |
| Qwen3-8B-final-blackbox | 平台化黑盒 | 80.75% |
| **Qwen3-8B-final-dual-async** | **平台化双机全异步 + Mooncake** | **83.23%** |

外部 agent 平台化闭环（用户侧 agent → 隧道 → 云端 Gateway → 轨迹 → 云侧
reward → GRPO）单步验证通过；双机正式训练全程 0 硬错误。
