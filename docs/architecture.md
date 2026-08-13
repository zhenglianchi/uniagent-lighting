# 整体架构

## 1. 项目定位

分布式代码智能体强化学习平台：白盒（mini-swe-agent）与黑盒（Claude Code）
双 harness 接入，以 HumanEvalFix / SWE-bench 为任务基准，GRPO（LoRA）训练，
agent 与执行环境分离（平台化），训练结果可直接服务化。

## 2. 物理部署

| 角色 | 位置 | 职责 |
|---|---|---|
| 开发台 | 本地 WSL2 | 代码 / 脚本 / 数据准备；平台化本地 agent |
| 训练机 | UCloud GPU（CC≥8.0） | verl + uni-agent（Gateway / vLLM / runner）、训练、评估 |
| 沙箱 | 腾讯云 Agent Runtime | 任务执行环境（/testbed），E2B 连接，实例级隔离 |

## 3. 训练链路

```
verl.trainer.main_ppo（GRPO + LoRA）
  → AgentFrameworkRolloutAdapter（uni-agent 框架）
    → runner（mini_swe / claude_code / external）
      → 腾讯沙箱执行（E2B）
      → 模型调用 → Gateway session（token-truth 轨迹）
  → TransferQueue → trainer 更新 → checkpoint
```

关键组件：

- **Gateway**：会话路由、OpenAI / Anthropic 协议适配、token 级轨迹物化
  （prompt_ids / response_ids / response_mask / logprobs / rm_scores）、
  last-assistant rollback 与链前缀复用
- **Runner**：AgentRunner 协议扩展点，支持任务解析、沙箱工厂、reward 评估
- **Reward**：pytest FAIL_TO_PASS 真实打分（无测试泄露，test_patch 仅评估阶段）
- **TransferQueue**：verl v1 数据平面，轨迹异步消费

## 4. 推理链路

- **采样推理**：agent 调模型端点（测试期阿里云百炼；正式期云端 Gateway /
  vLLM server）
- **训练 rollout**：verl 内置 vLLM，权重随训练轮次更新

## 5. 平台化架构（agent 与执行分离）

```
用户侧 agent（OpenAI 兼容 / Claude Code + MCP）
  → 云端 Gateway（公共端点，token-truth 轨迹）
    → TransferQueue → 云端 GRPO 训练 → checkpoint → 模型服务
腾讯沙箱：仅执行
```

- on-policy 约束由 Gateway 保证（轨迹 logprob 云侧生成，外部不可伪造）
- Claude Code 经 `sandbox_mcp_server.py`（手写 stdio JSON-RPC）将工具转发到
  云端沙箱执行
- 任意 OpenAI 兼容 harness 零改造接入（`base_url` 指向 Gateway session）

## 6. 关键设计

- 训练基座：Qwen/Qwen3-8B + LoRA（rank=32）+ FSDP2 + CPU offload
- rollout 优化：EAGLE-3 投机解码（+41.7% 吞吐）、LoRA 引擎热插（2.5s 权重
  同步）、Gateway 解析容错、双机全异步（colocate/separate_async）
- 可靠性：checkpoint 续训（`resume_mode=auto`）、逐步统计 watcher、训练日志
  与轨迹归档

## 7. 状态

- 三条全样本训练路径完成并评估：白盒 83.2% / 投机 82.61% / 黑盒 80.75%
  （基座 76.4%）
- 平台化单步闭环验证通过（白盒 / 黑盒外部 agent）
- 双机全异步脚本就绪，待同 VPC 网络实测
