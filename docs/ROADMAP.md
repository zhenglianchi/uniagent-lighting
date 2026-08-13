# Roadmap

## 1. 已完成：HumanEvalFix 全样本训练（三条路径）

数据：bigcode/humanevalpack Python 修复子集（161 条，无测试泄露）。

| 路径 | 结果 | 评测 |
|---|---|---|
| 白盒 baseline（26 步） | 基座 76.4% → **83.2%** | n=1 / temp 0.8 / 161 条 |
| 白盒 spec + EAGLE-3（25 步） | 吞吐 +41.7%，**82.61%** | 同上 |
| 黑盒 Claude Code（25 步） | **80.75%**（平台化通过率） | 同上 |

工具链：`make_humanevalfix_data.py`（数据构造 + 本地验证）、三套训练脚本、
`eval_humanevalfix.py`（并发评测）、`convert_verl_lora_to_hf.py`（权重合并）。

## 2. 黑盒平台化（agent 与执行分离）

- 组件：`external_agent_runner.py`（训练侧）、`sandbox_mcp_server.py`（MCP
  工具转发）、`platform_local_agent.py` / `platform_local_claude.py`（本地侧）
- 状态：白盒 / 黑盒单步闭环验证通过（reward 1.0、轨迹结构正确、权重未覆盖）
- 目标：任意 OpenAI 兼容 agent 零改造接入云端 Gateway，沙箱仅执行

## 3. 双机全异步（待网络就绪）

脚本：`run_grpo_multinode_async_ucloud.sh`（v0.35.0）。

- `colocate_async`（Trainer 与 rollout 同机重叠，官方 recipe 模式）先行，
  `separate_async`（训练机独立）实验性后测
- TransferQueue 跨节点数据平面（SimpleStorage）；Mooncake 不单跑（无 RDMA
  收益有限）
- 验收：step 墙钟 / 吞吐 vs 单机基线（实测 gen 53% + update 33%，理论
  1.6-1.8x）

## 4. 平台化完整实施（§D P0）

- 公共 Gateway + 会话创建 API（`POST /api/v1/tasks` → base_url/api_key）
- 轨迹异步入库 → 云端 GRPO → checkpoint → 模型服务
- 与双机全异步结合推进
