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
- ✅ mini-swe-agent 训练 runner 骨架（`uni_agent_ext/agents/mini_swe_agent_runner.py`）
- ✅ 7.2 任务数据（`scripts/make_agentic_data.py`，schema 待上机对齐）
- ✅ vLLM 访问方案（`docs/vllm_access.md`：SSH 隧道走 22 端口 + `scripts/vllm_tunnel.sh`）
- ⏳ 7.3 agentic 训练配置（multi_turn + agent_framework）
- ✅ 7.3 agentic 训练脚本（`scripts/run_grpo_single_agentic_ucloud.sh`，待上机验证）
- ⏳ 轨迹异步上传 + 云端重放训练（方案 2）
- ⏳ TQ 解耦改造（方案 1，正式改造目标）
- ✅ 部署到训练机打通（uni_agent_ext + Python 3.10 兼容补丁 + 数据/凭据）
- ✅ 沙箱→Gateway 访问打通（沙箱内 SSH 隧道走 22，v0.5.0）

## 使用

详见 `docs/architecture.md` 与各脚本头部注释。
