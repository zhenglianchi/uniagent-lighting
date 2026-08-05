# agentlighting 式改造方案

## 背景

标准 verl/uni-agent GRPO 是 **on-policy 闭环**：rollout（agent 在沙箱生成轨迹）与训练都在云端训练进程内，本地环境不参与训练运行时。

本项目目标：改造成 **agentlighting 式异步架构**——本地采样/rollout 与云端训练通过"轨迹存储"解耦，让本地（mini-swe-agent + 腾讯云沙箱）成为 rollout 侧的运行时组件。

## 三档改造路线

### 方案 1：verl TransferQueue 解耦（正式改造目标）

- 把 verl 的 agent-loop worker（runner）从训练 Ray 集群拆到本地；轨迹经云端 TransferQueue 异步喂给 trainer
- 本地 agent 的模型调用指向云端 Gateway（出方向，走 80/443 或 SSH 隧道）
- 关键待验证：TransferQueue 跨机网络连接（controller/storage 的 ZMQ 地址）、异步策略滞后处理

### 方案 2：轨迹文件异步上传 + 云端重放训练（务实先行）

- 本地现有采样链路产轨迹 JSONL → `trajectory_uploader.py` 上传
- 云端自定义数据集读轨迹，用当前模型重算 logprob/reward 训练（离线/近似 GRPO，可作对照实验）

### 方案 3：混合

- 开发/测试期走本地采样验证；正式训练先跑通云端 on-policy；本地轨迹另做 SFT/对照
- 后期升级到方案 1

## 关键决策（2026-08-05）

- 训练基线：LoRA rank=32 / AdamW(fp32) / offload 关 / fused kernels 关 / 梯度检查点
- 沙箱：腾讯云 Agent Runtime（`uni_agent_ext.sandbox.tencent_agent_runtime`，E2B 兼容）
- 模型端点：训练时指向 verl Gateway；沙箱/本地访问云端走公网（80/443 或 SSH 隧道）
- 无测试泄露：`test_patch` 只用于 reward 评估，不注入 agent
- 续训：`resume_mode=auto` + `save_freq>0` + `default_local_dir`，中断后可续
