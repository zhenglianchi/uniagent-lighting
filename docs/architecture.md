# 整体架构（2026-08-11：现行 + 目标形态，防跑偏锚点）

## 1. 项目定位

分布式**代码智能体强化学习平台**：mini-swe-agent 在代码沙箱里解决任务（多轮工具交互；
数据 = SWE-bench Lite → **HumanEvalFix，2026-08-06 定稿**，见 ROADMAP），用 GRPO
（LoRA 微调）训练模型，权重回传让"采样用的就是正在训练的模型"。

- **训练方式不是项目重点**：LoRA 保持轻量可扩展；
- **项目亮点在 rollout 侧优化**：LoRA 引擎常驻（adapter 热插）/ 投机解码（EAGLE-3，
  已实测 +41.7% 吞吐）/ 全异步（v1 colocate_async/separate_async，2026-08-11 定稿；
  **PD 分离已放弃**，详见 ROADMAP §2）；
- **改造目标**：把 uni-agent 改造成 agentlighting 式异步架构（见 §6）。

## 2. 物理部署（当前实际，2026-08-11）

| 角色 | 位置 | 职责 |
|---|---|---|
| 开发台 | 本地 WSL2 | 代码/脚本/数据准备；采样调试；GitHub 改造仓 |
| 训练机 node1 | UCloud 1×4090 48G / 94G（117.50.189.37，2026-08-08 新建） | verl+uni-agent、vLLM（训练 rollout）、runner 驱动 mini-swe-agent（harness 在训练机）、腾讯沙箱客户端、Gateway；全样本 baseline + 投机 run 都在此完成 |
| 沙箱 | 腾讯云 Agent Runtime | SWE-bench 实例（/testbed 执行环境），E2B 连接，跑完即毁 |
| node2 | UCloud 2×4090 24G（117.50.197.46） | 双机全异步目标机；**与 node1 不同 VPC，内网未通**，多机待网络就绪 |

## 3. 训练链路（agentic GRPO，全样本 26 步 + 投机 25 步已实测）

1. **数据**：SWE-bench Lite（→ HumanEvalFix，2026-08-06 定稿）→ `agentic_train/val.jsonl`
   （`raw_prompt + tools_kwargs{task/env/reward metadata} + reward_model.ground_truth`）
2. **训练入口**：`verl.trainer.main_ppo`（FSDP2 + LoRA），
   `rollout.multi_turn.enable=True` + `AgentFrameworkRolloutAdapter` + `agent_framework` 配置
3. **Gateway**：verl 起 Gateway actor（OpenAI 兼容端点，`/sessions/{id}/v1/chat/completions`），
   为每个 rollout 建 session 并**记录 agent 的模型调用（轨迹）**
4. **runner（uni_agent_ext.agents.mini_swe_agent_runner，ray_task 调度）**：
   - 建腾讯沙箱实例（`StartSandboxInstance` + E2B `Sandbox.connect`）
   - 生成本地 mini-swe config：环境类 `tencent_e2b` + `attach_instance_id`（连接已建实例）、
     `api_base = session.base_url`（本机 Gateway，无需隧道）、`model_name` = Gateway served model
   - **本地 subprocess 跑 mini-extra**（harness 在训练机；沙箱只是执行环境）
   - agent 多轮交互：命令发往沙箱 /testbed，模型调用走 Gateway session → 轨迹被记录
   - **真实 reward**：同沙箱写 `test_patch`（`git apply --3way` 回退 `patch -p1`）→
     跑 FAIL_TO_PASS 的 pytest → 分级打分 → POST `reward_info`
5. **GRPO 更新**：verl naive reward manager 读 rm_scores → LoRA 更新 → adapter 同步回 vLLM
   （~2s，引擎常驻）→ checkpoint（`resume_mode=auto` 可续训）→ 下一轮

训练配置（定稿）：LoRA rank=32 / AdamW(fp32) / offload 关 / **fused kernels 关**（与 LoRA 冲突）/
梯度检查点 / 全样本 batch=32 / mini=16 / micro=4 / n=4 / 并发 64 / lr=1e-5 / 5 epoch；
gateway 解析容错补丁（v0.31.8）已上机验证；投机 run 另加 `lora.merge=True` + EAGLE-3。

## 4. 两条推理链路（重要，勿混）

- **采样推理**：本地 mini-swe-agent 调模型端点（测试期=阿里云 API；正式期=云端 vLLM/Gateway）
- **训练 rollout**：verl 内置 vLLM + Gateway session；agent 的模型调用即训练侧 rollout，轨迹被记录

## 5. 关键决策与边界（防跑偏清单）

- **agent harness 在沙箱外**（思路 1.9）：沙箱只是执行环境，不把 agent 装进沙箱
- **agent 接入规范（2026-08-12 用户定序，勿图省事 ⛔）**：目标形态 = agent 跑在
  **用户侧/本地**（或任意位置），模型调用指向**云端 Gateway**（on-policy 只要求
  token-truth 轨迹由 Gateway 云侧记录，不要求 agent 在云端），沙箱永远只是执行
  环境。白盒 harness 放训练机、黑盒 claude 装沙箱均为"图省事"中间形态，**不作为
  终态**；Claude Code 工具本地化的远程执行适配是平台化必须解决的工程问题。
  当前黑盒正式训练跑完后，双机阶段即开始 §D 平台化（本地 agent 直连 Gateway +
  轨迹异步入库 + 双机全异步训练）。
- **无测试泄露**：`test_patch` 只用于 reward 评估，不注入 agent
- **腾讯云只用于沙箱**（HAI/COS 弃用）；训练在 UCloud；模型/数据/checkpoint 走 UCloud 本地
- **数据集（2026-08-06）**：换 HumanEvalFix（单函数修复，8B 60 轮内可出结果）；agent 不改
- **多机**：VPC 网络未通（node1/node2 不同 VPC）；网络就绪后只改并行参数 + Ray
- **改造仓约定**：每完成一项 commit + CHANGELOG + 语义化版本递增，推 main

## 6. 目标形态（agentlighting 式异步，改造方向）

- Algorithm ↔ **轨迹存储**（TransferQueue / 轨迹文件）↔ Runner（可分布在本地/多机）
- 当前：runner 在训练进程内（单机闭环已通）；目标：runner 拆出、轨迹异步进云端、
  训练消费（三档路线见 TODO，方案 1 = TQ 解耦为正式目标）
- **平台化终态（2026-08-12 定序）**：用户本地任意 agent（OpenAI 兼容 base_url）→
  云上公共 Gateway（会话 API + token-truth 轨迹）→ TQ 异步入库 → 云端 GRPO →
  checkpoint → 模型服务；agent 不在云端、沙箱只执行（详见 §5 接入规范）

## 7. 状态

- ✅ 单机 agentic GRPO 全链路（数据→沙箱→agent 轨迹→真实 reward→LoRA 更新）v0.15.1/v0.16.1
- ✅ HumanEvalFix 全样本 baseline（26 步，基座 76.4% → final 83.2%）+ 投机 run
  （25 步，82.61%，吞吐 +41.7%）2026-08-08~10
- ✅ gateway hermes 解析容错（方案 A）上机验证通过（2026-08-09）
- ✅ 黑盒 Claude Code runner（v0.35.1，腾讯 direct-URL）代码完成，待上机
- ⏳ 单机黑盒训练验证（下一步；复用 §3 链路，runner 换 claude_code_runner）
- ⏳ 双机全异步（VPC 就绪后第一优先；colocate_async → separate_async；PD 已放弃）
- ⏳ agentlighting 异步改造（方案 1，轨迹异步解耦）
