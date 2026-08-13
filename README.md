# uniagent-lighting

基于 verl / uni-agent 的**代码智能体训推平台**：白盒（mini-swe-agent）与黑盒
（Claude Code）双 harness 接入，agent 运行在用户侧或沙箱侧，模型调用经云端
Gateway 采集 token 级轨迹（on-policy，logprob 由云侧生成），腾讯云沙箱负责
执行，云端完成 GRPO（LoRA）训练并产出可服务权重。

## 平台化训练成果

以 HumanEvalFix（bigcode/humanevalpack Python 修复子集，161 条，无测试泄露）为
基准，三条训练路径均完成全样本 5 epoch（25-26 步）GRPO 训练：

| 权重 | Harness | 训练方法 | 评测（n=1 / temp 0.8 / 161 条） |
|---|---|---|---|
| `Qwen3-8B-final` | mini-swe-agent（白盒） | GRPO + LoRA(rank=32) | **83.2%**（134/161） |
| `Qwen3-8B-final-spec` | mini-swe-agent + EAGLE-3 | GRPO + LoRA + 投机解码 | **82.61%**（133/161） |
| `Qwen3-8B-final-blackbox` | Claude Code（黑盒） | GRPO + LoRA | **80.75%**（130/161） |

基座 Qwen/Qwen3-8B 评测 76.4%，三个变体均显著提升；成果由平台化训推链路产出，
平台化形态（用户侧 agent → 云端 Gateway）已验证可加载/继续这些权重。

## 核心能力

### 平台化训推链路（agent 与执行分离）

- **用户侧 agent**：任意 OpenAI 兼容 harness，`base_url` 指向云端 Gateway 即可
  接入，零改造；Claude Code 通过自研 MCP 工具转发层（`sandbox_mcp_server.py`，
  手写 stdio JSON-RPC）将 Bash/Read/Write/Edit 转发到云端沙箱执行
- **云端 Gateway**：会话路由、OpenAI/Anthropic 协议适配、token 级轨迹物化
  （prompt_ids / response_ids / response_mask / logprobs / rm_scores）
- **沙箱执行**：腾讯云 Agent Runtime（E2B 兼容），SWE-bench 托管镜像，
  实例级隔离
- **云端训练**：verl GRPO（LoRA + FSDP2 + CPU offload），TransferQueue 数据平面，
  checkpoint 续训（`resume_mode=auto`）

### 训练链路基础设施

- 白盒/黑盒双 runner（`mini_swe_agent_runner` / `claude_code_runner`）+
  外部 agent runner（`external_agent_runner`），支持 swewbench / humaneval_fix
  任务类型，真实 pytest reward（FAIL_TO_PASS，无测试泄露）
- 源码级排障：修复 verl/uni-agent 上游正确性 bug（LoRA merge 梯度未生效、
  EAGLE-3 logprobs 丢失等），8 个幂等部署补丁

### Rollout 性能优化

- **EAGLE-3 投机解码**：生成吞吐 +41.7%（199→282 tok/s），单 token 延迟
  -39.5%，25 步全样本对照验证训练质量无损
- **LoRA 引擎热插**：每步权重同步 2.5s（adapter）vs 全量 15GB refit
- **Gateway 解析容错**：修复 5.3% 工具调用解析错误率（JSON repair + 合成重试）
- **行为量化**：黑盒 vs 白盒（轮数/时长/吞吐/通过率）完整对比，定位长上下文
  prefill 平方级成本与 prefix cache 命中瓶颈

## 训练模式

| 模式 | 脚本 | 用途 | 关键参数 |
|---|---|---|---|
| 冒烟（纯 verl） | `run_grpo_smoke_ucloud.sh` | 链路验证 | batch2 / n=2 |
| 单机 LoRA | `run_grpo_single_lora_ucloud.sh` | LoRA + vLLM 共存验证 | rank=32 |
| agentic 单机 | `run_grpo_single_agentic_ucloud.sh` | 完整 agent 链路 | step_limit=60 |
| 全样本（白盒） | `run_grpo_humanevalfix_ucloud.sh` | train161 / 5 epoch | batch32 / 并发64 |
| 投机解码 | `spec_train_run.sh` | EAGLE-3 + LoRA merge | `LORA_MERGE=1` `SPEC_ON=1` |
| 黑盒全样本 | `run_grpo_humanevalfix_blackbox_ucloud.sh` | Claude Code harness | max_turns=60 |
| 双机全异步 | `run_grpo_multinode_async_ucloud.sh` | Trainer/rollout 重叠 | colocate_async |
| 平台化测试 | `run_grpo_platform_test_ucloud.sh` | 外部 agent 单步验证 | save_freq=-1 |

## 部署与使用

服务器部署步骤（版本链 torch 2.9.0+cu128 / vllm 0.11.1 / verl 0.9.0.dev /
ray 2.56.1）、补丁清单、数据准备、验证命令详见 `docs/deployment.md`；
架构设计见 `docs/architecture.md`；训练评测分析见 `docs/训练评测分析.md`；
规划见 `docs/ROADMAP.md`。

## 仓库结构

```
uni_agent_ext/       # uni-agent 扩展包（sandbox 腾讯后端 / 三套 runner）
scripts/             # 数据构建 / 训练 / 评测 / 平台化本地 agent / 运维脚本
patches/             # verl / uni-agent / mini-swe-agent 补丁
docs/                # 架构 / 部署 / 评测分析 / 规划 / 简历亮点
work/                # 数据 / 训练日志与轨迹归档
CHANGELOG.md         # 版本记录
```

## 许可

训练权重派生自 Qwen/Qwen3-8B（Apache 2.0）；代码遵循上游 verl / uni-agent
许可。
