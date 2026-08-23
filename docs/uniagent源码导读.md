# uni-agent 源码导读

> 面向已完成训练侧（verl）学习、要深入 uni-agent 侧的读者。本文按
> `vendor/uni-agent/uni_agent/` 的目录逐块拆解，每块给出文件、关键类/函数与含义，
> 并在最后给出学习主线。配套「数据流与 Gateway 导读」见 `docs/数据流与网关导读.md`。

## 0. 定位与整体架构

uni-agent 是 **agent 编排层**：把任意 agent harness 接入、Gateway 模型端点、
会话/链轨迹物化、沙箱执行、任务与工具抽象统一起来，产出 **verl 可直接消费的
TQ 数据**。它不启动训练（训练入口永远是 `verl.trainer.main_ppo`），而是通过
`agent_loop_manager_class=AgentFrameworkRolloutAdapter` 注入 verl 的 rollout 阶段。

```
verl trainer（训练/消费 TQ）
   ↑ TransferQueue（SimpleStorage / MooncakeStore）
uni-agent framework（写 TQ：_write_session_trajectories_to_tq）
   ↑ agent runner（mini_swe_agent_runner / claude_code_runner / external_agent_runner）
uni-agent gateway（模型端点 + session/链轨迹物化）
   ↑ OpenAI/Anthropic 兼容调用
agent（mini-swe-agent / Claude Code / 任意外部 agent） + sandbox（执行环境）
```

目录速览：

| 目录 | 职责 |
|---|---|
| `framework/` | 训练侧对接层：adapter、AgentFramework、runner 分发、写 TQ |
| `gateway/` | 模型端点 + 会话/链 + 轨迹物化 + OpenAI/Anthropic 协议适配 |
| `agents/` | 内置 agent（mini_swe_agent / react / claude_code）与抽象 |
| `sandbox/` | 沙箱抽象与实现（docker/local/modal/vefaas/...） |
| `tasks/` | 任务定义（swe_bench / swe_rebench）与 reward |
| `tools/` | 工具抽象（shell / edit_file / submit / finish） |
| `logging/` | 会话/轨迹日志（LogContext、脱敏） |
| `utils.py` | 通用工具 |

## 1. framework/ —— 训练侧对接层

### 1.1 `framework/entry.py`（6720B）—— verl 插件入口

| 代码 | 含义 |
|---|---|
| `build_gateway_manager(config, llm_client)` (31) | 从 config 读 `custom.agent_framework`，组装 `GatewayManager`（gateway_count、tool_parser_name、tokenizer、apply_chat_template_kwargs 都来自这里） |
| `build_agent_framework(config, gateway_manager, ...)` (58) | 按 `framework_class_fqn` 加载 AgentFramework 子类（默认 `OpenAICompatibleAgentFramework`） |
| `AgentFrameworkWorker`（@ray.remote）(78) | Ray actor：持有 AgentFramework，`generate_sequences` 真正执行处 |
| `AgentFrameworkRolloutAdapter` (98) | **verl 加载的类**：`create()` 起 GatewayManager + AgentFrameworkWorker；`generate_sequences()` 整批丢给 framework_worker（fire-and-forget，返回 None） |

### 1.2 `framework/base.py`（529B）—— 抽象

`AgentFramework(ABC)`：只有两个方法——`from_config` 和 `generate_sequences(prompts)`。
这就是 uni-agent 对 verl 的契约面。

### 1.3 `framework/framework.py`（44225B）—— 核心执行体

`OpenAICompatibleAgentFramework`（276 行起）是默认实现，方法链（学习主线）：

| 方法 | 含义 |
|---|---|
| `generate_sequences(prompts)` (384) | 入口：取 `partition_id=train/val`、`global_steps`、`n`（GRPO 组大小）、`uid`，调 `_run_batch_to_tq`；全部失败会 raise |
| `_run_batch_to_tq(...)` (430) | 一个 batch 的编排：为每个 prompt 起 `n` 个 session，统计成功/失败/未完成 |
| `_run_prompt_sessions_to_tq(...)` (486) | 单个 prompt 的 `n` 个 rollout session 并发调度 |
| `_run_session_with_concurrency_limit(...)` (576) | 并发限流（`max_concurrent_sessions`） |
| `_run_session(...)` (635) | **单个 session 全生命周期**：建 session_id → `gateway_manager.create_session` → 起 agent runner（按 `dispatch_mode`）→ agent 经 Gateway 多轮交互 → `finalize` 拿轨迹 → 写日志 |
| `_write_session_trajectories_to_tq(...)` (878) | 把轨迹转 TQ 字段写入 TransferQueue（**含我们补丁：跳过空响应轨迹**） |
| `_trajectory_to_tq_field_and_tag(...)` (925) | 单条轨迹 → TQ 字段组装（prompts/responses/input_ids/loss_mask/global_steps...） |
| `_score_trajectories` / `_score_from_reward_info` (829/811) | reward 打分（runner 回报或 reward loop worker） |
| `_select_session_trajectories` (143) | 从 session 多链轨迹里挑选（mask_unfinished_episode 逻辑） |

`AgentRunner(Protocol)` (37)：**runner 契约**——`async __call__(*, raw_prompt, session,
sample_index, tools_kwargs, run_timeout)`。所有 runner（含我们 uni_agent_ext 里的三个）
都满足这个签名。

### 1.4 `framework/task_runner.py`（3874B）—— runner 分发

`run_task(...)` (17)：按 `dispatch_mode` 执行 runner（`ray_task` → Ray task / `inline_async` → 直接 await）；
`_post_reward_info` (75)：runner 算完 reward 后 POST 回 session。

### 1.5 `framework/multi_modal_postprocess.py`（3579B）

多模态输入后处理（图像转 embed/位置编码等），训练链路少用，了解即可。

## 2. gateway/ —— 模型端点 + 轨迹物化

### 2.1 `gateway/config.py`（2678B）—— 配置

`GatewayActorConfig`：tokenizer / processor / tool_parser_name / prompt_length /
max_trajectory_length / apply_chat_template_kwargs 等。**`max_trajectory_length` 是
链分片的容量上限**（你们脚本没显式传，用默认，之前分析过 8192/16384 的事故）。

### 2.2 `gateway/manager.py`（5239B）—— actor 池管理

`GatewayManager`：driver 侧持有，管理 `gateway_count` 个 GatewayActor：
`create_session` / `finalize_session` / `abort_session` / `shutdown`，
按 session_id 哈希选 gateway（`_select_gateway_index`）。

### 2.3 `gateway/gateway.py`（12517B）—— 模型端点服务

`_GatewayActor`（@ray.remote，55 行起）：

| 代码 | 含义 |
|---|---|
| `_register_routes` (91) | FastAPI 路由：`POST /v1/chat/completions`（OpenAI 兼容）+ Anthropic `/v1/messages` |
| `_handle_openai_chat_completions` (162) | OpenAI 协议请求 → `adapters/openai.py` 转内部格式 → `GatewaySession.run_generation` |
| `_handle_anthropic_messages` (188) | Anthropic 协议请求 → `adapters/anthropic.py` 转换 |
| `create_session` (238) | 起 GatewaySession |
| `finalize_session` (269) | 结束会话，**返回 Trajectory 列表**（轨迹物化的出口） |
| `set_reward_info` (264) | runner 把 reward 写回 session（轨迹带 reward 进 TQ） |

Gateway 通过 `llm_client`（verl 的 LLMServerClient）调 vLLM 引擎——这就是
“agent 的 OpenAI 兼容端点背后是 verl 引擎”的连接点。

### 2.4 `gateway/session/session.py`（36649B）—— 轨迹诞生地（重点）

| 代码 | 含义 |
|---|---|
| `TrajectoryBuffer` (36) | 一条轨迹的 token 级缓冲：prompt_ids / response_ids / response_mask / response_logprobs / generation_versions |
| `ChainState` (79) | “链”：多轮内容累积的载体 |
| `MaterializedChain` (94) | 物化后的链（→ 一条 Trajectory） |
| `GatewaySession` (168) | 一个 agent 会话的完整状态机 |
| `run_generation(request, backend)` (220) | 每轮生成入口：prepare → 调模型 → commit 到链 |
| `_prepare_generation_inputs` (386) | 编码消息、选链、**容量检查**（触发分链） |
| `_select_chain` (526) | 通过消息 hash 匹配已有链（续链/回滚） |
| `finalize` (339) | 会话结束：物化所有链 → 返回 Trajectory 列表 |
| `_assert_response_logprob_alignment` (639) | logprobs 对齐校验（配合我们 vLLM logprobs 修复） |

链分片机制：链总长（prompt+response）到 `max_trajectory_length` → 冻结物化成一条
轨迹 → 开新链继续，保证 60 轮长任务单条轨迹不超限（详见数据流导读 §2.1）。

### 2.5 `gateway/session/codec.py`（15475B）—— 编解码

`MessageCodec`：`encode_full` / `encode_incremental`（消息 → token 输入）、
`decode_response`（vLLM/SGLang 输出 → 消息/tool call）、工具调用解析
（`_process_tool_calls_vllm/sglang`）、**JSON 修复重试**（`_repair_tool_call_json` +
`_synthetic_retry_call`，对应我们 hermes 解析容错补丁）。

### 2.6 `gateway/session/types.py`（2988B）

`InternalGenerationRequest`（内部生成请求）、`SessionHandle`（**runner 拿到的会话
句柄：base_url / session_id**，agent 的模型端点就来自这里）、`Trajectory`。

### 2.7 `gateway/adapters/` —— 协议转换

| 文件 | 含义 |
|---|---|
| `openai.py` | `openai_to_internal`（OpenAI 请求 → 内部消息）、`openai_build_response` / `openai_stream_response` |
| `anthropic.py` | `anthropic_to_internal`（Claude 请求 → 内部，含多模态/工具消息折叠）、流式响应 |
| `types.py` | 共享类型 |

这就是“任意 OpenAI/Anthropic 兼容 agent 都能接”的实现层。

## 3. agents/ —— 内置 agent（独立运行用）

| 文件 | 含义 |
|---|---|
| `base.py` | `Agent(ABC)`：`run()` 抽象；`ModelConfig` / `AgentConfig` / `AgentResult` |
| `registry.py` | `register_agent` / `get_agent_cls` / `build_agent`（按 name 注册表加载） |
| `mini_swe_agent/agent.py` (1182B) | mini-swe-agent 封装（**注意：训练链路不走这里**，走 uni_agent_ext 的 runner） |
| `react/agent.py` + `model.py` | ReAct 循环示例 agent |
| `claude_code/agent.py` (10491B) | Claude Code 封装（独立运行版） |

理解要点：`agents/` 是“agent 抽象 + 内置实现”，主要给**独立任务运行**（tasks/）用；
**训练链路的 agent 在 `uni_agent_ext/agents/*_runner.py`**（满足 AgentRunner 契约，
由 framework 分发），两者不要混淆。

## 4. sandbox/ —— 执行环境抽象

| 文件 | 含义 |
|---|---|
| `base.py` | `SandboxBackend(Protocol)`（exec/exec_shell/read_file/write_file/upload/download/expose_port）+ `Sandbox`（start/stop/open_shell/`entered` 上下文，带重试） |
| `registry.py` | `register_sandbox` / `build_sandbox` |
| `docker.py` / `local.py` | 本地/容器沙箱 |
| `modal.py` / `vefaas.py` / `openyuanrong.py` | 云沙箱实现 |

**我们的腾讯沙箱在 `uni_agent_ext/sandbox/tencent_agent_runtime.py`**（E2B 兼容端点
直连，不走 swerex 隧道），通过 `sandbox/registry.py` 注册为可用后端。

## 5. tasks/ —— 任务定义

| 文件 | 含义 |
|---|---|
| `base.py` | `Task(ABC)`（run/build_sandbox/build_agent）、`TaskConfig`、`TaskResult` |
| `config.py` | `TaskConfigResolver`：yaml 任务配置加载与深度合并 |
| `registry.py` | 任务注册表 |
| `swe_bench/`、`swe_rebench/` | 上游任务示例（本项目训练链路不走 tasks/，reward 思路见 uni_agent_ext runner） |

理解要点：`tasks/` 是**独立运行（非训练）**的入口：定义任务 → 建沙箱 → 建 agent →
跑 → 评 reward。训练链路不用它（训练用 framework + runner），但 reward 思路
（pytest FAIL_TO_PASS、防测试泄露）与训练侧一致。

## 6. tools/ —— 工具抽象

| 文件 | 含义 |
|---|---|
| `base.py` | `Tool(ABC)`（schema/run/start/close）、`Toolbox`（from_specs/all）、`ToolResult` |
| `shell.py` | Bash/Shell 工具 |
| `edit_file.py` | 文件编辑工具 |
| `submit.py` / `finish.py` | 提交/结束工具 |

理解要点：工具是 agent 的动作空间（工具调用 → 沙箱执行 → 观察结果），schema
由 `build_function_schema` 生成给模型。

## 7. logging/ —— 可观测性

`LogContext`（会话级日志上下文）、`handlers.py`（文件/控制台 handler）、
`redaction.py`（敏感信息脱敏）、`session.py`。训练时每个 session 的
`framework.log` / `task.log` 就来自这里（`framework._run_session` 里创建）。

## 8. uni_agent_ext/ —— 我们的扩展（训练链路实际用的）

| 文件 | 行数 | 含义 |
|---|---|---|
| `agents/mini_swe_agent_runner.py` | 651 | **白盒 runner（内部形态）**：建腾讯沙箱 → 写 solution.py → 跑 mini-swe-agent（base_url=session.base_url，直连 127.0.0.1:8001）→ pytest reward → POST reward_info；同时导出 `build_mini_swe_config` / `create_task_sandbox` / `evaluate_reward` 给外部形态复用 |
| `agents/claude_code_runner.py` | 362 | **黑盒 runner**：沙箱内装 claude-code 2.1.153，MCP 工具转发（Bash/Read/Write/Edit） |
| `agents/external_agent_runner.py` | 165 | **平台本地 runner**：建沙箱 → 写 task.json → 轮询 done → 云侧 reward（不跑 agent） |
| `sandbox/tencent_agent_runtime.py` | 289 | 腾讯云 E2B 沙箱后端（`SandboxBackend` 实现） |

三个 runner 都满足 `AgentRunner` 契约（raw_prompt / session / sample_index /
tools_kwargs / run_timeout），在训练脚本里通过
`agent_runners.<name>.runner_fqn` + `dispatch_mode` 接线。

## 9. 关键横切概念

**Runner 契约**：`async def runner(*, raw_prompt, session, sample_index, tools_kwargs, run_timeout)`——
uni-agent 框架不关心你用什么 agent，只要满足这个签名，就能被 `ray_task` / `inline_async` 分发。

**轨迹生命周期**：

```
GatewaySession（链累积多轮）
  → finalize → Trajectory 列表（prompt/response/mask/logprobs + reward）
  → _write_session_trajectories_to_tq → TQ fields（prompts/responses/input_ids/loss_mask/global_steps）
  → trainer 经 ReplayBuffer 拉走
```

**我们的补丁位置**（对应模块）：`session.py`（固定端口、logprobs 对齐）、
`codec.py`（hermes JSON 修复重试）、`framework.py`（空响应轨迹跳过）、
`gateway/adapters`（协议容错）——详见 `docs/修改与补丁汇总.md`。

## 10. 建议学习主线

1. `framework/entry.py`（adapter 怎么起 Gateway + framework worker）→ 已学
2. `framework/framework.py` 的 `generate_sequences → _run_session → _write_session_trajectories_to_tq`（**核心数据流**）
3. `gateway/gateway.py` + `session/session.py`（模型端点 + 链机制，**轨迹诞生地**）
4. `gateway/session/codec.py`（消息编解码 + 工具调用解析）
5. `gateway/adapters/openai.py`（OpenAI 兼容入口，理解“任意 agent 接入”）
6. `uni_agent_ext/agents/mini_swe_agent_runner.py`（我们的 runner 怎么用 session 跑 agent）
7. 对照 `docs/数据流与网关导读.md` 的端到端图收尾

> 一句话总结：**uni-agent = Gateway（端点+轨迹）+ framework（分发+写 TQ）+ runner
> 契约（任意 agent 接入）+ sandbox/tasks/tools（执行与任务抽象）**，训练侧只认
> “generate_sequences + 输出进 TQ”这个契约。
