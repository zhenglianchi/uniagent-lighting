# Changelog

本项目约定：**每完成一项任务 commit 一次**，按语义化版本递增。

## v0.32.0（2026-08-09）

- **gateway hermes 解析容错上机验证通过**（v0.31.8 补丁，服务器 uni-agent commit
  `5cc88ec`）：train3 × 1 epoch（3 样本 × n=4 = 12 会话）**3 个 step 全部 4/4 会话
  成功（0 failed / 0 unfinished）**；15 次 JSONDecodeError 全部被 repair 救回
  （0 次 unrecoverable）；per-session 通过率 step1 0/4 → step2 1/4 → step3 3/4，
  与修复前基线一致 → 修复不改模型行为，只消除解析错误杀会话
- **投机解码 A/B 量化（阶段 0 纯推理）**：新增 `scripts/spec_bench_ab.py` +
  `scripts/spec_ab_run.sh`——Qwen3-8B + RedHatAI EAGLE-3 speculator，32 HumanEvalFix
  prompt × n=4 × max_tokens 512、temp 0.8、max_num_seqs 16、max_model_len 8192、
  util 0.7：**tok/s 730.65 → 1155.57（+58%），E2EL p50 21.68ms → 13.95ms（-36%），
  平均接受长度 2.285**（官方 HumanEval k=3 ≈ 2.39）
- `run_grpo_humanevalfix_ucloud.sh`：新增 `CKPT_DIR` 环境变量（默认 checkpoint 目录
  不变），独立验证/评测 run 可覆盖，避免 resume_mode=auto 续训污染
- 排障沉淀：vLLM 0.11 offline LLM 不接受消息列表（tokenizer 套 chat template 传文本）、
  metrics 字段为 first_token_ts/latency、EAGLE-3 多占 ~2G 需 max_model_len 8192 或
  util 0.7（util 0.5 + 默认 40960 时 KV 不足）

## v0.32.1（2026-08-09）

- **投机解码全样本训练 run 启动**（train161 / batch 32 / mini 16 / micro 4 / 并发 64 /
  vllm 128 / util 0.8 / 5 epoch / ckpt keep 1，与 2026-08-08 基线一致）：
  `run_grpo_humanevalfix_ucloud.sh` 新增 `LORA_MERGE` / `SPEC_ON` / `SPEC_DRAFT` /
  `SPEC_TOKENS` / `LOG_DIR` 环境变量（`lora.merge=True` 是 LoRA×SD 互斥的前置；
  Hydra 传 JSON 需单引号包字符串 + `+` 前缀）；启动脚本 `spec_train_run.sh`、
  清理脚本 `kill_train.sh`、统计 watcher `start_stats_watch.sh`
- **EAGLE-3 logprobs 全丢 bug 定位与修复**：`logprobs_spec_test.py` 验证 vLLM 0.11.1
  + EAGLE-3 下 `logprobs=0` 只返回 2/32 个（vllm#30059，v0.12 才修），verl
  `vllm_async_server.py` 恰好设 0 → 一行 `0→1`（`patches/verl_vllm_logprobs_spec_fix.patch`，
  服务器 verl commit 5fa045e），实测 logprobs=1/3 返回 32/32
- **step 1 实测**：reward mean 0.301（基线 0.247）、吞吐 301 tok/s（基线 213.7，
  +41%）、update_weights 53s（merge 全量 15GB refit 代价，可接受）、per-sample
  耗时 -33%；5 epoch 预计 ~18-19h

## v0.29.0（2026-08-08）

- 新增 `scripts/collect_grpo_stats.py`：**GRPO 训练逐步统计收集器**
  （stdlib only）——解析 verl 主日志的 step 指标行 + AgentFrameworkWorker 的
  `generate_sequences summary` + 会话日志逐条 `evaluate_reward`，每个 batch 落一行
  JSONL，字段含：rollout 数量（sessions/outputs）、rollout 时长（`timing_s/gen`）、
  训练时长（`timing_s/update_actor`）、step 总时长、reward（mean/min/max +
  per_session 明细）、advantage、num_turns、tokens、throughput、grad_norm 等
- 支持 `--watch` 常驻监听（训练中增量落盘，可 nohup 后台跑），供全样本 20 轮
  长训逐步记录使用

## v0.29.1（2026-08-08）

- `run_grpo_humanevalfix_ucloud.sh`：`trainer.total_epochs` 改为环境变量
  `TOTAL_EPOCHS` 可覆盖（默认 1），全样本长训（如 20 轮）直接
  `TOTAL_EPOCHS=20 bash run_grpo_humanevalfix_ucloud.sh`；`resume_mode=auto` 保持默认
  （长训中断可从 checkpoint 续）

## v0.30.0（2026-08-08）

- `make_humanevalfix_data.py`：
  - 新增 `--include-unverified`：本地验证失败（buggy 版死循环超时）的样本也保留进
    数据集，`metadata.verified=false / deadloop=true` 标记，训练时按 reward=0 计入指标
    （全样本 164 条 = 161 已验证 + 3 死循环：Python-160/10/156）
  - 正常样本 metadata 增加 `verified=true / deadloop=false`
- `run_grpo_humanevalfix_ucloud.sh`：新增环境变量覆盖
  `TRAIN_BATCH_SIZE`（默认 1）/ `PPO_MINI_BATCH`（默认 2）/
  `PPO_MICRO_BATCH`（默认 1）/ `MAX_CKPT_KEEP`（默认不设 = 不轮换），
  全样本跑法示例：
  `TOTAL_EPOCHS=10 TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH=8 PPO_MICRO_BATCH=2 \
   MAX_CKPT_KEEP=1 TRAIN_FILE=.../humanevalfix_train164.jsonl bash run_grpo_humanevalfix_ucloud.sh`
  （checkpoint 只保留最近 1 个 = 一直覆盖最新，磁盘稳定 ~16G）

## v0.30.1（2026-08-08）

- **采样提速**（全样本 164 条跑太慢的根因修复）：
  - `tencent_agent_runtime.py`：新增 `TENCENT_SANDBOX_SKIP_TMUX=1` 跳过沙箱内
    tmux apt-get 安装——该步骤经常等到 E2B 180s 请求超时，白耗每会话 ~3 分钟；
    mini-swe-agent harness 自带 pexpect shell，实际不需要沙箱内 tmux（会话一直能跑通
    即为证明）
  - `run_grpo_humanevalfix_ucloud.sh`：默认 `export TENCENT_SANDBOX_SKIP_TMUX=1`；
    新增 `VLLM_MAX_NUM_SEQS`（默认 4）与 `CONCURRENCY`（默认 4，原已有）配套调高，
    例如 `CONCURRENCY=8 VLLM_MAX_NUM_SEQS=8` → rollout 吞吐约 2×

## v0.30.2（2026-08-08）

- `run_grpo_humanevalfix_ucloud.sh`：新增 `VLLM_GPU_MEM_UTIL`（默认 0.5）可覆盖，
  16 路并发建议 0.7（更多 KV cache，避免 vLLM 排队）

## v0.30.3（2026-08-08）

- 新增 `patches/verl_debug_metrics_logprobs_guard.patch`：修复全样本 batch=16 崩溃
  `KeyError: 'rollout_log_probs' not found`——多轮批次中重试/失败的 episode 缺
  rollout logprobs，导致整个 batch 缺该键；debug 指标非必需，缺键时跳过并记 warning
- 实测：并发 16 + 跳过 tmux 后 rollout 提速明显（step1 的 64+ 会话 ~25 分钟完成），
  CPU 内存稳定 62G/94G；崩溃发生在 step 完成后 debug metrics 阶段（非 OOM）

## v0.30.4（2026-08-08）

- `collect_grpo_stats.py`：KV 解析支持 `np.int64(...)`（verl num_turns min/max 是这个
  类型，此前只匹配 float64 导致 min/max 为 null）
- 全样本首步实测（step 1，并发 16）：16 样本 / 64 会话 / 19 条 reward=1.0（30%），
  reward mean 0.247 / advantage≠0；每步 34 分钟（rollout 22min + update 8.3min + 存盘），
  吞吐 213.7 tok/s（并发 4 时代的 3.4×）；CPU 峰值 63.8G 稳定；100 步 ETA ≈ 56h
  （164 样本 × n=4 × 10 epoch = 6560 会话，会话数是硬瓶颈）

## v0.30.5（2026-08-08）

- **修复死循环样本 reward 评估挂死**：`mini_swe_agent_runner.py` 的 `_run_tests_batch`
  在 pytest 脚本里加 `SIGALRM` 超时（`timeout-20` 秒强退 exit 124）+ `asyncio.wait_for`
  兜底（超时返回 reward 0）。根因：死循环样本 buggy 版会让 pytest 无限挂起，而 E2B
  命令超时后请求本身还可能无限挂（tmux 同款问题），导致整个 step 卡死
- 实测：step 1/2 正常完成（reward mean 0.267 → 0.465，10 epoch 50 步 ETA ≈ 38h），
  step 3 被死循环样本 Python-10 的 reward 评估卡住 → 用补丁 + resume 从 global_step_2 续跑

## v0.30.6（2026-08-08）

- **修复 agent 工具调用/循环挂死**（step 3 复现：Python-10 死循环命令让 E2B 请求无限挂）：
  - `tencent_e2b.py` / `tencent_agent_runtime.py`：`commands.run` 显式传
    `request_timeout=timeout+60`，命令超时后 HTTP 请求也强退，工具调用必返回
    （tmux 同款"Request timed out"无限挂的根因）
  - `run_mini_swe_agent_api`：`asyncio.wait_for(run_timeout)` 硬上限，agent 循环
    卡死时由 runner 收尾并上报失败
- 计划：等 step 3 完成后按用户要求加并发（32→64）+ 降 `MSA_AGENT_MAX_TURNS`（60→40）
  从 checkpoint 续训；step 3 被挂死会话阻塞无法出 checkpoint → 改为从 global_step_2
  续跑并重跑 step 3（带修复 + 新参数）

## v0.31.0（2026-08-09）

- 全样本 10 epoch 训练提前停止，**以 global_step_26（5 epoch + 1 步）为最终权重**，
  checkpoint 改名 `checkpoints/humanevalfix/final`（max_ckpt_to_keep=1 轮换掉了
  恰好 5 epoch 的 step 25，26 与 5 epoch 差异仅 1 步）
- 新增 `scripts/convert_verl_lora_to_hf.py`：verl FSDP2+LoRA checkpoint（PEFT 风格
  base_layer + lora_A/B）→ 合并为标准 HF 模型目录（safetensors + config/tokenizer 从
  基座复制），供 vLLM 直接 serve
- 新增 `scripts/eval_humanevalfix.py`：**n=1 通过率评测**——并发沙箱（默认 16）、
  mini-swe-agent harness + 腾讯沙箱 + 隐藏测试 reward（无测试泄露），输出 per-sample
  JSON + pass rate；用于"基座 vs final"对比（方案 A）
- 训练实测（10 epoch 中断于 26/50）：epoch 平均 reward 0.378 → 0.539 → 0.590 →
  0.657 → 0.699 →（epoch 6 起步 0.66）；最终权重 = step 26（reward mean 0.659）

## v0.31.6（2026-08-09）

- **方案 A 评测完成（n=1，温度 0.8，161 条，并发 24，与训练一致的 vLLM 参数）**：
  - 基座 Qwen3-8B：**123/161 = 76.4%**
  - final（step 26 LoRA 合并）：**134/161 = 83.2%**
  - 净提升 **+6.8pp**：26 条由失败转通过，15 条由通过转失败（单样本 temp 0.8 有
    采样噪声，净增 11 条）；per_test 全部为真实 pytest PASS
- **重要发现**：训练 step 1 的 reward mean（0.289）明显低估基座真实水平——同一批 32 样本
  在评测中通过率 81%；训练早期 gateway hermes 工具解析报错等链路问题压低了 rollout 成功率；
  最终模型 83.2% vs 基座 76.4%，RL 提升真实存在但幅度小于训练指标曲线给人的印象
- 评测工具链：`convert_verl_lora_to_hf.py`（verl FSDP2+LoRA → 合并 HF 模型）、
  `eval_humanevalfix.py`（并发沙箱 n=1 通过率评测）；腾讯沙箱并发实测上限 ~25
  （50 核配额 / 每沙箱 2 核），评测用 24

## v0.31.7（2026-08-09）

- 新增 `docs/训练评测分析.md`：全样本训练（26 步）+ 评测完整分析——逐步 reward/时长/
  吞吐表、时间效率提升（gen -31%、update -43%、轮数 -49%）、训练 vs 评测通过率
  对照（基座 76.4% / final 83.2%）、gateway 双解析根因量化（1,601 错误事件 /
  5.3% 调用 / 38% 会话受影响）与修复方案

## v0.31.8（2026-08-09）

- **gateway 工具解析容错（方案 A，代码已实现 + 本地逻辑验证通过，待服务器单步验证）**：
  - 新增 `patches/gateway_hermes_parse_guard.patch`（作用于
    `uni_agent/gateway/session/codec.py`）
  - 修复点：`_process_tool_calls_vllm` 解析 `<tool_call>JSON</tool_call>` 失败时，
    不再静默返回"无工具调用"文本（那会让 agent 误以为对话结束 → reward=0），而是：
    ① `_repair_tool_call_json` 轻量修复（去尾逗号、补相邻字符串间缺逗号、
      转义字符串内裸换行）后重试；② 仍失败则 `_synthetic_retry_call` 注入一个
      "重试提示"工具调用（无害 echo + JSON 错误说明），把错误反馈给 agent 让它重试
  - 本地单元验证：尾逗号 / 缺逗号 / 裸换行全部修复成功，合法输入零破坏
  - 待服务器验证：应用补丁后跑 1 个 step（3~8 样本），解析错误应趋近 0、
    per-session 通过率应回到 ~80%（对照：修复前 5.3% 调用报错 / 38% 会话受影响）

## v0.28.3（2026-08-08）

- `make_humanevalfix_data.py`：任务提示词增加 **heredoc 约束**——修复仅改 bug 逻辑、
  不要重写 docstring/未改动代码；必须整文件重写时用**一条 heredoc**
  （`cat > /testbed/solution.py <<'PYEOF' ... PYEOF`），禁止 `echo ... >> solution.py`
  逐行拼接
- 背景（阶段一首跑实测，8 样本 GRPO 单机 48G）：链路与工具调用格式**正常**
  （hermes parser 解析成功、bash 工具全部执行成功），但 **Qwen3-8B 行为退化是唯一瓶颈**：
  用 `echo ... >> solution.py` 逐行重建整个文件，13+ 轮连 docstring 都没写完，中途引号
  报错重试、两次触发 "No tool calls found"，最终 solution.py 语法不完整 → 4/4 reward=0
- 处理：按 TODO §8 规则止损停训 → 提示词修复 → 先跑 3 条（Python-61/104/105，qwen3.7-plus
  全过的对照组）快速验证 8B 是否有可训练 reward

## v0.28.2（2026-08-06）

- `run_grpo_humanevalfix_ucloud.sh`：`MODEL / TRAIN_FILE / VAL_FILE` 改为环境变量可覆盖
  （默认路径不变），便于阶段一扩冒烟数据（5~8 条）直接换数据跑
- 路线定稿（写入 TODO §8）：阶段一单机 48G+96G 验证 Qwen3-8B 有可训练 reward
  （验收：部分样本 reward>0、advantage≠0）→ 阶段二双机 24G+96G 多机测试
  （同 VPC/子网硬前提，offload=True + 16G swap）

## v0.28.1（2026-08-06）

- 新增 `scripts/run_humanevalfix_local.py`：本地开发用冒烟采样脚本
  （腾讯 E2B `code-interpreter-v1` 沙箱 + 阿里云百炼 API，不走 Gateway/训练机），
  复用单个沙箱实例逐样本重置 `/testbed`，轨迹落 `work/swebench/` + 汇总 JSON
- runner 修复（对正式训练路径同样生效）：
  - `SessionHandle` 改为守卫导入：无 ray 环境也能 import runner 纯函数部分
  - `run_mini_swe_agent_api` 先在主线程预导入 `tencent_e2b`
    （模块顶层 `signal.signal()` 只能主线程执行，否则 to_thread 里首次导入报
    `ValueError: signal only works in main thread`）
  - `get_agent(..., default_type="default")`：模板 yaml 无 `agent_class` 时
    `get_agent_class("")` 报 "Unknown agent type"
  - agent 失败返回附带 `traceback.format_exc`，便于定位
- 实测（qwen3.7-plus，3 条 train 样本全过）：交互轮数 **7 轮**（读码→复现→改→验证→提交，
  每轮 1 个 bash 调用），单条 35~43s，reward=1.0 / resolved=true；
  轨迹：`work/swebench/humanevalfix_humanevalfix-Python-{61,104,105}.traj.json`

## v0.1.0（2026-08-05）

- 初始化仓库：README / CHANGELOG / 架构文档
- 引入 `uni_agent_ext` 扩展包：
  - 腾讯云 Agent Runtime 沙箱后端（`sandbox/tencent_agent_runtime.py`）
  - mini-swe-agent 训练 runner 骨架（`agents/mini_swe_agent_runner.py`）
- 引入关键脚本：冒烟数据构建、单机 LoRA GRPO 训练、轨迹上传器、本地采样
- 记录：verl 续训机制（`resume_mode=auto` + `save_freq` + `default_local_dir`）

## v0.2.0（2026-08-05）

- 实测云端公网端口：仅 22 开放 → 新增 `docs/vllm_access.md`（SSH 隧道方案）+ `scripts/vllm_tunnel.sh`
- 新增 `scripts/make_agentic_data.py`（7.2 任务数据：SWE-bench Lite → raw_prompt + tools_kwargs，schema 待上机对齐）

## v0.3.0（2026-08-05）

- 新增 `scripts/run_grpo_single_agentic_ucloud.sh`（7.3 agentic 训练配置）：
  multi_turn.enable + AgentFrameworkRolloutAdapter + agent_framework.agent_runners
  （runner_fqn=uni_agent_ext.agents.mini_swe_agent_runner，dispatch=ray_task，
  concurrency 可控沙箱成本）；reward 走 naive manager（TQ rm_scores）；续训已启用
- 对齐官方 quickstart 接线（train_qwen3p5_dense.sh），待上机验证 TOOL_PARSER 与数据 schema

## v0.4.0（2026-08-05）

- 部署实测：uni_agent_ext 部署到训练机（tar + .pth 进 PYTHONPATH）+ agentic 数据 + 腾讯沙箱凭据
- **修复 uni-agent Python 3.10 兼容 bug**：`typing.NotRequired`（3.11+）→ typing_extensions 回退
  （`patches/uni_agent_py310_compat.patch`）；runner import 验证通过
- 新增 `docs/deployment.md`（部署步骤）

## v0.5.0（2026-08-05）

- 实测训练机公网仅 22 开放（3389/80/443/8000 均 filtered）→ **沙箱内 SSH 隧道方案落地**：
  runner 增加 `ensure_gateway_tunnel()`（注入专用密钥 → 沙箱内 `ssh -L` 转发 Gateway 端口 →
  轮询就绪 → 返回沙箱本地 api_base）；训练脚本加 `MSA_GATEWAY_SSH_HOST` 环境变量
- 专用隧道密钥：`~/.ssh/gateway_tunnel_key`（本地）+ 训练机 `/home/ubuntu/.ssh/gateway_tunnel_key`，公钥入 authorized_keys

## v0.6.0（2026-08-05）

- **修复 .pth 路径 bug**：内容应为包父目录 `/home/ubuntu`（而非 `/home/ubuntu/uni_agent_ext`）；
  `ray_import_test.py` 验证 MAIN-OK + RAY-WORKER-OK
- 训练机补装沙箱 SDK：`e2b-code-interpreter==2.9.0` + `tencentcloud-sdk-python-ags`（模块路径 v20250920）
- agentic 冒烟成本控制：train_batch_size 2→1、max_prompt_length 2048→4096（防过滤）

## v0.7.0（2026-08-05）

- agentic 训练脚本 source `tencent_sandbox.env`（修复 runner 建沙箱时 E2B_API_KEY 缺失）

## v0.8.0（2026-08-05）

- 补 E2B 环境映射：`E2B_API_KEY=${TENCENT_SANDBOX_E2B_TOKEN}` + `E2B_DOMAIN=ap-guangzhou.tencentags.com`
  （凭据文件只有 TENCENT_* 前缀，需显式映射后才被 e2b SDK 识别）

## v0.9.0（2026-08-06）

- **腾讯沙箱 SWE-bench 实例接入**：`tencent_agent_runtime` 对 `sweb.*` 镜像改走
  Cloud API `StartSandboxInstance`（ToolName=swebench-v1、ImageRegistryType=system、
  自动补 `swebench/` 前缀）+ E2B `Sandbox.connect(InstanceId)`；stop 时 E2B kill +
  `StopSandboxInstance` 双保险；非 sweb 镜像仍走 E2B template 路径

## v0.11.0（2026-08-06）

- **首条 agentic 链路全通验证**（沙箱实例→文件写入回退→隧道→runner→reward→销毁）：
  卡在 mini-extra rc=127（沙箱未装 mini-swe-agent）
- `MSA_INSTALL_AGENT=1`（沙箱内 pip install mini-swe-agent）；安装命令去掉 conda source
  activation（非交互 shell 不可靠）

## v0.12.0（2026-08-06）

- **Agent 部署形态修正（对齐思路 1.9，防跑偏）**：mini-swe-agent harness 在**训练机本地**驱动，
  沙箱只是执行环境（不再在沙箱内装/跑 agent——黑盒模式是 claude-code 那类自包含 CLI 的形态）
- runner：建沙箱 → 隧道 → 取 instance_id → **本地 subprocess 跑 `mini-extra swebench-single`**
  （environment_class=tencent_e2b，`attach_instance_id` 连接已建实例）→ 同沙箱 reward → 上报
- `tencent_e2b` 环境类新增 attach 模式（跳过 StartSandboxInstance 直接 connect）；
  `tencent_agent_runtime` 暴露 `instance_id` 属性
- 思路.md 新增 1.9「Agent 部署形态」

## v0.13.1（2026-08-06）

- mini-swe-agent tencent_e2b **attach 模式 cleanup 不销毁实例**（生命周期归 runner，
  否则 mini-extra 退出会停实例，reward 写 test_patch 报资源不存在）；部署文档 §6 记录
- agentic 链路实测进度：沙箱实例→隧道→本地 mini-extra（attach）→agent 运行 4.5min→
  reward 阶段被此坑拦截，已修

## v0.14.0（2026-08-06）

- **修正隧道方向**：harness 在训练机本地，直接调 `session.base_url`（本机 Gateway），
  无需沙箱内隧道（`MSA_GATEWAY_TUNNEL=0`）——此前隧道（沙箱→node2）是黑盒模式遗留，
  导致模型调用打到 node2 空端口 → 空轨迹
- runner 用真实 `model_name`（runner_kwargs 注入的 Gateway served model），不再写死 "default"

## v0.15.0（2026-08-06）

- **修复 LiteLLM Missing credentials**：mini-swe-agent 走 LiteLLM/OpenAI provider 必须带
  api_key（Gateway 接受任意非空值，用 "EMPTY"）；配置 model_kwargs + 子进程
  `OPENAI_API_KEY=EMPTY` 双保险——此前 1 次 API call 直接 Missing credentials → 空轨迹

## v0.15.1（2026-08-06）

- **验证通过**：单机 agentic GRPO 全链路跑通——`num_success_sessions=2 / outputs=2 / failed=0`，
  Training Progress 100% 2/2，agent 5 轮多轮轨迹，LoRA 更新 + adapter 同步 + checkpoint 全执行

## v0.16.0（2026-08-06）

- **真实 SWE-bench reward**：健壮测试列表解析（兼容 list/JSON/换行/逗号/字符级乱码）、
  test_patch `git apply --3way` + `patch -p1` 回退、分级打分（通过数/总数）、可选 PASS_TO_PASS、
  可配 testbed python 与超时

## v0.16.1（2026-08-06）

- **实测修复并验证**：FAIL_TO_PASS 经 verl tensordict 序列化到达 runner 时是**字符级列表**
  （JSON 字符串被逐字符拆散）→ runner 防御性合并后重解析，还原真实测试名；
  实测输出 `tests/unittest_nodes.py::AsStringTest::test_as_string_unknown: FAIL`（评分 0，agent 未修出补丁）

## v0.17.0（2026-08-06）

- **修复 agent 只交互 5-6 轮（RepeatedFormatError）的代码层根因**：
  - uni-agent gateway codec 引用的 `vllm.tool_parsers` 在 vllm 0.11.1 不存在 →
    工具调用解码失败 → 响应无 tool_calls → mini-swe-agent 连续 3 次格式错误退出；
    补丁：改从 `vllm.entrypoints.openai.tool_parsers` 导入（带旧版回退），
    见 `patches/uni_agent_vllm0111_toolparsers.patch`
  - TOOL_PARSER 由 hermes 改为 **qwen3_coder**（匹配 Qwen2.5-Coder 工具格式）
- 待验证：若模型仍不发工具调用（写散文），说明 tools 未渲染进 prompt，需继续修 backend 层

## v0.18.0（2026-08-06）

- **scripts/ 全量归档**：补齐此前只在本地/服务器上的有效脚本（24 个），覆盖四类：
  - UCloud 部署/升级：`install_ucloud_from_scratch.sh`（最终版本链 torch 2.9+cu128 / vllm 0.11.1 直装）、
    `upgrade_vllm_0111.sh`、`fix_strenum_ucloud.sh`、`patch_verl_ipc_cpu.py`、`fix_otel.sh`
  - 多机/集群（node1 备用，含 UCloud 版）：`run_grpo_multinode_ucloud.sh`、`run_grpo_dualgpu_ucloud.sh`、
    `fix_multinode_hosts.sh`、`nccl_multinode_test.py`、`ray_cluster_setup/join/restart.sh`、`setup_ssh_trust.py`
  - 腾讯沙箱工具：`tencent_create_sandbox_tool.py`、`tencent_list_sandbox_tools.py`、`tencent_start_swebench.py`、
    `tencent_sandbox_demo.py`、`run_tencent_sandbox_demo.py`
  - 运维/采样：`ssh_ucloud.py`、`proxy.sh`、`cc_connect.sh`、`start_sampling.sh`、`ssh_poll_node1.py`、`run_grpo_smoke_ucloud.sh`
- **凭据安全修复**：`setup_ssh_trust.py` 与 `check_node1.py` 曾硬编码旧机器明文密码 → `check_node1.py` 不入仓（废弃），
  `setup_ssh_trust.py` 改为从 `SSH_PASS`/`NODE1_IP`/`NODE2_IP` 环境变量读取，仓库内不再含任何明文凭据
- 不归档（已弃用）：HAI 时代脚本（`ssh_hai.py`/`setup_hai_uniagent.sh`/`run_grpo_multinode.sh`/`run_grpo_single_node.sh`）、
  vllm 0.8.5 兼容补丁（`fix_vllm085_compat.py`）、旧 StrEnum 脚本（`fix_strenum.sh`）

## v0.18.1（2026-08-06）

- **node2 部署 git 化（已上线）**：`/home/ubuntu/uniagent-lighting` = 仓库 clone（v0.18.0），
  `uni_agent_ext` 与 `swe-rl` 下全部脚本改为指向仓库的软链（旧拷贝备份为
  `uni_agent_ext.bak-20260806`）；import 验证通过（runner / reward env / 腾讯沙箱适配器）
- 更新 `docs/deployment.md`：新增「仓库管理」章节，更新流程 = 本地 commit+push → 服务器
  `git pull`；swe-rl 仅保留非仓库内容（凭据/数据/checkpoint/日志）

## v0.19.0（2026-08-06）

- **README 新增「从零部署」完整流程**：新机器从环境安装（`install_ucloud_from_scratch.sh`）
  到跑通 agentic GRPO 训练的 7 步指南 + 常见排坑 + 多机可选分支
- **补齐部署缺口**：`patches/tencent_e2b.py` 入仓（训练机 mini-swe-agent 必打的
  tencent_e2b 环境补丁，此前只在本地 mini-swe-agent 源码里，新机器无法获取）；
  部署流程第 4 步含安装 mini-swe-agent 2.4.6 + 覆盖补丁的命令
- README「当前进度」同步最新状态（agentic 全链路已验证、部署 git 化）

## v0.20.0（2026-08-06）

- **训练基座切换 Qwen/Qwen3-8B**（Qwen2.5-Coder-7B 不支持标准 function calling，
  导致 agent 只写散文、reward 恒 0、LoRA 不更新）：
  - `run_grpo_single_agentic_ucloud.sh`：MODEL=/home/ubuntu/models/Qwen3-8B、
    **TOOL_PARSER 改回 hermes**（Qwen3 官方推荐工具格式）、新增
    `++data.apply_chat_template_kwargs.enable_thinking=false`（gateway codec 渲染时关 thinking）
  - 全部训练/冒烟/安装脚本模型路径同步改为 Qwen3-8B
    （`run_grpo_smoke/multinode/single_lora/dualgpu_ucloud.sh`、`upgrade_vllm_0111.sh`、
    `install_ucloud_from_scratch.sh` 模型下载）
- 本地 + node2 已删除 Qwen2.5-Coder-7B-Instruct（15G×2），Qwen3-8B（15.27GiB）已下载并验证

## v0.20.1（2026-08-06）

- **修复腾讯沙箱采样无 /testbed**：pip 官方版 `swebench.py` 的镜像注入列表不含
  `tencent_e2b`（只有 docker/swerex_modal），导致实例启动时 image 为空、沙箱内无代码库。
  归档本地补丁版 `patches/miniswe_swebench.py`，README 部署第 4 步与 deployment.md §6
  补充覆盖该文件
- node2 已验证：补丁后沙箱实例正常注入 SWE-bench 镜像，agent 开始读代码

## v0.20.2（2026-08-06）

- **修复 gateway hermes 工具解析静默失败**：uni-agent codec 里
  `vllm.entrypoints.openai.chat_completion.protocol` 在 vllm 0.11.1 不存在
  （已移到 `openai.protocol`），导致 `_process_tool_calls_vllm` 抛 ImportError 被
  吞掉、直接返回原始 XML 文本 → mini-swe-agent "No tool calls found" →
  RepeatedFormatError（Qwen3-8B 输出是正确的 `<tool_call>` JSON）
- 更新 `patches/uni_agent_vllm0111_toolparsers.patch`（含两处 import 回退）；
  node2 已应用并验证：`_process_tool_calls_vllm` 解析出 `('bash', '{"command": "echo hi"}')`

## v0.20.3（2026-08-06）

- `run_grpo_single_agentic_ucloud.sh` 增加 `actor_rollout_ref.rollout.max_model_len=8192`：
  Qwen3-8B 最大上下文 40960，vLLM 按此预留 KV cache 会超显存（prompt+response 实际只需 8192）

## v0.20.4（2026-08-06）

- `actor_rollout_ref.rollout.load_format=safetensors`：HYBRID 引擎默认 dummy 加载，
  首次权重同步峰值（dummy+真实+FSDP）超 48GB；改为直接加载真实权重

## v0.20.5（2026-08-06）

- FSDP2 **CPU offload 全开**（offload_policy/param_offload/optimizer_offload=True）：
  解决 Qwen3-8B + HYBRID 引擎在 48GB 卡上的 OOM（实测训练显存峰值 15.7GB、CPU ~62GB）；
  LoRA 场景基座权重放 CPU 是安全取舍，训练吞吐下降但验证可接受
- **单机 agentic GRPO 完整跑通（Qwen3-8B）**：agent 真实工具调用 8~62 轮 →
  真实 SWE-bench reward → GRPO step 2 完成 → checkpoint 保存（global_step_1/2）
- ⚠️ 遗留：两条冒烟样本 reward 全 0 → advantage 全 0 → LoRA 未更新
  （`lora_B` 全 0、step1/2 权重逐字节相同）。链路已通，缺的是**正样本/学习信号**

## v0.23.0（2026-08-06）

- GRPO `n=2 → 4`：8B 对同一任务失败模式一致（reward 组内恒等），更大组提高
  组内差异概率（制造 advantage 非 0 的信号）

## v0.23.1（2026-08-06）

- **参数化测试名传递修复**：pytest 测试名经 shell `$(...)` 展开会被分词/转义破坏
  （如 `test_string_format_uninferable["I\ns"]`），改为 python 脚本列表传参
- **发现官方 SWE-bench Lite 数据集缺陷**：`pylint-dev__astroid-1866` 的
  PASS_TO_PASS 在官方数据里被截断（`test_string_format_uninferable["I` 处），
  该样本评估注定全 ERROR → 换样本

## v0.22.x（2026-08-06）

- v0.22.0：修复 runner 并发 session 共用固定 `/tmp/mini_swe_config.yaml` 导致
  agent attach 错沙箱/配置串扰（改 session 唯一路径）
- v0.22.1：**核心 reward 解析 bug**——`pytest -q` 输出点号进度条而非每测试一行，
  正则解析失败 → score 恒 0（真实 21 测试 20 过 = 0.9523）；改 `-v`
- v0.22.2：reward_info 键名不匹配——runner 上报 `reward_score`，framework 消费
  `reward` → rm_scores 恒 0；补 `reward` 键

## v0.26.0（2026-08-06）

- **回滚极简任务实验（simple-bench）**：删除 `scripts/run_simple_bench.py` /
  `scripts/make_simple_data.py` / runner 的 API 分支与任务文件注入，恢复 SWE-bench
  方案（`make_agentic_data.py`，temperature 回 0.8）
- **保留的修复/调参**：并发 /tmp 唯一路径（v0.22.0）、pytest -v 解析（v0.22.1）、
  reward 键（v0.22.2）、参数化测试列表传参（v0.23.1）、`n=4`、`CONCURRENCY=4`、
  `trainer.val_before_train=False`（跳过 val 验证，TQ val 查询对简单数据崩溃）
- **实验结论（入档）**：Qwen3-8B 在 SWE-bench 长程任务行为退化（60 轮不修改代码、
  死循环）；极简单函数任务可部分修复（simple_reverse 0.5）但组内无梯度；
  8B 模型能力是当前瓶颈，待换 Qwen3-Coder-30B-A3B 或 SFT 预热

## v0.27.0（2026-08-06）

- **路线定稿（用户拍板）**：agent 不改（保持 mini-swe-agent harness），**换数据集
  HumanEvalFix**（`bigcode/humanevalpack` Python 修复子集：单函数 buggy 代码 + 单测，
  8B 60 轮内可出结果，绕开 SWE-bench 长程探索退化）；SWE-bench Lite 留作对比基准
- **优化路线确认**：双机 TQ + Mooncake（双机网络就绪后第一优先）→ 投机解码
  （PD 分离为后续亮点），详见 `docs/ROADMAP.md` 与 TODO §C 6.5
- 新增 `docs/ROADMAP.md`（换数据集构造步骤 / 双机 TQ+Mooncake / 投机解码 /
  PD 分离 / 服务器恢复 checklist）
- 更新 `docs/architecture.md`（数据口径 → HumanEvalFix；状态与关键决策）
- 思路.md V4.3：黑盒 agent（Claude Code 类）调研结论留档（决定不改 agent，
  技术路线与 ToS 风险详见思路 1.10）
- **工作流**：服务器已关机、node2 镜像已保存；后续所有本地改动 commit + push 本仓，
  服务器恢复后 `git pull` 即可

## v0.27.1（2026-08-06）

- 按用户要求删除黑盒 agent（Claude Code 类）调研记录：TODO §8、`docs/ROADMAP.md` §1、
  `docs/architecture.md` 决策清单中的相关条目（思路.md 1.10 调研留档暂保留）

## v0.28.0（2026-08-06）

- **HumanEvalFix 数据构造**：新增 `scripts/make_humanevalfix_data.py`（原 SWE-bench
  `make_agentic_data.py` 保留不动）——humanevalpack python 子集 → `solution.py` +
  `test_solution.py`（check(candidate) 转 pytest 单测 `test_all`，`from solution import *`
  兼容测试引用同文件辅助函数）+ 本地 verify（buggy rc=1 / canonical rc=0，死循环超时跳过）
- 冒烟数据入库：`work/data/humanevalfix_train.jsonl`（3 条）+ `humanevalfix_val.jsonl`（2 条）
- runner 新增 `humaneval_fix` 任务类型（swe_bench 原路径不变）：沙箱 /testbed git 仓库 +
  solution.py 注入（`git add -A` 保证 patch 可 diff）+ mini-swe-agent API 直连（绕开
  swebench-single 数据集硬编码）+ reward 阶段写隐藏测试（无测试泄露）
- 新增 `scripts/run_grpo_humanevalfix_ucloud.sh`（数据/实验名/checkpoint 目录与 agentic 区分）
- 待上机验证：8B 通过率与 GRPO reward 组内差异（node2 恢复后 git pull 即可跑）
