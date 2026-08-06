# Changelog

本项目约定：**每完成一项任务 commit 一次**，按语义化版本递增。

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
