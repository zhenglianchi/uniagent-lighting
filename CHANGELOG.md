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
