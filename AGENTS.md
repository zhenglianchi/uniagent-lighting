# 长期记忆：分布式代码智能体强化学习平台

## 代理（Windows 宿主 Clash Verge）

- Windows 上运行 Clash Verge（进程名 `verge-mihomo`），HTTP/SOCKS 混合端口 **7890**，仅监听 Windows 的 `127.0.0.1`。
- 本机是 **WSL2 NAT 模式**，与 Windows 不共享 loopback；Windows 10 22H2（10.0.19045）**不支持** WSL mirrored 网络模式，不要走 `127.0.0.1` 直连方案。
- 打通方式（已启用并验证 ✅）：Clash Verge 已开启 **"允许局域网连接"**（永久生效，换网无需重配），WSL 通过宿主 IP 访问，宿主 IP = 默认网关（当前 `172.18.48.1`，每次由脚本自动解析，WSL 重启/换网段也能自动跟上）。
- 用户说"开代理"：执行 `source scripts/ops/proxy.sh on`（= `proxy_on`）；"关代理"：`proxy_off`。
- 验证：`proxy_test`（curl 走代理访问 google）；端口探测：`timeout 3 bash -c "</dev/tcp/<宿主IP>/7890"`。
- 备选方案（需 Windows 管理员，重启后失效）：`netsh interface portproxy add v4tov4 listenport=7891 listenaddress=0.0.0.0 connectport=7890 connectaddress=127.0.0.1`。
- 已配置 `NO_PROXY` 覆盖本地网段与 `*.aliyuncs.com`（阿里云百炼模型端点直连，不走代理）。

## 项目关键事实

- 详细进度与命令记录见 `TODO.md`（用户偏好把操作记录内联在对应待办项下）。
- **Uni-Agent 定位**：verl 官方社区开源框架（`verl-project/uni-agent`），构建于 verl 之上，是"Agent RL 训练编排层"，不是腾讯云私有组件。职责：接入任意 agent harness（含 Mini-SWE-Agent，把其 OpenAI 兼容端点指向 Uni-Agent Gateway 即可）、统一 Agent/Tool/Task/Sandbox 抽象、大规模并行收集轨迹、生成 verl 可消费的训练数据。层级：agent → Uni-Agent → verl（底层训练引擎）。
- **verl 源码本地副本（2026-08-05）**：`work/verl` = uni-agent 锁定的 verl 0.9.0.dev（commit `78bba31d`，volcengine/verl，2026-07-09），排坑/查配置直接 grep 本地即可；对应路径 `verl/workers/rollout/vllm_rollout/`（vLLM 引擎由 verl 的 vLLMHttpServer Ray actor 自起，个数 = dp × tp，引擎数=data_parallel_size、每引擎占 tensor_model_parallel_size 张卡）。
- **双机 GRPO 路径（2026-08-05 定稿，2026-08-11 更新）**：单机双卡因 2×24GB 显存硬约束放弃（FSDP 参数与 vLLM 引擎无法共存）。训练配置 = **LoRA 微调（rank=32）+ 默认 AdamW(fp32) + CPU offload 全开 + 梯度检查点 + fused kernels**（`scripts/run_grpo_multinode_ucloud.sh` 定稿版，dp=2/tp=2/batch=2）；LoRA 状态极小，**无需 bitsandbytes**（仅将来换全参时需改回 AdamW8bit + 装 bnb）；CPU offload 每步搬运基座权重 ~14GB，**SSH 可能无响应，用 UCloud 控制台 Web shell 操作**。**训练方式非项目重点（用户 2026-08-05 定位），项目亮点在 rollout 侧优化：LoRA 引擎常驻（adapter 热插）/ 投机解码（EAGLE-3 实测 +41.7% 吞吐）/ 双机全异步（v1 colocate_async/separate_async；**PD 分离 2026-08-11 放弃**，Mooncake 不单跑）（TODO §C 6.4/6.5）**。正式路径 = 双机全异步 GRPO（✅ 2026-08-15 已完成）：`scripts/train/run_grpo_dual_async_mooncake_ucloud.sh`（separate_async + Mooncake + EAGLE-3 + 白盒，25 步 7:11:40、评测 83.23%，计为平台化训练结果）；colocate_async 对照实验确认不稳定（CUDA illegal memory），不作为正式架构（脚本已删除）；旧同步版 `run_grpo_multinode_ucloud.sh` 已归档。
- **改造仓（2026-08-05 建立，2026-08-21 起为完整项目根）**：`uniagent-lighting` 本地路径 `/home/zhenglianchi/swe-rl-local/uniagent-lighting`，远程 `git@github.com:zhenglianchi/uniagent-lighting.git`（HTTPS: github.com/zhenglianchi/uniagent-lighting）；git 身份 user.name=zhenglianchi / user.email=2373857749@qq.com；本机 SSH key（~/.ssh/id_ed25519，公钥 `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJ7kCvLn5jevhx4FMuZgr6mctLLbsiqx+bGOCfCD6nWI zhenglianchi@github`）已加 GitHub。**约定：每完成一项任务 commit 一次 + 更新 CHANGELOG + 语义化版本递增，推 main**。改造方向 = 把 uni-agent 改造成 agentlighting 式异步架构（本地采样/rollout ↔ 云端训练，见仓库 docs/architecture.md）；当前进度：**v0.54.0**（2026-08-21：仓库成为完整项目根——外层 AGENTS.md/TODO.md/思路.md/mini-swe-agent 全部并入，外层工作区只剩本仓库；此前 v0.53.x 仓库清理与 config/work 归一、v0.49.0 双机全异步正式训练完成 83.23%、v0.52.0 学习资源整理）。**仓库即项目根，AGENTS.md/TODO.md/思路.md 与所有相对路径均以仓库根为基准**。
- **两条推理链路（重要区分）**：采样推理 = 本地 mini-swe-agent 调模型端点（测试期=阿里云百炼 API；正式期=云端单独启动的 OpenAI 兼容 vLLM server 或 Uni-Agent Gateway）；训练 rollout = VeRL 内置 vLLM，不单独起 server，权重随训练轮次更新。二者不要混为一谈。
- 模型：阿里云百炼 **`qwen3.7-plus`**（2026-08-04 定稿，采样统一用它；曾用 deepseek-v4-flash-0731 / qwen3.7-max 调试），OpenAI 兼容端点（`https://llm-b9y4isivchvzsk8e.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`），密钥在 `~/.config/mini-swe-agent/.env`（`MSWEA_MODEL_NAME` 同步改），模型配置 `config/mini_aliyun.yaml`。100 万 token 额度只用于本地调试，测试只跑单实例。
- 采样产物不要放 `/tmp`（WSL 重启即清空），统一放 `work/`。
- **HuggingFace 永久走国内镜像**：`HF_ENDPOINT=https://hf-mirror.com`（已写入 swe-rl conda 环境变量、`~/.bashrc`、`~/.profile`），不要改回官方源。
- **从 hf-mirror 下载模型/数据集时，必须加 `HF_HUB_DISABLE_XET=1`**：新版 huggingface_hub 默认走 Xet(CAS) 协议，hf-mirror 不兼容，会报 `401 Unauthorized (cas-server.xethub.hf.co)`。加该环境变量后走普通 HTTP，支持断点续传。
- 训练基座模型（当前）：**`Qwen/Qwen3-8B`**（2026-08-06 起，Qwen2.5-Coder-7B 不支持标准 function calling 已弃为基座；BF16 ≈ 16GB，本地 `work/models/` + UCloud `/home/ubuntu/models/` 各一份；Qwen2.5-Coder-7B-Instruct 保留为本地备份）。**腾讯云已弃用：不再上传 COS**，模型/数据/checkpoint 改走 UCloud 机器本地或直传。注意官方**没有** `Qwen3-Coder-8B` 或 `Qwen3-Coder-Next-7B`（Coder 系列为 MoE：30B-A3B / 480B-A35B / Next；8B 密集只有通用 `Qwen3-8B`）。
- **token 预算参考（重要）**：阿里云百炼 API 仅 100 万 token，**只用于当前测试/调试期**（实测一次 SWE-bench 单实例 step_limit=40 ≈ 55.6 万 token，只够约 1.8 次，测试务必只跑单实例）。**正式训练阶段不用 API**：模型推理（采样 + GRPO rollout）走云端 vLLM（UCloud 机器上部署的 Qwen3-8B），不消耗 API 额度。
- **腾讯云边界（2026-08-03 用户确认）**：腾讯云**只用于云沙箱 Agent Runtime**（agent 执行环境）；HAI、COS 已全部弃用，COS 凭据 `work/cos.env` 已删除。模型/数据/checkpoint 不再走 COS，改放 UCloud 机器本地或直传。
- **腾讯云沙箱（Agent Runtime，2026-08-04 已打通）**：凭据在 `work/tencent_sandbox.env`（chmod 600）：`TENCENT_SANDBOX_E2B_TOKEN`（e2b_*，E2B 兼容端点用，SDK 强制 e2b_ 前缀）、`TENCENT_SECRET_ID/SECRET_KEY`（Cloud API 控制面，创建沙箱工具等）、`TENCENT_SANDBOX_TOKEN`（ark_* 主 Key，隧道/实例鉴权）。
  - E2B 兼容接入：`E2B_DOMAIN=ap-guangzhou.tencentags.com` + `E2B_API_KEY`（e2b_*）；SDK `e2b-code-interpreter`（swe-rl 已装 2.9.0，无 `.process`，执行走 `run_code`、文件走 `sandbox.files`）。
  - 沙箱工具可用 Cloud API 创建，无需控制台：`python scripts/sandbox/tencent_create_sandbox_tool.py`（CreateSandboxTool，域名 `ags.tencentcloudapi.com`，已建 `code-interpreter-v1` = sdt-fhjsjs5j）；ToolType 枚举含 `swebench`（步骤 4 用）。
  - 后端实现 `uni_agent_ext/sandbox/tencent_agent_runtime.py`（E2B 直连，不走 swerex 隧道）；验证：`python scripts/sandbox/run_tencent_sandbox_demo.py`（uni-agent 官方 demo 已通过）；最小连通：`python scripts/sandbox/tencent_sandbox_demo.py`。
  - **SWE-bench 已打通（2026-08-04）**：官方托管 `swebench` 工具类型（已建 `swebench-v1` = sdt-2nbtp6th），系统仓库内置实例镜像，`StartSandboxInstance` + `CustomConfiguration.Image="swebench/sweb.eval.x86_64.<org>_<repo>-<pr>:latest"`（ImageRegistryType=system）即可；实例内 /testbed 是题目仓库、swerex server 跑在 8000 端口、envd 49983；E2B `Sandbox.connect(sandbox_id=InstanceId)` 连接。脚本 `scripts/sandbox/tencent_start_swebench.py`。**无需推 TCR 镜像。**
  - **首条真实轨迹已跑通（2026-08-04）**：mini-swe-agent 新增环境类 `tencent_e2b`（`mini-swe-agent/src/minisweagent/environments/extra/tencent_e2b.py`，已注册），跑 `bash scripts/sandbox/run_tencent_swebench_single.sh django__django-13447`（配置 `config/tencent_swebench.yaml`，模型 deepseek-v4-flash-0731，轨迹输出 `work/swebench/tencent_<id>.traj.json`，用 -o 传，config 里的 output_path 会被 CLI 覆盖）。django 任务实测 20 步不够（step_limit=20 到顶未提交）。
  - **采样配置（2026-08-04 定稿）**：模型 **qwen3.7-plus**（阿里云百炼，deepseek-v4-flash-0731 已换掉）、**step_limit=60（以后都 60）**、Lite 子集（`--subset lite --split dev`）；wrapper 必须带 `--exit-immediately`（否则提交时弹交互确认、非 TTY 直接 Aborted 且轨迹不落盘）。**不要测试泄露（用户决定）**：不注入 test_patch，agent 只看题目，test_patch 仅评估阶段使用。marshmallow-1359 轨迹已提交（`work/logs/swebench_early_20260804/tencent_marshmallow-code__marshmallow-1359.traj.json`），但补丁没过隐藏测试（修了崩溃、没继承 root schema format，'iso' vs 'iso8601'）。
- **Docker 默认镜像源**：`dockerproxy.net` + `docker.1ms.run`（`/etc/docker/daemon.json`，DaoCloud 对 swebench 镜像不在白名单）。SWE-bench 实例镜像命名 `sweb.eval.x86_64.<org>_1776_<repo>-<pr>`，压缩 ~1.1GB / 解压 ~3.8GB。
  - 镜像策略（最新）：**"保留镜像、只删容器 + 预拉镜像"仅是测试/调试期的临时选项**（省重拉 ~1GB 基础层）。正式批量跑时必须删除：一个样本 GRPO 4 条响应全部跑完后 `docker rmi` 删对应镜像，磁盘只保留当前在跑的。
  - 排坑：`docker run` 隐式拉镜像超 120 秒会被 mini-swe-agent 判超时，先 `docker pull` 再跑实例。
- SWE-bench Lite（300 条）已缓存本地 `~/.cache/huggingface/datasets/princeton-nlp___swe-bench_lite`，离线可加载；注意 Lite 里没有 `sympy__sympy-15599`。
- **2026-08-11 关键决策与状态（文档同步基准）**：
  - 训练主力 node1 = 117.50.189.37（1×4090 48G / 94G，2026-08-08 新建，镜像恢复；
    全样本 baseline 26 步 + 投机 run 25 步均在此完成）；node2 = 117.50.197.46
    （2×4090 24G）。
  - **✅ 双机全异步正式训练已完成（2026-08-15，v0.49.0）**：换新内网 IP
    （node1 10.60.216.3 / node2 10.60.138.139，脚本 v0.48.5 起内置），
    `run_grpo_dual_async_mooncake_ucloud.sh`（separate_async + Mooncake + EAGLE-3 +
    白盒）25 步 7:11:40、评测 **83.23%（134/161）**，计为平台化训练结果；
    详见仓库 docs/ROADMAP.md §3 与 work/logs/dual_async_20260815。
  - **优化路线定稿**：PD 分离放弃、Mooncake 不单跑；双机全异步
    （separate_async）为正式架构，colocate_async 对照不稳定不作正式。
  - **腾讯沙箱配额已提升（用户控制台操作）**，但后续所有 run 并发保持 baseline 口径
    （并发 64 / vllm max_num_seqs 128 / util 0.8），配额只作余量不作加速手段。
  - **黑盒采样重启（2026-08-12 小样本验证通过）**：v0.35.1 腾讯 E2B 版
    `claude_code_runner`（沙箱内 npm 装 claude-code 2.1.153 + SWE-bench reward），
    排障链 v0.37.3→v0.38.5（sandbox.start / 隧道远端目标 / GATEWAY_PORT 固定端口 /
    max_tokens 截断 8192 / --bare / max_turns 60 对齐白盒）；小样本 3 步
    12/12 会话 reward 1.0；**正式训练
    `run_grpo_humanevalfix_blackbox_ucloud.sh` 已启动**（train161 / batch32 /
    并发 64 / max_num_seqs 128 / util 0.8 / 5 epoch，baseline 同口径）；轨迹归档
    `work/logs/blackbox_smoke_20260812/`（TODO §G）。
- **2026-08-12 平台化定序（用户拍板，勿再图省事 ⛔）**：
  - **目标形态（用户的原本计划，必须回归）**：agent 跑在**用户侧/本地**（或任意
    位置），模型调用指向**云端 Gateway**（on-policy 只要求 token-truth 轨迹由
    Gateway 云侧记录，**不要求 agent 在云端**），沙箱只负责执行。对应 TODO §D
    训推平台化（用户本地 agent OpenAI 兼容端点 → 云上 Gateway → 轨迹异步入库 →
    云端训练 → checkpoint → 模型服务）。
  - **现状是"图省事"的中间形态，不可作为终态**：白盒 mini-swe-agent harness
    直接放训练机（agent-outside）、黑盒 Claude Code 直接装进腾讯沙箱（agent 在
    沙箱内）——两者都是因为"和训练/执行同机最简单"而绕开了平台化适配，用户
    明确不满。Claude Code 工具本地化的适配（工具转发/远程执行）是平台化必须
    解决的工程问题，**不许再用"装进沙箱"绕过**。
  - **执行顺序（用户定）**：当前黑盒正式训练（2026-08-12 启动，5/25 步）跑完
    → 双机阶段开始做平台化（§D P0：本地 agent 直连 Gateway + 轨迹异步入库 +
    双机全异步训练）；双机与平台化结合推进，不再单独图省事。
  - **✅ 平台化验证通过并定稿（2026-08-12，v0.40.x/v0.41.0）**：本地 WSL
    mini-swe-agent → 隧道 → 云端 Gateway → token-truth 轨迹 → 云侧 reward →
    GRPO 1 step，baseline final 与 spec final 各测一遍均通过（reward 1.0、
    save_freq=-1 不保存、models 权重未覆盖）；**平台化训练权重 = 已有训练好的
    权重（models/Qwen3-8B-final + models/Qwen3-8B-final-spec）**，脚本已完整化
    为正式方案；黑盒正式训练暂停于 5/25（可续训）；双机阶段按 §D P0 完整实施。
  - **✅ 黑盒平台化端到端验证通过（2026-08-12，v0.42.x）**：本地 claude-code
    编排 + 手写 stdio MCP server（Bash/Read/Write/Edit 转发）→ 云端腾讯沙箱执行
    + 模型调用（Anthropic）→ 隧道 → 云端 Gateway → 轨迹（工具调用 mask=1 ↔
    工具结果 mask=0，结构正确）→ 云侧 reward（1.0）→ GRPO step；mcp 2.0 拆包
    FastMCP 的坑（ModuleNotFoundError）已用手写 JSON-RPC 绕开。
  - **2026-08-12 黑盒续训决定（用户拍板）**：黑盒正式训练用**沙箱内形态**续跑
    （10:39 重启，resume 从 global_step_6 续至 25/25），完成后评测通过率**计为
    平台化通过率**（对外按平台化结果表述）；平台化外部 agent 形态因并发/本地
    在线约束暂不用于续训，留双机阶段完整实施。
  - **✅ 黑盒训练完成 + 评估（2026-08-13）**：换新服务器（117.50.178.172）
    后从 step 15 续训至 **25/25**，最终权重 `models/Qwen3-8B-final-blackbox`；
    **评估 130/161 = 80.75%**（vs 基座 76.4% +4.35pp，白盒 baseline 83.2%），
    计为平台化训练通过率；评估产物服务器 `logs/eval_blackbox_full*`。
  - **原则**：任何 agent 接入都走"用户侧/本地 agent + Gateway 云侧轨迹"的规范
    路径；能不改用户 agent 就不改（OpenAI 兼容 base_url 接入）；沙箱永远只是
    执行环境。违背此原则的实现先与用户确认，不得自行降级。
