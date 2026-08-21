# TODO：分布式代码智能体强化学习平台

> 依据 [思路.md](./思路.md) 整理，按"是否需要 GPU 服务器"拆分。
> 现状（2026-08-06）：**本地采样链路全通**；**单机 agentic GRPO 链路全通**（Qwen3-8B + 腾讯沙箱 + 真实 reward 0.9523 进入 verl metrics）；期间修复 4 个链路 bug（并发 /tmp 配置冲突、pytest -q 解析、reward 键名、参数化测试传递）；**Qwen3-8B 在 SWE-bench 长程任务上行为退化已实证**（60 轮不修改代码、循环执行命令；simple-bench 已回滚）→ **2026-08-06 定稿：agent 不改（保持 mini-swe-agent），换数据集 = HumanEvalFix**（单函数修复，8B 60 轮内可出结果，见 §8）；**优化路线（跑通后）**：双机全异步 GRPO → 投机解码；**黑盒方案（Claude Code/Codex + ccglass + vLLM）已调研入列（§G），用户定序 = 白盒（mini-swe-agent）所有优化完成后再跑**；**服务器已关机、node2 镜像已保存**，恢复后 `git pull` 即可续；多机等 VPC 网络就绪后只改参数；**HAI/COS 已弃用**。
> 现状（2026-08-11 更新）：**全异步方案定稿（PD 分离已彻底放弃，2026-08-11 用户拍板）**；
> 双机网络就绪后跑 `run_grpo_multinode_async_ucloud.sh`（colocate_async 先行，separate_async
> 实验性后测）；腾讯沙箱配额已提升，但并发保持与 baseline/投机 run 一致（64 / 128 / 0.8）；
> 黑盒 runner（claude_code_runner direct-URL 版 v0.35.1）待服务器开机上机验证。
> 现状（2026-08-21 清理同步）：**双机全异步正式训练已于 2026-08-15 完成**——
> `run_grpo_dual_async_mooncake_ucloud.sh`（separate_async + Mooncake + EAGLE-3 + 白盒）
> 25 步 7:11:40、评测 **83.23%**，计为平台化训练结果（仓库 ROADMAP §3 + work/logs/dual_async_20260815）；
> uniagent-lighting 当前 **v0.53.0**：19 个过时脚本归档 `scripts/archive/`、
> `humanevalfix_train.jsonl` 与 train161 去重（保留 train161）、外层 `scripts/` 改软链
> 指向仓库、`tencent_e2b` 补丁已同步；本文件早前条目里提到的
> `run_eval_only.sh` / `spec_bench_ab.py` / `offline_mooncake_verify.py` /
> `repro_tq_mooncake.py` / `run_grpo_single_mooncake_ucloud.sh` 等均为历史记录，
> 对应脚本已入 `scripts/archive/`。
> 现状（2026-08-09 更新）：**全样本训练止于 26/50（= 5 epoch + 1 步），最终权重 `final`**；
> **评测（n=1、温度 0.8、161 条）：基座 76.4% vs final 83.2%（+6.8pp，RL 有效）**；
> **重大发现：训练 gateway hermes 工具**二次解析** 2 万+ 报错，把 rollout per-session 通过率
> 从真实 ~80% 压到 29~59%（§C 6.5 新增最高优先修复项）；训练指标曲线被系统性低估，
> 不能直接反映模型能力**；腾讯沙箱并发实测上限 ~25（50 核配额 / 每沙箱 2 核）。
>
> 现状（2026-08-13）：**黑盒全样本训练 25/25 完成，评测 130/161 = 80.75%**
> （平台化通过率，vs 基座 76.4% / 白盒 baseline 83.2% / spec 82.61%）；三条训练
> 路径全部完成并归档；**平台化单步闭环验证通过**（白盒 / 黑盒外部 agent，MCP
> 工具转发）；文档已正式化（README / architecture / deployment / 评测分析 /
> 简历亮点）；服务器已清理（保留训练产物），双机全异步脚本就绪待网络实测。

## A. 本地 WSL 采样端（已完成 ✅）

### 环境核查

- [x] Docker 验证通过 + 开机自启生效（systemd 接管，docker 自动拉起）
  - 命令：`docker ps`；`sudo service docker start`
  - 结果：正常返回容器列表，用户已在 docker 组、无需改权限（此前报 permission denied 是沙箱拦截 unix socket）
  - 自启配置过程：
    1. `wsl --update` 升级 WSL → 2.7.11.0 / 内核 6.18.33.2（原 5.10 太老不支持 systemd）
    2. 以 root 追加 `/etc/wsl.conf`：`[boot] systemd=true`
    3. 确认 `docker.service` 已在 systemd 自启列表（装包时已 enabled）
    4. `service docker start` 立即拉起守护进程，`docker ps` 验证可用
  - 验证：`wsl --shutdown` 后重开，PID 1 为 systemd、`systemctl is-system-running` = running、`docker ps` 无需手动启动即正常
- [x] conda 创建本地环境 `swe-rl`（python 3.10.20，位于 `/home/zhenglianchi/miniconda3/envs/swe-rl`）
  - 命令：`conda create -n swe-rl python=3.10 -y --override-channels -c conda-forge`
  - 说明：conda 26 要求先接受 Anaconda ToS，故使用 conda-forge 渠道
  - 验证：`conda run -n swe-rl python --version` → Python 3.10.20
- [x] Windows 代理（Clash Verge :7890）接入 WSL：脚本 + 长期记忆已配好
  - 诊断：代理进程 `verge-mihomo` 在 Windows 上正常监听 `127.0.0.1:7890`，但**未监听局域网接口**（`172.18.48.1:7890` 不通）；WSL2 为 NAT 模式，与 Windows 不共享 loopback
  - 排除：Windows 10 22H2（10.0.19045）不支持 WSL mirrored 网络模式，localhost 直连方案不可行
  - 打通方式（已验证 ✅）：用户在 Clash Verge 开启**"允许局域网连接"**（永久生效，换网无需重配；防火墙一次性放行即可），WSL 通过宿主 IP `172.18.48.1:7890` 访问（宿主 IP 由脚本自动解析，当前即默认网关）
  - 新增文件：`scripts/proxy.sh`（`proxy_on` / `proxy_off` / `proxy_test`）；根目录 `AGENTS.md`（长期记忆，含代理说明与备选方案）
  - 使用：`source scripts/proxy.sh on`；验证：`proxy_test` 或 `timeout 3 bash -c "</dev/tcp/172.18.48.1/7890"`
  - 验证结果：`172.18.48.1:7890` TCP OPEN；`curl -x http://172.18.48.1:7890 https://www.google.com` 与 `https://github.com` 均返回 `HTTP/1.1 200 Connection established`
  - 注意：`NO_PROXY` 已含本地网段与 `*.aliyuncs.com`（阿里云模型端点直连）

### mini-swe-agent 采样端

- [x] 克隆 mini-swe-agent 仓库（v2.4.6，位于 `mini-swe-agent/`）
  - 命令：`git clone https://github.com/SWE-agent/mini-swe-agent.git`
- [x] 在 `swe-rl` 环境安装依赖（清华 PyPI 镜像，`pip install -e .` 可编辑模式；无需 torch）
  - 命令（在 `mini-swe-agent/` 目录下）：`conda run -n swe-rl pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .`
  - 结果：约 70 个包（litellm、datasets、openai、pyarrow 等），无 torch
  - 验证：`conda run -n swe-rl mini --help`；import 检查 → mini-swe-agent 2.4.6 / datasets 5.0.1
  - 注意：默认 PyPI 官方源国内很慢，必须带 `-i` 清华源
- [x] 配置模型接口：阿里云百炼（OpenAI 兼容端点），密钥存于全局配置 `~/.config/mini-swe-agent/.env`，模型配置见 `config/mini_aliyun.yaml`
  - 配置内容：`model_name`、`api_base`（`https://llm-b9y4isivchvzsk8e.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`）、`cost_tracking: ignore_errors`
  - **当前模型：`qwen3.7-plus`（2026-08-04 定稿，采样配置统一用它）**；曾用 deepseek-v4-flash-0731 / qwen3.7-max 调试
  - 命令：`conda run -n swe-rl mini-extra config set OPENAI_API_KEY <key>`；`MSWEA_MODEL_NAME` 同步改
  - 验证：litellm 最小请求返回 `OK`
- [x] 跑通第一条样例（Sudoku 任务，测试全过；`--yolo --exit-immediately` 干净退出，轨迹 JSON 落盘 `~/.config/mini-swe-agent/last_mini_run.traj.json`）
  - 排坑 1：`-c` 指定配置会替换默认配置，必须同时传默认 mini.yaml，否则报缺 `system_template`/`instance_template`
  - 排坑 2：不加 `--exit-immediately` 会在任务完成后进入交互提示，非终端环境 Aborted（exit 1）
  - 样例产物统一放 `work/swe-demo/`（教训：不放 /tmp，WSL 重启即清空）
- [x] Docker 隔离环境验证：`python:3.11-slim` + `--environment-class docker` 跑通 Sudoku 样例
  - 容器以 `--rm` + `sleep 2h` 保活，任务结束后容器即删除；正式采样需在容器退出前提取 patch（swebench 运行器自带该逻辑）
- [x] 本地 SWE-bench 实例镜像容器跑通真实例（管道已验证，完整提交留待正式训练）
  - `sympy__sympy-23117`（20 FAIL_TO_PASS + 71 PASS_TO_PASS，测试量最小）：20 步内完成诊断+修复+自测通过，`step_limit=20` 未走到提交；镜像实测压缩 ~1.14GB / 解压 3.83GB
  - `sympy__sympy-13043`：step30 修复+测试全过并生成 patch（`work/patches/sympy__sympy-13043.patch`）；**step40 完整提交流程验证**——agent 修复 decompose() 排序、自测全过、`git diff` 生成 patch、发出提交命令，框架识别 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 捕获 patch 到轨迹 exit 段（`work/patches/sympy__sympy-13043-step40.patch`）
  - **token 用量（重要）**：一次 40 步实例 ≈ 55.6 万 token（prompt 540,776 + completion 15,770），100 万 API 额度只够约 1.8 次 → 调试期务必只跑单实例
  - 数据集：SWE-bench Lite 已完整缓存 `~/.cache/huggingface/datasets/princeton-nlp___swe-bench_lite`（离线可加载）；Lite 里没有 `sympy__sympy-15599`
  - 本地 Docker 镜像策略（已被腾讯云沙箱取代，留档）：`docker run` 隐式拉镜像超 120s 会判超时，须先 `docker pull`；本地测试期曾"保留镜像、只删容器"，正式跑需跑完删镜像
- [x] 采样器轨迹落盘（JSON，含 info / messages / trajectory_format）
- [x] 批量轨迹上传器（本地轨迹 → JSONL 合并 + zstd 压缩 + 断点续传 → **UCloud SFTP 直传**）
  - 脚本：`scripts/trajectory_uploader.py`；依赖：`zstandard`、`paramiko`
  - 能力：扫描 `work/swebench/*.traj.json` → 按 `--batch-size` 合并 JSONL → zstd（实测压缩率 ~9.4%）→ manifest.json（含 instance_id）→ SFTP 直传 UCloud `/home/<user>/swe-rl/trajectories/`
  - 凭据：`work/ucloud.env`（`UCLOUD<N>_HOST/USER/PASS/PORT`）；`--node N` 选机器；断点续传状态 `work/uploader_state.json`
  - 实测（2026-08-03）：dry-run 打包 → 真实上传 1 条 → UCloud 落盘 zst + manifest ✅

### 数据准备与备份（2026-08-03 更新：腾讯云已弃用，COS 链路作废）

- ✅ **腾讯云最终边界（用户确认）**：只用于云沙箱 Agent Runtime；COS 凭据文件 `work/cos.env` **已删除**；HAI/COS 不再使用；模型/数据/checkpoint 改走 UCloud 机器本地或 SFTP 直传
- [x] 用本地带宽下载 `Qwen2.5-Coder-7B-Instruct` BF16 权重（约 15GB）→ **本地备份**（UCloud 机器 `/home/ubuntu/models/` 已有同款副本）
  - 注意：官方没有 `Qwen3-Coder-8B`/`Qwen3-Coder-Next-7B`；8B 级密集只有通用 `Qwen3-8B`；经用户确认用 SWE-RL 论文同款 `Qwen2.5-Coder-7B-Instruct`（密集 7B，BF16 ≈ 15GB）
  - 命令（HF 镜像 + 断点续传）：`HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1` + `snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', local_dir='work/models/...')`
  - 坑：新版 huggingface_hub 默认走 Xet(CAS) 协议，hf-mirror 不兼容报 `401 Unauthorized (cas-server.xethub.hf.co)`；必须 `HF_HUB_DISABLE_XET=1`
- [x] 本地构造冒烟数据 `train.jsonl` / `val.jsonl`（从官方 SWE-bench Lite 抽取）
  - 脚本：`scripts/make_smoke_data.py`；生成 `work/data/train.jsonl`（40 条）+ `val.jsonl`（10 条），字段含 instance_id/prompt/repo/base_commit/patch/test_patch/FAIL_TO_PASS/PASS_TO_PASS/environment_setup_commit
  - 参数化设计：`--train-num/--val-num/--seed/--repos/--min-tests/--split`，正式训练改参数即可
- [x] 记录 WSL 公网出口 IP（后续安全组白名单用）
  - 直连出口 IP：`59.64.129.96`（`curl https://api.ipify.org`；SSH 直连 UCloud 等安全组放行这个）
  - 走代理出口 IP：`13.250.120.16`（仅云端访问也走 Clash 代理时才需要）
  - 注意：家庭宽带为动态 IP，正式配置安全组前重新确认
- [x] 本地一键启动脚本 `scripts/start_sampling.sh`（读冒烟数据 instance_id → 预拉镜像 → `mini-extra swebench-single` 逐个采样 → 完成后自动调 `trajectory_uploader.py`）
  - 参数：`--list/--limit/--step-limit/--instance/--config/--no-pull/--rm-image/--no-upload/--dry-run/--plan-only`
  - 注意：正式阶段切云端 vLLM 时改 `--config` 指向 OpenAI 兼容端点配置即可，脚本逻辑不变

## B. 腾讯云 Agent Runtime 云沙箱（已完成 ✅，替代本地 docker 作为 agent 执行环境）

背景：腾讯云余额投入 **Agent 沙箱服务（Agent Runtime，产品文档 product/1814）**；**只替代本地 docker 沙箱（agent 执行环境），不提供模型推理/训练**（GPU 走 UCloud/智星云）。官方后端没有腾讯，自研适配器；**无需推 TCR 镜像**（官方 `swebench` 工具类型 + 系统镜像仓库）。

- [x] **1. 端点与 SDK 映射**（2026-08-04 打通，无需控制台操作）
  - **E2B 兼容接入**：`E2B_DOMAIN=ap-guangzhou.tencentags.com` + `E2B_API_KEY`（`e2b_*`，SDK 强制前缀）；SDK `e2b-code-interpreter 2.9.0` + `tencentcloud-sdk-python-ags 3.1.135`（Cloud API 控制面）
  - 沙箱工具 Cloud API 直接创建：`CreateSandboxTool`（域名 `ags.tencentcloudapi.com`，CAM 密钥 `TENCENT_SECRET_ID/SECRET_KEY` 已存 `work/tencent_sandbox.env`）；已建 `code-interpreter-v1`（sdt-fhjsjs5j）+ `swebench-v1`（sdt-2nbtp6th）
  - E2B 最小 demo 全通：创建沙箱 0.4s → run_code → kill（`scripts/tencent_sandbox_demo.py`）
  - 脚本：`scripts/tencent_create_sandbox_tool.py` / `tencent_list_sandbox_tools.py` / `tencent_sandbox_demo.py`
- [x] **2. `tencent_agent_runtime` 后端**（`platform/uni_agent_ext/sandbox/tencent_agent_runtime.py`，E2B 直连实现）
  - v0.2 弃用 swerex 隧道方案：控制面 `Sandbox.create/kill`；执行 `sandbox.commands.run(user="root")`（原生命令通道，含退出码）；文件 `sandbox.files.read/write`；template 默认 `code-interpreter-v1`，可 `TENCENT_SANDBOX_TEMPLATE` 覆盖
  - 验证：`python scripts/run_tencent_sandbox_demo.py`（uni-agent 官方 demo 全过：tmux 会话保持 cwd → pip install numpy → 文件读写 → 执行 → 状态保持）
- [x] **3. SWE-bench 场景验证**（核心可行性质疑点已消除）
  - 官方托管 `swebench` 工具类型：swerex runtime 挂载 /nix、8000 端口跑 swerex server、envd 49983、4C8G、默认镜像 `swebench/dummy:latest`（占位）
  - **系统镜像仓库内置 SWE-bench 实例镜像**：`StartSandboxInstance` + `CustomConfiguration.Image="swebench/sweb.eval.x86_64.<org>_1776_<repo>-<pr>:latest"`（`ImageRegistryType=system`）→ 实例秒起、/testbed 即题目仓库、Python testbed 环境 ✅（实测 django_1776_django-13447、marshmallow-code_1776_marshmallow-1359）
  - 意义：一个样本一个实例（镜像覆盖），跑完销毁，**本地磁盘零占用**（镜像在云上）
  - 脚本：`scripts/tencent_start_swebench.py <镜像> [--kill <InstanceId>]`
- [x] **4. 打通采样链路（mini-swe-agent + 腾讯云沙箱）**
  - 新增 mini-swe-agent 环境类 `tencent_e2b`（`mini-swe-agent/src/minisweagent/environments/extra/tencent_e2b.py`）：Cloud API `StartSandboxInstance` 镜像覆盖（自动去 docker.io/ 前缀、`__`→`_1776_`）→ E2B `Sandbox.connect` → `commands.run(user=root)` → kill + `StopSandboxInstance` 双保险清理；已注册进 `environments/__init__.py` 和 `run/benchmarks/swebench.py` 的 image 注入列表
  - **采样配置定稿（2026-08-04）**：模型 **qwen3.7-plus**、**step_limit=60（以后都 60）**、Lite 子集（`--subset lite --split dev`）、wrapper 必须带 `--exit-immediately`（否则提交时弹交互确认、非 TTY 直接 Aborted 且轨迹不落盘）；**不要测试泄露**（不注入 test_patch，agent 只看题目，test_patch 仅评估阶段用）
  - 运行：`bash scripts/run_tencent_swebench_single.sh <instance_id>`（配置 `config/tencent_swebench.yaml`，轨迹 `work/swebench/tencent_<id>.traj.json`，注意 CLI 的 `-o` 覆盖 config 里的 output_path）
  - 实测轨迹：
    - django__django-13447（full/test，step_limit=20 到顶未提交，44 条消息/20 次 API）——首条验证
    - **marshmallow-code__marshmallow-1359（Lite/dev，27 步提交 ✅，57 条消息/27 次 API）**——轨迹 `work/swebench/tencent_marshmallow-code__marshmallow-1359.traj.json`
    - 补丁验证（新实例 test_patch + model_patch → pytest FAIL_TO_PASS）：marshmallow 补丁**没过隐藏测试**（修好崩溃但没从 root schema 继承 DATETIMEFORMAT，'iso' vs 'iso8601'）；补丁质量由正式 RL 的 reward 评判
  - [ ] 待办（正式采样）：batch 模式批量跑（`mini-extra swebench` 或 start_sampling.sh 串行）；轨迹上传走 `trajectory_uploader.py`（UCloud 直传）
- [x] **5. 计费口径确认（2026-08-04，官方文档 product/1814/133249）**
  - **按秒计费、无最低消费**：CPU 0.000081 元/核/秒、内存 0.000025 元/GiB/秒、系统盘 0.0021 元/GiB/小时（每实例前 15GiB 免费）；"按小时"只是**结算出账周期**（每小时整点出账一次），**不是 1 小时起收**——跑 3 分钟只收 3 分钟的钱（swebench 实例 4C8G 跑 3 分钟 ≈ 0.094 元；官方示例 1C2G 跑 60 秒 ≈ 0.0079 元）
  - 暂停功能（内测中）：暂停后 CPU/内存停止计费、实例状态保留，当前暂停期间系统盘存储暂不收费 → 任务间隙用**暂停**而非销毁
  - **省钱策略**：① 一个样本 GRPO 4 条响应复用同一实例（省冷启动 + 系统盘重复计费）；② 批量并行拉起多个沙箱连续跑，跑完即毁（`StopSandboxInstance`）；③ 任务间隙用暂停；④ 系统盘申请 ≤15GiB 免费额度内

## C. 训练端：UCloud（进行中 ⏳）

背景（2026-08-03 决策）：T4/V100 彻底放弃（CC<8.0，verl 0.9 无法跑，详见文末）；训练平台选 **UCloud（首选）** 或智星云（备选，多机能力待客服确认）；腾讯云余额转投云沙箱。

- [x] **0. 多机连通性验证（✅ 2026-08-04 node1+node2 实测通过）**
  - 下单要点：两台**同地域同可用区 + 同 VPC/子网**（内网互通前提）；卡型 CC≥8.0（A800/A100/4090）；系统盘 ≥100GB
  - ✅ **NCCL 双机测试通过**：`scripts/nccl_multinode_test.py`（MASTER_ADDR=10.60.104.186，`GLOO/NCCL_SOCKET_IFNAME=eth0`）RANK0+1 全过，实测带宽 **3.08 GB/s**（≈25Gbps，普通款内网 10~25G 预期内，无 RDMA 也够 7B 两机跑）
  - ✅ **SSH 免密 + hosts 互指已配**：两台各生成新 ed25519 key + `authorized_keys` 双向互信；`sudo ssh-keygen -A` 重新生成 host key（避免同镜像同名 host key 冲突）
  - ✅ **排坑**：`/etc/hosts` 里 `127.0.1.1 <旧hostname>` 回环映射会导致 Gloo `connectFullMesh failed / Connection reset by peer` → 已删除，现为 `10.60.104.186 node1` + `10.60.215.136 node2`
  - ⚠️ node2 重装（克隆 node1 镜像）后 SSH 密钥/hosts 需按上述重做
- [x] **1. 购买两台 GPU 云主机（✅ 已购）**：node1（公网 117.50.46.41 / 内网 10.60.104.186）+ node2（内网 10.60.215.136，无公网走 node1 跳板），各 **2×RTX4090 24GB / 64GB 内存**，同地域同 VPC/子网（CC≥8.0）
- [ ] **2. 网络/防火墙**：安全组放行 SSH 22、Ray 6379/6381/6382/8265/10001、NCCL 动态端口（内网段全放最省事）；A800 按官方"多机 GPU 通信最佳实践"建 4 张虚拟网卡（普通卡跳过）
  - ❌ UAAA 应用仓库加速已购买 → 已弃用（GitHub 加速有效但官方源不稳，DNS 已还原，2026-08-03）；后续新机器不再配
- [x] **3. SSH 免密 + hosts（✅ 已配，node2 重装后需重做）**：`ssh-keygen` + `authorized_keys` 双向互信；`/etc/hosts` 互指内网 IP（详见条目 0 排坑）
- [ ] **4. 环境搭建（`scripts/setup_ucloud_uniagent.sh`，两台复用）**
  - 版本链（实测可用）：torch 2.7.1+cu126（PyPI，vllm 0.10.1 强制，勿升 2.8）/ vllm 0.10.1（verl 0.9 传 `logprobs_mode`，仅 vllm≥0.10 支持）/ transformers 4.57.6 / verl 0.9.0.dev（uni-agent 捆绑）/ ray 2.56.1 / TransferQueue==0.1.9
  - ✅ **node1 版本链升级（2026-08-04）**：torch 2.9.0+cu128 / vllm 0.11.1（多机硬性要求 vllm≥0.11.1，见 6.3 要点 1）；升级脚本 `scripts/upgrade_vllm_0111.sh`（node1 实测：vLLM 引擎 7B tp=2 冒烟通过、NCCL 双机 3.08 GB/s）；**node2 不单独装环境，直接克隆 node1 镜像**
  - ⚠️ verl 补丁（已内嵌 setup 脚本，幂等）：① py3.10 StrEnum 兼容；② 单卡 fsdp2 跳过冗余 `module.state_dict()` 拷贝+回灌
  - 模型：`$HOME/models/Qwen2.5-Coder-7B-Instruct`（hf-mirror 下载）
  - ✅ node1（117.50.183.168，RTX 4090 24G / 32G 内存）已装完（2026-08-03）：全套环境 + 24GB swap（fstab）+ vm.swappiness=10；清华 pip 源 + HF 镜像已永久配置；重建脚本已上传 `/home/ubuntu/`
  - ✅ **镜像保存前本地备份（2026-08-04）**：`work/ucloud_server_backup/` 已留存服务器现场（README 含恢复步骤）——版本链 torch 2.7.1+cu126 / vllm 0.10.1 / transformers 4.57.6 / verl 0.9.0.dev / ray 2.56.1、verl 两处补丁 diff（StrEnum + fsdp2 单卡）、pip freeze（236 包）、grpo_smoke.log、冒烟脚本（与本地同 MD5）；换新机器：保存镜像 → 新实例 ≥64GB 内存 → 验证 import/模型 → 更新 `work/ucloud.env` → 直接跑冒烟
  - ✅ **新机器（≥64GB 内存）不需要 25G swapfile**（用户确认）：swap 是 32GB 机器的兜底，64GB 下冒烟峰值 ~30GB 有 2 倍余量；镜像自带 swapfile 可留可删（删可省 25GB 磁盘），scale up 有压力时再加 8-16GB 小 swap 或调低 batch
  - ✅ **镜像保存前服务器已清理（2026-08-04）**：pip 缓存 ~12G + conda 包缓存 1.3G + journal/apt/__pycache__ + **swapfile 25G 已移除**（用户确认新机不需要）；清理后 /home/ubuntu 37G→24G、根盘已用 76G→39G；模型 4 分片完整、环境 import 验证 OK；镜像体积显著减小
- [x] **5. Ray 多机集群（✅ 曾拉起并验证 2 节点/4 GPU/64 CPU；vllm 升级前已 `ray stop`，node2 克隆后重做）**
  - node1（head）：`ray start --head --port=6379 --dashboard-port=8265 --metrics-export-port=0 --node-ip-address=10.60.104.186`
  - node2（join）：`ray start --address=10.60.104.186:6379 --node-ip-address=10.60.215.136`
  - 两台都必须先 `export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0`
  - 排坑：opentelemetry 版本冲突 → `fix_otel.sh`（两台都跑）；`pkill -f ray` 会误杀脚本自身，匹配写 `miniforge3/bin/ray`；`pkill -f 'pip install'` 同理会误杀执行脚本自身，匹配要精确
- [ ] **6. verl GRPO 冒烟（单卡先行 `run_grpo_smoke_ucloud.sh`，多机后续 `run_grpo_ucloud.sh`）**
  - ⏳ **单卡冒烟进度（node1，2026-08-03）**：已推进到 Training Progress 0/1（模型加载 → vLLM 生成 → 奖励打分 → sleep 释放权重 → FSDP2 训练步，全链路通了）；**最终训练步因 32GB 内存压力死机**（SSH 瘫掉，重启解决）
  - ✅ **结论：32GB 内存不够稳定跑 7B（fsdp2 + CPU offload + vllm 共存），换 ≥64GB 内存机型后再跑（用户已拍板）**；坑已全部填平，新机器装好 → 上传冒烟文件 → `bash run_grpo_smoke_ucloud.sh` 预期直接通过
  - ✅ **单机双卡冒烟脚本已备（2026-08-05）**：`scripts/run_grpo_dualgpu_ucloud.sh`（nnodes=1 / n_gpus_per_node=2 / rollout tp=2 / FSDP2 关 offload / bf16 / GRPO n=2）——node2 克隆前先在 node1 上跑这版（`grpo_dualgpu.log`），跑通后再进双机 `run_grpo_multinode_ucloud.sh`；dp=2/tp=1 的 A/B 变体注释在脚本头
  - ✅ **单机双卡实测（2026-08-05，node1 新 IP 117.50.173.5）**：链路各环节都验证到了，但**2×24GB 显存下 FSDP 参数与 vLLM 引擎无法共存**，结论与排坑：
    - 冒烟数据 2 条时 `train_batch_size` 必须 ≤2（写 4 会算出 0 步 → `ZeroDivisionError`），三个冒烟脚本已统一改 2
    - **默认 AdamW 全参不可行**：7B 状态 ~84GB（fp32 master+m+v），62GB 内存 OOM kill（dmesg 实证）；已装 bitsandbytes 0.50 备选 AdamW8bit，但 AdamW8bit 同样把机器顶死（SSH 无响应），全参训练确定走 SGD
    - **offload=False（参数驻 GPU）**：机器流畅，训练步能跑到 `Training Progress 0/1`，但训练后 vLLM 唤醒时 CUDA OOM（FSDP 7G/卡 + vLLM ≥12G/卡 > 24G）——单机双卡死路
    - **offload=True（参数驻 CPU）**：vLLM 显存够，但 FSDP2 的 CPU↔GPU 搬运 + 每步 14GB 权重同步把 CPU 打满，SSH 长时间无响应（15 分钟未恢复，慢 or 死锁未判定）——单机双卡同样不稳
    - **结论（用户 2026-08-05 拍板）**：放弃单机双卡；**正式路径 = 单卡冒烟（已验证）→ 双机 GRPO（4 GPU 分片，每 GPU 参数仅 3.5GB，offload=False 可共存）**
    - **双机首跑排坑（2026-08-05 实测，node1=10.60.92.91 / node2=10.60.61.9）**：
      - 双机链路本身通了：Ray 2 节点 4 GPU ✅、跨节点 worker 加载 ✅（node2 ip 出现在 WorkerDict）、vLLM 引擎双节点加载 ✅
      - **卡点 = 首次 LoRA 基座权重同步的瞬态峰值**（与 offload 开/关无关）：每节点 2 rank 全量 state_dict 物化 ~28G + vLLM 引擎内存侧加载 ~30G + Ray ≈ **60G/62G**，CPU 全核搬运 → sshd 饿死（端口通、心跳在、banner 不来）；node2 同样冲到 58G
      - 首次同步是一次性的（之后每步只传 adapter ~100MB），理论上撑过去就轻；但 15+ 分钟未恢复 ×3 次（含 offload 开关两版）
      - **第三次尝试（offload=False）结果（2026-08-05 17:13）**：node2 内存 58G→1G、GPU 清空（训练进程被清理，无 dmesg OOM 记录，疑似集群级联/内核 OOM）；node1 仍卡死 → **运行失败，需重启重跑**
      - **对策（已定，下次重跑必做）**：① 两台加 16-24GB swapfile（峰值缓冲，防 OOM——实测 60G 峰值无 swap 必挂）；② `checkpoint_engine.update_weights_bucket_megabytes` 2048→512（已改脚本）；③ Ray 启动带 `RAY_memory_monitor_refresh_ms=0`（已做）；④ 若仍卡死，控制台抓 `top`/`dmesg` 判断"极端慢 vs 死锁"
      - **第四次尝试（补丁版，2026-08-05 17:32）**：
        - ✅ swap 20G 生效：峰值 61G 内存 + 15G swap 成功兜住，全程 SSH 在线、无 OOM
        - ✅ **IndexError 根因定位并修复**：bucket 512 时 embedding(~1.09GB) > bucket 触发 `_direct_send_large_weight`，对 CPU 张量 reduce_tensor 句柄不足 7 项 → `rebuild_ipc list_args[6]` 越界；修复 = bucket 改回 2048 + 补丁 `scripts/patch_verl_ipc_cpu.py`（发送前 CPU→CUDA，两台已打）
        - ❌ 但 vLLM 引擎 init 又崩（EngineCore_DP1 初始化失败，WorkerProc 异常）——内存压力下双引擎并发初始化的偶发
      - **结论：62GB RAM 是硬瓶颈，升级内存（128GB）是最直接解法**（显存 24G/卡一直够，峰值 13G；瓶颈纯粹是 RAM）；升级后配置基本不用改，重跑即可
  - ✅ **单机 LoRA GRPO 跑通（2026-08-05，node2：1×RTX 4090 48GB / 94GB RAM，IP 117.50.197.46 / 10.60.46.121）**
    - 机器 = 镜像恢复成功（torch 2.9.0+cu128 / vllm 0.11.1 / verl 0.9.0.dev / IPC 补丁全在）
    - 脚本 `scripts/run_grpo_single_lora_ucloud.sh`（nnodes=1 / tp=1 / LoRA rank=32 / AdamW fp32 / offload 关 / batch=2 / n=2 / lr=1e-5 / **fused kernels 关**）
    - **里程碑：Training Progress 100% 1/1**，一步 58.4s（gen 45.3s / old_log_prob 6.3s / update_actor 4.5s / **update_weights 2.1s = adapter 热插生效**）；critic/score/mean=1.0；训练显存峰值 18.9G、内存峰值 65G（94G 无 swap 兜住）
    - **排坑：`use_fused_kernels=True` 与 LoRA(PEFT) 冲突** → training 步 `aten.mm: mixed torch.Tensor and DTensor`（fused monkey patch 替换 linear 层后 LoRA 张量混算）；**LoRA 下必须 fused kernels=False**（多机脚本已同步改）
    - 收尾瑕疵：step 完成后最终验证阶段 DataLoader worker 被 Killed（dmesg 无 OOM，疑似 Ray teardown 清理），`Final validation metrics: None`——不影响训练步本身，后续可查
    - **node1 同配置复跑（2026-08-05）**：`Training Progress 100% 1/1 [00:49.7s]`；step 49.7s（gen 37.2s / old_log_prob 6.2s / update_actor 4.3s / update_weights 1.8s）；吞吐 65.6 tok/s；训练显存峰值 **18.77G**、训练 CPU **44.8G**（verl 内部统计）；首次同步外部内存峰值 ~65G（node2 观测同量级）
    - **单机双机基线对照**（冒烟口径 2 prompt × n=2）：node1 step 49.7s / 65.6 tok/s，node2 step 58.4s / 57.7 tok/s——差异来自生成时长（37s vs 45s），训练/同步部分基本一致

### 7. 单机完整 agentic 链路（2026-08-05 定稿：先单机跑通，多机只改参数）

背景：多机网络暂未就绪（两台不同 VPC，见上文）；单机训练引擎已验证（纯 verl LoRA GRPO 100%）。接下来在**单机**上把「采样 → runner（用户 agent）→ agentic 多轮 GRPO 训练 → 真实 reward」整条链路跑通，多机只是并行参数扩展。

- [~] **7.1 Runner 编写/接入（核心）**
  - ✅ **runner 骨架已完成（2026-08-05）**：`platform/uni_agent_ext/agents/mini_swe_agent_runner.py`（编译通过）——仿 `claude_code_runner.py`，职责拆分：`extract_task`（SWE-bench 元数据）/ `create_task_sandbox`（腾讯沙箱工厂，扩展点）/ `build_agent_command`（沙箱内跑 `mini-extra swebench-single` 指向 Gateway）/ `evaluate_reward`（test_patch + FAIL_TO_PASS，不注入 agent）/ 主流程（建沙箱→跑→打分→上报 reward_info→清理）；mini-swe-agent 沙箱内用 `environment_class=local`
  - ✅ **Gateway 暴露方案（用户 2026-08-05 定：公网 IP）**：训练机 Gateway 监听 `0.0.0.0:<port>`（选非默认端口如 38197/8001）→ UCloud 安全组放行该端口 + 腾讯沙箱出口可达 → 沙箱内 `api_base=http://<训练机公网IP>:<port>/v1`；鉴权用 Gateway 的 api_key（任意非空即可）；**安全提示：训练结束即关安全组端口，勿长期暴露**
  - 待办：① 沙箱内 mini-swe-agent 安装/预装方案（现为可选 `MSA_INSTALL_AGENT=1` 现场 pip install，正式建议预装镜像）；② mini-swe-agent `local` 环境类 + 生成配置 schema 需上机实测对齐；③ 部署：`uni_agent_ext` 包需放到训练机 PYTHONPATH
- [ ] **7.2 任务数据（tools_kwargs + task config）**
  - SWE-bench Lite 样本 → `raw_prompt` + `tools_kwargs.task`（序列化 Task Config）；冒烟先用 2 条，逐步扩
  - sandbox = 腾讯云 Agent Runtime（swebench 工具类型，`platform/uni_agent_ext/sandbox/tencent_agent_runtime.py` 已打通）
- [~] **7.3 Agentic 训练配置**
  - ✅ **脚本已写（v0.3.0，对齐官方 quickstart 接线）**：`uniagent-lighting/scripts/run_grpo_single_agentic_ucloud.sh`——`multi_turn.enable=True` + `agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter` + `custom.agent_framework`（gateway_count=1 / agent_runners.mini_swe_agent.runner_fqn=uni_agent_ext.agents.mini_swe_agent_runner.mini_swe_agent_runner / dispatch=ray_task / max_concurrent_sessions=2 控沙箱成本 / mask_unfinished_episode=False / use_reward_loop_worker=False）+ `reward.reward_manager.name=naive`（TQ rm_scores）+ 续训启用
  - 训练侧沿用定稿：LoRA rank=32 / AdamW fp32 / offload 关 / **fused kernels 关** / 梯度检查点 / batch=2 / n=2 / lr=1e-5
  - 待上机验证：TOOL_PARSER（hermes vs qwen3_coder，需匹配 Qwen2.5-Coder chat template）、agentic 数据 schema、Gateway/隧道、uni_agent_ext 部署
  - ✅ **2026-08-06 深夜实测进度（改造仓 uniagent-lighting 已同步，v0.12~v0.14）**：
    - 架构修正为 **agent-outside**（思路 1.9）：harness 在训练机本地驱动 mini-swe-agent，沙箱只是执行环境
    - 全链路已通：SWE-bench 沙箱实例（StartSandboxInstance+E2B connect）→ 文件写入 base64 回退 →
      tencent_e2b attach 模式 → 本地 subprocess mini-extra → agent 实际运行 4~6 分钟 → reward 评估
    - 逐个排坑（均已 commit）：.pth 父目录、py310 typing.NotRequired、e2b SDK、E2B_API_KEY 映射、
      sweb 实例接入、write_file 回退、attach cleanup 不销毁实例、**v0.14.0 隧道方向修正**
      （harness 本地直连 Gateway，无需沙箱内隧道）+ 真实 model_name
    - **最后状态**：v0.14.0 修复"空轨迹"（隧道打空端口 + model_name=default）后**尚未验证**——
      明天第一步 = 在 node2 重跑 `run_grpo_single_agentic_ucloud.sh`，预期：agent 调本机 Gateway →
      轨迹被 session 记录 → reward（实例存活）→ GRPO 训练步；若 TOOL_PARSER/数据 schema 有问题再排
    - ✅ **2026-08-06 上午：单机 agentic GRPO 全链路跑通（v0.15.0 后验证）**
      - v0.15.0 修复 LiteLLM `Missing credentials`（api_key=EMPTY + OPENAI_API_KEY 兜底）
      - 结果：`num_success_sessions=2 / outputs=2 / failed=0`，**Training Progress 100% 2/2**，
        5 轮多轮轨迹、LoRA 更新 + adapter 同步 + checkpoint 全部执行；reward=0（agent 未产出通过补丁，
        冒烟可接受，7.4 再上真实 reward 调优）
      - **意义：harness 在外 + 腾讯沙箱 + Gateway session 记录 + GRPO 训练闭环完整成立**
    - ✅ **7.4 真实 reward 完成并验证（2026-08-06，v0.16.x）**：
      - runner 重写：健壮测试解析（兼容字符级乱码）、git apply --3way 回退、分级打分、可选 P2P
      - 实测：FAIL_TO_PASS 到达时是字符级列表（verl 序列化），防御性合并后还原真实 pytest node id
        （`tests/unittest_nodes.py::AsStringTest::test_as_string_unknown`），评分 0（agent 未修出补丁，符合预期）
      - 下一步：换 HumanEvalFix 数据后扩大冒烟样本量跑像样的一轮（见 §8）；然后多机（VPC 网络）
- [x] **7.4 真实 reward（测试通过率）**（✅ 2026-08-06 v0.16.x，见上）
  - 替换 `reward_smoke.py`：腾讯云沙箱跑 test_patch 的 pytest FAIL_TO_PASS → reward 0/1（或分级）；**不注入 test_patch 给 agent**（无测试泄露，用户决定）
- [x] **7.5 单机验证**（✅ 2026-08-06 v0.15.0/16.x 全链路跑通，全样本 2026-08-08~09）
  - 单机跑通 1 步 agentic GRPO：agent 在沙箱多轮工具轨迹 → reward → LoRA 更新 → adapter 同步；记录步耗时/吞吐/显存/内存峰值（基线：纯 verl 单机 step ~50s、训练显存峰值 18.8G、内存峰值 ~65G）
  - ✅ **续训机制确认（2026-08-05）**：verl `trainer.resume_mode=auto` 自动从 `default_local_dir` 最新 checkpoint 续训（`latest_checkpointed_iteration.txt`）；需 `save_freq>0`。单机脚本已加：`save_freq=1 / resume_mode=auto / default_local_dir=/home/ubuntu/swe-rl/checkpoints/single_lora_smoke`——沙箱/机器中断后重跑同一脚本即自动续，agentic 脚本沿用
- [ ] **7.6 多机扩展（后续）**
  - 同 VPC 网络就绪后：`nnodes=2 / n_gpus_per_node=1 / dp=2 / tp=1`（脚本已按 2×48G 改好），配置不变只改并行参数 + Ray 集群
  - ⛔ **双机网络前提不满足（2026-08-05 确认）**：node1（10.60.173.163）与 node2（10.60.46.121）**不在同一 VPC**（虽都是 10.60.0.0/16、网关 10.60.0.1，但子网 173/46 双向 ping 100% 丢包）；公网 IP 为 NAT 映射（eth0 仅内网 IP），NCCL 走公网实测不可行（socket err=-3，rank 超时）→ **多机必须同 VPC/子网**（新建时选同 VPC，或控制台/客服做内网互通）。网络就绪前：单机路径已通（node2 完整跑完 GRPO 步），可先跑单机基线
    - **恢复后 checklist（从 2026-08-04 镜像创建 node1+node2）**：① 更新 `work/ucloud.env` 新 IP；② 上传 `run_grpo_multinode_ucloud.sh`（LoRA 定稿版）+ `fix_multinode_hosts.sh`（UCloud 版）+ `nccl_multinode_test.py`；③ 两台跑 fix_multinode_hosts.sh（写 10.60 内网映射 + 删 127.0.1.1）；④ SSH 密钥重做（ssh-keygen -A + 用户密钥互信）；⑤ 起 Ray（node1 head 10.60.104.186:6379 + node2 join，带 GLOO/NCCL_SOCKET_IFNAME=eth0）；⑥ node1 跑 `run_grpo_multinode_ucloud.sh`（**LoRA rank=32 / AdamW fp32（无需 bitsandbytes）/ CPU offload 全开 / 梯度检查点 / fused kernels / dp=2 / tp=2 / batch=2**）
  - ✅ **配置定稿（2026-08-05 用户三改）**：放弃全参，改 **LoRA 微调 + 默认 AdamW(fp32) + CPU offload 全开 + 梯度检查点 + fused kernels**；LoRA 可训练参数 ~0.05B，fp32 优化器状态 ~560MB 分片后每卡 ~140MB，**不需要 8bit 也不需要 bitsandbytes**；CPU 压力大导致 SSH 断连时用 UCloud 控制台 Web shell 操作（用户确认）

### 8. 换数据集：HumanEvalFix（2026-08-06 定稿，下一步执行；agent 不改）

背景：Qwen3-8B 在 SWE-bench 长程任务上行为退化已实证（60 轮不修改代码、循环执行命令；
simple-bench 极简实验已回滚）；**用户拍板：agent 不改（保持 mini-swe-agent harness），
通过降低任务难度换取 8B 可出结果**。

- **数据集**：`bigcode/humanevalpack` 的 Python 修复子集（HumanEvalFix）——单函数
  buggy 代码 + 单元测试；远小于 SWE-bench（短 prompt、少轮、单文件）；8B 级模型有公开
  pass@1 基准（Granite 8B ≈ 25~48%），**60 轮内可出结果**，正好绕开长程探索退化
- **构造步骤（本地可做，无需 GPU）**：
  1. hf-mirror 拉 `bigcode/humanevalpack`（`HF_HUB_DISABLE_XET=1`），过滤 Python 子集
  2. 每样本生成 `solution.py`（buggy 代码）+ `test_solution.py`（隐藏测试）+
     FAIL_TO_PASS / PASS_TO_PASS 清单
  3. 转 verl agentic 数据：`raw_prompt`（修复指令）+ `tools_kwargs.task`（文件注入：
     预写 solution.py + test_solution.py 到沙箱工作目录）+ `reward_model.ground_truth` +
     `ability`（沿用 `make_agentic_data.py` 的 schema）
  4. 冒烟 2 条 → 3~5 条验证 8B 通过率（预期 ≥50%）
- **runner 改动**：`uni_agent_ext/agents/mini_swe_agent_runner.py` 恢复/新增"任务文件
  注入"路径（建沙箱后先写 solution.py / test_solution.py 再启动 agent）；reward 沿用 7.4
  已通的 pytest FAIL_TO_PASS 真实打分（无测试泄露：test_solution.py 只在 reward 阶段注入）
- **验收标准**：3~5 条样本至少 1 条修出通过补丁 → 跑一轮 GRPO 观察 reward 分布出现
  组内差异（advantage ≠ 0）→ 支撑完整训练链路，作为校招亮点（agentic 修复 + 沙箱系统）
- **分阶段路线（2026-08-06 用户拍板）**：
  - **阶段一：单机 48G + 96G，先答"Qwen3-8B 有没有可训练奖励"**
    - 冒烟数据扩到 5~8 条：`make_humanevalfix_data.py --train-num 8 --val-num 2`
      （死循环自动过滤；train/val 路径已可在 `run_grpo_humanevalfix_ucloud.sh` 里用
      `TRAIN_FILE/VAL_FILE` 环境变量覆盖，默认 `/home/ubuntu/swe-rl/data/`）
    - 跑 `run_grpo_humanevalfix_ucloud.sh`：48G 卡 offload 关 / util 0.5 / LoRA rank=32 /
      fused kernels 关；显存峰值 ~19G、内存峰值 ~65G，96G 无压力（已实测基线）
    - **验收**：`num_success_sessions > 0` 且部分样本 reward > 0（组内差异 →
      advantage ≠ 0）；若全 0 → 先别训练，回本地 `run_humanevalfix_local.py`
      换 8B 端点跑 2~3 条轨迹诊断（工具调用格式问题 vs 模型能力问题）
  - **阶段二：双机 24G + 96G，多机测试**
    - 硬前提：两台**同 VPC/同子网**（UCloud 新建时选同一 VPC；上次跨 VPC 内网
      ping 不通、NCCL 走公网 socket err=-3，这是唯一没过的关）
    - 形态：每节点 1×24G，dp=2 / tp=1，LoRA + offload=True + fused kernels 关；
      每节点内存峰值 ~50–60G，96G 够，swap 留 16G 保险（每步 16G CPU↔GPU 搬运
      会短暂压满 CPU/SSH，用控制台 Web shell）
    - 配置只改 `nnodes=2` + Ray 集群 + hosts，训练参数不变（见 7.6）
- **成本**：单条远小于 SWE-bench（prompt 短、轮数少），腾讯沙箱按秒计费更低
- **已完成（2026-08-06，v0.28.x，本地可做部分全部落地）**：
  - `scripts/make_humanevalfix_data.py`（**新增**，原 `make_agentic_data.py` 保留不动）：
    拉 humanevalpack python 子集 → `solution.py`（prompt+buggy_solution）+ `test_solution.py`
    （check(candidate) 转 pytest 单测 `test_all`；`from solution import *` 兼容测试引用
    同文件辅助函数）+ 本地 verify（buggy 必须 rc=1、canonical 必须 rc=0；死循环任务
    超时自动跳过）
  - `work/data/humanevalfix_train.jsonl`（3 条）+ `humanevalfix_val.jsonl`（2 条）已入库
  - runner 新增 `humaneval_fix` 分支（swe_bench 原路径不变）：沙箱 /testbed git 仓库 +
    solution.py 注入（`git add -A` 保证提交时 `git diff` 有输出）+ mini-swe-agent API
    直连（绕开 swebench-single 数据集硬编码）+ reward 阶段写隐藏测试（无测试泄露）
  - `scripts/run_grpo_humanevalfix_ucloud.sh`（数据/实验名/checkpoint 目录与 agentic 区分）
  - v0.28.1：`scripts/run_humanevalfix_local.py`（腾讯 E2B 沙箱 + 百炼 API 本地冒烟采样，
    复用单沙箱逐样本；不依赖 Gateway/训练机）；runner 修复（主线程预导入 tencent_e2b、
    agent_class 默认 default、无 ray 可 import、错误带 traceback）
  - **实测（2026-08-06，qwen3.7-plus）**：3/3 样本全修好，**每条约 7 轮交互、35~43s、
    reward=1.0**（读码→复现→改→验证→提交）；轨迹
    `work/swebench/humanevalfix_humanevalfix-Python-{61,104,105}.traj.json` —— 说明
    任务/沙箱/reward 链路正确，**8B 通过率与 reward 组内差异留给阶段一上机验证**
  - **阶段一上机验证（✅ 2026-08-08，新机 117.50.81.187 = 单卡 48G + 94G 内存）**：
    - 首跑（train8 原始提示词）：**4/4 reward=0，止损停训**；解码轨迹定位根因 =
      **Qwen3-8B 用 `echo '...' >> solution.py` 逐行重建整个文件**（13+ 轮连 docstring
      都没写完，引号报错重试、两次 "No tool calls found"，solution.py 语法不完整）——
      工具调用格式本身正常（hermes parser 解析成功、bash 执行成功）
    - 修复（v0.28.3）：`make_humanevalfix_data.py` 提示词加 **heredoc 约束**
      （整文件重写必须一条 `cat > solution.py <<'PYEOF' ... PYEOF`，禁止逐行 echo）
    - 重跑 train3（resume 续训，覆盖 P61/P104）：**P61 4/4=1.0、P104 1/4=1.0（3/4=0）**；
      step3 指标 reward mean=0.2 / min=0 / max=1，**advantage≠0（max 1.5 / min -0.5）**，
      num_turns 7~25、response_len max 6941、actor 显存峰值 11.9G / CPU 63.7G
    - **结论：验收通过 ✅ —— Qwen3-8B 在 HumanEvalFix 上存在可训练奖励（组内差异 → GRPO 有梯度）**
    - 遗留：① resume 续训混入旧 run checkpoint，P105 未覆盖 → 需清 checkpoint 干净复跑
      train8 记录完整通过率；② 训练收尾时 DataLoader worker 被 OOM kill（CPU 峰值 63.7G
      接近 94G 上限，扩批/双机时注意内存预算）
- **服务器状态（2026-08-08，2026-08-11 更新）**：新机 **117.50.189.37**（单卡 48G +
  94G，无 swap）已恢复镜像、humanevalfix 数据（161+3+2）、腾讯沙箱凭据全部就绪；
  **全样本训练已完成（2026-08-08~09，26 步，见 §9）+ 投机 run 完成（25 步，见 §9.1/9.3）**；
  下一步 = 单机黑盒训练验证（§G）→ 阶段二双机全异步（同 VPC/子网）

### 9. 全样本单机 GRPO（✅ 2026-08-08~09 完成，止于 26/50 = 5 epoch + 1 步）

### 9.1 投机解码全样本训练 run（2026-08-09 启动，进行中）

- **目标**：与 2026-08-08~09 全样本 run 同配置（train161 / batch 32 / mini 16 / micro 4 /
  并发 64 / vllm max_num_seqs 128 / util 0.8 / 5 epoch / ckpt keep 1 / resume auto），
  仅新增：**gateway 修复（v0.31.8 补丁）+ LoRA merge（lora.merge=True）+ EAGLE-3 投机解码**，
  对比 step 时长 / 吞吐 / 通过率（vs 修复前 26 步基线）
- **关键前置（已落地）**：
  - LoRA×SD 互斥 → `actor_rollout_ref.model.lora.merge=True`（verl FSDP2 逐层 merge +
    全量 state_dict 同步，vLLM enable_lora=False；代价 = 每步全量 ~15GB refit，待实测）
  - `+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config='{"method":
    "eagle3", "model": "…/Qwen3-8B-speculator.eagle3", "num_speculative_tokens": 3,
    "draft_tensor_parallel_size": 1}'`（Hydra 注意：JSON 值需单引号包字符串 + `+` 前缀，
    否则按 dict 解析或 "not in struct" 报错）
  - 日志/checkpoint 全隔离：`grpo_humanevalfix_spec.log` + `logs/humanevalfix_spec/` +
    `checkpoints/humanevalfix_spec/`（不碰 final / 旧 logs）
- **启动即踩坑：EAGLE-3 下 logprobs 全丢（RL 生命线问题，已修复）**
  - 现象：Gateway `RuntimeError: backend logprobs must align with token_ids: got 0
    logprobs for 26 tokens`（会话全部失败）
  - 根因（小测试定位）：vLLM 0.11.1 + EAGLE-3 下 `SamplingParams.logprobs=0` 只返回
    2/32 个 logprobs（vllm#30059 的 top_logprobs=0 bug，v0.12 才修）；verl
    `vllm_async_server.py` 恰好 `sampling_params["logprobs"] = 0 if ... else None`
  - 修复：verl 一行 `0 → 1`（`patches/verl_vllm_logprobs_spec_fix.patch`，服务器
    verl commit 5fa045e）；实测 logprobs=1/3 返回 32/32 ✓（=1 时开销可忽略）
  - 教训：TODO §C 6.5 预判的"spec decode 下 logprobs 精确性"风险真实存在，但问题不是
    数值不精确而是 0 全丢；logprobs>=1 后理论无损（rejection sampling 保分布）
- **step 1 实测（✅ 2026-08-09 14:25）**：128 会话 / reward mean 0.301（基线 0.247）/
  gen 24.1min / update_actor 15.0min（batch 2x 合理）/ **update_weights 53s（merge 全量
  15GB refit，vs as-adapter 2.5s；每步多 ~50s，占 step 2%，可接受）** / step 45.3min /
  **吞吐 301 tok/s（基线 213.7，+41%）** / CPU 峰值 42.2G（健康）；per-sample 耗时
  1.42min vs 基线 2.13min（**-33%**）；5 epoch 25 步预计 ~18-19h
- **全程 25 步平均对比（✅ 2026-08-10 训练完成后统计）**：rollout 生成吞吐
  **282.4 tok/s vs 基线 199.2（+41.7%）**（watcher `throughput_tok_s` 口径；日志
  `perf/throughput` 口径 271 vs 199，+36%）；每 token 生成延迟 **4.05ms vs 6.70ms
  （-39.5%）**；每步生成耗时 1064s vs 1433s（-25.7%）。分阶段：step1-5 +42% /
  6-15 +47% / 16-25 +36%，提速全程稳定。前提：投机 run 每步生成 token 更多
  （48.4 万 vs 42.5 万，轮数下降但响应变长），更大生成量下仍快 ~40%，EAGLE-3
  收益扎实（与 step 1 +41% 一致）
- **状态（2026-08-09 14:30）**：step 2 rollout 进行中；逐步统计 watcher 已挂
  （`collect_grpo_stats.py --watch` → `logs/grpo_stats_spec.jsonl`）

### 9.3 投机 run 训练完成（✅ 2026-08-10 09:38，25/25 步 = 5 epoch）

- **merge bug 修复后重跑全量 25 步**（2026-08-09 21:14 启动，2026-08-10 09:38 完成，
  中间经历 2 小时 UCloud 公网断连，训练未受影响）
- **关键过程（修复验证）**：
  - step 1-2：基座水平（reward 0.32/0.24、轮数 25.6/23.6）——正常起步
  - step 3 起：**轮数开始下降、reward 上行**（17.5 轮 / 0.482），与基线轨迹完全重合
  - step 6-25：reward 中枢 ~0.55（0.42-0.70 波动），轮数降到 8.8-13（策略高效收敛）
  - 后 10 步（16-25）：0.60/0.65/0.59/0.64/0.45/0.61/**0.70**/0.63/0.66/0.68
- **对比基线 26 步 run**：修复后走势逐 step 几乎一致（step 3 0.482 vs 0.461、step 9
  0.620 vs 0.659、step 14 0.652 vs 0.677、step 22 0.702 vs 0.705）——merge bug 修复
  后训练完全恢复正常；投机解码（EAGLE-3）在训练链路中无副作用
- **产出**：checkpoint `global_step_25`（16G，旧 step 已按 keep=1 清理）；最终权重
  `/home/ubuntu/models/Qwen3-8B-final-spec`（convert_verl_lora_to_hf.py 合并）
- **训练日志完整性核对（✅ 2026-08-10 10:45）**：`grpo_humanevalfix_spec.log`
  （10.8MB，最后写入 09:36:55）完整无断档：25/25 步进度与指标行全在、无缺步；
  配套 `logs/grpo_stats_spec.jsonl`（39 行）覆盖到 step 25（reward mean 0.677 /
  128 sessions / 0 failed）；checkpoint `global_step_25`（16G）与
  `latest_checkpointed_iteration.txt` 均已落盘。EAGLE-3 确认生效（vLLM 侧
  speculative_algorithm=EAGLE、num_draft_tokens=4、num_steps=3）。唯一瑕疵：
  退出收尾时 `DataLoader worker (pid 428257) killed by signal: Killed`
  （OOM killer，且无干净 "Training completed" 横幅），但发生在 step 25 指标与
  ckpt 落盘之后，不影响训练结果
- **评测结果（✅ 2026-08-10 11:26，run 重跑）**：**final-spec 133/161 = 82.61%**
  （n=1 / temp 0.8 / 161 条 / 并发 24，与基线评测口径一致；全程 0 沙箱错误）
  - 对比：基座 Qwen3-8B 76.4%（123/161）、baseline final 83.2%（134/161）→
    投机 run 与基线 final 几乎持平（差 1 条），RL 提升有效（+6.2pp vs 基座）
  - run1-4 失效复盘：首轮 41/161 = 25.5% 无效——115 条因腾讯沙箱 E2B 批量故障
    未测（93 `AUTHENTICATION_FAILED` + 22 `resource does not exist`，10:42 起）；
    真实评测的 46 条中 41 通过（89%）；已删除全部失效产物（run1-4 日志/汇总/轨迹），
    保留 08-09 baseline 成功结果
  - 重跑：复用现有 vLLM（`Qwen3-8B-final-spec`，确认 = step25 转换权重），
    `run_eval_only.sh` 并发 24（与 baseline 一致），11:02 启动 → 11:26 完成
  - 脚本 `eval_spec_final.sh`（注意：eval_humanevalfix.py 的 load_envs 路径推断在
    服务器布局失效，脚本内直接 source tencent_sandbox.env 注入凭据）

### 9.1a 重大 bug：merge=True 时 rollout 同步基座权重（2026-08-09 发现并修复）

- **现象**：投机 run 训练 11 步 reward 始终在基座水平波动（0.28-0.44）、agent 轮数
  不降（~23 轮）、策略不收敛；对比基线（as-adapter）同阶段 reward 0.5-0.7、轮数
  降到 10-14；**前 4 步两边一致（都是基座初始行为），step 5 后基线上升我们徘徊**
- **根因（verl PR #7014 bug，我们的 78bba31 未包含）**：`get_per_tensor_param()` 的
  merge 分支在 `merged_lora_context` 内提取 `state_dict()`（返回 live 存储**别名**），
  但返回**生成器**，consumer（`engine_workers.update_weights`）在 context 退出
  （基座权重已恢复）后才迭代物化 → 每步同步给 vLLM 的是**没有 LoRA 的基座权重**，
  训练梯度从未作用到 rollout
- **修复**：backport #7014——新增 `_merged_lora_per_tensor_param()` 生成器，在 context
  内逐个 `full_tensor()`/`clone()` 物化（`patches/verl_merged_lora_materialize_fix.patch`，
  服务器 verl commit c2049af，v0.32.2）
- **处置**：11 步无效训练 checkpoint/logs 改名备份后已删除（用户确认）→ 从头重跑
- **修复后 5 步验证（✅ 2026-08-10 00:33）**：reward 0.32→0.24→0.48→0.44→0.46
  （脱离基座水平，稳定 0.44-0.48）、轮数 25.6→23.6→17.5→19.3→16.7（持续下降）、
  吞吐 ~300 tok/s；与基线 step 5（0.50/12.7 轮）趋势吻合 → **修复彻底生效**
- **经验**：merge 配置的权重同步必须"context 内物化"；换 merge/全参配置后先跑 3-5 步
  验证轮数下降/reward 上行，再投入长训

- **用户拍板配置**：全样本 = HumanEvalFix 全部 164 条（含 3 条死循环样本，
  `--include-unverified` 保留并标记 `verified=false/deadloop=true`，reward 按 0 计入
  指标）；`TOTAL_EPOCHS=10`、`TRAIN_BATCH_SIZE=16`、`PPO_MINI_BATCH=8`、
  `PPO_MICRO_BATCH=2`、`MAX_CKPT_KEEP=1`（checkpoint 一直覆盖只留最新，磁盘稳定 ~16G）、
  `resume_mode=auto`（长训续跑）
- **提速改造（v0.30.1）**：① `TENCENT_SANDBOX_SKIP_TMUX=1` 跳过沙箱内 tmux apt-get
  （该步骤常卡到 E2B 180s 超时，每会话白耗 ~3 分钟；mini-swe-agent harness 自带
  pexpect shell 不需要它）；② `CONCURRENCY=16` + `VLLM_MAX_NUM_SEQS=16` +
  `VLLM_GPU_MEM_UTIL=0.7`（沙箱在腾讯云不占本机，vLLM 侧同步扩容 KV）
- **崩溃修复（v0.30.3）**：`patches/verl_debug_metrics_logprobs_guard.patch`——
  batch=16 时个别 episode（重试/失败）缺 `rollout_log_probs`，整个 batch 缺键 →
  `calculate_debug_metrics` KeyError 崩训练；补丁缺键时跳过 debug 指标
- **目录混杂教训**：多次重启没清 `logs/humanevalfix/`，三个 run 的会话目录叠加进
  同一个 step_1（84 个目录 ≠ 64）——**每次重启前必须清空该目录**，否则统计被污染
- **step 1 实测（并发 16）**：16 样本 / 64 会话 / 19 条 reward=1.0（30%）/
  reward mean=0.247、advantage≠0；每步 34 分钟（rollout 22min + update 8.3min +
  存盘 1min），吞吐 **213.7 tok/s**（并发 4 时代 62 的 3.4×）；CPU 峰值 63.8G/94G
  稳定（无 swap 也扛住）；checkpoint global_step_1 已存
- **逐步统计**：`logs/grpo_stats_full.jsonl`（watcher 每 30s 增量，每 step 一行：
  样本、会话数、rollout/训练/step 时长、reward mean/min/max + 逐条、advantage、
  num_turns、tokens、throughput、grad_norm）；100 步 ETA ≈ **56h**
  （164 样本 × n=4 × 10 epoch = 6560 会话，会话吞吐是硬瓶颈）
- **训练中止与最终权重（2026-08-09）**：用户定 5 epoch 即止，实际止于 step 26
  （5 epoch + 1 步）；`max_actor_ckpt_to_keep=1` 轮换掉了 step 25，保留 **step 26**，
  目录改名 `checkpoints/humanevalfix/final`（26 与 5 epoch 仅差 1 步，视为 5 epoch 基准）
- **评测结果（方案 A，n=1、温度 0.8、161 条、并发 24、vLLM 参数与训练一致）**：
  - **基座 Qwen3-8B：123/161 = 76.4%**；**final：134/161 = 83.2%**（+6.8pp；
    26 条失败→通过、15 条通过→失败，净 +11；per_test 全为真实 pytest PASS）
  - 工具链：`scripts/convert_verl_lora_to_hf.py`（verl FSDP2+LoRA → 合并 HF 模型，
    注意 DTensor 需 `to_local()`、保留 bf16）、`scripts/eval_humanevalfix.py`
    （并发沙箱 n=1 通过率评测；v0.31.4 起真并发 asyncio.gather + return_exceptions）
  - **腾讯沙箱配额**：实测同时存活上限 ~25（50 核 / 每沙箱 2 核），评测并发取 24；
    一次性 gather 建 64 个会触发 `LimitExceeded.CPU`（训练渐进派发所以没触发）
- **重大发现：训练 gateway 双解析压低 rollout 成功率**（详见 §C 6.5 新增项）：
  训练 reward 本身正确（1972 PASS / 1375 FAIL、0 collect 错误），但 gateway 对
  hermes 工具调用**二次解析**累计 **1,601 次解析错误事件**（日志含堆栈约 2 万行）→
  同一批 32 样本：
  训练 step1 per-session 通过率 29% vs 干净评测 81% → **训练指标曲线被系统性低估**；
  最终模型 83.2% > 基座 76.4% 证明 RL 有效，但真实提升幅度小于曲线给人的印象

### 6.1 当前训练配置（`run_grpo_smoke_ucloud.sh`，单卡 7B / 4090）

```bash
# 环境变量
export VLLM_USE_V1=1                 # verl 0.9 只用 vllm V1（需 CC≥8.0）
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_memory_monitor_refresh_ms=0  # 关 Ray OOM 杀手：95% 阈值会抢在 swap 前杀进程
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1

# 数据（冒烟口径）
data.train_files=smoke_train.jsonl    # 2 条（SWE-bench Lite）
data.val_files=smoke_val.jsonl        # 1 条
data.train_batch_size=2  data.val_batch_size=1
data.max_prompt_length=1024  data.max_response_length=512
data.filter_overlong_prompts=True  data.truncation=error

# 模型 / actor（FSDP2 + 原生 CPU 卸载）
actor_rollout_ref.model.path=Qwen2.5-Coder-7B-Instruct
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.model.override_config.attn_implementation=sdpa  # 显式关 flash-attn，免编译
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.actor.fsdp_config.offload_policy=True   # 权重常驻 CPU，rollout 时 GPU 让给 vllm
actor_rollout_ref.actor.fsdp_config.model_dtype=float16
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.optim.optimizer=SGD               # 省内存：无 Adam 状态
actor_rollout_ref.actor.ppo_mini_batch_size=2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.use_dynamic_bsz=True  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
actor_rollout_ref.actor.use_kl_loss=False                 # 跳过 ref 模型，省 ~14GB
actor_rollout_ref.actor.entropy_coeff=0

# rollout（vllm，与 FSDP2 错峰共用 24GB 显存）
actor_rollout_ref.rollout.name=vllm  actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.8      # 7B bf16 权重 14.3GB，0.6 不够（KV 分不到）
actor_rollout_ref.rollout.n=4                             # GRPO group size
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.free_cache_engine=True          # 训练前 vllm sleep(level=2) 连权重一起释放
actor_rollout_ref.rollout.max_num_seqs=4

# 奖励（假奖励：非空回答 +1，只验证链路）
reward.custom_reward_function.path=reward_smoke.py
reward.num_workers=1

# transfer_queue（v1 trainer 无法关闭，只能减参省内存）
transfer_queue.backend.SimpleStorage.num_data_storage_units=2
transfer_queue.backend.SimpleStorage.total_storage_size=1000

# trainer
trainer.balance_batch=True  trainer.logger='["console"]'
trainer.n_gpus_per_node=1  trainer.nnodes=1
trainer.save_freq=-1  trainer.test_freq=-1  trainer.total_epochs=1
```

### 6.2 我做的修改清单

| # | 修改 | 文件/位置 | 要点 |
|---|------|-----------|------|
| 1 | FSDP1 → FSDP2 + 原生 CPU 卸载 | `run_grpo_smoke_ucloud.sh`：`strategy=fsdp2` + `offload_policy=True` | FSDP1 手动 offload 在 torch 2.7.1 单卡下失效（参数到 CPU 但 GPU 残留同尺寸缓冲，最小复现证实） |
| 2 | verl 单卡 fsdp2 加载峰值补丁 | verl `transformer_impl.py`（`work/patch_fsdp2_singlerank.py`，已内嵌 setup 脚本） | 单卡跳过 `module.state_dict()` 拷贝+回灌：加载峰值 28GB→~16GB；多卡分支保留广播逻辑 |
| 3 | vllm 显存利用率 0.6→0.8 | `run_grpo_smoke_ucloud.sh` | 7B bf16 权重 14.3GB > 0.6×24GB，KV cache 分不到报错 |
| 4 | 关 Ray OOM 监控 + swap 兜底 | `run_grpo_smoke_ucloud.sh` + 24GB swap（fstab）+ `vm.swappiness=10`（sysctl.conf） | Ray 95% 阈值提前杀进程，swap 根本用不上 |
| 5 | 数据 schema 补齐 | `scripts/make_smoke_data.py` + `work/data/smoke_*.jsonl` | verl agentic RL 需 `data_source` + `prompt`（消息列表）+ `reward_model.ground_truth`（gold patch）+ `ability` |
| 6 | 版本链修正 | `setup_ucloud_uniagent.sh` | torch 2.7.1+cu126 / vllm 0.10.1 / transformers 4.57.6；`TransferQueue==0.1.9`；StrEnum+fsdp2 补丁内嵌 |

### 6.3 关键要点（经验）

1. **版本链**：verl 0.9 传 `logprobs_mode`，只有 vllm≥0.10 支持（0.9.2 报 unrecognized arguments）；vllm 0.10.1 强制 `torch==2.7.1`（PyPI 默认 cu126），不要升 torch 2.8
   - **多机分支（2026-08-04 定稿）**：多机 GRPO 必须 **torch 2.9.0+cu128 / vllm 0.11.1**（0.10.1 下 tp=4 报 `AssertionError: multi-node MP 需 dp>1 或 vllm≥0.11.1`，dp=2/tp=2 又不认 `--master-addr/--node-rank/--nnodes` → exit 2）；升级脚本 `scripts/upgrade_vllm_0111.sh`；**verl 保持 0.9.0.dev 不降级**
2. **24GB 显存错峰设计成立**：rollout 时 FSDP2 权重在 CPU、vllm 用 0.8×24GB；训练时 vllm `sleep(level=2)` 连权重释放、FSDP2 参数上 GPU（14GB）——两者永不共存
3. **32GB 内存是瓶颈**：fsdp2 加载峰值修掉后 ~16GB 可加载，但训练步整体（WorkerDict RSS ~28GB + vllm + ray）仍超 32GB → **必须 ≥64GB 内存**；换新机后不再需要 swap
4. **数据格式**：verl agentic RL 输入 = `data_source` / 消息列表 `prompt` / `reward_model.ground_truth`；漏字段会分别在 agent loop 和 reward 阶段报 `KeyError`
5. **多机注意**：上述单卡补丁只影响 world_size=1 分支，多卡走原逻辑（rank0 广播），无需额外处理

### 6.4 多机后续

- 参数：`trainer.nnodes=2`、`n_gpus_per_node=1~2`；冒烟口径：2 条 prompt、`n=4`、`batch_size=2`、`micro_batch_size=1`、`total_training_steps=2~5`
- 验收标准：模型加载 → vllm rollout 生成 4 条 → 奖励函数打分 → 梯度更新 → loss 变化；记录每 step 耗时、GPU 利用率、吞吐
- 若多机带宽受限，后续评估 Mooncake / P2P 权重分发（阶段三）
- **镜像保存前整理（✅ 2026-08-10）**：服务器 swe-rl 80G→32G——删除 gateway_test
  checkpoint（48G，3 步 gateway 测试产物）、broken/pre-gateway 训练会话与日志
  （120M）、一次性日志与已存档 tar 包；保留 baseline `humanevalfix/final` + spec
  `humanevalfix_spec/global_step_25`（各 16G）、models（48G）、训练日志、正式评测
  （eval_final_spec 133/161）。代码仓 uniagent-lighting 同步到 **v0.33.2**
  （v0.33.1 补全 eval 结果/config/pip-freeze/修正空 val 数据 + smoke 数据；
  v0.33.2 补归 7 个服务器在用脚本，全部 sha256 核对一致），服务器 git pull 同步。
  磁盘 193G 用 107G（56%），剩余 87G 可保存镜像
- **下一步：双机测试全异步 + TQ mooncake 存储后端（2026-08-10 用户定序）**：
  保存 node1 镜像 → 恢复 node1+node2 → 双机 GRPO 冒烟（§6.4 参数）→ 验证全异步
  rollout 链路 → 评估 TransferQueue 的 mooncake 存储后端 vs 默认，对比训练吞吐/
  每 step 耗时/带宽占用，看是否提速
- **双机对照实验完成（✅ 2026-08-14，A-E 五组）**，详见 docs/训练评测分析.md §7：
  - A sync=79.4s/步（两遍干净）→ B colocate_async=77.5s（首跑干净，重跑崩）→
    C colocate+Mooncake=崩 2/2 → **D separate_async=48.1s（-39%）→
    E separate+Mooncake=48.2s（无差异但稳定）**；吞吐 25-36 → 79-167 tok/s
  - **架构定稿 = separate_async**（trainer 1 卡 + 独立 rollout 1 卡）；colocate
    的 vLLM 0.11.1 CUDA illegal memory access 为上游竞态（sleep/wake + 多轮
    resume），与 Mooncake 无关
  - **投机解码恢复**：separate_async 独立引擎 dp=1 单节点，避开 EAGLE+dp>1
    死锁（A 首炸根因），正式训练 EAGLE-3 全开
  - 排障链：mooncake_master 预启动（PATH + libcudart）/ node2 补装
    mooncake-transfer-engine / local_hostname 置空 / `MC_STORE_MEMCPY=0` /
    TQ clear_data 补丁 / verl NestedTensor num_turns 补丁（均已归档）
- **正式双机训练（⏳ 2026-08-14 启动）**：`run_grpo_dual_async_mooncake_ucloud.sh`
  = 黑盒 HumanEvalFix 161 条 + separate_async + Mooncake + EAGLE-3，batch32/mini16/
  micro4/pss2、并发 64/128/0.8、5 epoch（26 步）、save_freq=1、resume=auto；
  checkpoint `checkpoints/humanevalfix_dual_async_mooncake`、日志
  `logs/humanevalfix_dual_async_mooncake`；train3 全链路验证通过后启动，
  完成后自动接全量评估（结果写回本文件 + 训练评测分析）
- **Mooncake num_turns 13B 排查 + 离线验证（✅ 2026-08-15）**，详见
  docs/训练评测分析.md §7.6：
  - 现象：双机真实训练（128 会话）偶发 `Buffer too small for key 'X@num_turns':
    required=13, available=8`，坏 key 随机；13B 内容 = TQ msgpack 小整数打包
    （`010000000c0000000100000000` = int 0），即 num_turns 被非张量字节路径写入，
    训练端按 int64（8B）读 → 写读类型不一致
  - 官方社区无匹配 open issue；最近三个 C++ bug（#1704 TENT merge 边界 /
    #2086+#2850 TCP 撕裂写 / #2477+#2714 TENT 数据竞争）均已修复且包含在已装
    0.3.12.post1；TQ/C++ 写路径顺序与 size 透传无逻辑错误
  - **离线真实轨迹验证通过**：508 条真实轨迹构造框架同款字段 → TQ+MooncakeStore
    写 2048 keys（8 writer 并发 × 4 轮 + kv_clear 复用），TQDBG 全 Tensor 路径、
    store dump 每轮 512 个 `X@num_turns` 全 8B、0 异常 → 单机/离线无法复现 13B，
    依赖双机跨节点特定条件
  - **防御已落地**：读端 `_get_with_padded_buffers` 兜底 + 写端 num_turns size!=8
    断言日志（TQ mooncake_client，node2 待重启同步）；脚本
    `scripts/offline_mooncake_verify.py` / `scripts/repro_tq_mooncake.py` /
    `scripts/run_grpo_single_mooncake_ucloud.sh`
  - 排查期环境问题：GatewayActor 不继承 `GATEWAY_PORT`（需 ray start 前 export）、
    腾讯沙箱 CPU 配额被残留 RUNNING 实例占满（脚本字段已确认 `InstanceSet`）、
    node1 sshd 默认 `MaxStartups 10` 高并发隧道限流
- **单机在线验证 + 两个新 bug 修复（✅ 2026-08-15）**，详见
  docs/训练评测分析.md §7.7：
  - **vLLM EAGLE-3 illegal memory（与 Mooncake 无关）**：48 会话 util 0.6 验证在
    rollout 全部成功后 EngineCore 崩（`Failed to reset prefix cache ... 939 blocks
    not freed` → rejection_sampler 越界）。vLLM 0.11.1 老版本上游竞态，历史白盒
    run 均 0 次；**util 恢复 0.8（baseline/spec 同口径）后 24 会话 0 崩溃**
  - **Mooncake INVALID_PARAMS 空 slice**：`max_trajectory_length` 截断的空响应轨迹
    （response_ids 空）被 framework 照常写入，4 字段 0 字节 slice 被 master 拒绝、
    session 写入失败。修复 = framework 跳过空响应轨迹 +
    TQ mooncake_client 空 slice 告警（补丁
    `patches/uni_agent_skip_empty_response_trajectory.patch` /
    `patches/tq_mooncake_zero_slice_warn.patch`）
  - **修复后 24 会话验证全绿**：24/24 成功、0 INVALID_PARAMS / 0 Buffer too small /
    0 vLLM 崩溃；GRPO step 1 完成（202348 tokens）并保存 global_step_1
  - **环境变量坑（已写入 docs/deployment.md §4）**：Ray worker 不继承脚本内
    export，`E2B_API_KEY` / `E2B_DOMAIN` / `GATEWAY_PORT` 等必须 `ray start`
    前 export，否则 agent 起沙箱报 `AuthenticationException`（08-15 实测踩坑，
    24 会话全挂一次）
  - **为何白盒历史 run 不崩（docs/训练评测分析.md §7.8）**：baseline/spec 日志
    `transfer_queue.enable=False`（不走 TQ KV + Mooncake），故 13B / 空 slice 不
    可能出现；vLLM illegal memory 是 util 0.6 触发（spec run util 0.8 从未
    reset 失败），与白盒/黑盒无关
  - **§7.9 深挖定稿（2026-08-15 补充）**：空轨迹完整机制（链分片 + agent 不认
    finish_reason=length + 新链首轮必空，spec 实例 25 轮有效 + 26 轮空）；
    13B 完整机制（verl padding_utils Python int → TQ msgpack 13B vs int64 8B）；
    统一归因 = SimpleStorage 容忍空/任意大小，Mooncake 严格要求；空轨迹
    baseline 37/spec 45 个之前就有，换后端才引爆
- **新双机就绪（2026-08-15，镜像恢复）**：node1 外网 117.50.188.83 /
  内网 10.60.216.3（4090 48G）；node2 外网 117.50.216.11 / 内网 10.60.138.139
  （4090 48G）。两台镜像完整（models/checkpoints/补丁全部在）；
  hosts 新映射 + SSH 互信已配；三个脚本 IP 已更新（v0.48.5），双机脚本按用户
  决定改为白盒 mini_swe_agent runner。待办：mooncake_master + Ray 组网 →
  双机小样本对照（mooncake 收益 + separate_async 收益）→ 正式训练
- **✅ 双机平台化正式训练完成（2026-08-15）**，详见 docs/训练评测分析.md §8：
  - 架构：separate_async（trainer node1 + rollout node2）+ MooncakeStore +
    EAGLE-3 + 白盒 mini-swe-agent，配置与 baseline/spec 完全同口径
    （train161/batch32/mini16/micro4/并发64/util0.8/5epoch/LoRA r32）
  - 过程：25 步 7:11:40，0 硬错误；checkpoint 滚动保留
    （max_actor_ckpt_to_keep 修复 v0.48.6 + 守护进程兜底，磁盘 ≤81%）
  - **评估 83.23%（134/161）**：与白盒 baseline 83.2% 持平，验证全异步 +
    Mooncake + 投机不损失质量；**计为平台化训练结果**
  - 产物：models/Qwen3-8B-final-dual-async（合并权重）、
    logs/eval_dual_async_final{,.json,_dir}、25 步轨迹
  - 脚本：scripts/run_grpo_dual_async_mooncake_ucloud.sh（正式训练）、
    scripts/eval_dual_async_final.sh（评估）
- **全异步 + TQ/mooncake 调研结论（✅ 2026-08-11，基于本地 verl 78bba31 源码 +
  uni-agent 官方 recipe + TransferQueue 0.1.8 wheel）**：
  - **TQ 已在用**：verl v1 数据流全程走 TQ（trainer_base 无条件 kv_batch_put/get：
    rollout 输出/reward/old_log_prob/ref_log_prob/value），agent framework 层
    `AgentFrameworkWorker.tq.init()` + `async_kv_put` 写轨迹（tag finished/failure）；
    当前 backend = **SimpleStorage**（CPU 内存 + TCP）
  - **全异步可开（当前未开）**：我们训练是 `trainer_mode=sync`（默认），step 时间 =
    gen+update+杂项串行相加。官方 fully async recipe =
    `trainer.v1.trainer_mode=colocate_async`（单机）/ `separate_async`（双机）+
    `transfer_queue.enable=True` + `num_warmup_batches` + `data.train_batch_size=0/
    gen_batch_size=1`；verl 78bba31 已内置 PPOTrainerColocateAsync（partial rollout）
  - **mooncake 可换**：`transfer_queue.backend.storage_backend: MooncakeStore`
    （experimental）；TransferQueue 0.1.8（服务器已装版本）内置
    mooncake_manager/client/bootstrap；需另装 mooncake-transfer-engine + 启动 master；
    无 RDMA 时 TCP fallback，双机小数据量收益有限（传输非瓶颈）
  - **加速估算（spec run 25 步实测）**：每步 1970s（gen 1064 + update 619 + save 47 +
    杂项 ~240）；colocate_async 重叠后理想每步 ≈ max(gen, update+杂项) ≈ 1100-1200s →
    **1.6-1.7x**（分阶段：step1-8 1.89x / 9-17 1.87x / 18-25 1.78x）；双机
    separate_async + 投机解码预计 **1.8-2.2x**；uni-agent 官方 8×A100 partial
    rollout 2.1x（95.6h→45.8h），verl fully_async README 2.35-2.67x（128 GPU）
  - **实施参数（双机）**：`trainer.v1.trainer_mode=separate_async` +
    `trainer.v1.separate_async.num_warmup_batches=1~2` + `transfer_queue.enable=True`
    + `data.train_batch_size=0` + `data.gen_batch_size=1`；mooncake 先作为对照实验
    （SimpleStorage vs MooncakeStore）单独测
- **uni-agent 创新点保留情况审查（✅ 2026-08-11）**：保留 = Gateway（会话路由/
  协议转换/轨迹物化）、Token 级轨迹（response_mask/loss_mask/rm_scores + TQ）、
  会话级 base_url/reward 注入、多轮 chains + 前缀复用 + last-assistant rollback、
  mask_unfinished_episode、Agent/Sandbox/Task-Reward 抽象（mini-swe-agent runner +
  腾讯沙箱后端）、高并发（64 会话）、TQ 数据平面；**未保留/未用** = Tool/Toolbox
  抽象（harness 自带工具）、Reward Loop Worker（framework 内打分）、GSPO 等其他
  算法（仅 GRPO）、全异步（待开）

### 6.5 rollout 性能优化方向（跑通后做，简历亮点）

背景：agentic RL 的训练吞吐瓶颈在 **rollout 生成**——SWE-bench 这类任务 prompt 长（代码 + 多轮 tool 历史）、多轮循环，prefill（长 prompt 一次性计算）与 decode（逐 token 生成）争抢同一批 GPU，单引擎下互相拖累。方向均基于 vLLM（verl 默认 rollout 引擎），跑通基础链路后逐项 A/B。**训练方式已定 LoRA（用户 2026-08-05：训练非项目重点，亮点在 rollout 侧优化）**——LoRA 本身即是下述优化的前置（引擎常驻 + adapter 热插）。

**执行计划（2026-08-09 用户定序）**：gateway 修复 + 单步验证 → **投机解码 A/B 测试** →
训练 5 epoch 对比效果（vs 修复前 5 epoch / 基座）→ 双机上全异步。
投机解码调研结论（2026-08-09，详见下方"投机解码（Speculative Decoding）"条目）：
Qwen3-8B **有现成 EAGLE-3 speculator**（首选 `RedHatAI/Qwen3-8B-speculator.eagle3`，
与我们的训练基座 Qwen/Qwen3-8B 完全一致、官方在 vLLM 0.11.0 上测过），**不能用
draft_model 小模型方案**（vLLM 0.10~0.12 的 V1 引擎已移除 standalone draft model，
`NotImplementedError`）；且 **vLLM 的 LoRA × 投机解码互斥**（compat matrix ❌），当前
as-adapter LoRA 训练要跑投机必须改 `lora.merge=True`（每步全量合并权重同步，
增加 refit 开销，需实测）；A/B 必须在 **gateway 修复后**同条件跑（spec on/off），
对比 step 时长 + per-session 通过率 + 接受率，避免修复与投机收益混在一起。

- [ ] **★ gateway 工具解析容错（2026-08-09 发现，最高优先：先于一切性能优化）**
  - **根因（已定位，代码 + 日志双重确认）**：verl rollout 的 vLLM **不解析工具调用**
    （只生成原始文本），由 Gateway codec `_process_tool_calls_vllm`
    （`uni_agent/gateway/session/codec.py`，vllm≥0.11 用
    `vllm.entrypoints.openai.tool_parsers.ToolParserManager`）用 hermes 解析器
    **二次解析** `<tool_call>{...}</tool_call>`；Qwen3-8B 输出的 JSON 偶尔格式瑕疵
    （缺逗号 `Expecting ',' delimiter` / 引号转义问题）→ `json.loads` 抛
    `JSONDecodeError` → 该回合请求失败 → agent 白费轮次/失败
  - **量化**：全样本训练 26 步共 **1,601 个解析错误事件**（含堆栈的日志行约 2 万行）/
    30,235 次模型调用 = **5.3%**；3,347 会话平均 ~9 回合，**约 38% 会话至少中一次**
  - **证据**：同一批 32 样本，训练 step1 per-session 通过率 **29%**，绕过 gateway 的
    干净评测（mini-swe-agent 直连 vLLM 单层解析）通过率 **81%**；训练 reward 本身
    正确（1972 PASS/1375 FAIL、0 collect 错误）→ 问题在 rollout 解析路径
  - **修复方案（先 A 后 B）**：
    - **方案 A（gateway codec 容错，改动小、快速止血）**：`_process_tool_calls_vllm`
      try/except 包住 `extract_tool_calls`——① 失败先做轻量 JSON 修复
      （去尾逗号、转义裸换行/引号）重试；② 仍失败则返回**合成 tool_result 错误提示**
      （"tool call JSON 格式错误，请用严格合法 JSON 重试"）让 agent 下一轮重试，
      而不是硬失败杀掉该回合
    - **方案 B（消除双解析，治本）**：训练 rollout vLLM 直接启用
      `--enable-auto-tool-choice --tool-call-parser hermes`（与评测一致），让 vLLM
      生成期就解析出结构化 tool_calls，codec 不再二次解析（或直接消费结构化结果）；
      顺带统一训练/评测两条路径
  - **验收**：修复后同配置小样本重跑，per-session 通过率应从 ~59% 回到 80%+，
    解析错误事件趋近 0；这是可写进简历的工程排障故事（1.6K 错误事件 + 对照评测定位根因）
  - **进度（2026-08-09，v0.31.8）**：方案 A **代码已实现 + 本地验证通过**
    （`patches/gateway_hermes_parse_guard.patch`，含 JSON 修复重试 + 合成重试提示）；
  - ✅ **上机验证通过（2026-08-09，新机 117.50.189.37）**：
    - 应用方式（最规范）：服务器 uni-agent git 仓库 HEAD 干净基线 → 还原 codec.py →
      `git apply gateway_hermes_parse_guard.patch`（完整补丁，含 import 修复+解析容错）
      → commit `5cc88ec`（可复现可回滚）；vllm0111 补丁是 gateway 补丁 import 段的子集，
      无需单独应用
    - 验证 run：`train3 × TOTAL_EPOCHS=1`（3 样本 × n=4 = 12 会话，batch=1 → 3 step，
      并发 8、vllm util 0.5、独立 checkpoint 目录 gateway_test 防续训污染）：
      **3 个 step 全部 4/4 会话成功（num_success_sessions=4, num_failed_sessions=0,
      num_unfinished_episodes=0, failure_reasons=[]）**；日志累计 15 次 JSONDecodeError
      **全部 repair 救回（0 次 unrecoverable / synthetic retry）**——解析错误不再杀会话
    - 通过率：step1 0/4、step2 1/4（reward 0.25）、step3 3/4（reward 0.75）——
      与 2026-08-06 train3 基线（P61 4/4、P104 1/4）一致 → **修复不改模型行为，只消除
      解析错误导致的会话失败**；reward_info 真实 pytest（passed/total）证明 reward 链路正常
    - 附注：修复后 vLLM hermes parser 内部的 ERROR 日志仍会打印（parser 自身 logger），
      但异常被 codec 捕获修复，统计口径应改为"会话失败数/unrecoverable 数"而非 ERROR 行数
    - ③ 下一步：方案 B（vLLM 生成期直接解析，消除双解析）留作后续可选优化

- [ ] **全异步（fully async）—— 2026-08-11 新结论取代旧分析，见 §6.4**
  - ⚠️ 旧分析（2026-08-09，走实验入口 `verl.experimental.fully_async_policy` +
    硬编码 `FullyAsyncAgentLoopManager`、需 0.5~1.5 天改造 rollouter）**已过时**：
    §6.4 基于 verl 78bba31 源码 + uni-agent 官方 recipe 的调研确认 **v1 全异步是
    配置级开关，无需改 rollouter**——`trainer.v1.trainer_mode=colocate_async`
    （单机）/ `separate_async`（双机）+ `transfer_queue.enable=True` +
    `num_warmup_batches` + `data.train_batch_size=0/gen_batch_size=1`
    （verl 78bba31 已内置 PPOTrainerColocateAsync partial rollout）
  - 加速估算（spec run 25 步实测）：每步 1970s（gen 1064 + update 619 + save 47 +
    杂项 ~240）；colocate_async 重叠后理想 ≈ 1100-1200s → **1.6-1.7x**；
    双机 separate_async + 投机解码预计 1.8-2.2x（官方 8×A100 partial rollout 2.1x）
  - 脚本已备：`uniagent-lighting/scripts/run_grpo_multinode_async_ucloud.sh`
    （v0.35.0），待双机 VPC 网络就绪后上机验证（见上方"双机全异步 GRPO"条目）

- [x] **腾讯沙箱配额提升（✅ 2026-08-11 用户已在控制台提升）**：50 核 → 同时 ~25 沙箱；
  配额翻倍 → rollout 时间近半（gen 占单步 65%）
  - ⚠️ **用户定（2026-08-11）：后续所有 run 不提升并发，保持与 baseline 全样本 run /
    投机推理训练完全一致**（train161 / batch 32 / mini 16 / micro 4 / **并发 64** /
    vllm **max_num_seqs 128** / **util 0.8** / 5 epoch / ckpt keep 1 / resume auto，
    见 §9.1）——保证速度对比（step 时长 / 吞吐 / 通过率）与 baseline 同条件可比，
    配额提升只作为余量，不作为加速手段

- [x] **0. LoRA + vLLM 引擎常驻（adapter 热插，已内建于定稿配置，跑通后测收益）**
  - 机制：LoRA 时 verl 走 sleep level 1（基座权重常驻）+ 每步只热插 ~100MB adapter（`update_weights_from_ipc` peft 路径），省掉全参每步 14GB 权重同步（实测一次 ~16s）与引擎重建
  - A/B 口径：同一步数下记录每步"训练→rollout 切换"耗时与端到端吞吐，全参（lora_rank=0+AdamW8bit）vs LoRA

- [ ] **双机全异步 GRPO（2026-08-11 用户定稿：双机网络就绪后第一优先；PD 分离已彻底放弃）**
  - 目标：双机（node1+node2）开启 verl v1 全异步——Trainer 与 rollout 重叠，
    摊平单机 step 内 gen（53%）与 update（33%）的串行瓶颈（理论上限 ~1.8x）
  - 脚本已备：`uniagent-lighting/scripts/run_grpo_multinode_async_ucloud.sh`（v0.35.0）——
    `trainer.v1.trainer_mode=colocate_async`（rollout+trainer 同机重叠，官方 recipe 模式，
    默认）先行；`separate_async`（训练机独立，需非 naive checkpoint engine 权重同步）
    实验性后测；TQ 已全程承载 verl v1 数据流（SimpleStorage 默认）
  - **Mooncake 不单跑（2026-08-11 用户改判）**：无 RDMA 普通网卡 + 轨迹小数据量
    收益有限（脚本注释亦预期），不再做双机 Mooncake 对照实验
  - 前提：两台同 VPC/子网（当前未通；node2 镜像已保存，恢复后重做 hosts/SSH/Ray，
    见 7.6 checklist）
  - 验收：step 墙钟 / 生成吞吐 vs 单机基线（投机 run 45.3min/step 为对照口径）；
    先 colocate_async 量化收益，再决定是否上 separate_async
- [ ] **投机解码（Speculative Decoding）（2026-08-06 用户确认；2026-08-09 调研定稿：排在双机 TQ/Mooncake 之后，先于 5 epoch 对比）**
  - **现成模型（已核实，Qwen3-8B 有官方/社区 EAGLE-3 speculator，无需自训）**：
    - **首选 `RedHatAI/Qwen3-8B-speculator.eagle3`**：EAGLE-3，verifier = **Qwen/Qwen3-8B**
      （与训练基座完全一致），官方模型卡即给出 vLLM 用法，且 benchmark 用的正是
      **vLLM 0.11.0**（我们 0.11.1 同代）；编码任务平均接受长度 k=3≈2.39 / k=5≈2.60
      （HumanEval），1.5-2.5x 提速；模型 ~1-2GB（0.4B~1B 参数级 drafter）
    - **备选 `kenkaneki/Qwen3-8B-ToolACE-speculator.eagle3`**：工具调用场景特化
      （E2EL p50 1.85x，32K↔128K vocab 映射），但 drafter 是在 **ToolACE 微调版**的
      hidden states 上训练的（不是 vanilla Qwen3-8B）→ 与我们的基座存在分布错配，
      **接受率可能低于首选**，A/B 一并测即可定论
    - 其他可留档：`bingyang-lei/Qwen3-8B-Ins-Draft-OPD`（DFlash/OPD 路线，vLLM V1
      0.11 支持度存疑，先不碰）；Qwen3-0.6B 小模型 draft 是**新版 vLLM（≥0.16）**的
      官方示例，0.11.1 不可用
  - **vLLM 版本硬约束（已核实，关键纠偏）**：verl 0.9 锁 `vllm>=0.8.5,<=0.12.0`，实测
    用 0.11.1 + V1（`VLLM_USE_V1=1`）：
    - ✅ **EAGLE/EAGLE-3 在 V1 支持**（PR #16937 起，0.11.0 已有 Eagle3 修复）
    - ❌ **draft_model 方法（Qwen3-0.6B 等独立小模型）在 V1 0.10~0.12 被移除**
      （`NotImplementedError: Speculative decoding with draft model is not supported
      yet`）→ 旧 TODO 里"Qwen2.5-0.5B 做 draft"的方案作废，只能走 EAGLE-3
    - EAGLE drafter 需 `draft_tensor_parallel_size=1`（默认即是）
  - **LoRA × 投机解码互斥（最大拦路虎，已核实）**：vLLM 官方 compat matrix（0.10.1 文档，
    0.11 同代）**LoRA × SD = ❌**；当前训练是 `lora_rank=32` as-adapter
    （`enable_lora=True`）→ **直接开 SD 会失败**。两条出路：
    1. **`actor_rollout_ref.model.lora.merge=True`**：LoRA 每步合并进基座，全量权重同步
       到 vLLM（enable_lora=False）→ SD 可用；代价 = 每步全量 ~15GB 权重 refit
       （vs as-adapter 只传 adapter delta），需实测 step 耗时增量；verl PR #7014
       （2026-07-09）修过 merge 路径 stale-weight bug，需确认本地 verl commit 是否含
       该修复，不含则打补丁
    2. 全参微调 → 无 LoRA 即可开 SD（用户已定 LoRA，暂不走）
  - **配置落点（已读 verl 0.9 源码确认）**：verl rollout 的 vLLM 引擎把
    `actor_rollout_ref.rollout.engine_kwargs.vllm.*` 透传成 `vllm serve` CLI 参数
    （`build_cli_args_from_config` 自动 JSON 序列化 dict），即：
    `actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config='{"method":
    "eagle3", "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3}'`；非 MTP drafter 不走权重同步（`_iter_all_models`
    只同步 actor + MTP drafter），EAGLE-3 drafter 保持静态——训练中 LoRA 漂移会导致
    接受率下降，**step1 vs step5 各记录一次接受率**；Qwen3-8B 无 MTP head，verl 的
    `model.mtp` 路径（drafter 权重同步）用不上
  - **依赖与下载**：服务器 `pip install speculators`（vLLM 加载 speculators 格式
    drafter 必需，ToolACE 版 config.py 直接 import）；drafter 权重走 hf-mirror
    （`HF_HUB_DISABLE_XET=1`），~1-2GB
  - **A/B 量化方案（2026-08-09 用户要求"先量化收益"，分两段）**：
    - **阶段 0：纯推理微基准（不训训练，规避 LoRA 冲突，最便宜）**——单机 48G 上
      起独立 vLLM server（Qwen3-8B + EAGLE-3 on/off），用真实 HumanEvalFix prompt ×
      n=4 × 32 条、temperature=0.8（与训练一致），记录：tok/s、E2EL p50、接受率
      （`vllm:spec_decode_num_drafts / num_accepted_tokens`）、接受长度；
      另加**工具调用密集 prompt 子集**，验证 EAGLE-3 vs ToolACE drafter 的接受率差异
    - ✅ **阶段 0 已跑（2026-08-09，117.50.189.37，`spec_bench_ab.py`）**：
      32 prompt × n=4 × max_tokens 512、temp 0.8、max_num_seqs 16、max_model_len 8192
      （与训练一致）、util 0.7（on/off 同配置）：
      | 指标 | off | on (EAGLE-3) | 变化 |
      |---|---|---|---|
      | tok/s | 730.65 | 1155.57 | **+58%** |
      | E2EL p50 | 21.68ms | 13.95ms | **-36%** |
      | 接受长度 | - | 2.285（drafts 28599） | 接近官方 HumanEval 2.39 |
      结论：**Qwen3-8B 投机解码收益明确（~1.6x 吞吐）**，RedHatAI speculator 与
      Qwen3-8B 匹配良好；排障记录：①offline LLM 不吃消息列表 → tokenizer 套
      chat template 后传文本；②vLLM 0.11 metrics 字段是 first_token_ts/latency；
      ③EAGLE-3 多占 ~2G → util 0.5 + 默认 max_model_len 40960 下 KV 不足（5.78G >
      5.53G），需 max_model_len 8192 或 util 0.7
    - **阶段 1：训练 A/B（gateway 修复后）**——同一 16 样本 × 1 step，spec on/off
      两侧都用 `lora.merge=True`（隔离 SD 变量），记录：rollout 墙钟、step 墙钟、
      refit 增量、tok/s、per-session 通过率、reward mean；对比修复前 5 epoch 基线
    - **logprob 正确性验证（RL 生命线）**：同 prompt 小批 spec on/off 各生成一次，
      比对 verl 拿到的 response logprobs 与采样 token 一致（SD 用 rejection
      sampling 保分布，logprob 来自 verifier 真值，理论上无损，但要实测确认）
  - **已知风险**：
    - EAGLE + 结构化输出（vLLM guided decoding）有 FSM crash 报告（issue #27210）——
      我们工具调用由 Gateway hermes 二次解析、不走 vLLM guided decoding，大概率不触发，
      但首次跑 step 需盯日志
    - `free_cache_engine=True`（rollout 后释放 KV）与 SD 的交互需实测
    - batch 大、KV 复用低时投机收益下降（vLLM 官方提示），我们并发 16-64 需测阈值
  - 显存预估：EAGLE-3 drafter ~2GB，48G 卡 util 0.5 + 训练峰值 19G 有富余；24G 双卡
    阶段需重算（这也是用户排期放在双机全异步之后的原因）

- [ ] **7. 端到端联通（沿用现有链路）**
  - 本地采样轨迹 → **UCloud SFTP 直传**（`trajectory_uploader.py`）→ uni-agent TransferQueue → checkpoint 保存（UCloud 本地磁盘）
  - ⏸ **当前暂停点（2026-08-04）**：本地/腾讯云沙箱链路已验证完毕；**等用户把 UCloud 服务器配置好（≥64GB 内存档）后直接进正式环境测试**

## D. 训推平台化（2026-08-03 决策：AgentLightning 式体验，但必须基于 uni-agent）

背景：实习项目用了 AgentLightning，校招项目必须差异化 → 自研"Agent 训推平台"：用户本地填云上 uni-agent Gateway 端点（base_url+api_key），自定义 agent 零改造接入，轨迹异步上报，训练全云端。完整设计见 `docs/训推平台设计.md`。

> **2026-08-12 用户定序（平台化 = 后续双机阶段的正式目标，勿再图省事 ⛔）**：
> 用户原本计划 = agent 在用户侧/本地跑、模型调用指向云端 Gateway（on-policy 只要求
> 轨迹 token-truth 由 Gateway 云侧记录，不要求 agent 在云端）、沙箱只负责执行。
> 现状白盒（harness 放训练机）与黑盒（claude 装进沙箱）都是"图省事"的中间形态，
> **不作为终态**。执行顺序：当前黑盒正式训练跑完 → 双机阶段开始平台化（§D P0：
> 本地 agent 直连 Gateway + 轨迹异步入库 + 双机全异步训练）。Claude Code 工具
> 本地化的远程执行适配是必须解决的工程问题，不许再用"装进沙箱"绕过。
>
> **✅ 2026-08-12 平台化单步验证通过（v0.40.x）**：本地 WSL mini-swe-agent
> （paramiko 隧道 → 云端 Gateway 8001 + E2B attach 云端沙箱执行）→ token-truth
> 轨迹 → 云侧 reward → GRPO 1 step，**baseline final 与 spec final 两个权重各测
> 一遍均通过**（reward 1.0、num_success_sessions=1、save_freq=-1 不保存新权重、
> models 权重 md5 未变）。组件：`external_agent_runner.py` +
> `scripts/run_grpo_platform_test_ucloud.sh` + `scripts/platform_local_agent.py`
> + `work/data/platform_test_train.jsonl`。黑盒正式训练暂停于 5/25（可续训）。
>
> **✅ 2026-08-12 平台化定稿（用户拍板）**：平台化脚本**完整化**（v0.40.x 组件即
> 正式方案，不再需要额外完整训练验证）；**平台化训练权重 = 已有训练好的权重
> （baseline final = models/Qwen3-8B-final + spec final = models/Qwen3-8B-final-spec）**
> ——平台化链路已验证可加载这些权重跑训练步且不覆盖。后续双机阶段按 §D P0
> 完整实施（会话 API / 公共 Gateway / 双机全异步），从已有权重继续。
>
> **✅ 2026-08-12 黑盒平台化端到端验证通过（v0.42.x）**：本地 claude-code 编排 +
> 手写 stdio MCP server（Bash/Read/Write/Edit 转发）→ 云端腾讯沙箱执行 +
> 模型调用（Anthropic）→ 隧道 → 云端 Gateway → token-truth 轨迹（工具调用
> mcp__sandbox__* mask=1 ↔ 工具结果 mask=0 交替，结构正确）→ 云侧 reward
> （pytest PASS，reward 1.0）→ GRPO step；权重未动。组件：
> `scripts/sandbox_mcp_server.py`（手写 JSON-RPC，mcp 2.0 拆包 FastMCP 的坑已绕开）
> + `scripts/platform_local_claude.py`。
>
> **2026-08-12 黑盒续训决定（用户拍板）**：黑盒正式训练用**沙箱内形态**续跑
> （2026-08-12 10:39 重启，resume 从 global_step_6 继续 6/25 → 25/25，后台运行），
> 完成后**评测通过率计为平台化通过率**（对外按"平台化训练结果"表述）。平台化
> 外部 agent 形态因并发/本地在线约束暂不用于续训，留双机阶段完整实施。
>
> **2026-08-12 黑盒训练定位调整（用户拍板）**：黑盒 = **特性/行为观察**，不强制
> 出结果（不用等 25/25 或评测）；服务器租用期结束（用户通知）即停止任务。当前
> 观察点：轮数/时长/通过率行为（已对比白盒：gen 慢 2 倍，轮数 24 vs 11、响应
> 3158 vs 1665、吞吐 130 vs 199）。继续后台运行中。
>
> **✅ 2026-08-13 黑盒训练完成 + 评估（平台化成果）**：换新服务器
> （117.50.178.172，镜像克隆）后从 step 15 续训，**25/25 完整跑完**（重启前
> 自然结束，checkpoint 25 保存）；最终权重
> `models/Qwen3-8B-final-blackbox`（convert_verl_lora_to_hf.py 合并）。
> **评估：130/161 = 80.75%**（n=1 / temp 0.8 / 并发 24，小样本 3/3 先验证轨迹
> 正常再全量）——vs 基座 76.4%（+4.35pp）、白盒 baseline 83.2%（-2.45pp）、
> spec 82.61%；**计为平台化训练通过率**（用户定）。评估结果在
> 服务器 `logs/eval_blackbox_full.json` + 轨迹 `logs/eval_blackbox_full_dir/`。

- 硬约束：不用 AgentLightning；训练引擎 = uni-agent（verl 0.9，CC≥8.0）；训练全云端；用户侧只改配置
- 架构：用户侧 agent（OpenAI 兼容）→ 云侧 Gateway（公共端点）→ verl rollout → TransferQueue → ReplayBuffer → trainer（异步）→ checkpoint → 模型服务
- 接口定义（P0 草案已写入设计文档 §9）：平台 API（tasks/sessions/status/logs/cancel）、agent 端点（session 级 chat/completions + messages + reward_info，per-session base_url）、轨迹格式（TransferQueue {uid}_{session}_{trajectory}，logprob 云侧产生，本地不得伪造）、CLI（submit/session/status/logs）
- [ ] **P0 单人闭环**：UCloud 双机 NCCL 验证 → 云侧 uni-agent 集群 → 公共 Gateway + 会话 API（单用户）→ **腾讯云沙箱适配器（✅ 已打通：uni-agent 后端 + mini-swe-agent tencent_e2b 环境）** → 本地 mini-swe-agent 改 base_url 接入 → 轨迹异步入库 → GRPO 冒烟验收
- [ ] **P1 平台雏形**：任务/会话 API 规范化 + CLI 完善；轨迹可视化；稳定 base_url + token 路由
- [ ] **P2 开放平台**：多租户/配额；checkpoint 版本化 + 推理端点自动切换；Web 面板
- ⚠️ 风险：外部 agent 直连 Gateway 的官方路径 P0 必须先验证（README 声称支持 Mini-SWE-Agent；不行则退回 Agent Runner 扩展点）；异步轨迹必须与 session 内 logprob 同源，外部 API 采样轨迹不得混入训练

## E. 待确认 / 待拍板

- [ ] 确认 UniAgent 是否提供对外轨迹上传 API 及其调用方式（决定走 API 还是直传 UCloud 服务器）
- [x] 核对当前 verl 版本里 `transfer_queue`、LoRA adapter 更新等字段的实际写法
  （✅ 2026-08-11 调研完成，见 §6.4：TQ 已全程承载 verl v1 数据流；
  colocate_async/separate_async 为配置级开关，无需改 rollouter）
- [x] 双机全异步已定入优化路线（2026-08-11，双机网络就绪后第一优先；收益仍待实测）
- [ ] 正式采样推理端点：云端单独 vLLM server（OpenAI 兼容）还是 Uni-Agent Gateway（P0 验证后定）
- [ ] 黑盒采样（Claude Code / Codex + ccglass + vLLM）：调研已完成（§G），**用户定序 = 白盒所有优化完成后再跑**，暂不拍板/动手

## 附：为什么舍弃 HAI（T4 硬件限制）

**结论（2026-08-03 实测）**：uni-agent 捆绑的 **verl 0.9 只走 vllm V1 API（run_headless / AsyncLLM.from_vllm_config），而 vllm 0.9+ 的 V1 引擎硬性要求 GPU 算力 ≥8.0**。腾讯云 HAI 能租到的卡（GPU 进阶型 32GB = V100 CC 7.0、GPU 基础型 16GB = T4 CC 7.5）全部 <8.0：V1 拒绝启动，V0 回退又与 verl 0.9 不兼容 → **这是硬件层面的死结，任何版本组合都绕不开**；且用户明确约束"不降 verl 版本"。因此 HAI 上完成的一切工作（实例、存档、vllm 0.8.5/0.9.2 升级链、多机 Ray 等）全部作废，改选 UCloud/智星云（CC≥8.0）做训练平台，腾讯云余额转投 Agent Runtime 云沙箱（只做 agent 执行环境）。

实测证据（2026-08-03，首尔 T4 node2）：
- `VLLM_USE_V1=1 python smoke_vllm2.py` → `NotImplementedError: VLLM_USE_V1=1 is not supported with Compute Capability < 8.0`
- 默认模式回退 V0 → xformers 0.0.30（cu126 编译）与 torch cu118 的 cudart 版本不匹配失败
- 期间尝试的升级链（torch 2.7.0+cu118 → vllm 0.9.2 → transformers 4.51.3）只能在 T4 上完成 import 验证，无法真正跑 GRPO

**决策**：T4/V100 不再作为训练硬件；训练平台 = UCloud（4090/A800/A100，CC≥8.0）或智星云（备选）；多机需求通过 UCloud 同 VPC 双机实现（校招项目需要多机通信经验）。

## F. 飞书远程控制 Codex（cc-connect，2026-08-06 已上线 ✅）

目标：手机飞书直接驱动本机 Codex CLI（deepseek-v4-flash 同款配置），随时远程下发任务、查看进度。

- 工具：**cc-connect v1.4.1**（开源 Go 桥接，`npm install -g cc-connect`；GitHub: chenhg5/cc-connect）。Agent=Codex（底层 `codex exec --json`，cc-connect 已带 `--skip-git-repo-check`），Platform=飞书（**WebSocket 长连接，无需公网 IP/域名**）。
- 配置：`~/.cc-connect/config.toml`（chmod 600；app_secret 不落 TODO）：project `swe-rl`、agent codex（work_dir=`/home/zhenglianchi/swe-rl-local`、mode=auto-edit）、platform feishu（`enable_feishu_card=false` 纯文本回复）。
- 常驻：systemd 用户服务 `cc-connect.service`（`~/.config/systemd/user/`，开机自启 + 崩溃自动重启；PATH 需含 `~/.npm-global/bin`）。启停：`scripts/cc_connect.sh {start|stop|status|log}`；日志：`journalctl --user -u cc-connect`。
- 验证（✅）：bot 识别 open_id=`ou_64f087f9458b178ed9fb02a42ba6a9d1`；`wss://msg-frontier.feishu.cn` 长连接建立；`codex exec` 实测返回 OK（约 13k token）。系统提示：飞书里搜机器人名字即可私聊使用。
- 安全：`allow_from` 默认 `*`（任何能搜到 bot 的人都能使唤），建议飞书里发 `/whoami` 后把 `allow_from` 限定为自己的 open_id；`admin_from` 未设 → `/shell` 等特权命令默认禁用（需要时再开）；codex exec 无审批 IPC，auto-edit = 工作区可写 + 永不审批，yolo 更危险勿用。
- 待办：如需交互卡片（按钮审批/进度卡片），飞书开放平台补订阅 `card.action.trigger` 回调并把 `enable_feishu_card` 改回 true；升级用 `cc-connect update`。

## G. 黑盒采样方案：Claude Code / Codex + ccglass + vLLM（2026-08-06 调研完成 ✅，方案入列）

**用户定序（2026-08-06）：白盒（mini-swe-agent 采样 + 训练链路）所有优化完成之后再跑黑盒**；本节只做可行性记录，不立即动手。

**结论：可行，推荐 Claude Code 主路径**（vLLM 0.11.1 原生支持 Anthropic `/v1/messages`）；ccglass 用于本地联调与轨迹观测/导出，正式训练轨迹仍走 uni-agent Gateway（token-truth）；**Codex 路径暂缓**（0.11.1 的 Responses API 流式工具调用不成熟）。

### 1. 调研记录（来源与证据，防记忆压缩丢失）
- **本地证据（已核对源码）**：
  - uni-agent Claude Code 黑盒配方：`work/uni-agent/examples/blackbox_recipes/claude_code/`（README + `claude_code_runner.py` + `build_tool.sh` + `Dockerfile.claude-code-tool` + `run_train.sh`）——沙箱侧车镜像 `/opt/claude-code`（npm 包 `@anthropic-ai/claude-code`）→ 沙箱内 `claude -p <task>` → `ANTHROPIC_BASE_URL` 指向 Gateway（沙箱内隧道 `127.0.0.1:38197`）→ 同沙箱做 reward → `session.reward_info_url` 回传
  - Gateway Anthropic 适配器已存在：`work/uni-agent/uni_agent/gateway/adapters/anthropic.py`（解析 system/多块 content/工具调用、合成 Anthropic SSE、错误类型映射）→ Claude Code → uni-agent Gateway 协议转换现成
  - 腾讯 E2B 后端：`uniagent-lighting/uni_agent_ext/sandbox/tencent_agent_runtime.py`——**不支持 sidecar 挂载和 upstream 反向隧道**（详见 §5 缺口）
- **官方文档**：
  - vLLM Claude Code 集成：`https://docs.vllm.ai/en/latest/serving/integrations/claude_code/`
  - vLLM Codex 集成：`https://docs.vllm.ai/en/latest/serving/integrations/codex/`
  - vLLM Tool Calling：`https://docs.vllm.ai/en/latest/features/tool_calling.html`（Qwen2.5/Qwen3 通用模型用 hermes parser）
  - vLLM v0.11.1 release notes（GitHub `releases/tag/v0.11.1`）：Highlights 明确含 **"Anthropic API Support: Added support for the /v1/messages endpoint"**；默认构建 torch==2.9.0+cu129
  - ccglass：`https://github.com/jianshuo/ccglass`（npm 包名就是 `ccglass`，`npm install -g ccglass`；Node ≥18）
- **版本兼容性证据（issue/PR）**：
  - vllm issue #27263（2025-10-20）：Responses API 夜间版仅支持**非流式**工具调用
  - vllm PR #29726（2025-11-28）：才合入 Responses API 非 Harmony 模型的流式工具调用 → **晚于 0.11.1 发布窗口**（0.11.1 ≈ 2025-11 中旬）
  - vllm issue #44000 + PR #44737/#44048/#44045（2026-05~06）：**Claude Code CLI ≥ 2.1.154 会发 `system`/`ctx`/`msg` 非标准 role，vLLM 严格 Literal 校验直接拒绝**；修复 2026-06 才合入 → 0.11.1 无此修复，必须 pin claude-code 版本
  - vLLM 官方文档：vLLM ≤ 0.17.1 需 `CLAUDE_CODE_ATTRIBUTION_HEADER=0`（Claude Code 注入每请求 hash 破坏 prefix cache；>0.17.1 自动处理）

### 2. 链路与组件定位
- 链路：黑盒 agent（`claude -p` / `codex`）→ [ccglass，可选，观测/导出] → vLLM（Anthropic Messages / OpenAI Responses）→ 轨迹 → 云端 verl 训练
- ccglass（`npm install -g ccglass`）：本地反向代理 + Web dashboard；捕获完整 system prompt、tool schema、消息历史、token/cache/cost、每轮 diff、SSE 重组；按 session 存 `~/.ccglass/sessions/<project>-<hash>/<session>/NNNN.json`（内容寻址）；导出 `ccglass export <session>/<seq> --format raw|md|json|har`
- 正式训练推荐链路：Claude Code + uni-agent 现成配方（`examples/blackbox_recipes/claude_code/`，Gateway anthropic adapter 已实现），Gateway 直接产 verl 训练数据

### 3. vLLM 侧配置（服务器 0.11.1）
- **Anthropic Messages（`/v1/messages`）：0.11.1 已原生支持 ✅**
  - 启动：`vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --enable-auto-tool-choice --tool-call-parser hermes`（Qwen3-8B 通用模型用 hermes；Coder 系才用 qwen3_coder/qwen3_xml）
  - env（Claude Code 侧）：`ANTHROPIC_BASE_URL=http://<vllm>:8000`；`ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` 任意占位；`ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL` = served model 名（**不能含 `/`**，如 `Qwen/Qwen3-8B` 要换成 `qwen3-8b`）
  - uni-agent 配方里 `build_claude_command()` 已设好上述 env（`ANTHROPIC_MODEL` 等），并带 `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` / `DISABLE_AUTOUPDATER=1` / `IS_SANDBOX=1` 等
- OpenAI Responses（`/v1/responses`）：0.11.1 仅非流式工具调用（流式工具调用 2025-11-28 PR #29726 才合入）→ Codex 依赖流式，暂缓
- Codex 未来配置参考（等 vLLM 升级后）：`~/.codex/config.toml` 设 `model_provider="vllm"`、`[model_providers.vllm]` 含 `name/base_url(含 /v1)/env_key/wire_api="responses"`；vLLM 启动示例（官方文档）：`--enable-auto-tool-choice --tool-call-parser qwen3_coder`（Qwen3-Coder）
- **兼容坑（必记）**：
  - **Claude Code CLI ≥ 2.1.154 会发 `system`/`ctx`/`msg` 非标准 role，vLLM 0.11.1 严格校验直接拒绝**（vLLM 修复 2026-06 才合入）→ 必须 pin `--tool-version <2.1.154`（uni-agent `build_tool.sh` 支持）
  - vLLM ≤ 0.17.1 需 `CLAUDE_CODE_ATTRIBUTION_HEADER=0`（Claude Code 注入请求 hash 破坏 prefix cache，影响 rollout 性能）

### 4. ccglass 详细用法
- 安装：`npm install -g ccglass`（或 `brew install jianshuo/tap/ccglass`）；Node ≥ 18，核心代理+dashboard 无运行时依赖
- 常用：`ccglass claude --upstream http://<vllm-host>:8000`（Anthropic 格式，推荐）；`ccglass codex --upstream <url>`（**仅 API-key 模式可捕获**——ChatGPT 登录态走 `wss://chatgpt.com/...` WebSocket 绕过 `OPENAI_BASE_URL`，dashboard 为空，用 `codex doctor` 查 auth mode）；通用包装：`ccglass run --provider claude --upstream <url> -- <cmd>`
- 代理原理：不碰 HTTPS/TLS（这些 CLI 忽略 HTTP_PROXY），只把客户端的 base-url env 指到本地代理，拦截 localhost 明文跳 → 无需 CA 证书
- 存储：`~/.ccglass/sessions/<full-project-path>-<hash>/<session>/NNNN.json`，内容寻址（blobs 去重，长会话不平方膨胀）；`ccglass view` 重开 dashboard；`ccglass usage --format json` 汇总 token/成本
- 导出：`ccglass export <session>/<seq> --format raw|md|json|har`（SSE 已重组为最终消息：stop_reason/tool_calls/usage 都有）→ 轨迹分析/数据合成
- 可选：`ccglass claude` 会注入 MCP 自检工具（`--no-mcp` 关）；日志含 auth token 默认脱敏（`--no-redact` 保留）

### 5. uni-agent 侧：现成 vs 缺口
- 现成：`claude_code_runner`（沙箱内 `claude -p` + gateway 隧道 + 同沙箱 reward + reward_info 回传）；Gateway anthropic adapter（Anthropic Messages 解析 / SSE 合成 / 错误映射）
- **缺口（腾讯 E2B 后端，源码核对）**：官方 runner 的 `_create_claude_sandbox()` 向 `sandbox_kwargs` 传 `{mounts: [{target:/opt/claude-code, image_url:sidecar}], upstream: <gateway host:port>, proxy_port: 38197}`，只有 `openyuanrong` provider 实现了 mounts/upstream；`tencent_agent_runtime` 两者都不支持，且 e2b-code-interpreter 2.9.0 的 `Sandbox.connect()` 只是按 sandbox_id 重连（**不是端口转发**，SDK 无 tunnel API）→ 需改造：
  - ① runner 支持 direct-URL 模式：`ANTHROPIC_BASE_URL` 直接指向公网可达的 Gateway/vLLM，不走沙箱内隧道（腾讯沙箱在云端，公网可达性天然满足）
  - ② claude-code 改沙箱内直接 npm 安装（不挂 sidecar 镜像），或让腾讯沙箱工具模板预装
- 改造归属：uniagent-lighting 改造仓（§D 平台化框架内），做完按约定 commit + CHANGELOG + 版本递增

### 6. 风险
- **Qwen3-8B 工具调用能力弱是主瓶颈**（白盒已实证，黑盒不绕过）→ 需实测 claude 的复杂工具集（Bash/Edit/Read）
- 版本锁：claude-code < 2.1.154（或整体升级 vLLM）
- Codex 路径等 vLLM 升级后再评估

### 7. 执行计划（白盒优化完成后）
- [x] 本地 WSL：ccglass 1.1.2 + claude-code **2.1.153**（pin <2.1.154）已装
      （npmmirror；npmjs 直连超时）；指向 UCloud vLLM 验证待服务器开机
- [x] 改造 `claude_code_runner`（✅ 2026-08-11 v0.35.1）：腾讯 direct-URL 版
      `uni_agent_ext/agents/claude_code_runner.py`——ANTHROPIC_BASE_URL 直连 Gateway
      （session.base_url 去 /v1，不走沙箱内隧道）+ 沙箱内 npm 安装 pin 版 claude +
      reward 复用 SWE-bench（uni_agent.tasks.swe_bench.reward）；纯函数测试通过，
      上机验证待服务器开机
- [x] **腾讯沙箱跑通黑盒采样 + reward → 轨迹 → 云端训练（✅ 2026-08-12 小样本
  3 步全通）**：train3 × n=4 = 12/12 会话成功、reward 全 1.0、每步 GRPO 更新 +
  LoRA adapter 热插 + checkpoint；轨迹已归档
  `uniagent-lighting/work/logs/blackbox_smoke_20260812/`（12 轨迹 + 检查脚本 +
  排障记录）；**正式训练已启动**（`run_grpo_humanevalfix_blackbox_ucloud.sh`，
  train161 / batch32 / mini16 / micro4 / 并发 64 / max_num_seqs 128 / util 0.8 /
  5 epoch，与 baseline/投机同口径）
  - 排障链（v0.37.3→v0.38.5）：漏 sandbox.start() → 隧道远端目标用 gateway hostname
    → GATEWAY_PORT 固定端口补丁（direct-URL 方案，公网 8001 未放行暂走隧道）→
    max_tokens=32000 超 vLLM 容量（adapter 截断 8192）→ 完整模式系统提示误判
    output 超限（加 --bare）→ max_turns 100→60 对齐白盒
- [ ] （可选）Codex 路径：升级 vLLM 后验证 Responses API
