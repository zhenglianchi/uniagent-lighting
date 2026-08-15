# 部署与运维指南

## 1. 版本链

| 组件 | 版本 | 说明 |
|---|---|---|
| torch | 2.9.0+cu128 | vllm 0.11 配套 |
| vllm | 0.11.1 | verl 0.9 支持 logprobs_mode；多机硬性要求 ≥0.11.1 |
| transformers | 4.57.x | |
| verl | 0.9.0.dev | uni-agent 捆绑（本地副本 commit 78bba31） |
| ray | 2.56.1 | |
| TransferQueue | 0.1.9 | verl v1 数据平面 |
| mini-swe-agent | 2.4.6 | 白盒 harness |
| claude-code | 2.1.153 | 黑盒 harness（pin < 2.1.154，vLLM 0.11.1 role 校验） |

## 2. 基础环境安装

```bash
cd ~ && curl -fsSL -o install_ucloud_from_scratch.sh \
  https://raw.githubusercontent.com/zhenglianchi/uniagent-lighting/main/scripts/install_ucloud_from_scratch.sh
bash install_ucloud_from_scratch.sh
```

脚本完成：Miniforge + swe-rl env（Python 3.10）、清华 pip 源、HF 镜像
（`HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`）、
clone uni-agent（含 verl 子模块）、verl 补丁（见 §3）、模型下载
（`Qwen3-8B` 至 `~/models/`）。

## 3. 补丁清单

服务器侧 verl / uni-agent 补丁（部署时应用，`patches/` 内）：

| 补丁 | 作用 |
|---|---|
| `fix_strenum_ucloud.sh` | py3.10 StrEnum 兼容 |
| `patch_verl_ipc_cpu.py` | IPC CPU 大权重传输（bucket 2048 + 发送前 CPU→CUDA） |
| `verl_vllm_logprobs_spec_fix.patch` | EAGLE-3 下 logprobs 0 全丢修复（0→1） |
| `verl_merged_lora_materialize_fix.patch` | merge=True 同步基座权重 bug（backport verl#7014） |
| `verl_debug_metrics_logprobs_guard.patch` | batch 缺 rollout_log_probs 防崩 |
| `gateway_hermes_parse_guard.patch` | hermes 解析容错（JSON repair + 合成重试） |
| `verl_gateway_fixed_port.patch` | Gateway 固定端口（`GATEWAY_PORT`） |
| `gateway_fixed_port.patch` | run_uvicorn 支持固定端口 |
| `uni_agent_py310_compat.patch` | uni-agent py3.10 兼容 |
| `debug_launcher_py310.patch` | 官方调试工具 py3.10 兼容 |
| `uni_agent_skip_empty_response_trajectory.patch` | framework 跳过空响应轨迹（Mooncake 0 字节 slice 修复，2026-08-15） |
| `tq_mooncake_zero_slice_warn.patch` | TQ mooncake_client 0 字节 slice 告警（2026-08-15） |
| padding_utils num_turns 修复 | `verl/trainer/ppo/padding_utils.py` 第 109 行 `num_turns=0` → `torch.tensor(0, dtype=torch.long)`（padding 行 13B→8B，2026-08-15，node2 重启后需同步） |

## 4. 环境变量与启动顺序（重要）

**核心原则：Ray worker 的环境变量在 `ray start` 时固定，不继承训练脚本内的
`export`。凡 agent/沙箱/Gateway 运行需要的变量，必须在 `ray start` 之前
`export`，否则 Ray task 内拿不到（2026-08-15 实测踩坑：`E2B_API_KEY` 未传入，
24 会话全部 `AuthenticationException`）。**

### 4.1 标准启动顺序（node1 / node2 相同）

```bash
# 1) 沙箱/腾讯云凭据（必须在 ray start 前，Ray worker 需要）
source /home/ubuntu/uniagent-lighting/scripts/bootstrap_ray_env.sh

# 2) Ray（带 GPU 资源；如需跨节点先配好 hosts/SSH 互信）
/home/ubuntu/miniforge3/envs/swe-rl/bin/ray stop --force
/home/ubuntu/miniforge3/envs/swe-rl/bin/ray start --head --port=6379 --num-gpus=1
# 双机：node2 用 --address=node1内网IP:6379 加入

# 3) Mooncake master（node1，独立进程，不受 ray stop 影响）
# 见 scripts/run_grpo_dual_async_mooncake_ucloud.sh 内 MOONCAKE_AUTO_INIT 逻辑

# 4) 跑训练脚本（脚本内的 export 仅训练进程可见）
cd /home/ubuntu/swe-rl && bash run_grpo_xxx.sh
```

### 4.2 环境变量清单（脚本内已 export，供参考）

| 变量 | 值 | 作用 |
|---|---|---|
| `VLLM_USE_V1` | 1 | verl v1 引擎 |
| `RAY_memory_monitor_refresh_ms` | 0 | 关 Ray 内存监控（大对象误杀） |
| `HF_ENDPOINT` | https://hf-mirror.com | HF 国内镜像 |
| `HF_HUB_DISABLE_XET` | 1 | hf-mirror 不走 Xet 协议 |
| `MC_STORE_MEMCPY` | 0 | Mooncake 禁用 GPU memcpy（与 vLLM 同卡冲突） |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 1 | 多流稳定 |
| `GATEWAY_PORT` | 8001 | Gateway 固定端口 |
| `MSA_GATEWAY_TUNNEL` | 0 | 白盒本地直连 Gateway，不走沙箱内隧道 |
| `TENCENT_SANDBOX_SKIP_TMUX` | 1 | 跳过沙箱 tmux 安装（省 180s） |
| `E2B_DOMAIN` / `E2B_API_KEY` | tencent_sandbox.env | 腾讯沙箱 E2B 兼容端点 |

## 4. 改造仓部署

```bash
cd /home/ubuntu
git clone https://github.com/zhenglianchi/uniagent-lighting.git
ln -sfn /home/ubuntu/uniagent-lighting/uni_agent_ext /home/ubuntu/uni_agent_ext
echo "/home/ubuntu" > /home/ubuntu/miniforge3/envs/swe-rl/lib/python3.10/site-packages/uni_agent_ext.pth
mkdir -p /home/ubuntu/swe-rl/{data,logs,checkpoints,outputs}
```

运行脚本软链到 `/home/ubuntu/swe-rl/`（训练/评测/运维脚本，
`git pull` 后自动更新）。数据与凭据：

```bash
scp work/data/humanevalfix_train161.jsonl work/data/humanevalfix_val.jsonl \
  ubuntu@<服务器IP>:/home/ubuntu/swe-rl/data/
scp work/tencent_sandbox.env ubuntu@<服务器IP>:/home/ubuntu/swe-rl/
chmod 600 /home/ubuntu/swe-rl/tencent_sandbox.env
```

mini-swe-agent 补丁（白盒/评测需要）：

```bash
/home/ubuntu/miniforge3/envs/swe-rl/bin/pip install "mini-swe-agent==2.4.6"
cp patches/tencent_e2b.py \
  /home/ubuntu/miniforge3/envs/swe-rl/lib/python3.10/site-packages/minisweagent/environments/extra/tencent_e2b.py
cp patches/miniswe_swebench.py \
  /home/ubuntu/miniforge3/envs/swe-rl/lib/python3.10/site-packages/minisweagent/run/benchmarks/swebench.py
```

## 5. 训练

### 5.1 冒烟验证

```bash
bash /home/ubuntu/swe-rl/run_grpo_smoke_ucloud.sh 2>&1 | tee grpo_smoke.log
```

### 5.2 全样本训练（白盒 / 黑盒）

```bash
# 白盒 baseline / spec（环境变量可覆盖）
bash /home/ubuntu/swe-rl/run_grpo_humanevalfix_ucloud.sh 2>&1 | tee grpo_humanevalfix.log
SPEC_ON=1 LORA_MERGE=1 bash /home/ubuntu/swe-rl/spec_train_run.sh 2>&1 | tee grpo_spec.log

# 黑盒（Claude Code，沙箱内）——后台运行
CLAUDE_GATEWAY_TUNNEL=1 MSA_GATEWAY_SSH_HOST=<公网IP> \
  setsid nohup bash /home/ubuntu/swe-rl/run_grpo_humanevalfix_blackbox_ucloud.sh \
  > grpo_humanevalfix_blackbox.log 2>&1 < /dev/null &
```

### 5.3 续训

`resume_mode=auto` + `default_local_dir`（checkpoint 目录）→ 直接重跑同一脚本
自动从 `latest_checkpointed_iteration.txt` 续训；checkpoint keep 由
`MAX_CKPT_KEEP` 控制（=1 只留最新）。

### 5.4 平台化训练（外部 agent）

### 5.5 双机平台化正式训练（separate_async + Mooncake，2026-08-15 定稿）

**定位**：平台化训推链路的双机形态——白盒 mini-swe-agent（训练机本地）→
Gateway → 轨迹 → TQ/Mooncake → 云端训练（node1 trainer + node2 独立 rollout
引擎）→ LoRA 合并 → 全量评估。**25 步评估 83.23%，计为平台化训练结果。**

**前置**（两台均执行）：
```bash
# hosts / SSH 互信（新 IP 需先改 fix_multinode_hosts.sh 内 NODE1_IP/NODE2_IP）
source /home/ubuntu/uniagent-lighting/scripts/bootstrap_ray_env.sh
bash /home/ubuntu/uniagent-lighting/scripts/fix_multinode_hosts.sh

# node1：Ray head + Mooncake master
/home/ubuntu/miniforge3/envs/swe-rl/bin/ray stop --force
/home/ubuntu/miniforge3/envs/swe-rl/bin/ray start --head --port=6379 --num-gpus=1
# node2 加入：ray start --address=<node1内网>:6379 --num-gpus=1
# Mooncake master（node1，独立进程）见 scripts/run_grpo_dual_async_mooncake_ucloud.sh 注释
```

**正式训练**（node1 后台）：
```bash
cd /home/ubuntu/swe-rl
setsid nohup bash run_grpo_dual_async_mooncake_ucloud.sh \
  > logs/grpo_humanevalfix_dual_async_mooncake.log 2>&1 < /dev/null &
```
（默认配置即正式口径：train161/batch32/并发64/util0.8/5epoch/EAGLE-3/Mooncake；
`MAX_CKPT_KEEP=1` 已传递，checkpoint 自动滚动；另可挂
`ckpt_cleanup_daemon.sh` 独立守护兜底磁盘）

**评估**：
```bash
cd /home/ubuntu/swe-rl
bash eval_dual_async_final.sh   # 合并 LoRA + vLLM serve + 161 条全量评估
```

```bash
# 训练侧（后台）——runner = external_agent_runner，等本地 agent 完成
MODEL=/home/ubuntu/models/Qwen3-8B-final \
  setsid nohup bash run_grpo_platform_test_ucloud.sh > grpo_platform.log 2>&1 < /dev/null &

# 本地（WSL）——读任务、起隧道、跑 agent、回传完成标记
PYTHONPATH=... python uniagent-lighting/scripts/platform_local_agent.py --wait
```

## 6. 评估

```bash
# 1. 合并 LoRA 到 HF 权重（不覆盖基座）
python convert_verl_lora_to_hf.py \
  --ckpt <ckpt>/actor/model_world_size_1_rank_0.pt \
  --base /home/ubuntu/models/Qwen3-8B \
  --out /home/ubuntu/models/Qwen3-8B-final-<tag>

# 2. 起 vLLM + 评估（先小样本 3 条验证，再全量）
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/Qwen3-8B-final-<tag> --port 8001 \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 \
  --gpu-memory-utilization 0.8 --enable-prefix-caching &
python eval_humanevalfix.py \
  --data <val|train161>.jsonl --base-url http://127.0.0.1:8001/v1 \
  --model <served> --temperature 0.8 --concurrency 24 --out eval_<tag>.json
```

评估前必须 `source tencent_sandbox.env`（eval 脚本的 load_envs 路径推断在
服务器布局下失效）。

## 7. 训练日志与轨迹归档

训练产物（主日志 / 会话轨迹 / 评估结果）归档至代码仓 `work/logs/`：

| 归档 | 内容 |
|---|---|
| `humanevalfix_full_20260809/` | 白盒 baseline + spec 轨迹、主日志、逐步统计 |
| `blackbox_smoke_20260812/` | 黑盒小样本 3 步轨迹、排障记录 |
| `blackbox_full_20260812/` | 黑盒正式训练 step 1-25 轨迹、全量评估结果 |
| `spec_run_20260810/` | spec 日志、逐步统计、评测 |

## 8. 常见问题

- HF 下载必须 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`
- LoRA 训练必须 `use_fused_kernels=False`（与 PEFT 冲突）
- EAGLE-3 需 `lora.merge=True`（LoRA×SD 互斥）+ logprobs 补丁
- CPU offload 训练时 SSH 可能无响应（控制台 Web shell 操作）
- 腾讯沙箱并发超限会 `LimitExceeded.CPU`（渐进派发，评测并发取 24）
- 黑盒 claude-code 需 pin < 2.1.154 + `CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192`
- **Ray worker 不继承脚本内 export**：`E2B_API_KEY` / `E2B_DOMAIN` /
  `GATEWAY_PORT` 等必须在 `ray start` 前 export（见 §4.1），否则 agent 起沙箱报
  `AuthenticationException: API key is required`
- **Mooncake 0 字节 slice**：空响应轨迹（`max_trajectory_length` 截断）写入会被
  master 以 `INVALID_PARAMS` 拒绝 → framework 已跳过（补丁 #20），训练日志出现
  `skip empty-response trajectory` 为正常
- **vLLM EAGLE-3 illegal memory**：util 必须 ≥0.8（baseline/spec 同口径），
  util 0.6 下长上下文会触发 `Failed to reset prefix cache` → rejection sampler 越界

## 9. 权重与镜像清单（2026-08-15 定稿）

### 9.1 权重命名规范

统一命名 `Qwen3-8B-final-<训练形态>`，由 `convert_verl_lora_to_hf.py` 合并
对应训练最后一步 checkpoint 到基座生成（16G，BF16）：

| 权重 | 训练形态 | 通过率 | 状态 |
|---|---|---|---|
| `Qwen3-8B-final` | 单机白盒 baseline（sync 26 步） | 83.2% | 已归档（旧镜像/文档） |
| `Qwen3-8B-final-spec` | 单机白盒投机（25 步） | — | 已归档（旧镜像/文档） |
| `Qwen3-8B-final-blackbox` | 单机黑盒（25 步） | 80.75% | 已归档（旧镜像/文档） |
| **`Qwen3-8B-final-dual-async`** | **双机 separate_async + Mooncake + EAGLE-3（25 步）** | **83.23%** | **当前镜像唯一保留权重（平台化训练结果）** |

### 9.2 当前镜像内容（2026-08-15 清理后）

- **models/**：仅 `Qwen3-8B-final-dual-async`（16G，最终合并权重）；
  基座/投机器/旧 final 权重已删除（如需复现训练可重新下载基座 Qwen3-8B）
- **checkpoints/**：仅 `humanevalfix_dual_async/global_step_25`
  （双机训练最终 checkpoint，16G，LoRA adapter）
- **logs/**：全部保留（训练轨迹 25 步、grpo 日志、评估轨迹、合并/清理日志），
  已同步本地代码仓 `uniagent-lighting/work/logs/dual_async_20260815/`
- **data/**：训练数据保留（humanevalfix_train161.jsonl 等）
- 训练/评估脚本与全部补丁：见仓库 `scripts/` + `patches/`

### 9.3 复现要点

- 正式训练：`scripts/run_grpo_dual_async_mooncake_ucloud.sh`（双机
  separate_async + Mooncake + EAGLE-3 + 白盒，配置见训练评测分析 §8）
- 评估：`scripts/eval_dual_async_final.sh`（合并 + vLLM serve + 161 条全量）
- 双机环境：`scripts/bootstrap_ray_env.sh`（ray start 前环境变量）+
  `scripts/fix_multinode_hosts.sh`（内网映射，新 IP 需先改脚本）
