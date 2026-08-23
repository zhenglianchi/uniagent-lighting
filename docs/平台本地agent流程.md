# 平台本地 agent（外部形态）完整流程导读

> 覆盖：训练侧 `uni_agent_ext/agents/external_agent_runner.py`（165 行）
> ＋ 本地侧 `scripts/platform/platform_local_agent.py`（221 行）＋ 启动脚本
> `scripts/train/run_grpo_platform_test_ucloud.sh`。
> 前置阅读：`docs/uniagent源码导读.md`（framework/Gateway 部分）、
> `docs/数据流与网关导读.md`。

## 0. 一句话总览

**训练侧先起**：verl 训练（加载已有权重、`save_freq=-1` 不保存）→ framework 调
`external_agent_runner` → 建腾讯沙箱 + 写 `task.json` → 轮询 `done` 标记。
**用户后起**（WSL）：`platform_local_agent.py --wait` → SSH 读 `task.json` → 起
隧道 → 跑 mini-swe-agent（模型调用走隧道到云端 Gateway）→ 建 `done` 标记。
训练侧收到 done → 沙箱 pytest 算 reward → POST 回 session → 轨迹 + reward 进 TQ
→ GRPO 一步完成。

## 1. 谁先起、都起什么（启动顺序）

```text
┌─ 训练侧（node1）────────────────────────────────────────────┐
│ 1. source scripts/ops/bootstrap_ray_env.sh   # ray start 前注入凭据
│ 2. 单机验证：verl 自建 Ray（脚本没传 ray_init.address）
│ 3. MODEL=/home/ubuntu/models/Qwen3-8B-final \
│      setsid nohup bash run_grpo_platform_test_ucloud.sh \
│      > grpo_platform_test_baseline.log 2>&1 < /dev/null &
│    └─ main_ppo → AgentFrameworkRolloutAdapter.create
│       ├─ GatewayManager（GATEWAY_PORT=8001）
│       └─ AgentFrameworkWorker（Ray actor）
│    训练循环需要采样时 → external_agent_runner：
│       建沙箱 → 注入 solution.py → 写 task.json → 轮询 done
└──────────────────────────────────────────────────────────────┘
                           │ task.json（SFTP 可见）
                           ▼
┌─ 用户侧（WSL）───────────────────────────────────────────────┐
│ 4. cd uniagent-lighting
│    PYTHONPATH=$PWD/vendor/uni-agent:$PWD \
│    python scripts/platform/platform_local_agent.py --wait --timeout 1800
│    ├─ paramiko 连训练机（work/ucloud.env）
│    ├─ wait_for_task：轮询 task.json
│    ├─ TunnelForwarder：本地随机端口 → 训练机内网 Gateway:8001
│    ├─ 生成 mini-swe config（base_url=隧道后 URL，attach 沙箱）
│    ├─ 跑 mini-swe-agent（每轮模型调用走隧道 → Gateway → vLLM）
│    └─ SSH touch <session_id>.done
└──────────────────────────────────────────────────────────────┘
                           │ done 标记
                           ▼
┌─ 训练侧（继续）──────────────────────────────────────────────┐
│ 5. external_agent_runner 检测到 done
│    → evaluate_reward（沙箱 pytest FAIL_TO_PASS）
│    → POST reward_info 到 session.reward_info_url
│    → framework finalize 拿轨迹 → 写 TQ → GRPO step 完成
└──────────────────────────────────────────────────────────────┘
```

**结论：训练侧先起，用户后起。** 用户侧脚本带 `--wait`，即使先启动也只是
空转轮询，直到训练侧写出 task.json 才真正干活。

## 2. 训练侧：external_agent_runner 逐步解析

文件：`uni_agent_ext/agents/external_agent_runner.py`

### 2.1 入口（`external_agent_runner`，86 行）

满足 AgentRunner 契约（raw_prompt / session / sample_index / tools_kwargs /
run_timeout），由 framework 的 `_run_session` 以 ray_task 方式调用。

### 2.2 建沙箱 + 注入任务文件（96-124 行）

```python
sandbox = create_task_sandbox(image=image, gateway_url=session.base_url)
await sandbox.start()                       # 跳转 mini_swe_agent_runner.py:177
```

`create_task_sandbox` 在 `uni_agent_ext/agents/mini_swe_agent_runner.py:177`：
用 `tencent_agent_runtime.py`（腾讯 E2B 后端）建沙箱，把 `session.base_url`
传进去（沙箱内 agent 可达 Gateway）。

随后在沙箱里 `git init` + 写入任务文件（`solution.py` 等，`tools_kwargs.env.files`），
`git add -A`——**只注入题目文件，隐藏测试在 reward 阶段才写，无测试泄露**。

### 2.3 写 task.json（`_write_task_file`，47-75 行）

写到 `<PLATFORM_TEST_DIR>/<session_id>.task.json`，字段：

| 字段 | 内容 |
|---|---|
| `session_id` | Gateway 会话 id |
| `base_url` | Gateway 为该 session 暴露的模型端点（**本地 agent 要连的就是它**） |
| `instance_id` | 已建腾讯沙箱实例（本地 agent attach 用） |
| `image` | 沙箱镜像 |
| `raw_prompt` / `tools_kwargs` / `task` | 任务信息（issue、模型配置等） |
| `done_marker` | 本地 agent 完成后要 touch 的远程文件路径 |

### 2.4 等 done 标记（`_wait_for_done`，77-86 行）

每 5s（`PLATFORM_POLL_INTERVAL`）检查 `<session_id>.done` 是否存在，直到
`run_timeout`（默认 7200s）。**这是训练侧的主 hang 点**：本地 agent 一直不建
done，这里就一直等。

### 2.5 云侧 reward + POST（105-127 行）

```python
score, eval_result = await evaluate_reward_msa(sandbox, task, timeout=600)
    # 跳转 mini_swe_agent_runner.py:421：沙箱内写隐藏测试 → pytest FAIL_TO_PASS
reward_info = {"reward": score, "reward_score": score, "external_agent": True, **eval_result}
await client.post(session.reward_info_url, json={"reward_info": reward_info})  # httpx 30s 超时
```

`framework._score_from_reward_info` 消费 `reward` 键 → 轨迹带上 reward。

### 2.6 清理（finally）

删 task.json / done 文件，`sandbox.stop()`。

## 3. 本地侧：platform_local_agent.py 逐步解析

文件：`scripts/platform/platform_local_agent.py`

### 3.1 main()：凭据与 SSH（128-158 行）

1. `load_sandbox_env()`：读 `work/tencent_sandbox.env` → `E2B_DOMAIN` /
   `E2B_API_KEY`（本地 attach 沙箱用）；
2. `load_ucloud_env()`：读 `work/ucloud.env` → `UCLOUD1_HOST/USER/PASS/PORT`；
3. `paramiko.SSHClient().connect(host, port, user, password, timeout=30)` →
   `open_sftp()` + `get_transport()`。

### 3.2 wait_for_task：等 task.json（69-87 行）

每 5s `sftp.listdir(remote_dir)` 找 `*.task.json`，取最新的读回 payload；
超时（默认 1800s）raise `TimeoutError`。**用户侧第一个 hang 点**——训练侧没
起 / 沙箱建得慢 / Gateway 没就绪时就在这里空转。

### 3.3 TunnelForwarder：SSH 隧道（35-67 行）

paramiko `direct-tcpip` 通道：本地 `127.0.0.1:<随机端口>` → 训练机内网
`<base_url host>:8001`。实现：`_serve` 循环 accept，`_forward` 开双向 pump
线程。**隧道只是转发，本身不 hang**；hang 出现在“目标端口不通时 agent 请求
挂起”（靠 mini-swe-agent 的请求超时兜底）。

### 3.4 build_config：生成 mini-swe config（89-112 行）

调 `mini_swe_agent_runner.build_mini_swe_config`（uni_agent_ext，264 行）：

```python
base_url = f"http://127.0.0.1:{local_port}{parsed.path}"   # 隧道后的 Gateway 端点
model    = $PLATFORM_MODEL        # 默认 Qwen3-8B
max_turns= $PLATFORM_MAX_TURNS     # 默认 60
instance_id = payload["instance_id"]   # attach 云端已建沙箱
image    = payload["image"]
```

### 3.5 run_local_agent：跑 agent（114-127 行）

`ThreadPoolExecutor` 里执行：`get_sb_environment(config, instance)` →
`get_agent(...)` → `agent.run(problem_statement)`，`future.result(timeout=...)`
兜底超时。**agent 每轮模型调用走隧道 → 训练机 Gateway → vLLM；工具调用 attach
云端沙箱实例执行。**

### 3.6 创建 done 标记（158-165 行）

```python
client.exec_command(f"mkdir -p {remote_dir} && touch {done_marker}")
```

训练侧轮询到这个文件就继续。**注意：done 标记创建失败 / 网络断，训练侧会等到
7200s 超时**（超时后仍会评估，但轨迹可能不完整）。

## 4. 两端通信协议

```text
训练侧 ──写──▶ <dir>/<session_id>.task.json   （任务清单）
用户侧 ──读──▶ task.json → 隧道 + config + 跑 agent
用户侧 ──写──▶ <dir>/<session_id>.done        （完成信号）
训练侧 ──等──▶ done → reward → POST reward_info
```

全程没有“训练侧和用户侧互相同步”，只有文件系统两个标记：task.json（任务下发）
和 done（完成回报）。reward 在云侧，轨迹在 Gateway 云侧物化——所以
on-policy 保证不依赖 agent 位置。

## 5. 哪里会 hang（排查清单）

| 位置 | 现象 | 兜底 |
|---|---|---|
| 本地 `wait_for_task` | 训练侧没起 / 沙箱慢 / Gateway 没就绪 | `--timeout`（默认 1800s）→ TimeoutError |
| 本地 SSH connect | 网络不通 / 凭据错 | paramiko `timeout=30` |
| 本地 agent 请求 | 隧道目标端口不通（Gateway 没监听 / 内网不可达） | mini-swe-agent 请求超时（取决于其 request_timeout） |
| 本地 `run_local_agent` | agent 循环挂起（E2B 工具调用卡住） | `ThreadPoolExecutor` + `future.result(timeout)` → rc=-1 |
| 训练侧 `_wait_for_done` | 用户没跑 / agent 卡住 / done 没建成 | `run_timeout=7200s` → done=False，**仍会评估**（reward 可能 0） |
| 训练侧 `evaluate_reward` | 沙箱 pytest 慢/挂 | `SWE_AGENT_EVAL_TIMEOUT`（600s） |
| 训练侧 reward POST | Gateway 不可达 | httpx `timeout=30` |

常见“看起来卡住”其实是正常等待：本地 `wait_for_task` 轮询日志每 5s 打一行
`waiting for task.json`；训练侧 `_wait_for_done` 每 5s 轮询但**不打日志**
（只有超时才有 warning）。

## 6. 用户操作速查

训练侧（node1，先起）：

```bash
source scripts/ops/bootstrap_ray_env.sh
MODEL=/home/ubuntu/models/Qwen3-8B-final \
  setsid nohup bash scripts/train/run_grpo_platform_test_ucloud.sh \
  > grpo_platform_test_baseline.log 2>&1 < /dev/null &
```

用户侧（WSL，后起；训练起来后 task.json 出现即自动开始）：

```bash
cd /home/zhenglianchi/swe-rl-local/uniagent-lighting
PYTHONPATH=$PWD/vendor/uni-agent:$PWD \
  python scripts/platform/platform_local_agent.py --wait --timeout 1800
```

前置：`work/ucloud.env`（SSH 凭据）、`work/tencent_sandbox.env`（E2B 凭据）、
swe-rl conda 环境（paramiko/mini-swe-agent）。训练脚本加载的是**已有权重**
（`Qwen3-8B-final` / `final-spec`），`save_freq=-1` 不写新权重。

## 6.1 多实例（多个 WSL 并行认领，2026-08-23）

`wait_for_task` 已改为 **SFTP 原子认领**：把 task.json `rename` 成
`<name>.claimed`，谁 rename 成功谁拥有该任务，其余实例跳过继续找下一个；
处理完本地删除 claimed 文件。因此起 N 个实例即可并行消化 N 个任务：

```bash
# 每个 WSL 终端起一个实例；--max-tasks 控制单实例连续处理数（默认 1）
PYTHONPATH=$PWD/vendor/uni-agent:$PWD \
  python scripts/platform/platform_local_agent.py --wait --timeout 1800 --max-tasks 1
```

训练侧要让多个 task.json **并发**出现，需调整
`run_grpo_platform_test_ucloud.sh`（默认 1/1/1 = 单步验证；环境变量可调，口径与
内部形态一致，并发统一由 framework 的 `max_concurrent_sessions` 控制）：

- `ROLLOUT_N>1` 或 `TRAIN_BATCH_SIZE>1`（产生多个并发 session）；
- `CONCURRENCY>1`（= `max_concurrent_sessions`，external_agent_runner 每个 session
  写一个 task.json，framework 并发调度受此上限约束）。

```bash
# 例：与内部形态同口径的并发配置（4 组 × 并发 64）
ROLLOUT_N=4 TRAIN_BATCH_SIZE=1 CONCURRENCY=64 \
  bash run_grpo_platform_test_ucloud.sh
```

本地实例数 × `--max-tasks` ≥ 训练侧并发数即可消化全部任务。

**清理与内部形态一致**：

- 本地实例正常处理完即删 claimed 文件；**过期 claimed 自动回收**——认领后
  mtime 超过 `--timeout` 视为实例崩溃，改回 task.json 供其他实例重新认领；
- 训练侧 `external_agent_runner` finally 同时清理 `<id>.task.json` 与
  `<id>.task.json.claimed` 两种名字 + done 标记，并 `sandbox.stop()`（与内部
  形态的沙箱清理一致）；
- 实例崩溃且无人回收时，训练侧 `_wait_for_done` 仍会在 `run_timeout` 后超时并
  继续评估（reward 可能 0）。

## 7. 一句话总结

**训练侧 = 建沙箱 + 发任务（task.json）+ 等完成（done）+ 云侧 reward；用户侧 =
读任务 + 隧道连 Gateway + 跑 agent + touch done。两边只通过文件系统两个标记
交互，轨迹和 reward 全在云端，用户侧永远不碰训练数据。**
