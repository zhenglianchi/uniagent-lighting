# uni_agent_ext —— uni-agent 平台扩展

不修改 uni-agent 源码，通过注册表懒加载扩展能力。当前包含：

- `sandbox/tencent_agent_runtime.py`：腾讯云 Agent Runtime（云沙箱）后端（**E2B 兼容实现，已通过官方 demo 验证**）
- `agents/mini_swe_agent_runner.py`：白盒训练 runner（mini-swe-agent + 腾讯沙箱 +
  Gateway session + 真实 reward，HumanEvalFix/SWE-bench 任务类型）
- `agents/claude_code_runner.py`：黑盒训练 runner（Claude Code `-p` + 腾讯沙箱
  direct-URL 直连 Gateway + SWE-bench reward；v0.35.1，待上机验证）

## 接入 uni-agent

```bash
cd platform && pip install -e .   # 安装本扩展包（或把 platform/ 加入 PYTHONPATH）
```

在应用启动处（或直接 import）注册扩展：

```python
import uni_agent_ext            # 导入即注册
from uni_agent.sandbox import SandboxConfig, build_sandbox

config = SandboxConfig(
    provider="tencent_agent_runtime",
    image="python:3.12",
    runtime_timeout=3600,
    sandbox_kwargs={"startup_timeout": 180},
)
sandbox = build_sandbox(config)
```

## 环境变量（2026-08-04 定稿，凭据在 `work/tencent_sandbox.env`，chmod 600）

| 变量 | 用途 |
|---|---|
| `E2B_DOMAIN` | `ap-guangzhou.tencentags.com`（E2B 兼容端点） |
| `E2B_API_KEY` | `e2b_*` 兼容 Key（SDK 强制 e2b_ 前缀） |
| `TENCENT_SANDBOX_TEMPLATE` | 沙箱工具名（默认 `code-interpreter-v1`），可用 `sandbox_kwargs["template"]` 覆盖 |
| `TENCENT_SECRET_ID` / `TENCENT_SECRET_KEY` | 腾讯云 CAM 密钥（Cloud API 控制面：创建/查询沙箱工具） |
| `SANDBOX_PROXY` | 可选 HTTP 代理（与 uni-agent 其它后端一致） |

## 验证

- 最小连通：`python scripts/tencent_sandbox_demo.py`（创建 → run_code → kill）
- uni-agent 官方 demo：`python scripts/run_tencent_sandbox_demo.py`（安装包→写文件→执行→状态保持，2026-08-04 全过）
- 创建/查询沙箱工具：`python scripts/tencent_create_sandbox_tool.py` / `python scripts/tencent_list_sandbox_tools.py`

## TODO（后续）

- [x] SWE-bench 场景可行性（2026-08-04 已验证）：官方托管 `swebench` 工具类型 + 系统镜像仓库内置实例镜像，实例级覆盖镜像即可，无需推 TCR
- [x] uni-agent 后端 SWE-bench 适配（2026-08-04 已实现）：`start()` 检测
  `sweb.eval.` 开头镜像 → Cloud API `StartSandboxInstance`（镜像覆盖）→ E2B
  `Sandbox.connect` 而非 `Sandbox.create`
- [ ] `startup_timeout` 与 uni-agent `SANDBOX_STARTUP_TIMEOUT`/`SANDBOX_STARTUP_CONCURRENCY` 联动
