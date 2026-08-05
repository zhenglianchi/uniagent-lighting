"""腾讯云 Agent Runtime（云沙箱）后端 —— uni-agent Sandbox。

Provider 名：``tencent_agent_runtime``

实现方式（2026-08-04 重构）：控制面与数据面全部走**官方 E2B 兼容 SDK**
（``e2b_code_interpreter``），不依赖沙箱内 swerex server：

- 控制面：``Sandbox.create(template=<沙箱工具名>, timeout=...)`` / ``sandbox.kill()``
- 执行：``sandbox.commands.run``（原生命令通道，返回 stdout/stderr/exit_code）
- 文件：``sandbox.files.read/write``（原生数据通道，覆盖基类的 base64-exec 地板）

配置：
- 环境变量：``E2B_DOMAIN``（ap-guangzhou.tencentags.com）、``E2B_API_KEY``
  （``e2b_`` 兼容 Key，见 ``work/tencent_sandbox.env``）
- template：默认 ``code-interpreter-v1``（须已在腾讯云创建同名沙箱工具）；
  可用 sandbox_kwargs 的 ``template`` 或环境变量 ``TENCENT_SANDBOX_TEMPLATE`` 覆盖；
  SWE-bench 镜像（``sweb.eval.x86_64.*``）→ 工具名的映射在步骤 4 接入。
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uni_agent.sandbox.base import ExecResult, Sandbox
from uni_agent.sandbox.registry import register_sandbox

if TYPE_CHECKING:
    from uni_agent.sandbox.base import SandboxConfig

logger = logging.getLogger(__name__)

def _to_tencent_template(image: str) -> str:
    """把 uni-agent 通用镜像名映射为腾讯云沙箱工具名。

    E2B 兼容路径里 template 就是腾讯云“沙箱工具”名称。目前只做常见占位名映射，
    SWE-bench 场景（``sweb.eval.*``）在步骤 4 接入官方 ``swebench`` 工具类型。
    """
    if image in ("python:3.12", "python:3.11", "python:3.10"):
        return "code-interpreter-v1"
    logger.warning(
        "tencent_agent_runtime: 镜像 %r 无内置映射，按“沙箱工具名”透传（请确认已创建同名工具）",
        image,
    )
    return image


@register_sandbox("tencent_agent_runtime")
class TencentAgentRuntimeSandbox(Sandbox):
    """腾讯云 Agent Runtime 云沙箱后端（E2B 兼容实现）。

    用法（需先 ``import uni_agent_ext`` 注册懒加载模块）：

        config = SandboxConfig(
            provider="tencent_agent_runtime",
            image="python:3.12",
            runtime_timeout=3600,
            sandbox_kwargs={"startup_timeout": 180},
        )
        sandbox = build_sandbox(config)
        async with sandbox:
            result = await sandbox.exec_shell("python --version")
    """

    def __init__(
        self,
        *,
        image: str = "python:3.12",
        runtime_timeout: float = 3600.0,
        startup_timeout: float = 180.0,
        template: str | None = None,
    ) -> None:
        self.image = image
        self.template = (
            template or os.getenv("TENCENT_SANDBOX_TEMPLATE") or _to_tencent_template(image)
        )
        self.runtime_timeout = runtime_timeout
        self.startup_timeout = startup_timeout
        self._sbx: Any = None  # e2b_code_interpreter.Sandbox（阻塞 SDK，经 to_thread 调用）
        self._cloud_instance_id: str = ""  # SWE-bench 实例（Cloud API）的 InstanceId

    @classmethod
    def from_config(cls, config: SandboxConfig) -> TencentAgentRuntimeSandbox:
        return cls(
            image=config.image,
            runtime_timeout=config.runtime_timeout,
            **config.sandbox_kwargs,
        )

    # ----- 控制面 -----
    async def start(self) -> None:
        if self._sbx is not None:
            return  # already started

        from e2b_code_interpreter import Sandbox

        if self._is_swebench_image(self.image):
            await self._start_swebench_instance(Sandbox)
        else:
            self._sbx = await asyncio.to_thread(
                Sandbox.create, template=self.template, timeout=int(self.runtime_timeout)
            )
            logger.info("tencent sandbox created id=%s template=%s", self._sbx.sandbox_id, self.template)

        # uni-agent shell 工具在无原生 shell 时用 tmux；sweb 镜像直接跑黑盒 agent，跳过
        if not self._is_swebench_image(self.image):
            result = await self.exec_shell(
                "which tmux >/dev/null 2>&1 || (DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux)",
                timeout=180.0,
            )
            if result.exit_code != 0:
                logger.warning("tencent sandbox tmux install failed: %s", result.stderr.strip()[-500:])

    async def stop(self) -> None:
        sbx, self._sbx = self._sbx, None
        if sbx is not None:
            try:
                await asyncio.to_thread(sbx.kill)
                logger.info("tencent sandbox killed")
            except Exception:
                logger.debug("tencent sandbox kill failed", exc_info=True)
        if self._cloud_instance_id:
            try:
                await asyncio.to_thread(self._stop_cloud_instance, self._cloud_instance_id)
                logger.info("tencent swebench instance stopped: %s", self._cloud_instance_id)
            except Exception:
                logger.debug("StopSandboxInstance failed", exc_info=True)
            self._cloud_instance_id = ""

    # ----- SWE-bench 实例（Cloud API StartSandboxInstance + E2B connect）-----
    @staticmethod
    def _is_swebench_image(image: str) -> bool:
        return image.startswith("sweb.") or image.startswith("swebench/") or "sweb.eval" in image

    @staticmethod
    def _normalize_swebench_image(image: str) -> str:
        """StartSandboxInstance 需要系统仓库前缀 ``swebench/``。"""
        image = image.replace("docker.io/", "").replace("__", "_1776_")
        return image if image.startswith("swebench/") else f"swebench/{image}"

    def _ags_client(self, region: str):
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.ags.v20250920 import ags_client

        cred = credential.Credential(
            os.environ["TENCENT_SECRET_ID"], os.environ["TENCENT_SECRET_KEY"]
        )
        http_profile = HttpProfile(endpoint="ags.tencentcloudapi.com")
        return ags_client.AgsClient(cred, region, ClientProfile(httpProfile=http_profile))

    async def _start_swebench_instance(self, sandbox_cls) -> None:
        from tencentcloud.ags.v20250920 import models

        region = os.getenv("TENCENT_SANDBOX_REGION", "ap-guangzhou")
        tool_name = os.getenv("TENCENT_SANDBOX_SWEBENCH_TOOL", "swebench-v1")
        image = self._normalize_swebench_image(self.image)
        client = await asyncio.to_thread(self._ags_client, region)

        def _start() -> str:
            req = models.StartSandboxInstanceRequest()
            req.ToolName = tool_name
            req.Timeout = os.getenv("TENCENT_SANDBOX_TIMEOUT", "10m")
            req.CustomConfiguration = models.CustomConfiguration()
            req.CustomConfiguration.Image = image
            req.CustomConfiguration.ImageRegistryType = "system"
            resp = client.StartSandboxInstance(req)
            return resp.Instance.InstanceId

        instance_id = await asyncio.to_thread(_start)
        self._cloud_instance_id = instance_id
        self._sbx = await asyncio.to_thread(
            sandbox_cls.connect, sandbox_id=instance_id, api_key=os.environ.get("E2B_API_KEY")
        )
        logger.info("tencent swebench instance connected id=%s image=%s", instance_id, image)

    def _stop_cloud_instance(self, instance_id: str) -> None:
        from tencentcloud.ags.v20250920 import models

        region = os.getenv("TENCENT_SANDBOX_REGION", "ap-guangzhou")
        client = self._ags_client(region)
        req = models.StopSandboxInstanceRequest()
        req.InstanceId = instance_id
        client.StopSandboxInstance(req)

    def _require_sandbox(self) -> Any:
        if self._sbx is None:
            raise RuntimeError("TencentAgentRuntimeSandbox not started; call start() first")
        return self._sbx

    @property
    def instance_id(self) -> str:
        """底层实例 id（sweb 走 Cloud API InstanceId；其它走 E2B sandbox_id）。"""
        return self._cloud_instance_id or getattr(self._sbx, "sandbox_id", "")

    # ----- 数据面 -----
    async def is_alive(self) -> bool:
        sbx = self._sbx
        if sbx is None:
            return False
        try:
            return bool(await asyncio.to_thread(sbx.is_running))
        except Exception:
            return False

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        sbx = self._require_sandbox()
        import shlex

        cmd = shlex.join(str(a) for a in argv)
        cmd_timeout = timeout if timeout else 600.0
        try:
            result = await asyncio.to_thread(
                sbx.commands.run,
                cmd,
                cwd=workdir,
                envs=env or None,
                user="root",
                timeout=cmd_timeout,
            )
            return ExecResult(
                exit_code=int(result.exit_code),
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
        except Exception as exc:
            exit_code = getattr(exc, "exit_code", None)
            if exit_code is not None:
                return ExecResult(
                    exit_code=int(exit_code),
                    stdout=getattr(exc, "stdout", "") or "",
                    stderr=getattr(exc, "stderr", "") or "",
                )
            return ExecResult(exit_code=-1, stdout="", stderr=f"e2b commands.run failed: {exc}")

    async def _ensure_parent_dir(self, path: str) -> None:
        parent = str(Path(path).parent)
        try:
            await asyncio.to_thread(self._require_sandbox().files.make_dir, parent)
        except Exception:
            pass  # 目录可能已存在，写文件时容错

    async def write_file(self, path: str, content: bytes | str) -> None:
        data = content.encode("utf-8") if isinstance(content, str) else content
        try:
            await self._ensure_parent_dir(path)
            await asyncio.to_thread(self._require_sandbox().files.write, path, data)
        except Exception as exc:
            # sweb 镜像没有默认 'user' 用户，e2b filesystem API 会报
            # "error looking up user 'user'" → 退回 base64-exec 写入
            import base64

            logger.warning("files.write failed (%s); fallback to base64-exec", exc)
            b64 = base64.b64encode(data).decode()
            result = await self.exec_shell(f"echo {b64} | base64 -d > {path}", timeout=60)
            if result.exit_code != 0:
                raise RuntimeError(f"base64 write failed: {result.stderr[:500]}")

    async def read_file(self, path: str) -> bytes:
        data = await asyncio.to_thread(self._require_sandbox().files.read, path, "bytes")
        return bytes(data)

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        data = Path(local_path).read_bytes()
        await self._ensure_parent_dir(remote_path)
        await asyncio.to_thread(self._require_sandbox().files.write, remote_path, data)

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        data = await asyncio.to_thread(self._require_sandbox().files.read, remote_path, "bytes")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(data))
