"""Tencent Cloud Agent Runtime（腾讯云）SWE-bench 环境 for mini-SWE-agent。

流程：Cloud API ``StartSandboxInstance`` 用实例镜像覆盖启动托管 ``swebench``
工具（系统镜像仓库，无需推 TCR）→ E2B ``Sandbox.connect`` 连接 → 
``commands.run`` 执行命令 → ``StopSandboxInstance`` 销毁。

必需环境变量：
    TENCENT_SECRET_ID / TENCENT_SECRET_KEY    Cloud API 密钥（起/停实例）
    E2B_DOMAIN / E2B_API_KEY                  E2B 兼容端点（默认 ap-guangzhou.tencentags.com）
可选：
    TENCENT_SANDBOX_REGION                    默认 ap-guangzhou

配置（YAML environment 节）：
    template        沙箱工具名（默认 swebench-v1，ToolType=swebench）
    image           SWE-bench 镜像（get_sb_environment 注入，自动去 docker.io/ 前缀、
                    __ → _1776_ 归一化）
    cwd             默认 /testbed
    timeout         命令超时（秒）
    user            root
    sandbox_timeout 实例超时（秒，默认 1800）
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import weakref
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge

_active_sandboxes: list[weakref.ref] = []


def _cleanup_all_sandboxes(signum=None, frame=None):
    """进程退出/信号时清理所有沙箱。"""
    for ref in _active_sandboxes:
        env = ref()
        if env is not None:
            env.cleanup()
    _active_sandboxes.clear()
    if signum is not None:
        raise SystemExit(1)


atexit.register(_cleanup_all_sandboxes)
for _sig in (signal.SIGINT, signal.SIGTERM):
    _prev = signal.getsignal(_sig)

    def _handler(signum, frame, _prev=_prev):
        _cleanup_all_sandboxes(signum, frame)
        if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
            _prev(signum, frame)

    signal.signal(_sig, _handler)


class TencentE2BEnvironmentConfig(BaseModel):
    template: str = "swebench-v1"
    image: str = ""
    cwd: str = "/testbed"
    timeout: int = 60
    user: str = "root"
    sandbox_timeout: int = 1800
    env: dict[str, str] = {}
    interpreter: list[str] = ["bash", "-c"]
    attach_instance_id: str | None = None  # 已有实例直接 connect（runner 已建好沙箱）


class TencentE2BEnvironment:
    def __init__(self, *, logger: logging.Logger | None = None, **kwargs):
        self.logger = logger or logging.getLogger("minisweagent.tencent_e2b")
        self.config = TencentE2BEnvironmentConfig(**kwargs)
        self._killed = False
        self._instance_id: str | None = None
        self._sandbox: Any = None
        self._create()
        _active_sandboxes.append(weakref.ref(self))

    # ----- 控制面 -----
    def _normalize_image(self, image: str) -> str:
        image = image.strip()
        if image.startswith("docker.io/"):
            image = image[len("docker.io/"):]
        # SWE-bench 数据集 id 用 __ 分隔 org/repo，Docker/腾讯云命名用 _1776_
        image = image.replace("__", "_1776_")
        return image

    def _create(self) -> None:
        from e2b_code_interpreter import Sandbox

        if self.config.attach_instance_id:
            # runner 已通过 Cloud API 建好实例，这里只连接不创建
            self._instance_id = self.config.attach_instance_id
            self._sandbox = Sandbox.connect(sandbox_id=self._instance_id)
            self.logger.info("E2B attached to sandbox: %s", self._instance_id)
            return

        from tencentcloud.ags.v20250920 import ags_client, models
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        region = os.environ.get("TENCENT_SANDBOX_REGION", "ap-guangzhou")
        cred = credential.Credential(
            os.environ["TENCENT_SECRET_ID"], os.environ["TENCENT_SECRET_KEY"]
        )
        http_profile = HttpProfile(endpoint="ags.tencentcloudapi.com")
        client = ags_client.AgsClient(cred, region, ClientProfile(httpProfile=http_profile))

        image = self._normalize_image(self.config.image)
        req = models.StartSandboxInstanceRequest()
        req.ToolName = self.config.template
        req.Timeout = f"{self.config.sandbox_timeout}s"
        req.CustomConfiguration = models.CustomConfiguration()
        req.CustomConfiguration.Image = image
        req.CustomConfiguration.ImageRegistryType = "system"
        resp = client.StartSandboxInstance(req)
        self._instance_id = resp.Instance.InstanceId
        self.logger.info(
            "Tencent swebench instance started: %s image=%s", self._instance_id, image
        )
        self._sandbox = Sandbox.connect(sandbox_id=self._instance_id)
        self.logger.info("E2B connected to sandbox: %s", self._instance_id)

    # ----- 数据面 -----
    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        effective_timeout = timeout or self.config.timeout
        full_cmd = f"cd {cwd} && {command}"

        try:
            result = self._sandbox.commands.run(
                full_cmd,
                user=self.config.user,
                timeout=effective_timeout,
            )
            combined = result.stdout or ""
            if result.stderr:
                combined = combined + result.stderr if combined else result.stderr
            output = {
                "output": combined or "",
                "returncode": result.exit_code,
                "exception_info": "",
            }
        except Exception as e:
            exit_code = getattr(e, "exit_code", -1)
            out = getattr(e, "stdout", "") or ""
            err = getattr(e, "stderr", "") or ""
            combined = (out + err) if out and err else (out or err)
            output = {
                "output": combined,
                "returncode": exit_code,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }

        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def cleanup(self) -> None:
        if self._killed:
            return
        self._killed = True
        if self.config.attach_instance_id:
            # attach 模式：实例生命周期归 runner（建沙箱 + reward 评估都由 runner 负责），
            # 退出时只断开连接，不 kill / 不停实例
            self._sandbox = None
            return
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception:
                self.logger.debug("E2B kill failed", exc_info=True)
            self._sandbox = None
        if self._instance_id:
            try:
                from tencentcloud.ags.v20250920 import ags_client, models
                from tencentcloud.common import credential
                from tencentcloud.common.profile.client_profile import ClientProfile
                from tencentcloud.common.profile.http_profile import HttpProfile

                region = os.environ.get("TENCENT_SANDBOX_REGION", "ap-guangzhou")
                cred = credential.Credential(
                    os.environ["TENCENT_SECRET_ID"], os.environ["TENCENT_SECRET_KEY"]
                )
                http_profile = HttpProfile(endpoint="ags.tencentcloudapi.com")
                client = ags_client.AgsClient(
                    cred, region, ClientProfile(httpProfile=http_profile)
                )
                req = models.StopSandboxInstanceRequest()
                req.InstanceId = self._instance_id
                client.StopSandboxInstance(req)
                self.logger.info("Tencent swebench instance stopped: %s", self._instance_id)
            except Exception as e:
                if "STOPPED state" in str(e):
                    # E2B kill 已先停实例，Cloud API stop 幂等即可
                    self.logger.debug("instance already stopped: %s", self._instance_id)
                else:
                    self.logger.warning(
                        "Failed to stop Tencent instance %s: %s", self._instance_id, e
                    )
            self._instance_id = None

    def __del__(self):
        self.cleanup()
