"""uni-agent 平台扩展包。

本包在不修改 uni-agent 源码的前提下扩展其能力：
  - sandbox.tencent_agent_runtime：腾讯云 Agent Runtime（云沙箱）后端

使用前先注册扩展（把 provider 名映射到本包模块，供 uni-agent 懒加载）：

    import uni_agent_ext
    # 或显式调用：
    uni_agent_ext.register_extensions()

之后即可正常使用：SandboxConfig(provider="tencent_agent_runtime", ...)
"""

from __future__ import annotations


def register_extensions() -> None:
    """把扩展沙箱后端注册进 uni-agent 的 SANDBOX_MODULES（懒加载表）。"""
    from uni_agent.sandbox.registry import SANDBOX_MODULES

    SANDBOX_MODULES["tencent_agent_runtime"] = "uni_agent_ext.sandbox.tencent_agent_runtime"


# 导入即注册，保持与 uni-agent 其它后端一致的“自注册”体验。
register_extensions()
