"""黑盒平台化 MCP server（2026-08-12）：把 Claude Code 的工具转发到云端腾讯沙箱。

Claude Code（本地编排）通过 MCP 调用本 server 暴露的工具，工具内部经 E2B 在
**云端沙箱**执行（沙箱只执行，agent 在用户侧）：

- Bash(command, workdir)   → 沙箱 ``commands.run``
- Read(file_path)          → 沙箱 ``files.read``
- Write(file_path, ...)    → 沙箱 ``files.write``
- Edit(file_path, old, new)→ 沙箱读 + 替换 + 写
- Glob(pattern, path)      → 沙箱 ``find``

环境变量：``E2B_SANDBOX_ID``（训练侧 runner 建的沙箱实例）、``E2B_API_KEY``、
``E2B_DOMAIN``（本地 tencent_sandbox.env）。

用法（由 claude 通过 --mcp-config 拉起，stdio 通信）：
  python scripts/sandbox_mcp_server.py
"""

from __future__ import annotations

import asyncio
import os

from mcp.server.fastmcp import FastMCP

try:
    from e2b_code_interpreter import Sandbox
except Exception:  # noqa: BLE001
    Sandbox = None  # type: ignore[assignment]

mcp = FastMCP("sandbox")

_sandbox = None


def _get_sandbox():
    """按需连接云端沙箱（E2B_SANDBOX_ID）。"""
    global _sandbox
    if _sandbox is None:
        sandbox_id = os.environ.get("E2B_SANDBOX_ID", "")
        api_key = os.environ.get("E2B_API_KEY", "")
        if not sandbox_id:
            raise RuntimeError("E2B_SANDBOX_ID not set (训练侧 runner 建的沙箱实例)")
        if not api_key:
            raise RuntimeError("E2B_API_KEY not set (腾讯沙箱凭据)")
        if Sandbox is None:
            raise RuntimeError("e2b_code_interpreter not installed")
        _sandbox = Sandbox.connect(sandbox_id=sandbox_id, api_key=api_key)
    return _sandbox


def _format_result(result) -> str:
    exit_code = getattr(result, "exit_code", -1)
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return f"exit_code={exit_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"


@mcp.tool()
async def Bash(command: str, workdir: str | None = None, timeout: int = 120) -> str:
    """在云端沙箱内执行 shell 命令（cwd 默认 /testbed）。"""
    sbx = await asyncio.to_thread(_get_sandbox)
    result = await asyncio.to_thread(
        sbx.commands.run,
        command,
        cwd=workdir or "/testbed",
        user="root",
        timeout=timeout,
        request_timeout=timeout + 60,
    )
    return _format_result(result)


@mcp.tool()
async def Read(file_path: str) -> str:
    """读取云端沙箱内文件内容。"""
    sbx = await asyncio.to_thread(_get_sandbox)
    data = await asyncio.to_thread(sbx.files.read, file_path)
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


@mcp.tool()
async def Write(file_path: str, content: str) -> str:
    """把内容写入云端沙箱文件（覆盖）。"""
    sbx = await asyncio.to_thread(_get_sandbox)
    await asyncio.to_thread(sbx.files.write, file_path, content.encode("utf-8"))
    return f"written {file_path} ({len(content)} chars)"


@mcp.tool()
async def Edit(file_path: str, old_string: str, new_string: str) -> str:
    """在云端沙箱文件里替换 old_string 为 new_string（精确单次替换）。"""
    sbx = await asyncio.to_thread(_get_sandbox)
    data = await asyncio.to_thread(sbx.files.read, file_path)
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    count = text.count(old_string)
    if count == 0:
        raise RuntimeError(f"old_string not found in {file_path}")
    if count > 1:
        raise RuntimeError(f"old_string appears {count} times in {file_path}; make it unique")
    new_text = text.replace(old_string, new_string, 1)
    await asyncio.to_thread(sbx.files.write, file_path, new_text.encode("utf-8"))
    return f"edited {file_path} (1 replacement)"


@mcp.tool()
async def Glob(pattern: str, path: str = "/testbed") -> str:
    """在云端沙箱内按 glob 模式查找文件（走 find）。"""
    sbx = await asyncio.to_thread(_get_sandbox)
    result = await asyncio.to_thread(
        sbx.commands.run,
        f"find {path} -name {pattern!r} -not -path '*/node_modules/*' 2>/dev/null | head -100",
        cwd="/testbed",
        user="root",
        timeout=60,
        request_timeout=120,
    )
    return _format_result(result)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
