"""黑盒平台化 MCP server（手写 stdio JSON-RPC，2026-08-12）。

把 Claude Code 的工具转发到云端腾讯沙箱（agent 在用户侧、沙箱只执行）：
- Bash(command, workdir)   → 沙箱 ``commands.run``
- Read(file_path)          → 沙箱 ``files.read``
- Write(file_path, ...)    → 沙箱 ``files.write``
- Edit(file_path, old, new)→ 沙箱读 + 替换 + 写
- Glob(pattern, path)      → 沙箱 ``find``

MCP stdio 传输 = 每行一个 JSON-RPC 消息（newline-delimited），不依赖 FastMCP
（mcp 2.0 已拆包），手写协议避免版本兼容问题。

环境变量：``E2B_SANDBOX_ID``、``E2B_API_KEY``、``E2B_DOMAIN``。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

try:
    from e2b_code_interpreter import Sandbox
except Exception:  # noqa: BLE001
    Sandbox = None  # type: ignore[assignment]

PROTOCOL_VERSION = "2025-03-26"

_sandbox = None


def _get_sandbox():
    global _sandbox
    if _sandbox is None:
        sandbox_id = os.environ.get("E2B_SANDBOX_ID", "")
        api_key = os.environ.get("E2B_API_KEY", "")
        if not sandbox_id:
            raise RuntimeError("E2B_SANDBOX_ID not set")
        if not api_key:
            raise RuntimeError("E2B_API_KEY not set")
        if Sandbox is None:
            raise RuntimeError("e2b_code_interpreter not installed")
        _sandbox = Sandbox.connect(sandbox_id=sandbox_id, api_key=api_key)
    return _sandbox


def _run(cmd: str, workdir: str | None = None, timeout: int = 120):
    sbx = _get_sandbox()
    return sbx.commands.run(
        cmd,
        cwd=workdir or "/testbed",
        user="root",
        timeout=timeout,
        request_timeout=timeout + 60,
    )


def _format_result(result) -> str:
    exit_code = getattr(result, "exit_code", -1)
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return f"exit_code={exit_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"


async def _tool_bash(args: dict) -> str:
    result = await asyncio.to_thread(
        _run, args["command"], args.get("workdir"), int(args.get("timeout", 120))
    )
    return _format_result(result)


async def _tool_read(args: dict) -> str:
    sbx = await asyncio.to_thread(_get_sandbox)
    data = await asyncio.to_thread(sbx.files.read, args["file_path"])
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


async def _tool_write(args: dict) -> str:
    sbx = await asyncio.to_thread(_get_sandbox)
    await asyncio.to_thread(sbx.files.write, args["file_path"], args["content"].encode("utf-8"))
    return f"written {args['file_path']} ({len(args['content'])} chars)"


async def _tool_edit(args: dict) -> str:
    sbx = await asyncio.to_thread(_get_sandbox)
    data = await asyncio.to_thread(sbx.files.read, args["file_path"])
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    count = text.count(args["old_string"])
    if count == 0:
        raise RuntimeError(f"old_string not found in {args['file_path']}")
    if count > 1:
        raise RuntimeError(f"old_string appears {count} times; make it unique")
    await asyncio.to_thread(
        sbx.files.write,
        args["file_path"],
        text.replace(args["old_string"], args["new_string"], 1).encode("utf-8"),
    )
    return f"edited {args['file_path']} (1 replacement)"


async def _tool_glob(args: dict) -> str:
    pattern = args["pattern"]
    path = args.get("path", "/testbed")
    result = await asyncio.to_thread(
        _run, f"find {path} -name {pattern!r} -not -path '*/node_modules/*' 2>/dev/null | head -100"
    )
    return _format_result(result)


TOOLS: dict[str, dict] = {
    "Bash": {
        "description": "在云端沙箱内执行 shell 命令（cwd 默认 /testbed）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
                "workdir": {"type": "string", "description": "工作目录，默认 /testbed"},
                "timeout": {"type": "integer", "description": "超时秒数，默认 120"},
            },
            "required": ["command"],
        },
    },
    "Read": {
        "description": "读取云端沙箱内文件内容。",
        "inputSchema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    "Write": {
        "description": "把内容写入云端沙箱文件（覆盖）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    "Edit": {
        "description": "在云端沙箱文件里把 old_string 替换为 new_string（唯一匹配）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    "Glob": {
        "description": "在云端沙箱内按 glob 模式查找文件。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "搜索起点，默认 /testbed"},
            },
            "required": ["pattern"],
        },
    },
}

_HANDLERS = {
    "Bash": _tool_bash,
    "Read": _tool_read,
    "Write": _tool_write,
    "Edit": _tool_edit,
    "Glob": _tool_glob,
}


def _rpc(id, result=None, error=None) -> str:
    msg: dict = {"jsonrpc": "2.0", "id": id}
    if error is not None:
        msg["error"] = {"code": error[0], "message": error[1]}
    else:
        msg["result"] = result
    return json.dumps(msg)


async def _handle(msg: dict) -> str | None:
    method = msg.get("method")
    params = msg.get("params") or {}
    msg_id = msg.get("id")
    if method == "initialize":
        return _rpc(
            msg_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sandbox-mcp", "version": "0.1.0"},
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return _rpc(msg_id, {})
    if method == "tools/list":
        return _rpc(
            msg_id,
            {"tools": [{"name": name, "description": spec["description"], "inputSchema": spec["inputSchema"]} for name, spec in TOOLS.items()]},
        )
    if method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        handler = _HANDLERS.get(name)
        if handler is None:
            return _rpc(msg_id, error=(-32602, f"unknown tool: {name}"))
        try:
            text = await handler(arguments)
            return _rpc(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as exc:  # noqa: BLE001
            return _rpc(
                msg_id,
                {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True},
            )
    return _rpc(msg_id, error=(-32601, f"method not found: {method}"))


async def main() -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    writer_transport, writer_protocol = await loop.connect_write_pipe(
        lambda: asyncio.streams.FlowControlMixin(asyncio.Lock()), sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)

    while True:
        line = await reader.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = await _handle(msg)
        if response is not None:
            writer.write((response + "\n").encode())
            await writer.drain()


if __name__ == "__main__":
    asyncio.run(main())
