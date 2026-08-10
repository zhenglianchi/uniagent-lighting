#!/usr/bin/env python3
"""列出并清理腾讯云沙箱实例（训练/评测残留），释放 CPU 配额。"""

from __future__ import annotations

import os


def main() -> None:
    from e2b_code_interpreter import Sandbox

    domain = os.environ.get("E2B_DOMAIN", "ap-guangzhou.tencentags.com")
    api_key = os.environ.get("E2B_API_KEY", "")
    if not api_key:
        raise SystemExit("E2B_API_KEY not set")

    print(f"listing sandboxes @ {domain} ...")
    try:
        paginator = Sandbox.list(api_key=api_key, domain=domain, limit=100)
    except TypeError:
        paginator = Sandbox.list(limit=100)
    items = list(paginator.next_items()) if hasattr(paginator, "next_items") else []
    while paginator.has_next:
        items.extend(paginator.next_items())
    print(f"found {len(items)} sandbox(es)")
    killed = 0
    for sbx in items:
        sid = getattr(sbx, "sandbox_id", None) or str(sbx)
        print(f"killing {sid}")
        try:
            Sandbox.kill(sid, api_key=api_key, domain=domain)
            killed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  kill failed: {exc}")
    print(f"killed {killed} sandbox(es)")


if __name__ == "__main__":
    main()
