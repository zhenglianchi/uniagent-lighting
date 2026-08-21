"""验证 Ray worker 能否 import uni_agent_ext（.pth vs PYTHONPATH 诊断）。

用法（训练机上）：
  python scripts/ray_import_test.py
通过标准：输出 MAIN-OK 与 RAY-WORKER-OK。
"""

from __future__ import annotations

import ray


@ray.remote
def check_import() -> str:
    import uni_agent_ext  # noqa: F401
    from uni_agent_ext.agents.mini_swe_agent_runner import mini_swe_agent_runner

    return f"RAY-WORKER-OK {mini_swe_agent_runner.__name__}"


def main() -> None:
    import uni_agent_ext  # noqa: F401

    print("MAIN-OK")
    ray.init(num_cpus=1, log_to_driver=False)
    print(ray.get(check_import.remote()))
    ray.shutdown()


if __name__ == "__main__":
    main()
