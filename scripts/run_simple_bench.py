"""极简 Python 修复任务验证：验证 8B 是否真的会修改代码（SWE-bench 之外的控制实验）。

用法（node2，需 vLLM server 已起在 127.0.0.1:8000）：
  cd /home/ubuntu/swe-rl && set -a && source tencent_sandbox.env && set +a && \
  export E2B_DOMAIN=ap-guangzhou.tencentags.com E2B_API_KEY=$TENCENT_SANDBOX_E2B_TOKEN && \
  /home/ubuntu/miniforge3/envs/swe-rl/bin/python \
    /home/ubuntu/uniagent-lighting/scripts/run_simple_bench.py simple_is_even
"""

from __future__ import annotations

import base64
import json
import sys

import yaml
from minisweagent.agents import get_agent
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment


TASKS: dict[str, dict[str, str]] = {
    "simple_is_even": {
        "solution.py": "def is_even(n: int) -> bool:\n    return n % 2 == 1  # BUG: logic inverted\n",
        "test_solution.py": (
            "from solution import is_even\n"
            "def test_even():\n    assert is_even(4) is True\n"
            "def test_odd():\n    assert is_even(3) is False\n"
        ),
    },
    "simple_max_of": {
        "solution.py": "def max_of(a, b):\n    if a > b:\n        return b  # BUG: returns smaller\n    return a\n",
        "test_solution.py": (
            "from solution import max_of\n"
            "def test_first_larger():\n    assert max_of(5, 3) == 5\n"
            "def test_second_larger():\n    assert max_of(1, 2) == 2\n"
        ),
    },
    "simple_reverse": {
        "solution.py": "def reverse(s: str) -> str:\n    return s  # BUG: no reversal\n",
        "test_solution.py": (
            "from solution import reverse\n"
            "def test_reverse_abc():\n    assert reverse('abc') == 'cba'\n"
            "def test_reverse_empty():\n    assert reverse('') == ''\n"
        ),
    },
}


def build_instance(instance_id: str) -> dict:
    files = TASKS[instance_id]
    payload = {
        f"f{index}_{name.split('.')[0]}_b64": base64.b64encode(content.encode()).decode()
        for index, (name, content) in enumerate(files.items())
    }
    code = files["solution.py"]
    problem = (
        f"Fix the bug in /testbed/solution.py.\n\n"
        f"Buggy code currently in /testbed/solution.py:\n```python\n{code}\n```\n\n"
        f"Requirements:\n"
        f"1. Read /testbed/solution.py and /testbed/test_solution.py\n"
        f"2. Edit /testbed/solution.py to fix the bug (prefix every command with `cd /testbed &&`)\n"
        f"3. Run: cd /testbed && python -m pytest test_solution.py -q\n"
        f"4. When all tests pass, submit with EXACTLY this command:\n"
        f"   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cd /testbed && git diff -- solution.py > patch.txt && cat patch.txt\n"
    )
    return {
        "instance_id": instance_id,
        "problem_statement": problem,
        "image_name": "python:3.12",
        **payload,
    }


def main() -> None:
    instance_id = sys.argv[1] if len(sys.argv) > 1 else "simple_is_even"
    if instance_id not in TASKS:
        raise SystemExit(f"unknown task {instance_id}; available: {list(TASKS)}")
    instance = build_instance(instance_id)

    config = yaml.safe_load(open("/home/ubuntu/swe-rl/config_qwen3_vllm.yaml"))
    config["environment"] = {
        "environment_class": "tencent_e2b",
        "image": "python:3.12",
        "cwd": "/",  # /testbed 由 startup 命令创建，agent 需 cd /testbed
        "timeout": 120,
        "user": "root",
        "sandbox_timeout": 1800,
    }
    config["run"] = {
        "env_startup_command": (
            "mkdir -p /testbed && cd /testbed && git init -q && "
            "echo '{{f0_solution_b64}}' | base64 -d > solution.py && "
            "echo '{{f1_test_solution_b64}}' | base64 -d > test_solution.py && "
            "ls -la /testbed"
        )
    }
    config["agent"]["step_limit"] = 30

    env = get_sb_environment(config, instance)
    agent = get_agent(get_model(config=config["model"]), env, config["agent"])
    try:
        agent.run(instance["problem_statement"])
    finally:
        pass

    # 沙箱内评估：跑测试（agent 结束后沙箱可能已停，尽力而为）
    try:
        out = env.execute({"command": "cd /testbed && python -m pytest test_solution.py -q --tb=no"})
        print("=== POST-RUN pytest ===")
        print((out.get("stdout") or out.get("output") or "")[-2000:])
        print("returncode:", out.get("returncode"))
    except Exception as exc:
        print("post-run eval failed:", exc)


if __name__ == "__main__":
    main()
