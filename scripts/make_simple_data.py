"""生成极简 Python 修复任务的 agentic 训练数据（8B 行为控制实验）。

任务：单函数 bug 修复（solution.py + test_solution.py），沙箱用腾讯 E2B
``code-interpreter-v1`` 模板（python:3.12），由 runner 注入文件（v0.25.1）。

用法：
  conda run -n swe-rl python scripts/make_simple_data.py --out-dir work/data
"""

from __future__ import annotations

import argparse
import json
import os


PROBLEM_TEMPLATE = (
    "Fix the bug in /testbed/solution.py.\n\n"
    "Buggy code currently in /testbed/solution.py:\n```python\n{code}\n```\n\n"
    "Requirements:\n"
    "1. Read /testbed/solution.py and /testbed/test_solution.py\n"
    "2. Edit /testbed/solution.py to fix the bug (prefix every command with `cd /testbed &&`)\n"
    "3. Run: cd /testbed && python -m pytest test_solution.py -q\n"
    "4. When all tests pass, submit with EXACTLY this command:\n"
    "   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cd /testbed && git diff -- solution.py > patch.txt && cat patch.txt\n"
)


TASKS: dict[str, dict] = {
    "simple_is_even": {
        "solution.py": "def is_even(n: int) -> bool:\n    return n % 2 == 1  # BUG: logic inverted\n",
        "test_solution.py": (
            "from solution import is_even\n"
            "def test_even():\n    assert is_even(4) is True\n"
            "def test_odd():\n    assert is_even(3) is False\n"
        ),
        "F2P": ["test_solution.py::test_even", "test_solution.py::test_odd"],
    },
    "simple_max_of": {
        "solution.py": "def max_of(a, b):\n    if a > b:\n        return b  # BUG: returns smaller\n    return a\n",
        "test_solution.py": (
            "from solution import max_of\n"
            "def test_first_larger():\n    assert max_of(5, 3) == 5\n"
            "def test_second_larger():\n    assert max_of(1, 2) == 2\n"
        ),
        "F2P": ["test_solution.py::test_first_larger", "test_solution.py::test_second_larger"],
    },
    "simple_reverse": {
        "solution.py": "def reverse(s: str) -> str:\n    return s  # BUG: no reversal\n",
        "test_solution.py": (
            "from solution import reverse\n"
            "def test_reverse_abc():\n    assert reverse('abc') == 'cba'\n"
            "def test_reverse_empty():\n    assert reverse('') == ''\n"
        ),
        "F2P": ["test_solution.py::test_reverse_abc", "test_solution.py::test_reverse_empty"],
    },
}


def task_config_for(instance_id: str) -> dict:
    return {
        "name": "simple_fix",
        "sandbox": {"provider": "tencent_agent_runtime", "image": "python:3.12"},
        "agent": {"name": "mini_swe_agent", "max_steps": 30},
        "model": {"temperature": 1.0, "max_total_tokens": 8192},
    }


def to_record(instance_id: str) -> dict:
    task = TASKS[instance_id]
    files = {k: v for k, v in task.items() if k.endswith(".py")}
    problem_statement = PROBLEM_TEMPLATE.format(code=files["solution.py"])
    metadata = {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "FAIL_TO_PASS": task["F2P"],
        "PASS_TO_PASS": [],
        "test_patch": "",
        "environment_setup_commit": "",
    }
    return {
        "data_source": "simple-bench",
        "prompt": [{"role": "user", "content": problem_statement}],
        "extra_info": {
            "tools_kwargs": {
                "task": task_config_for(instance_id),
                "env": {"image": "python:3.12", "instance_id": instance_id, "files": files},
                "reward": {"metadata": metadata},
            }
        },
        "reward_model": {"ground_truth": metadata},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="work/data")
    p.add_argument("--train-ids", nargs="+", default=["simple_is_even", "simple_max_of"])
    p.add_argument("--val-ids", nargs="+", default=["simple_reverse"])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in (("agentic_train.jsonl", args.train_ids), ("agentic_val.jsonl", args.val_ids)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for iid in ids:
                f.write(json.dumps(to_record(iid), ensure_ascii=False) + "\n")
        print(f"written {path} ({len(ids)} samples)")


if __name__ == "__main__":
    main()
