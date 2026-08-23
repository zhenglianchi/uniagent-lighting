#!/usr/bin/env python3
"""从 ``bigcode/humanevalpack`` 生成 **HumanEvalFix** agentic 训练数据（verl/uni-agent）。

任务口径：单函数 bug 修复。沙箱预置 ``/testbed/solution.py``（buggy 代码，git 已跟踪），
agent 修好后按 mini-swe 模板提交 git patch；隐藏测试 ``test_solution.py`` **只在 reward
阶段注入**（无测试泄露）。

schema（与 verl agentic RL 一致）：
``data_source / prompt / extra_info.tools_kwargs{task, env, reward} / reward_model.ground_truth``。
默认输出 ``humanevalfix_train{--train-num}.jsonl``（如 161 → ``humanevalfix_train161.jsonl``）
/ ``humanevalfix_val.jsonl``。

用法：
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \\
  conda run -n swe-rl python scripts/data/make_humanevalfix_data.py --train-num 3 --val-num 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path


DATASET_NAME = "bigcode/humanevalpack"
DATASET_CONFIG = "python"

PROBLEM_TEMPLATE = """Fix the bug in /testbed/solution.py.

Problem:
{docstring}

Function signature:
{signature}

The buggy implementation is already in /testbed/solution.py.

Requirements:
1. Read /testbed/solution.py.
2. Edit /testbed/solution.py to fix the bug. Prefix every command with `cd /testbed &&`.
   Keep the change minimal: fix only the buggy logic, do not rewrite the docstring or unchanged code.
   If you must rewrite the whole file, write it with ONE heredoc command, for example:
   cat > /testbed/solution.py <<'PYEOF'
   ...complete file content...
   PYEOF
   NEVER build the file line by line with `echo ... >> solution.py` — that wastes turns and breaks on quotes.
3. Verify your fix yourself (e.g., run the docstring examples). Hidden tests will be run after submission.
4. When done, submit with EXACTLY this command:
   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cd /testbed && git diff -- solution.py > patch.txt && cat patch.txt
"""


def instance_id_for(task_id: str) -> str:
    """humanevalfix 任务 id（``Python/0`` -> ``humanevalfix-Python-0``）。"""
    return f"humanevalfix-{task_id.replace('/', '-')}"


def build_solution_code(prompt: str, buggy_solution: str) -> str:
    """solution.py = 原题 prompt（imports+签名+docstring）+ buggy 函数体。"""
    return prompt.rstrip() + "\n" + buggy_solution.rstrip() + "\n"


def build_test_file(entry_point: str, test: str, test_setup: str = "") -> str:
    """把 humanevalpack 的 ``check(candidate)`` 风格测试转成可被 pytest 收集的文件。

    原 test 结尾有模块级 ``check(<entry_point>)`` 调用（pytest import 时就会执行，
    会变成 collect error 或提前断言失败），这里去掉并在最后包一层 ``test_all()``。
    生成单测 ``test_solution.py::test_all``，FAIL_TO_PASS 即该 node id。
    ``from solution import *``：humanevalpack 的测试会引用同文件的辅助函数
    （如 decode_cyclic 的测试用 encode_cyclic），只导入 entry_point 会 NameError。
    """
    body = test.rstrip()
    lines = body.splitlines()
    if lines:
        last = lines[-1].strip()
        if re.fullmatch(rf"check\(\s*{re.escape(entry_point)}\s*\)", last):
            lines = lines[:-1]
            body = "\n".join(lines).rstrip()
    parts: list[str] = ["from solution import *"]
    if test_setup and test_setup.strip():
        parts.append(test_setup.rstrip())
    if body:
        parts.append(body)
    parts.append(f"def test_all():\n    check({entry_point})")
    return "\n\n".join(parts) + "\n"


def problem_statement_for(row: dict) -> str:
    return PROBLEM_TEMPLATE.format(
        docstring=(row.get("docstring") or "").strip(),
        signature=(row.get("signature") or "").strip(),
    )


def verify_task(
    entry_point: str,
    solution_code: str,
    test_code: str,
    *,
    expect_pass: bool,
    timeout: int = 20,
) -> tuple[bool, int, str]:
    """本地验证：buggy 版应 FAIL（rc=1），canonical 版应 PASS（rc=0）。

    严格区分 rc：0=全过、1=有断言失败、2=collect/import 错误、5=没有收集到测试；
    buggy 只接受 rc=1（排除"测试文件本身写错"的假阴性）。
    """
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "solution.py").write_text(solution_code, encoding="utf-8")
            (p / "test_solution.py").write_text(test_code, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider"],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        # bug 版可能死循环（HumanEval 部分任务 while 缺失步进等），视为不可验证 → 跳过
        return False, -1, f"timeout after {timeout}s"
    tail = (proc.stdout or "")[-300:].strip().replace("\n", " | ")
    if expect_pass:
        ok = proc.returncode == 0
    else:
        ok = proc.returncode == 1
    return ok, proc.returncode, tail


def task_config_for(instance_id: str, *, max_steps: int, temperature: float, max_total_tokens: int) -> dict:
    return {
        "name": "humaneval_fix",
        "sandbox": {"provider": "tencent_agent_runtime", "image": "python:3.12"},
        "agent": {"name": "mini_swe_agent", "max_steps": max_steps},
        "model": {"temperature": temperature, "max_total_tokens": max_total_tokens},
    }


def to_agentic_record(row: dict, *, max_steps: int, temperature: float, max_total_tokens: int) -> dict:
    task_id = row["task_id"]
    instance_id = instance_id_for(task_id)
    entry_point = row["entry_point"]
    solution_code = build_solution_code(row["prompt"], row["buggy_solution"])
    test_code = build_test_file(entry_point, row["test"], row.get("test_setup") or "")
    problem_statement = problem_statement_for(row)
    metadata = {
        "instance_id": instance_id,
        "task_type": "humaneval_fix",
        "entry_point": entry_point,
        "problem_statement": problem_statement,
        # 本地验证结果：正常 True；死循环/不可验证但保留计入指标的样本为 False
        "verified": True,
        "deadloop": False,
        "FAIL_TO_PASS": ["test_solution.py::test_all"],
        "PASS_TO_PASS": [],
        "test_patch": "",
        # 隐藏测试：只给 reward 用，不注入 agent（无测试泄露）
        "hidden_files": {"test_solution.py": test_code},
    }
    return {
        "data_source": "humanevalfix-python",
        "prompt": [{"role": "user", "content": problem_statement}],
        "extra_info": {
            "tools_kwargs": {
                "task": task_config_for(instance_id, max_steps=max_steps, temperature=temperature, max_total_tokens=max_total_tokens),
                # files = 可见任务文件（agent 工作区）；hidden_files 在 metadata 里
                "env": {
                    "image": "python:3.12",
                    "instance_id": instance_id,
                    "workdir": "/testbed",
                    "files": {"solution.py": solution_code},
                },
                "reward": {"metadata": metadata},
            }
        },
        "reward_model": {"ground_truth": metadata},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-num", type=int, default=2)
    p.add_argument("--val-num", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="work/data")
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-total-tokens", type=int, default=8192)
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify-timeout", type=int, default=20)
    p.add_argument(
        "--include-unverified",
        action="store_true",
        help="本地验证失败（如 buggy 版死循环超时）的样本也保留进数据集，"
        "metadata.verified=False / deadloop=True 标记，训练时计入指标（reward=0 口径）",
    )
    args = p.parse_args()

    from datasets import load_dataset

    print(f"loading {DATASET_NAME}/{DATASET_CONFIG} ...")
    rows = list(load_dataset(DATASET_NAME, DATASET_CONFIG, split="test"))
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    selected: list[tuple[str, dict]] = []
    skipped: list[str] = []
    for idx, row in enumerate(rows):
        instance_id = instance_id_for(row["task_id"])
        if not row.get("entry_point") or not row.get("buggy_solution") or not row.get("test"):
            skipped.append(f"{instance_id}: missing fields")
            continue
        if args.verify:
            solution_code = build_solution_code(row["prompt"], row["buggy_solution"])
            canonical_code = build_solution_code(row["prompt"], row["canonical_solution"])
            test_code = build_test_file(row["entry_point"], row["test"], row.get("test_setup") or "")
            if idx % 5 == 0:
                print(f"verifying {instance_id} ... ({idx + 1}/{len(rows)})")
            buggy_ok, buggy_rc, buggy_tail = verify_task(
                row["entry_point"], solution_code, test_code,
                expect_pass=False, timeout=args.verify_timeout,
            )
            canon_ok, canon_rc, canon_tail = verify_task(
                row["entry_point"], canonical_code, test_code,
                expect_pass=True, timeout=args.verify_timeout,
            )
            if not (buggy_ok and canon_ok):
                if args.include_unverified:
                    rec = to_agentic_record(
                        row, max_steps=args.max_steps, temperature=args.temperature,
                        max_total_tokens=args.max_total_tokens,
                    )
                    meta = rec["extra_info"]["tools_kwargs"]["reward"]["metadata"]
                    meta["verified"] = False
                    meta["deadloop"] = buggy_rc == -1
                    selected.append((instance_id, rec))
                    skipped.append(
                        f"{instance_id}: include-unverified (buggy_rc={buggy_rc} "
                        f"canon_rc={canon_rc})"
                    )
                else:
                    skipped.append(
                        f"{instance_id}: verify buggy_rc={buggy_rc} canon_rc={canon_rc} "
                        f"(buggy:{buggy_tail[:100]} canon:{canon_tail[:100]})"
                    )
                continue
        selected.append((instance_id, to_agentic_record(
            row, max_steps=args.max_steps, temperature=args.temperature,
            max_total_tokens=args.max_total_tokens,
        )))

    if len(selected) < args.train_num + args.val_num:
        print(
            f"WARNING: only {len(selected)} valid tasks (need {args.train_num + args.val_num}); "
            "用 --no-verify 跳过本地验证或减小数量",
            file=sys.stderr,
        )

    for name, num in (
        (f"humanevalfix_train{args.train_num}.jsonl", args.train_num),
        ("humanevalfix_val.jsonl", args.val_num),
    ):
        path = Path(args.out_dir) / name
        chosen = selected[:num]
        with open(path, "w", encoding="utf-8") as f:
            for instance_id, rec in chosen:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"written {path} ({len(chosen)} samples): {[i for i, _ in chosen]}")
        selected = selected[num:]

    if skipped:
        print(f"skipped {len(skipped)} tasks:")
        for line in skipped[:20]:
            print("  -", line)


if __name__ == "__main__":
    main()
