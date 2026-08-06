#!/usr/bin/env python3
"""从 SWE-bench Lite 生成 **agentic 训练数据**（verl/uni-agent agent loop 输入）。

与 make_smoke_data.py（纯文本 prompt）不同，本脚本为每条样本补齐：
  - ``raw_prompt``：issue 文本消息
  - ``tools_kwargs``：{task: uni-agent Task Config, env: {image: sweb 实例镜像},
    reward: {metadata: problem_statement / FAIL_TO_PASS / test_patch / instance_id}}
  - ``reward_model.ground_truth``：给 reward 函数用的真值

输出 jsonl（每条一个 JSON 对象），schema 对齐 verl AgentDataset
（``extra_info.tools_kwargs``，见 verl/utils/dataset/rl_dataset.py），
上机前需用真实训练配置实测对齐。

用法：
  conda run -n swe-rl python scripts/make_agentic_data.py          # 默认 2 条 train + 1 条 val
  conda run -n swe-rl python scripts/make_agentic_data.py --train-num 40 --val-num 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_smoke_data import build_prompt, find_cache, load_lite  # noqa: E402


def sweb_image_for(instance_id: str) -> str:
    """sweb.eval 实例镜像名：sweb.eval.x86_64.<org>_1776_<repo>-<pr>:latest。"""
    org, _, rest = instance_id.partition("__")
    return f"sweb.eval.x86_64.{org}_1776_{rest}:latest"


def task_config_for(instance_id: str) -> dict:
    """uni-agent Task Config（tools_kwargs.task）。"""
    return {
        "name": "swe_bench",
        "sandbox": {"provider": "tencent_agent_runtime", "image": sweb_image_for(instance_id)},
        "agent": {"name": "mini_swe_agent", "max_steps": 60},
        "model": {"temperature": 1.0, "max_total_tokens": 65536},
    }


def to_agentic_record(rec: dict) -> dict:
    """把 SWE-bench Lite 记录转成 verl agentic jsonl 行。"""
    instance_id = rec["instance_id"]
    problem_statement = rec.get("problem_statement") or _first_user_text(build_prompt(rec))
    metadata = {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "FAIL_TO_PASS": list(rec.get("FAIL_TO_PASS", [])),
        "PASS_TO_PASS": list(rec.get("PASS_TO_PASS", [])),
        "test_patch": rec.get("test_patch", ""),
        "environment_setup_commit": rec.get("environment_setup_commit", ""),
    }
    return {
        "data_source": "swe-bench-lite",
        "prompt": [{"role": "user", "content": build_prompt(rec)}],
        "extra_info": {
            "tools_kwargs": {
                "task": task_config_for(instance_id),
                "env": {"image": sweb_image_for(instance_id), "instance_id": instance_id},
                "reward": {"metadata": metadata},
            }
        },
        "reward_model": {"ground_truth": metadata},
    }


def _first_user_text(prompt: str) -> str:
    return prompt.strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-num", type=int, default=2)
    p.add_argument("--val-num", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="work/data")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    args = p.parse_args()

    cache = find_cache(os.path.expanduser("~/.cache/huggingface/datasets/princeton-nlp___swe-bench_lite"))
    records = load_lite(cache, args.split)
    rng = random.Random(args.seed)
    rng.shuffle(records)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, num in (("agentic_train.jsonl", args.train_num), ("agentic_val.jsonl", args.val_num)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records[:num]:
                f.write(json.dumps(to_agentic_record(rec), ensure_ascii=False) + "\n")
        print(f"written {path} ({num} samples)")


if __name__ == "__main__":
    main()
