#!/usr/bin/env python3
"""从本地缓存的 SWE-bench Lite 数据集抽取子集，生成 VeRL 训练用的
train.jsonl / val.jsonl（冒烟数据）。

数据来源：HF 缓存的 princeton-nlp/SWE-Bench_Lite（300 条），离线读取。
输出格式（每条一个 JSON 对象）：
  instance_id, prompt, repo, base_commit, patch, test_patch,
  FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit

用法示例：
  # 默认：train 40 条 + val 10 条，随机种子 42
  conda run -n swe-rl python scripts/make_smoke_data.py

  # 正式训练：用 Lite 全量或更大子集
  conda run -n swe-rl python scripts/make_smoke_data.py \
      --train-num 200 --val-num 50 --seed 42

  # 只抽 sympy / astropy 仓库、且测试量 >= 2 的样本
  conda run -n swe-rl python scripts/make_smoke_data.py \
      --repos sympy astropy --min-tests 2
"""

import argparse
import glob
import json
import os
import random
import sys


DEFAULT_CACHE = os.path.expanduser(
    "~/.cache/huggingface/datasets/princeton-nlp___swe-bench_lite"
)


def find_cache(cache_root: str) -> str:
    """定位本地 HF 缓存中的 SWE-bench Lite 数据目录。"""
    if not os.path.isdir(cache_root):
        sys.exit(f"未找到 SWE-bench Lite 缓存：{cache_root}\n"
                 "请先离线加载一次数据集：\n"
                 "  HF_HUB_OFFLINE=1 conda run -n swe-rl python -c \"from datasets import load_dataset; "
                 "load_dataset('princeton-nlp/SWE-Bench_Lite')\"")
    # 缓存结构：default/0.0.0/<hash>/swe-bench_lite-{dev,test}.arrow
    matches = glob.glob(os.path.join(cache_root, "default", "0.0.0", "*", "swe-bench_lite-*.arrow"))
    if not matches:
        sys.exit(f"缓存目录存在但未找到 arrow 文件：{cache_root}")
    return cache_root


def load_lite(cache_root: str, split: str) -> list[dict]:
    """读取指定 split（dev/test）的全部记录。"""
    hash_dirs = [
        d for d in os.listdir(os.path.join(cache_root, "default", "0.0.0"))
        if d.endswith(".lock") is False
        and os.path.isfile(
            os.path.join(cache_root, "default", "0.0.0", d, f"swe-bench_lite-{split}.arrow")
        )
    ]
    if not hash_dirs:
        sys.exit(f"找不到 {split} 分片，请检查缓存目录结构：{cache_root}")
    path = os.path.join(
        cache_root, "default", "0.0.0",
        hash_dirs[0],
        f"swe-bench_lite-{split}.arrow",
    )
    if not os.path.isfile(path):
        sys.exit(f"找不到 {split} 分片：{path}")

    import pyarrow.ipc as ipc

    with ipc.open_stream(path) as reader:
        table = reader.read_all()
    records = []
    for i in range(table.num_rows):
        rec = {c: table.column(c)[i].as_py() for c in table.column_names}
        records.append(rec)
    return records


def build_prompt(rec: dict) -> str:
    """构造发给 agent 的任务 prompt（与 mini-swe-agent 的 PR 描述一致）。"""
    return rec["problem_statement"].strip()


def to_verl_record(rec: dict) -> dict:
    """把 SWE-bench 原始字段转换为 verl agentic RL 训练数据行。"""
    return {
        "instance_id": rec["instance_id"],
        "data_source": "swe-bench-lite",
        "prompt": [{"role": "user", "content": build_prompt(rec)}],
        "ability": "swe-bench",
        "reward_model": {"style": "rule", "ground_truth": rec["patch"]},
        "repo": rec["repo"],
        "base_commit": rec["base_commit"],
        "patch": rec["patch"],
        "test_patch": rec["test_patch"],
        "FAIL_TO_PASS": rec.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": rec.get("PASS_TO_PASS", []),
        "environment_setup_commit": rec.get("environment_setup_commit", ""),
    }


def parse_args():
    p = argparse.ArgumentParser(description="生成 SWE-bench Lite 冒烟/训练数据")
    p.add_argument("--split", default="test", choices=["dev", "test"],
                   help="从哪个官方 split 抽样本（默认 test，与采样一致）")
    p.add_argument("--train-num", type=int, default=40,
                   help="train.jsonl 样本数（默认 40）")
    p.add_argument("--val-num", type=int, default=10,
                   help="val.jsonl 样本数（默认 10）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--repos", nargs="*", default=None,
                   help="只保留指定 repo（如 sympy astropy；默认不过滤）")
    p.add_argument("--min-tests", type=int, default=1,
                   help="只保留 FAIL_TO_PASS 数量 >= 该值的样本（默认 1）")
    p.add_argument("--output-dir", default="work/data",
                   help="输出目录（默认 work/data）")
    return p.parse_args()


def main():
    args = parse_args()
    cache_root = find_cache(DEFAULT_CACHE)
    records = load_lite(cache_root, args.split)
    print(f"读取 {args.split} split：{len(records)} 条")

    # 过滤
    if args.repos:
        before = len(records)
        records = [r for r in records if r["repo"] in args.repos]
        print(f"按 repo={args.repos} 过滤：{before} -> {len(records)}")
    before = len(records)
    records = [r for r in records if len(r.get("FAIL_TO_PASS", [])) >= args.min_tests]
    print(f"按 FAIL_TO_PASS>={args.min_tests} 过滤：{before} -> {len(records)}")

    total = args.train_num + args.val_num
    if total > len(records):
        sys.exit(f"需要 {total} 条，过滤后只剩 {len(records)} 条，请调小数量或放宽过滤条件")

    rng = random.Random(args.seed)
    sampled = rng.sample(records, total)
    train = sampled[: args.train_num]
    val = sampled[args.train_num:]

    os.makedirs(args.output_dir, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        out_path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(to_verl_record(rec), ensure_ascii=False) + "\n")
        print(f"写出 {out_path}：{len(rows)} 条")

    ids = [r["instance_id"] for r in train + val]
    print(f"样本 instance_id 列表：\n{json.dumps(ids, indent=0, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
