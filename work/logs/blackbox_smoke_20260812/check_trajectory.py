"""黑盒（Claude Code）小样本轨迹检查脚本。

用法：conda run -n swe-rl python work/blackbox_trajectories/check_trajectory.py
  [--dir work/blackbox_trajectories/step_1] [--session xxx]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


TOKENIZER_DIR = "/home/zhenglianchi/swe-rl-local/work/models/Qwen3-8B"


def load_session(session_dir: Path):
    traj = json.loads((session_dir / "trajectory.json").read_text())
    npz = np.load(session_dir / "trajectory.npz", allow_pickle=True)
    return traj, npz


def describe(npz, tokenizer, *, max_text: int = 300):
    prompt_ids = npz["traj0_prompt_ids"]
    response_ids = npz["traj0_response_ids"]
    response_mask = npz["traj0_response_mask"]
    logprobs = npz.get("traj0_response_logprobs")

    gen_mask = response_mask.astype(bool)
    n_gen = int(gen_mask.sum())
    n_tool = int((~gen_mask).sum())
    n_lp = int((logprobs != 0).sum()) if logprobs is not None else 0

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)

    # 响应内 mask 分段（1=模型生成，0=工具结果 continuation）
    segments = []
    cur = int(response_mask[0])
    start = 0
    for i in range(1, len(response_mask)):
        if int(response_mask[i]) != cur:
            segments.append((start, i, cur))
            start = i
            cur = int(response_mask[i])
    segments.append((start, len(response_mask), cur))

    seg_desc = []
    for s, e, m in segments[:12]:
        seg_text = tokenizer.decode(response_ids[s:e], skip_special_tokens=False)
        seg_desc.append(f"[{m}] {seg_text[:80]!r}")
    if len(segments) > 12:
        seg_desc.append(f"... ({len(segments)} segments total)")

    return {
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "model_gen_tokens": n_gen,
        "tool_continuation_tokens": n_tool,
        "logprob_nonzero": n_lp,
        "prompt_head": prompt_text[:max_text],
        "response_head": response_text[:max_text],
        "segments": seg_desc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="work/blackbox_trajectories/step_1")
    parser.add_argument("--session", default=None, help="只检查指定 session 目录名（子串匹配）")
    args = parser.parse_args()

    root = Path(args.dir)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    session_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.session:
        session_dirs = [d for d in session_dirs if args.session in d.name]

    for sd in session_dirs:
        print("=" * 100)
        print("SESSION:", sd.name)
        traj, npz = load_session(sd)
        meta = traj["trajectories"][0]
        print(
            f"turns={meta['num_turns']} reward={meta['reward_score']} "
            f"prompt_len={meta['prompt_len']} response_len={meta['response_len']} "
            f"model_token_count={meta['model_token_count']} has_logprobs={meta['has_logprobs']}"
        )
        info = describe(npz, tokenizer)
        print(
            f"tokens: prompt={info['prompt_tokens']} response={info['response_tokens']} "
            f"model_gen={info['model_gen_tokens']} tool_cont={info['tool_continuation_tokens']} "
            f"logprob_nonzero={info['logprob_nonzero']}"
        )
        print("--- prompt head ---")
        print(info["prompt_head"])
        print("--- response head ---")
        print(info["response_head"])
        print("--- response mask segments ---")
        for s in info["segments"]:
            print("  ", s)


if __name__ == "__main__":
    main()
