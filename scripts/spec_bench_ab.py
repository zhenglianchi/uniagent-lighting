#!/usr/bin/env python3
"""投机解码 A/B 纯推理微基准（阶段 0，不碰训练）。

同批 HumanEvalFix prompt、同采样参数下对比：
  - spec off（Qwen3-8B 基线）
  - spec on（Qwen3-8B + EAGLE-3 drafter）
指标：墙钟、output tok/s、TTFT p50、E2EL p50、接受率（vLLM spec_decode metrics）。

用法（训练机 node2 上，swe-rl 环境）：
  python spec_bench_ab.py --prompts /home/ubuntu/swe-rl/data/humanevalfix_train164.jsonl \
      --num-prompts 32 --n 4 --max-tokens 512 --temperature 0.8 \
      --model /home/ubuntu/models/Qwen3-8B \
      --draft /home/ubuntu/models/Qwen3-8B-speculator.eagle3 \
      --gpu-memory-utilization 0.5 --max-num-seqs 16

只测 decode 阶段的生成速度（prompt 固定一次 prefill），对比 spec on/off。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time


def load_prompts(path: str, num: int) -> list[str]:
    """从 verl agentic jsonl 取前 num 条的首轮 user prompt 文本。"""
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("prompt") or []
            user_text = next(
                (m.get("content") for m in msgs if m.get("role") == "user"),
                None,
            )
            if user_text:
                prompts.append(user_text)
            if len(prompts) >= num:
                break
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def run_bench(args: argparse.Namespace, speculative_config: dict | None) -> dict:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    prompts = load_prompts(args.prompts, args.num_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # 与训练一致的 ChatML（thinking 关闭、user 单轮、生成前缀），避免 offline API
    # 不接受消息列表的限制（vLLM 0.11 offline LLM 只吃文本/token ids）
    prompt_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        speculative_config=speculative_config,
        disable_log_stats=False,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        n=args.n,
    )
    t0 = time.time()
    outputs = llm.generate(prompt_texts, params)
    wall = time.time() - t0

    n_req = len(outputs)
    n_out_tok = sum(
        len(o.outputs[i].token_ids) for o in outputs for i in range(len(o.outputs))
    )
    ttft = []
    e2el = []
    for o in outputs:
        for i in range(len(o.outputs)):
            m = o.metrics
            if m is not None:
                # vLLM 0.11 V1: RequestStateStats —— first_token_latency 已算好 TTFT
                ftt = getattr(m, "first_token_latency", None)
                if ftt is None:  # 兜底：旧版字段
                    ft = getattr(m, "first_token_time", None) or getattr(m, "first_token_ts", None)
                    at = getattr(m, "arrival_time", None)
                    if ft is not None and at is not None:
                        ftt = ft - at
                if ftt is not None:
                    ttft.append(ftt)
                last_ts = getattr(m, "last_token_ts", None)
                first_ts = getattr(m, "first_token_ts", None)
                if last_ts is not None and first_ts is not None:
                    n = len(o.outputs[i].token_ids)
                    if n > 1:
                        e2el.append((last_ts - first_ts) / (n - 1))

    result = {
        "mode": "on" if speculative_config else "off",
        "draft": (speculative_config or {}).get("model", "-"),
        "requests": n_req,
        "responses": n_req * args.n,
        "output_tokens": n_out_tok,
        "wall_sec": round(wall, 2),
        "tok_per_sec": round(n_out_tok / wall, 2),
        "ttft_p50_ms": round(statistics.median(ttft) * 1000, 2) if ttft else None,
        "e2el_p50_ms": round(statistics.median(e2el) * 1000, 2) if e2el else None,
    }

    # vLLM spec decode 接受率（V1 metrics；若不可用则置 None，靠日志 stats 兜底）
    try:
        metrics = llm.get_metrics()
        num_drafts = num_accepted = None
        for m in metrics:
            name = getattr(m, "name", "")
            val = getattr(m, "value", None)
            if name == "vllm:spec_decode_num_drafts":
                num_drafts = val
            elif name == "vllm:spec_decode_num_accepted_tokens":
                num_accepted = val
        if num_drafts:
            result["num_drafts"] = num_drafts
            result["num_accepted"] = num_accepted
            result["mean_accept_len"] = round(1 + num_accepted / num_drafts, 3)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] spec metrics unavailable: {exc}")

    del llm
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", required=True)
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B")
    p.add_argument("--draft", default="/home/ubuntu/models/Qwen3-8B-speculator.eagle3")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--only", choices=["off", "on", "both"], default="both")
    args = p.parse_args()

    spec_on = {
        "method": "eagle3",
        "model": args.draft,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
    }
    results = []
    if args.only in ("off", "both"):
        print(f"\n===== RUN: spec OFF ({args.num_prompts} prompts x n={args.n}) =====", flush=True)
        r = run_bench(args, None)
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
        results.append(r)
    if args.only in ("on", "both"):
        print(f"\n===== RUN: spec ON ({args.num_prompts} prompts x n={args.n}) =====", flush=True)
        r = run_bench(args, spec_on)
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
        results.append(r)

    if len(results) == 2:
        off, on = results
        print("\n===== A/B 对比 =====")
        print(f"tok/s      : off {off['tok_per_sec']} -> on {on['tok_per_sec']}  "
              f"({on['tok_per_sec'] / off['tok_per_sec']:.2f}x)")
        if off.get("e2el_p50_ms") and on.get("e2el_p50_ms"):
            print(f"E2EL p50   : off {off['e2el_p50_ms']}ms -> on {on['e2el_p50_ms']}ms  "
                  f"({on['e2el_p50_ms'] / off['e2el_p50_ms']:.2f}x)")
        if on.get("mean_accept_len"):
            print(f"接受长度   : {on['mean_accept_len']} (drafts={on.get('num_drafts')})")


if __name__ == "__main__":
    main()
