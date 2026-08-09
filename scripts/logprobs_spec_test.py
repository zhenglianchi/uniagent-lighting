#!/usr/bin/env python3
"""vLLM 0.11.1 EAGLE-3 投机解码下 logprobs 返回行为验证。

对比 SamplingParams.logprobs = 0 / 1 / 3（EAGLE-3 on）时 output.logprobs 是否非空，
定位 Gateway "backend logprobs must align with token_ids: got 0 logprobs" 根因。
"""

from __future__ import annotations

import json
from vllm import LLM, SamplingParams

MODEL = "/home/ubuntu/models/Qwen3-8B"
DRAFT = "/home/ubuntu/models/Qwen3-8B-speculator.eagle3"
PROMPT = "Fix the bug in /testbed/solution.py.\n\nProblem:\nbrackets is a string of \"(\" and \")\".\nreturn True if every opening bracket has a corresponding closing bracket."


def main() -> None:
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_num_seqs=4,
        max_model_len=4096,
        enforce_eager=True,
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
        },
        disable_log_stats=False,
    )
    for lp in (0, 1, 3):
        params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32, n=1, logprobs=lp)
        out = llm.generate([PROMPT], params)[0]
        o = out.outputs[0]
        n_log = len(o.logprobs) if o.logprobs is not None else 0
        first = None
        if n_log:
            first = len(o.logprobs[0]) if isinstance(o.logprobs[0], dict) else "?"
        print(json.dumps({
            "logprobs_param": lp,
            "num_tokens": len(o.token_ids),
            "num_logprobs_returned": n_log,
            "topk_in_first": first,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
