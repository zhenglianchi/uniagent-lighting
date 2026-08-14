#!/usr/bin/env python3
"""Repro TQ+Mooncake num_turns 13B corruption: dual-node concurrent writers + reader.

Mirrors the real verl+uni-agent pattern:
- fields include num_turns as int64 tensor (8B), session_id/global_steps as int (13B bytes path)
- writers run as Ray tasks across both nodes, each doing async_kv_batch_put
- reader does kv_batch_get(select_fields=["num_turns"]) afterwards
- kv_clear simulates consumption (triggers TQ global-index release/reuse)

Set TQ_DEBUG_WRITE=1 to trace every @num_turns put (type + upsert size).
"""
import argparse
import os
import sys
import time
import uuid

import torch
import ray
from omegaconf import OmegaConf

import transfer_queue as tq

sys.path.insert(0, "/home/ubuntu/uni-agent")
from uni_agent.framework.framework import _list_of_tq_fields_to_tensordict  # noqa: E402


MOONCAKE_CONF = OmegaConf.create(
    {
        "backend": {
            "storage_backend": "MooncakeStore",
            "MooncakeStore": {
                "auto_init": False,
                "metadata_server": "10.60.188.85:50123",
                "master_server_address": "10.60.188.85:50124",
                "protocol": "tcp",
                "local_hostname": "",
                "global_segment_size": 8589934592,
                "local_buffer_size": 2147483648,
            },
        },
        "controller": {"sampler": "SequentialSampler", "polling_mode": False},
    }
)


def make_session_fields(uid: str, session_idx: int, ntraj: int = 4, base: int = 5, seq_len: int = 3000):
    """One session = ntraj trajectories, num_turns as 0-dim long tensor.

    Mirrors the real framework fields including large input_ids/position_ids
    (thousands of tokens) which force multi-slice transfers in mooncake.
    """
    fields = []
    for i in range(ntraj):
        resp_len = 1024
        input_ids = torch.tensor(list(range(4, 4 + seq_len)), dtype=torch.long)
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len, dtype=torch.long)
        rm_scores = torch.zeros(resp_len, dtype=torch.float32)
        fields.append(
            {
                "prompts": input_ids[:32],
                "responses": input_ids[:resp_len],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": torch.ones(resp_len, dtype=torch.long),
                "loss_mask": torch.ones(resp_len, dtype=torch.long),
                "rm_scores": rm_scores,
                "rollout_log_probs": torch.zeros(resp_len, dtype=torch.float32),
                "num_turns": torch.tensor(base + i, dtype=torch.long),
                "session_id": session_idx,
                "global_steps": 1,
                "min_global_steps": 1,
                "max_global_steps": 1,
                "uid": uid,
                "extra_info": {"reward_extra_info": {"acc": 0.5, "finished": True}},
                "data_source": "humanevalfix",
                "reward_model": "naive",
            }
        )
    return fields


@ray.remote(num_cpus=1)
class Writer:
    def __init__(self, tag: str):
        self.tag = tag
        tq.init()
        print(f"[Writer-{tag}] tq initialized (storage backend in use)", flush=True)

    def write_round(self, round_id: int, n_sessions: int) -> list[str]:
        keys: list[str] = []
        fields: list[dict] = []
        tags: list[dict] = []
        for s in range(n_sessions):
            uid = f"w{self.tag}_r{round_id}_s{s}"
            session_idx = s % 4
            fs = make_session_fields(uid, session_idx, ntraj=4, base=5)
            for i, f in enumerate(fs):
                keys.append(f"{uid}_{session_idx}_{i}")
                fields.append(f)
                tags.append(
                    {
                        "status": "success",
                        "global_steps": 1,
                        "min_global_steps": 1,
                        "max_global_steps": 1,
                        "prompt_len": 3,
                        "response_len": 8,
                        "seq_len": 11,
                        "uid": uid,
                    }
                )
        td = _list_of_tq_fields_to_tensordict(fields)
        tq.kv_batch_put(keys=keys, partition_id="train", fields=td, tags=tags)
        print(f"[Writer-{self.tag}] round {round_id}: put {len(keys)} keys", flush=True)
        return keys


@ray.remote(num_cpus=1)
class Reader:
    def __init__(self):
        tq.init()
        print("[Reader] tq initialized", flush=True)

    def read_round(self, keys: list[str], round_id: int) -> dict:
        bad: dict[str, str] = {}
        try:
            data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=["num_turns"])
            nt = data["num_turns"]
            print(f"[Reader] round {round_id}: read {len(keys)} keys, num_turns col ok (shape={tuple(nt.shape) if hasattr(nt, 'shape') else type(nt).__name__})", flush=True)
        except Exception as e:
            print(f"[Reader] round {round_id}: GET FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            bad["exception"] = str(e)[:500]
        return bad


@ray.remote(num_cpus=1)
class Cleaner:
    def __init__(self):
        tq.init()

    def clear_round(self, keys: list[str]):
        try:
            tq.kv_clear(keys=keys, partition_id="train")
        except Exception as e:
            print(f"[Cleaner] clear failed: {type(e).__name__}: {str(e)[:200]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writers", type=int, default=8)
    ap.add_argument("--sessions-per-writer", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--check-size", action="store_true", default=True)
    args = ap.parse_args()

    # Driver creates the controller with Mooncake config first.
    tq.init(MOONCAKE_CONF)
    print("controller initialized with MooncakeStore", flush=True)

    writers = [Writer.remote(f"w{i}") for i in range(args.writers)]
    readers = [Reader.remote()]
    cleaner = Cleaner.remote()

    total_bad = 0
    for rnd in range(args.rounds):
        all_keys = []
        write_refs = [w.write_round.remote(rnd, args.sessions_per_writer) for w in writers]
        # read the previous round's keys concurrently with the next write (pipeline)
        if rnd > 0 and prev_keys:
            chunk = 128
            for start in range(0, len(prev_keys), chunk):
                keys = prev_keys[start : start + chunk]
                res = ray.get(readers[0].read_round.remote(keys, f"{rnd}-prev"))
                if res:
                    total_bad += 1
                    print(f"[main] prev round {rnd-1} chunk@{start}: FAILED -> {res}", flush=True)
        all_keys = [k for ref in write_refs for k in ray.get(ref)]
        print(f"[main] round {rnd}: {len(all_keys)} keys written, reading...", flush=True)
        # read current round as well
        chunk = 128
        for start in range(0, len(all_keys), chunk):
            keys = all_keys[start : start + chunk]
            res = ray.get(readers[0].read_round.remote(keys, rnd))
            if res:
                total_bad += 1
                print(f"[main] round {rnd} chunk@{start}: FAILED -> {res}", flush=True)
        # clear to release global indexes (simulate consumption)
        ray.get(cleaner.clear_round.remote(all_keys))
        prev_keys = all_keys
        print(f"[main] round {rnd} done, cumulative failures={total_bad}", flush=True)

    print(f"RESULT: total_failed_chunks={total_bad}", flush=True)


if __name__ == "__main__":
    main()
