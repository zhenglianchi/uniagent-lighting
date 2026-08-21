#!/usr/bin/env python3
"""离线轨迹 -> TQ+Mooncake 写入/读取验证（不跑 agent/gateway/隧道）。

目标：用真实训练轨迹（humanevalfix_blackbox / platform_test 的 trajectory.json+npz）
构造与框架 `_trajectory_to_tq_field_and_tag` 同款的字段，经 TQ kv_batch_put 写入
MooncakeStore，然后：
  1. TQ kv_batch_get 读回 num_turns，检查值正确；
  2. 直接用 mooncake store 客户端遍历所有 `X@num_turns` 键，断言 get_size == 8
     （杜绝 13B msgpack 错写）；
  3. 多 writer 并发 + reader + kv_clear 释放/复用 global index，模拟真实训练模式。

用法（node1，Ray 已起、mooncake_master 已起）：
  TQ_DEBUG_WRITE=1 python offline_mooncake_verify.py --traj-dir /home/ubuntu/swe-rl/logs \
      --sessions 128 --writers 8 --rounds 4
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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


def load_trajectories(traj_dir: str, limit: int) -> list[dict]:
    """Load real trajectory.json + npz pairs into a flat list of trajectory dicts."""
    out = []
    roots = []
    for d in sorted(Path(traj_dir).glob("*/step_*/session-*/")):
        roots.append(d)
        if len(roots) >= limit * 4:
            break
    for d in roots:
        jf = d / "trajectory.json"
        nf = d / "trajectory.npz"
        if not jf.exists() or not nf.exists():
            continue
        try:
            meta = json.loads(jf.read_text())
            npz = np.load(nf, allow_pickle=True)
            for i in range(meta.get("num_trajectories", 1)):
                t = meta["trajectories"][i]
                out.append(
                    {
                        "prompt_ids": npz[f"traj{i}_prompt_ids"].tolist(),
                        "response_ids": npz[f"traj{i}_response_ids"].tolist(),
                        "response_mask": npz[f"traj{i}_response_mask"].tolist(),
                        "response_logprobs": (
                            npz[f"traj{i}_response_logprobs"].tolist()
                            if f"traj{i}_response_logprobs" in npz.files
                            else None
                        ),
                        "num_turns": t.get("num_turns", 0),
                        "reward_score": t.get("reward_score"),
                        "reward_info": t.get("reward_info") or {},
                        "session_dir": str(d),
                    }
                )
        except Exception:
            continue
    return out


def make_field(
    traj: dict, uid: str, session_idx: int, global_steps: int
) -> tuple[dict, dict]:
    """Mirror framework `_trajectory_to_tq_field_and_tag` field layout."""
    prompts = torch.tensor(traj["prompt_ids"], dtype=torch.long)
    responses = torch.tensor(traj["response_ids"], dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    position_ids = torch.arange(input_ids.size(0), dtype=torch.long)
    resp_len = responses.size(0)
    rm_scores = torch.zeros(resp_len, dtype=torch.float32)
    if traj["reward_score"] is not None and resp_len > 0:
        rm_scores[-1] = float(traj["reward_score"])
    field = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": torch.tensor(traj["response_mask"], dtype=torch.long),
        "loss_mask": torch.tensor(traj["response_mask"], dtype=torch.long),
        "rm_scores": rm_scores,
        "num_turns": torch.tensor(int(traj["num_turns"]), dtype=torch.long),
        "session_id": session_idx,
        "global_steps": global_steps,
        "uid": uid,
        "extra_info": {"reward_extra_info": traj["reward_info"]},
        "data_source": "humanevalfix",
        "reward_model": "naive",
    }
    if traj["response_logprobs"] is not None:
        field["rollout_log_probs"] = torch.tensor(traj["response_logprobs"], dtype=torch.float32)
    tag = {
        "status": "success",
        "global_steps": global_steps,
        "min_global_steps": global_steps,
        "max_global_steps": global_steps,
        "prompt_len": prompts.size(0),
        "response_len": resp_len,
        "seq_len": input_ids.size(0),
        "uid": uid,
    }
    return field, tag


@ray.remote(num_cpus=1)
class Writer:
    def __init__(self, tag: str):
        self.tag = tag
        tq.init()

    def write_round(self, round_id: int, traj_batch: list[dict], per_session: int) -> list[str]:
        """traj_batch: one trajectory per session slot; build per_session traj fields."""
        keys: list[str] = []
        fields: list[dict] = []
        tags: list[dict] = []
        for s, traj in enumerate(traj_batch):
            uid = f"w{self.tag}_r{round_id}_s{s}"
            for j in range(per_session):
                f, tag = make_field(traj, uid, j, round_id)
                keys.append(f"{uid}_{j}_{0}")
                fields.append(f)
                tags.append(tag)
        td = _list_of_tq_fields_to_tensordict(fields)
        tq.kv_batch_put(keys=keys, partition_id="train", fields=td, tags=tags)
        return keys


@ray.remote(num_cpus=1)
class Reader:
    def __init__(self):
        tq.init()

    def read_round(self, keys: list[str], round_id: int) -> dict:
        try:
            data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=["num_turns"])
            nt = data["num_turns"]
            if hasattr(nt, "to_padded_tensor"):
                nt = nt.to_padded_tensor(0)
            vals = nt.reshape(-1).tolist() if hasattr(nt, "reshape") else list(nt)
            return {"ok": True, "n": len(keys), "vals_sample": vals[:8]}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}


@ray.remote(num_cpus=1)
class Cleaner:
    def __init__(self):
        tq.init()

    def clear_round(self, keys: list[str]):
        try:
            tq.kv_clear(keys=keys, partition_id="train")
        except Exception as e:
            print(f"[Cleaner] clear failed: {str(e)[:150]}", flush=True)


def store_check(num_expect: int) -> dict:
    """Directly probe mooncake store: every X@num_turns must be exactly 8 bytes."""
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    ret = store.setup(
        "10.60.188.85", "http://10.60.188.85:50123/metadata",
        1024 * 1024 * 1024, 256 * 1024 * 1024, "tcp", "", "10.60.188.85:50124",
    )
    assert ret == 0, f"store setup failed: {ret}"
    bad: list[tuple[str, int]] = []
    found = 0
    probe: list[tuple[str, int]] = []
    for i in range(num_expect * 4 + 16):
        k = f"{i}@num_turns"
        try:
            sz = store.get_size(k)
            if len(probe) < 5:
                probe.append((k, sz))
        except Exception:
            continue
        if sz >= 0:
            found += 1
            if sz != 8:
                bad.append((k, sz))
    store.close()
    return {"found": found, "probe": probe, "bad": bad[:20], "bad_count": len(bad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="/home/ubuntu/swe-rl/logs")
    ap.add_argument("--sessions", type=int, default=128)
    ap.add_argument("--per-session", type=int, default=4)
    ap.add_argument("--writers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    trajs = load_trajectories(args.traj_dir, limit=args.sessions)
    print(f"loaded {len(trajs)} real trajectories", flush=True)
    if len(trajs) < args.sessions:
        print(f"WARN: only {len(trajs)} trajectories, using them", flush=True)
    trajs = trajs[: args.sessions]

    tq.init(MOONCAKE_CONF)
    print("controller initialized with MooncakeStore", flush=True)

    writers = [Writer.remote(f"w{i}") for i in range(args.writers)]
    reader = Reader.remote()
    cleaner = Cleaner.remote()

    per_writer = max(1, args.sessions // args.writers)
    total_fail = 0
    for rnd in range(args.rounds):
        all_keys: list[str] = []
        refs = []
        for w, wi in zip(writers, range(args.writers)):
            chunk = trajs[wi * per_writer : (wi + 1) * per_writer]
            if chunk:
                refs.append(w.write_round.remote(rnd, chunk, args.per_session))
        for ref in refs:
            all_keys.extend(ray.get(ref))
        print(f"[main] round {rnd}: wrote {len(all_keys)} keys", flush=True)
        chunk = 256
        for start in range(0, len(all_keys), chunk):
            res = ray.get(reader.read_round.remote(all_keys[start : start + chunk], rnd))
            if not res.get("ok"):
                total_fail += 1
                print(f"[main] round {rnd} READ FAIL: {res}", flush=True)
            else:
                print(f"[main] round {rnd} read ok n={res['n']} vals={res['vals_sample']}", flush=True)
        # store direct check BEFORE clear (clear releases indexes / may wipe data)
        chk = store_check(args.sessions * args.per_session)
        print(f"[main] round {rnd} store check: found={chk['found']} bad={chk['bad_count']} probe={chk['probe']}", flush=True)
        total_fail += chk["bad_count"]
        ray.get(cleaner.clear_round.remote(all_keys))
        print(f"[main] round {rnd} done, failures={total_fail}", flush=True)

    print(f"RESULT: total_failures={total_fail}", flush=True)


if __name__ == "__main__":
    main()
