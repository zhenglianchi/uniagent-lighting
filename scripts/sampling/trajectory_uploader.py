#!/usr/bin/env python3
"""本地采样轨迹批量上传器（trajectory uploader，UCloud SFTP 直传版）。

链路：sampler -> JSONL 缓冲 -> zstd 压缩 -> SFTP 直传 UCloud 训练机
      （默认目标：<UCloud 用户主目录>/swe-rl/trajectories/）

能力：
  - 扫描采样产物目录（*.traj.json），按批次合并为 JSONL 并用 zstd 压缩；
  - 断点续传：状态文件记录已成功上传的轨迹（按文件路径+mtime 指纹），
    失败批次不会标记，下次重跑自动补齐；
  - SFTP 直传：凭据读取 work/ucloud.env（与 scripts/ops/ssh_ucloud.py 同源）；
    --dry-run 时只打包不上传。

用法：
  # 只打包，预览将上传哪些批次（推荐先跑这个）
  conda run -n swe-rl python scripts/sampling/trajectory_uploader.py --dry-run

  # 打包并上传所有未上传轨迹到 UCloud node1
  conda run -n swe-rl python scripts/sampling/trajectory_uploader.py

  # 指定轨迹目录 / 批大小 / 目标机器
  conda run -n swe-rl python scripts/sampling/trajectory_uploader.py \
      --input-dir work/swebench --batch-size 8 --node 1
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import zstandard


def load_ucloud_env(path: str, node: int) -> dict:
    """从 work/ucloud.env 读取指定机器的 SSH/SFTP 凭据（UCLOUD<N>_HOST/USER/PASS/PORT）。"""
    env = load_env_file(path)
    prefix = f"UCLOUD{node}"
    creds = {
        "host": env.get(f"{prefix}_HOST"),
        "user": env.get(f"{prefix}_USER"),
        "password": env.get(f"{prefix}_PASS"),
        "port": int(env.get(f"{prefix}_PORT", "22")),
    }
    missing = [k for k, v in creds.items() if not v]
    if missing:
        sys.exit(f"凭据不完整（node {node}，缺 {missing}）：请检查 {path}")
    return creds


def load_env_file(path: str) -> dict:
    """读取 KEY=VALUE 格式的配置文件（跳过注释和空行）。"""
    env = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def parse_args():
    p = argparse.ArgumentParser(description="轨迹批量上传器")
    p.add_argument("--input-dir", default="work/swebench",
                   help="采样产物目录，扫描 *.traj.json（默认 work/swebench）")
    p.add_argument("--queue-dir", default="work/upload_queue",
                   help="打包输出目录（默认 work/upload_queue）")
    p.add_argument("--state-file", default="work/uploader_state.json",
                   help="断点续传状态文件（默认 work/uploader_state.json）")
    p.add_argument("--batch-size", type=int, default=8,
                   help="每个批次合并的轨迹条数（默认 8）")
    p.add_argument("--remote-prefix", default="swe-rl/trajectories",
                   help="UCloud 远程目录（相对主目录，默认 swe-rl/trajectories）")
    p.add_argument("--env-file", default="work/ucloud.env",
                   help="UCloud 凭据文件（默认 work/ucloud.env）")
    p.add_argument("--node", type=int, default=1,
                   help="目标机器编号（默认 1，对应 ucloud.env 的 UCLOUD1_*）")
    p.add_argument("--dry-run", action="store_true",
                   help="只打包不上传，并打印计划")
    return p.parse_args()


def load_state(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"uploaded": {}, "batches": []}


def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def fingerprint(path: str) -> str:
    """轨迹文件指纹：路径 + 大小 + mtime，用于断点续传去重。"""
    st = os.stat(path)
    return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"


def extract_instance_id(path: str) -> str:
    """从轨迹文件名提取 instance_id。

    SWE-bench instance_id 形如 <org>__<repo>-<PR>（如 sympy__sympy-13043），
    文件名可能是 <instance_id>.traj.json 或 <instance_id>-step40.traj.json。
    用正则匹配 org__repo-PR 前缀，避免把 instance_id 内部的 '-' 切错。
    """
    base = os.path.basename(path)[: -len(".traj.json")]
    m = re.match(r"^(.+?__[^-]+-\d+)", base)
    return m.group(1) if m else base


def scan_trajectories(input_dir: str) -> list[str]:
    if not os.path.isdir(input_dir):
        sys.exit(f"轨迹目录不存在：{input_dir}")
    files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".traj.json")
    )
    return files


def pack_batch(traj_files: list[str], out_path: str):
    """把一批轨迹合并为 JSONL，再用 zstd 压缩成 .jsonl.zst。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        cctx = zstandard.ZstdCompressor(level=3)
        with cctx.stream_writer(f) as writer:
            for path in traj_files:
                with open(path, encoding="utf-8") as tf:
                    traj = json.load(tf)
                writer.write(json.dumps(traj, ensure_ascii=False).encode("utf-8"))
                writer.write(b"\n")
    os.replace(tmp, out_path)


def write_manifest(traj_files: list[str], manifest_path: str, batch_id: str,
                   remote_key: str, zst_path: str) -> dict:
    """写批次 manifest（含 instance_id 列表，供云端对齐任务）。"""
    manifest = {
        "batch_id": batch_id,
        "remote_key": remote_key,
        "zst_file": os.path.basename(zst_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trajectories": [
            {"file": os.path.basename(f), "instance_id": extract_instance_id(f)}
            for f in traj_files
        ],
        "instance_ids": [extract_instance_id(f) for f in traj_files],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def upload_to_ucloud(local_path: str, remote_path: str, creds: dict) -> None:
    """通过 paramiko SFTP 上传单个文件到 UCloud 机器。"""
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=creds["host"], port=creds["port"],
        username=creds["user"], password=creds["password"], timeout=60,
    )
    try:
        sftp = client.open_sftp()
        try:
            _mkdir_p(sftp, os.path.dirname(remote_path))
            sftp.put(local_path, remote_path)
        finally:
            sftp.close()
    finally:
        client.close()


def _mkdir_p(sftp, remote_dir: str) -> None:
    """递归创建远程目录（SFTP 无 -p，需逐级 mkdir，已存在则忽略）。"""
    parts = remote_dir.split("/")
    cur = ""
    for part in parts:
        if not part:
            continue
        cur += "/" + part
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            try:
                sftp.mkdir(cur)
            except OSError:
                pass  # 并发/已存在等竞态，忽略


def main():
    args = parse_args()
    state = load_state(args.state_file)
    uploaded = state.setdefault("uploaded", {})
    batches = state.setdefault("batches", [])

    traj_files = scan_trajectories(args.input_dir)
    pending = [f for f in traj_files if fingerprint(f) not in uploaded]
    print(f"扫描到 {len(traj_files)} 条轨迹，未上传 {len(pending)} 条")
    if not pending:
        print("没有新轨迹，退出。")
        return

    # 检查上传凭据：读 UCloud 机器凭据（work/ucloud.env），缺失时只打包
    try:
        creds = load_ucloud_env(args.env_file, args.node)
        can_upload = not args.dry_run
    except SystemExit as e:
        creds = None
        can_upload = False
        print(f"警告：{e}，本次只打包不上传。", file=sys.stderr)

    batches_meta = []
    for i in range(0, len(pending), args.batch_size):
        chunk = pending[i : i + args.batch_size]
        batch_id = time.strftime("%Y%m%d-%H%M%S") + f"-{i // args.batch_size:03d}"
        zst_path = os.path.join(args.queue_dir, f"{batch_id}.jsonl.zst")
        remote_prefix = f"/home/{creds['user']}/{args.remote_prefix.rstrip('/')}" if creds else args.remote_prefix
        remote_key = f"{args.remote_prefix.rstrip('/')}/{batch_id}.jsonl.zst"
        manifest_path = os.path.join(args.queue_dir, f"{batch_id}.manifest.json")

        ids = [extract_instance_id(f) for f in chunk]
        print(f"[批次 {batch_id}] {len(chunk)} 条轨迹: {ids}")
        if not args.dry_run or not os.path.isfile(zst_path):
            pack_batch(chunk, zst_path)
            raw = sum(os.path.getsize(f) for f in chunk)
            comp = os.path.getsize(zst_path)
            print(f"  打包完成：{raw} bytes -> {comp} bytes"
                  f"（压缩率 {comp / raw * 100:.1f}%）")
        manifest = write_manifest(chunk, manifest_path, batch_id, remote_key, zst_path)

        if can_upload:
            upload_to_ucloud(zst_path, f"{remote_prefix}/{batch_id}.jsonl.zst", creds)
            upload_to_ucloud(manifest_path, f"{remote_prefix}/{batch_id}.manifest.json", creds)
            for f in chunk:
                uploaded[fingerprint(f)] = remote_key
            batches.append({"batch_id": batch_id, "remote_key": remote_key,
                            "instance_ids": ids})
            print(f"  已上传：{remote_prefix}/{batch_id}.jsonl.zst")
        elif args.dry_run:
            print(f"  [dry-run] 将上传：{remote_prefix}/{batch_id}.jsonl.zst")
        else:
            print(f"  [未上传] 打包完成但无凭据：{remote_prefix}/{batch_id}.jsonl.zst")

    save_state(args.state_file, state)
    print(f"\n状态已保存：{args.state_file}")
    if can_upload:
        print(f"本次上传 {len(pending)} 条轨迹，共 {len(batches)} 个批次")
    elif args.dry_run:
        print("dry-run 结束：未上传任何文件")
    else:
        print("提醒：配置 work/ucloud.env 的 UCLOUD<N>_* 后重跑本脚本即可断点续传补齐上传。")


if __name__ == "__main__":
    main()
