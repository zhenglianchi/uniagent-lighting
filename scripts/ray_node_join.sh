#!/usr/bin/env bash
# 在 node1 上执行：加入 node2 的 Ray 集群
# 用法：ssh root@10.2.0.7 'bash /root/ray_node_join.sh'

set -e

echo "== 清理 node1 上旧的 Ray 进程 =="
pkill -f "miniforge3/bin/ray" 2>/dev/null || true
pkill -f "raylet" 2>/dev/null || true
pkill -f "gcs_server" 2>/dev/null || true
sleep 5

echo "== 加入 Ray 集群（head=10.2.0.16:6379）=="
/root/miniforge3/bin/ray start --address=10.2.0.16:6379 \
  --num-cpus=16 \
  2>&1 | tail -8
