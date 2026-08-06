#!/usr/bin/env bash
# 重启双机 Ray 集群（node2 head + node1 join）
set -e

echo "== 停止 node1 Ray =="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no root@10.2.0.7 \
  "pkill -f miniforge3/bin/ray || true; pkill -f raylet || true; pkill -f gcs_server || true; sleep 3; echo node1-stopped"

echo "== 停止 node2 Ray =="
pkill -f miniforge3/bin/ray 2>/dev/null || true
pkill -f raylet 2>/dev/null || true
pkill -f gcs_server 2>/dev/null || true
sleep 4

echo "== node2 启动 head =="
/root/miniforge3/bin/ray start --head \
  --port=6379 --dashboard-port=8265 --metrics-export-port=0 --num-cpus=16 \
  2>&1 | tail -3

echo "== node1 加入 =="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no root@10.2.0.7 \
  "bash /root/ray_node_join.sh" 2>&1 | tail -3

echo "== 集群状态 =="
sleep 6
/root/miniforge3/bin/ray status 2>&1 | sed -n '1,20p'
