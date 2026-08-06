#!/usr/bin/env bash
# 在 node2 上执行：启动 Ray head 集群
# 用法：bash /root/ray_cluster_setup.sh

set -e

echo "== 清理旧的 Ray 进程 =="
pkill -f "miniforge3/bin/ray" 2>/dev/null || true
pkill -f "raylet" 2>/dev/null || true
pkill -f "gcs_server" 2>/dev/null || true
sleep 5

echo "== 再次确认 opentelemetry 修复 =="
/root/miniforge3/bin/python -c "from opentelemetry.semconv._incubating.attributes.otel_attributes import OtelComponentTypeValues; print('otel OK')" 2>&1 | tail -1

echo "== 启动 Ray head（node2，内网 10.2.0.16）=="
/root/miniforge3/bin/ray start --head \
  --port=6379 \
  --dashboard-port=8265 \
  --metrics-export-port=0 \
  --num-cpus=16 \
  2>&1 | tail -8

echo ""
echo "== Ray 状态 =="
/root/miniforge3/bin/ray status 2>&1 | head -20
