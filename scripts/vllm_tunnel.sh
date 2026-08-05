#!/usr/bin/env bash
# 本地访问云端 vLLM 的 SSH 隧道（走 22 端口；常驻用 autossh）。
# 用法：
#   bash vllm_tunnel.sh <云端公网IP> [本地端口] [云端端口]
#   例：bash vllm_tunnel.sh 117.50.197.46 8000 8000
set -euo pipefail

REMOTE_HOST="${1:?用法: vllm_tunnel.sh <云端公网IP> [本地端口] [云端端口]}"
LOCAL_PORT="${2:-8000}"
REMOTE_PORT="${3:-8000}"
USER="${VLLM_TUNNEL_USER:-ubuntu}"

echo "== 启动 SSH 隧道：127.0.0.1:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT} =="
exec autossh -M 0 -N \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${USER}@${REMOTE_HOST}"
