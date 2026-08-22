#!/usr/bin/env bash
# cc-connect 启停脚本（飞书远程控制 Codex，2026-08-06）
# 基于 systemd 用户服务（开机自启 + 崩溃自动重启）
# 用法: scripts/ops/cc_connect.sh {start|stop|status|log [N]}
set -euo pipefail

case "${1:-}" in
  start)
    systemctl --user start cc-connect
    sleep 2
    systemctl --user status cc-connect --no-pager | head -8
    ;;
  stop)
    systemctl --user stop cc-connect
    echo "已停止"
    ;;
  status)
    if systemctl --user is-active cc-connect >/dev/null 2>&1; then
      echo "运行中"
      systemctl --user status cc-connect --no-pager | head -6
    else
      echo "未运行"
    fi
    ;;
  log)
    journalctl --user -u cc-connect -n "${2:-30}" --no-pager
    ;;
  *)
    echo "用法: $0 {start|stop|status|log [N]}"
    exit 1
    ;;
esac
