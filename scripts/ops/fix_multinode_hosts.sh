#!/usr/bin/env bash
# UCloud 多机主机名/网络修复（2026-08-05 重写，取代 HAI 版）：两台都执行
# 1) /etc/hosts 写入 UCloud 内网映射（Gloo/NCCL 需要按名字解析）
# 2) 删除 127.0.1.1 回环映射（否则 Gloo connectFullMesh 失败）
# 3) 按内网 IP 设置 hostname（node1/node2）
# 注意：若恢复后的内网 IP 变化，先改本脚本里的 NODE1_IP/NODE2_IP 再执行。
set -e

# 2026-08-14 双机新内网 IP（可按环境变量覆盖）
NODE1_IP=${NODE1_IP:-10.60.188.85}
NODE2_IP=${NODE2_IP:-10.60.253.166}

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "本机 IP: $HOST_IP"

# 1. 写内网映射（幂等）
sudo sh -c "grep -qF '$NODE1_IP' /etc/hosts || echo '$NODE1_IP node1' >> /etc/hosts"
sudo sh -c "grep -qF '$NODE2_IP' /etc/hosts || echo '$NODE2_IP node2' >> /etc/hosts"

# 2. 删除 127.0.1.1 回环映射（Ubuntu 默认会写，必须清）
sudo sed -i '/^127\.0\.1\.1[[:space:]]/d' /etc/hosts

echo "== /etc/hosts =="
grep -E "node1|node2" /etc/hosts

# 3. hostname
if [ "$HOST_IP" = "$NODE1_IP" ]; then
  sudo hostnamectl set-hostname node1
else
  sudo hostnamectl set-hostname node2
fi
hostname
