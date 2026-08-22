#!/usr/bin/env bash
# WSL <-> Windows 宿主代理开关（Clash Verge :7890）
#
# 前提：Clash Verge 已开启"允许局域网连接"，
#       否则 Windows 只监听 127.0.0.1，WSL2(NAT) 无法访问。
# 用法：
#   source scripts/ops/proxy.sh      # 只定义函数，不自动开启
#   source scripts/ops/proxy.sh on   # 定义并立即开启
#   proxy_on                     # 开启（自动解析宿主 IP）
#   proxy_docker                 # 给 docker daemon 配代理并重启（需 sudo）
#   proxy_off                    # 关闭
#   proxy_test                   # 验证代理连通（curl google）

PROXY_PORT=7890
export PROXY_PORT

# 自动解析 Windows 宿主 IP：NAT 模式下默认网关即宿主机
_proxy_host() {
  local gw
  gw=$(ip route show default 2>/dev/null | awk '/default/{print $3; exit}')
  if [ -z "$gw" ]; then
    gw=$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf 2>/dev/null)
  fi
  echo "$gw"
}

proxy_on() {
  local host
  host=$(_proxy_host)
  if [ -z "$host" ]; then
    echo "错误：无法解析 Windows 宿主 IP" >&2
    return 1
  fi
  export PROXY_HOST="$host"
  export HTTP_PROXY="http://$host:$PROXY_PORT"
  export HTTPS_PROXY="http://$host:$PROXY_PORT"
  export ALL_PROXY="http://$host:$PROXY_PORT"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
  export all_proxy="$ALL_PROXY"
  # 本地网段与国内直连域名（阿里云百炼等）不走代理
  export NO_PROXY="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.aliyuncs.com,*.maas.aliyuncs.com"
  export no_proxy="$NO_PROXY"
  echo "代理已开启：$HTTP_PROXY"
}

proxy_off() {
  unset PROXY_HOST HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy NO_PROXY no_proxy
  echo "代理已关闭"
}

proxy_docker() {
  local host conf
  host=$(_proxy_host)
  if [ -z "$host" ]; then
    echo "错误：无法解析 Windows 宿主 IP" >&2
    return 1
  fi
  conf=/etc/systemd/system/docker.service.d/http-proxy.conf
  sudo mkdir -p "$(dirname "$conf")"
  printf '[Service]\nEnvironment="HTTP_PROXY=http://%s:7890"\nEnvironment="HTTPS_PROXY=http://%s:7890"\nEnvironment="NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.aliyuncs.com"\n' "$host" "$host" \
    | sudo tee "$conf" >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl restart docker
  echo "docker daemon 代理已配置并重启：http://$host:7890"
}

proxy_test() {
  local host
  host=$(_proxy_host)
  curl -sI --max-time 10 -x "http://$host:$PROXY_PORT" https://www.google.com | head -1
}

if [ "$1" = "on" ]; then
  proxy_on
fi
