# 网络访问方案

## 1. 云端 Gateway 访问

训练机 Gateway 监听内网固定端口（`GATEWAY_PORT`，默认 8001，补丁
`verl_gateway_fixed_port.patch` + `gateway_fixed_port.patch`）。访问路径：

- **白盒 harness（任意位置 / 用户侧）**：调云端 Gateway `session.base_url`
  （训练机内网直连，或经 SSH 隧道 / 安全组放行从外部接入）
- **腾讯沙箱 / 远端**：SSH 隧道（`ssh -L 127.0.0.1:8001:<内网IP>:8001`，
  走公网 22）或安全组放行 8001 后公网直连（`CLAUDE_GATEWAY_PUBLIC_HOST`）

黑盒 runner 默认隧道模式（`CLAUDE_GATEWAY_TUNNEL=1` + `MSA_GATEWAY_SSH_HOST`），
仅需 22 端口；direct-URL 模式需安全组放行 Gateway 端口。

## 2. 腾讯沙箱连接

训练机 / 本地 agent 经 E2B 兼容端点（`ap-guangzhou.tencentags.com`）出方向连接
腾讯云沙箱，无需公网入方向。

## 3. 本地平台化 agent

`platform_local_agent.py` / `platform_local_claude.py` 用 paramiko
direct-tcpip 隧道将本地端口转发到训练机内网 Gateway，agent 模型调用走
`http://127.0.0.1:<port>/sessions/<id>/v1`。
