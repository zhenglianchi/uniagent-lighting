# 本地/沙箱 agent 如何访问云端 vLLM（2026-08-11 更新）

> **当前路径（2026-08-05 起定稿，不再需要隧道）**：
> 训练机 Gateway 监听 `0.0.0.0:<port>`（非默认端口如 38197/8001）→ UCloud 安全组放行
> 该端口 + 腾讯沙箱出口可达 → 白盒 harness（训练机本地）直接调
> `session.base_url`（本机 Gateway）；黑盒 Claude Code（腾讯沙箱内）用
> `ANTHROPIC_BASE_URL` **直连公网 Gateway**（direct-URL，v0.35.1）。安全提示：
> 训练结束即关安全组端口，勿长期暴露。
>
> 下方 SSH 隧道方案为**历史记录**（2026-08-05 前公网仅开 22 端口时使用；
> v0.14.0 起 agent-harness-outside 模式不再需要；仅作排障参考）。

## 问题

本地（WSL）mini-swe-agent / runner 需要把模型端点指向云端训练机的 vLLM
（OpenAI 兼容端点），但实测云端公网**只开放 22 端口**（2026-08-05 对
`117.50.197.46` 探测：22 通，80/443/8000/38197/8080 均 filtered）。

## 历史方案：SSH 隧道（-L 端口转发，走 22）

```bash
# 云端 vLLM 监听 127.0.0.1:8000（或 Gateway 端口）时：
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 ubuntu@<云端公网IP>
# 本地 agent 配置：
#   api_base = http://127.0.0.1:8000/v1
```

- 不开放新端口、加密传输，安全
- 常驻用 `autossh -M 0 -N -L ...`（断线自动重连）
- 训练/采样前先拉起隧道，结束可关

## 其它路径（备忘）

- 若 UCloud 安全组后续放行自定义端口（如 8000），可直接 `vllm serve --port 8000`
  走公网，隧道可省；
- 腾讯云沙箱 → 云端 Gateway 的访问路径同理（沙箱内 `api_base=http://<云端公网IP>:<端口>`，
  需确认沙箱出口能到达对应端口；必要时也走 SSH 隧道方案的服务端侧）。

## 历史：沙箱侧隧道（v0.5.0，已废弃）

实测训练机公网仅 22 开放（3389/80/443/8000 均 filtered），故 runner 改为**沙箱内起 SSH 隧道**：

- runner 把专用密钥（`/home/ubuntu/.ssh/gateway_tunnel_key`，公钥已在训练机 authorized_keys）
  注入沙箱，执行 `ssh -N -L 127.0.0.1:8000:127.0.0.1:<gateway_port> ubuntu@<公网IP>`
- 沙箱内 agent 的 `api_base = http://127.0.0.1:8000/v1`
- 环境变量：`MSA_GATEWAY_SSH_HOST`（训练机公网 IP）、`MSA_GATEWAY_SSH_KEY_PATH`、
  `MSA_GATEWAY_LOCAL_PORT`（默认 8000）

> **v0.14.0 更新**：agent-harness-outside 模式（思路 1.9）下 harness 在训练机本地，
> 直接调 `session.base_url`（本机 Gateway），**不再需要沙箱内隧道**（`MSA_GATEWAY_TUNNEL=0`）。
>
> **v0.35.1 更新**：黑盒 Claude Code 走 direct-URL——沙箱内 `ANTHROPIC_BASE_URL`
> 直指训练机公网 Gateway（`session.base_url` 去 `/v1`），腾讯沙箱在云端天然公网可达，
> 无需任何隧道（见 `uni_agent_ext/agents/claude_code_runner.py`）。
