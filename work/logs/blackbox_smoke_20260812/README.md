# 黑盒（Claude Code）小样本 GRPO 测试记录（2026-08-11 ~ 08-12）

## 结果

- **3/3 step 全部完成，12/12 会话成功（num_failed_sessions=0）**，reward 全 1.0
  （3 条 HumanEvalFix 样本都被 Claude Code 修好，pytest PASS）
- 每步完整执行：Claude Code 多轮工具调用（沙箱内）→ Gateway 轨迹 → GRPO 更新 →
  LoRA adapter 热插（update_weights ~2.5s）→ checkpoint 保存
- `actor/loss=0, advantage=0` 符合预期（reward 全 1.0 → 组内无差异）
- 配置：train3 × n=4 / 并发 4 / max_turns 60（v0.38.5 起）/ 隧道模式
  （CLAUDE_GATEWAY_TUNNEL=1，GATEWAY_PORT=8001 固定内网监听）

## 轨迹检查结论

- 12/12 轨迹结构完整：prompt_ids / response_ids / response_mask / response_logprobs /
  reward_info，与白盒同格式；materialization_reason=None；无测试泄露
- step_1（by_length）：7-10 轮，Read → Bash 验证 → 修复 → 提交，干净
- step_2（unique_digits）：2 条 102 轮 Edit 循环（199 段、response 11K tokens、
  无 Bash 验证盲改）→ v0.38.5 将 max_turns 100→60（对齐白盒）截断该类成本
- step_3：7 轮，干净
- logprob：mask=1 生成 token 仅 10-25% 非零（工具调用段开头几个），白盒同款；
  verl 训练默认 recompute old_log_probs（`_compute_old_log_prob`），
  **不影响训练**，只影响 debug 指标（rollout_corr）参考性

## 排障过程（5 轮，v0.37.3 → v0.38.5）

1. **v0.37.3**：`claude_code_runner` 漏 `sandbox.start()` → 4 会话全失败
   （TencentAgentRuntimeSandbox not started）
2. **v0.37.4**：SSH 隧道远端目标写死 `127.0.0.1:{port}`，Gateway 实际监听
   Ray node IP → 隧道转发目标改为 `gateway_url.hostname`（ECONNRESET 消失）
3. **v0.38.0**：回到 direct-URL 定稿（用户拍板）——`GATEWAY_PORT` 固定端口补丁
   （verl `run_uvicorn` port 参数 + gateway.py 读环境变量）；公网 8001 未放行，
   实测不可达 → 按用户指示用隧道（CLAUDE_GATEWAY_TUNNEL=1）
4. **v0.38.2**：claude-code 默认 `max_tokens=32000` 原样传给 vLLM →
   `max_model_len(16384) - 32000 = -15616` → vLLM 400（debug_launcher 定位）；
   Gateway anthropic adapter 截断 max_tokens（`GATEWAY_MAX_GENERATION_TOKENS=8192`）
5. **v0.38.4**：仍报 "response exceeded 8192"——完整模式系统提示把 input_tokens
   顶过 8192，被 claude 客户端误判；加 `--bare`（与 debug_launcher 成功路径一致）后
   首轮 4/4 reward=1.0

## 工具

- 官方 `examples/gateway/debug_launcher.py`（py3.10 补丁
  `patches/debug_launcher_py310.patch`）：fake/真实 vLLM backend + claude-code
  复现定位（无需完整训练）
- `check_trajectory.py`：轨迹 token 还原 + mask 分段检查（服务器/本地跑）
- 轨迹存档 `blackbox_trajectories.tgz`：12 session（trajectory.json + npz）

## 文件

| 文件 | 说明 |
| --- | --- |
| `blackbox_trajectories.tgz` | 12 条轨迹（step_1/2/3 各 4 会话） |
| `check_trajectory.py` | 轨迹检查脚本（Qwen3-8B tokenizer 还原） |
