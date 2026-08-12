# 黑盒（Claude Code）HumanEvalFix 正式训练归档（2026-08-12，step 1-15）

## 说明

- 配置：train161 / batch32 / mini16 / micro4 / n=4 / 并发 64 / max_turns 60 /
  5 epoch（目标 25 步）；step 1-5 于 2026-08-11 夜 ~ 08-12 晨完成，
  step 6-15 于 08-12 10:39 续训（resume 从 global_step_6）→ **止于 step 15/25**
  （服务器租用期结束前停止，checkpoint 15 已保存可续训）
- **定位：特性/行为观察**（用户拍板）——不强制出最终结果，重点观察黑盒
  （Claude Code 编排 + Qwen3-8B 推理）的训练行为
- 形态：沙箱内 claude-code（agent 在沙箱），平台化外部 agent 形态验证通过后
  留双机阶段实施

## 结果摘要（step 1-15）

- 通过率（reward=1.0 占比）：0.43 ~ 0.74 波动，累计 ~60%
- 轮数：mean 16-28（白盒 11-13），部分会话吃满 60 轮 → 每步 65-110 分钟
  （白盒 30-35 分钟），gen 占 75%+
- 与白盒对比：黑盒 per-session 通过率整体高 ~15-20pp（Claude Code 编排更强），
  代价是轮数翻倍、时长翻倍、吞吐低 35%（长上下文 prefill）

## 文件

| 文件 | 说明 |
| --- | --- |
| `blackbox_trajectories.tgz` | step_1-15 会话轨迹（framework/task/trajectory，9756 文件） |
| `grpo_stats_blackbox*.jsonl` | 逐步统计（服务器 logs/ 同步） |

> 主日志：服务器 `/home/ubuntu/swe-rl/grpo_humanevalfix_blackbox.log`（1-5）+
> `grpo_humanevalfix_blackbox_resume.log`（6-15）；checkpoint 15（16G）在
> 服务器 `checkpoints/humanevalfix_blackbox/global_step_15`
