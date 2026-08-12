# 白盒 HumanEvalFix 全样本训练归档（baseline + spec，2026-08-08~10）

## baseline（grpo_humanevalfix_full.log，26 步 = 5 epoch + 1）

- 配置：train161 / batch32 / mini16 / micro4 / n=4 / 并发 64 / 5 epoch
- 评测：基座 76.4% → **final 83.2%**（+6.8pp，n=1 / temp 0.8 / 161 条）
- 逐步数据见 `grpo_stats_full.jsonl`；详细分析见 docs/训练评测分析.md

## spec（grpo_humanevalfix_spec.log，25 步 = 5 epoch）

- 配置同 baseline + `lora.merge=True` + EAGLE-3 投机解码
- 吞吐 **+41.7%**（199.2 → 282.4 tok/s），评测 **82.61%**（133/161）
- 期间修复 verl#7014 merge 权重物化 bug

## 文件

| 文件 | 说明 |
| --- | --- |
| `humanevalfix_trajectories.tgz` | 会话轨迹（step_1-26 + step_1-25，framework/task/trajectory） |
| `whitebox_logs.tgz` | 主训练日志（full + spec，10.8M/10.9M） |
| `grpo_stats_full.jsonl` / `grpo_stats_spec.jsonl` | 逐步统计（watcher） |

> 本地完整备份：`swe-rl-local/work/server_logs/`（解包会话目录 + 主日志 +
> `swe_rl_logs_full/spec.tar.gz`）
