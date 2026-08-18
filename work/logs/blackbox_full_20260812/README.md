# 黑盒（Claude Code）HumanEvalFix 正式训练归档（2026-08-12 ~ 08-13，step 1-25 + 评估）

## 说明

- 配置：train161 / batch32 / mini16 / micro4 / n=4 / 并发 64 / max_turns 60 /
  5 epoch（25 步）；step 1-5 于 2026-08-11 夜 ~ 08-12 晨完成，step 6-15 于
  08-12 10:39 续训（旧服务器），**换新服务器（117.50.178.172）后 step 15 → 25/25
  完整跑完**（2026-08-13，重启前自然结束，checkpoint 25 保存）
- **定位：特性/行为观察**（用户拍板）——不强制出最终结果，重点观察黑盒
  （Claude Code 编排 + Qwen3-8B 推理）的训练行为
- 形态：黑盒 Claude Code 经云端 Gateway 接入（Anthropic 协议适配 + MCP 工具
  转发到腾讯沙箱执行），轨迹云侧物化，成果计为平台化训练产出

## 结果摘要（step 1-25）

- 通过率（reward=1.0 占比）：0.43 ~ 0.74 波动，累计 ~60%
- 轮数：mean 16-28（白盒 11-13），部分会话吃满 60 轮 → 每步 65-110 分钟
  （白盒 30-35 分钟），gen 占 75%+
- 与白盒对比：黑盒 per-session 通过率整体高 ~15-20pp（Claude Code 编排更强），
  代价是轮数翻倍、时长翻倍、吞吐低 35%（长上下文 prefill）
- **最终评估：130/161 = 80.75%**（n=1 / temp 0.8 / 并发 24；先 3 条小样本验证
  轨迹/真实 pytest 后全量）——vs 基座 76.4%（+4.35pp）、白盒 baseline 83.2%、
  spec 82.61%；**计为平台化训练通过率**（用户定）

## 文件

| 文件 | 说明 |
| --- | --- |
| `blackbox_trajectories.tgz` | step_1-15 会话轨迹（9756 文件） |
| `humanevalfix_blackbox/` | step_16-25 会话轨迹（解包） |
| `eval_blackbox_full.json` + `eval_blackbox_full_dir/` | 全量评估结果 + 161 条轨迹 |

> 主日志：服务器 `/home/ubuntu/swe-rl/grpo_humanevalfix_blackbox.log`（1-5）+
> `grpo_humanevalfix_blackbox_resume.log`（6-15）+ `grpo_humanevalfix_blackbox_resume2.log`
> （16-25）；最终权重 `models/Qwen3-8B-final-blackbox`（16G）
