# 投机解码 GRPO 训练日志存档（2026-08-10）

## 来源

- 服务器：UCloud node1（117.50.189.37），`/home/ubuntu/swe-rl/`
- 训练 run：投机解码全样本训练（`spec_train_run.sh`，2026-08-09 21:14 ~ 2026-08-10 09:38，
  25/25 步 = 5 epoch，EAGLE-3 + LoRA merge）

## 文件

| 文件 | 说明 | sha256 |
| --- | --- | --- |
| `grpo_humanevalfix_spec.log.gz` | 完整训练日志（原始 10.8MB，gzip 压缩） | 3f4c69797c791d397aab39e74906416df34aa39ad35f618f2e9b5599ec7b950a |
| `grpo_stats_spec.jsonl` | 逐步统计（39 行，覆盖 step 1-25） | 3e2417b3ffa8891a7558ad53938693297af029340483f949523ee2c942f3d9c0 |

## 训练配置摘要

- train161 / batch 32 / mini 16 / micro 4 / 并发 64 / vllm max_num_seqs 128 / util 0.8
- 5 epoch（25 步）；`lora.merge=True`；EAGLE-3（`Qwen3-8B-speculator.eagle3`，
  num_speculative_tokens=3）；logprobs 修复（vllm#30059 一行补丁）
- 最终权重：`checkpoints/humanevalfix_spec/global_step_25` →
  `/home/ubuntu/models/Qwen3-8B-final-spec`（convert_verl_lora_to_hf.py）
- 训练收尾有 DataLoader worker OOM kill（step 25 指标与 ckpt 已落盘后，不影响结果）

## 本地完整存档

完整会话目录（25 个 step）与原始日志另存本地
`swe-rl-local/work/server_logs/`（`swe_rl_logs_spec.tar.gz` 15.8MB）。
