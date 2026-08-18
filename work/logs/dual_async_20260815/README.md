# 双机全异步平台化训练归档（2026-08-15，step 1-25 + 评估）

## 说明

- 配置：train161 / batch32 / mini16 / micro4 / n=4 / 并发 64 / util 0.8 /
  5 epoch（25 步）——与白盒 baseline 完全同口径
- 架构：双机 `separate_async`（trainer node1 + 独立 rollout 引擎 node2）+
  TransferQueue **MooncakeStore** + EAGLE-3 投机 + 白盒 mini-swe-agent
- 结果：25 步 7:11:40、0 硬错误；**全量评估 83.23%（134/161）**，计为平台化
  训练结果
- **轨迹说明**：token 级轨迹（trajectory.json/npz）在训练时经 TQ 实时传输未落
  盘，本目录仅保留每会话 `task.log`（agent 行为）与评估轨迹
  （`eval_dual_async_final_dir/`，161 条）

## 文件

- `humanevalfix_dual_async/`：25 步会话 task.log
- `eval_dual_async_final_dir/`：161 条全量评估轨迹
- `eval_dual_async_final.json`：评估汇总（134/161 = 83.23%）
- `grpo_humanevalfix_dual_async_mooncake.log`：训练日志
- `ckpt_cleanup.log` / `convert_dual_async.log` / `mooncake_master.log`：运维记录
