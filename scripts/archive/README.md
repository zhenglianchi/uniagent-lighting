# 归档脚本（superseded / 一次性排障）

2026-08-21 仓库清理（v0.53.0）归档，git 历史完整保留，需要时可 `git log --follow` 找回。
这些脚本已不再被 README / deployment / ROADMAP 引用，且大多已被当前工具链取代：

| 脚本 | 归档原因 | 替代 |
|---|---|---|
| `ray_cluster_setup.sh` / `ray_cluster_restart.sh` / `ray_node_join.sh` | 无任何引用 | `bootstrap_ray_env.sh` + `fix_multinode_hosts.sh` |
| `ray_import_test.py` | 一次性 import 验证 | — |
| `setup_ssh_trust.py` / `ssh_poll_node1.py` | 一次性运维 | `ssh_ucloud.py` |
| `vllm_tunnel.sh` | 一次性 SSH 隧道 | docs/vllm_access.md 记录方案 |
| `kill_eval.sh` / `kill_train.sh` / `start_stats_watch.sh` | 一次性进程管理 | 服务器手动操作 |
| `spec_ab_run.sh` / `spec_bench_ab.py` | 投机 A/B 量化已完成（2026-08-09） | 结果见 docs/训练评测分析.md |
| `run_eval_only.sh` / `run_eval_final_spec.sh` | 旧评测包装 | `eval_spec_final.sh` / `eval_dual_async_final.sh` |
| `run_grpo_single_blackbox_ucloud.sh` | 黑盒小样本版 | `run_grpo_humanevalfix_blackbox_ucloud.sh` |
| `run_grpo_single_mooncake_ucloud.sh` | Mooncake 不单跑（2026-08-11 决策） | `run_grpo_dual_async_mooncake_ucloud.sh` |
| `run_grpo_multinode_ucloud.sh` | 同步版双机，被全异步取代 | `run_grpo_dual_async_mooncake_ucloud.sh`（正式） |
| `offline_mooncake_verify.py` / `repro_tq_mooncake.py` | 13B bug 排障专用（已定论） | docs/训练评测分析.md §7.6 |
